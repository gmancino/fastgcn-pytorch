#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create the GCN model (w/ sampling abilities)

    In this file, we assume that instead of given an edge list
    to represent the connectivity, we are given a sparse matrix
"""

# Import files
import time
import torch
import typing
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch_geometric.utils as utils

# Import custom files
from models.custom_gcn_layers import GCNLayer
from models.helper_functions import *


# Create the class
class FastGCN(nn.Module):
    """
    FastGCN model from Chen, et. al.
    """

    def __init__(self, input_dim: int, hidden_dims: typing.List[int], output_dim: int,
                 dropout: float, samp_probs: np.array = None, num_nodes: int = None,
                 device: typing.Optional = torch.device("cpu")):

        # Declare super
        super().__init__()

        # Save the device
        self.device = device

        # Check hidden dims
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        else:
            pass

        # Set up the layers
        layer_list = [GCNLayer(in_channels=input_dim, out_channels=hidden_dims[0], device=self.device)]
        for i in range(len(hidden_dims) - 1):
            layer_list.append(GCNLayer(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1], device=self.device))
        layer_list.append(GCNLayer(in_channels=hidden_dims[-1], out_channels=output_dim, device=self.device))

        # Create a module list
        self.layers = nn.ModuleList(layer_list).to(self.device)

        # Set activation functions and dropout
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

        # Save the sampler
        self.samp_probs = samp_probs
        self.num_nodes = num_nodes
        # self.tensor_probs = torch.tensor(samp_probs).to(self.device).float().unsqueeze(1) / sum(samp_probs)

        # Save the global adjacency matrix for full batch GCN
        self.full_adj = None
        self.precompute = None

    def forward(self, x: torch.Tensor,
                csr_mat: sp.csr_matrix,
                drop: bool = False,
                stochastic: bool = False,
                batch_sizes: typing.List[int] = None,
                possible_training_nodes: list = None) -> tuple:
        """Forward pass of the model"""

        # One way is to perform full pass
        if stochastic is False:

            # Create adjacency matrix
            if self.full_adj is None:
                self.full_adj = csr_to_torch_coo(csr_mat).to(self.device)

            # Perform pre-computation
            if self.precompute is None:
                self.precompute = torch.sparse.mm(self.full_adj, x)

            # No sampling is performed
            init_batch = None

            # Loop through the parameters
            for ind, p in enumerate(self.layers):

                # Check index:
                if ind == 0:

                    # Dropout and activation
                    if drop:
                        x = self.drop(self.activation(p.precomputed_forward(self.precompute)))
                    else:
                        x = self.activation(p.precomputed_forward(self.precompute))

                elif 0 < ind < (len(self.layers) - 1):

                    # Dropout and activation
                    if drop:
                        x = self.drop(self.activation(p(x, self.full_adj)))
                    else:
                        x = self.activation(p(x, self.full_adj))
                else:

                    # Softmax
                    x = self.final_activation(p(x, self.full_adj))

        # Another is sampling using FastGCN
        else:

            # First get one batch
            init_batch = np.random.choice(possible_training_nodes,
                                        size=batch_sizes[0],
                                        replace=False)

            # Then compute the subgraphs
            batch_adjs = self.get_subgraphs_concated_sampling(init_batch=init_batch,
                                                     batch_sizes=batch_sizes[1:],
                                                     csr_adj_mat=csr_mat)

            # Perform precomputation
            if self.precompute is None:

                # Create adjacency matrix
                if self.full_adj is None:
                    self.full_adj = csr_to_torch_coo(csr_mat).to(self.device)

                # Save precomputation
                self.precompute = torch.sparse.mm(self.full_adj, x)

            # Propagate through the network
            for ind, p in enumerate(self.layers):

                # ALWAYS perform precomputation
                if ind == 0:

                    # Dropout and activation
                    if drop:
                        x = self.drop(self.activation(p.precomputed_forward(self.precompute[batch_adjs[ind]])))

                    else:
                        x = self.activation(p.precomputed_forward(self.precompute[batch_adjs[ind]]))

                # Final layer
                elif ind == (len(self.layers) - 1):

                    # Softmax
                    x = self.final_activation(p(x, batch_adjs[ind]))

                # Check index:
                else:

                    # Dropout and activation
                    if drop:
                        x = self.drop(self.activation(p(x, batch_adjs[ind])))

                    else:
                        x = self.activation(p(x, batch_adjs[ind]))

            # Return the result
        return x, init_batch

    @torch.no_grad()
    def get_subgraphs(self, init_batch: typing.List[int], batch_sizes: typing.List[int],
                    csr_adj_mat: sp.csr_matrix) -> list:
        """Here we sample from just the neighbors of the current nodes"""

        # Create a dummy version of the batch_sizes
        batch_sizes.insert(0, 0)

        # Get the initial batch
        batch = [init_batch]
        adj_out_mats = []

        # Get the next layer of nodes
        new_nodes = np.unique(csr_adj_mat[batch[0], :].nonzero()[1])

        # Get only the subset of nodes that are 1-hop away
        new_nodes = np.setdiff1d(new_nodes, batch[0])

        # Loop over the remaining batches
        for i in range(1, len(batch_sizes)):

            # Save denominator to avoid re-computing
            denom = sum(self.samp_probs[new_nodes])

            # Save the probs
            probs = self.samp_probs[new_nodes] / denom

            # Get a batch
            batch.append(np.random.choice(new_nodes, size=batch_sizes[i],
                                                       replace=False, p=probs))

            # Create the probability distribution over these nodes
            subgraph_probs = self.samp_probs[batch[i]] / denom

            # Save adjmats
            adj_out_mats.append(csr_to_torch_coo(csr_adj_mat[batch[i - 1], :][:, batch[i]].multiply(1. / (subgraph_probs * len(subgraph_probs)))).to(self.device))

            # Get the next layer of nodes
            new_nodes = np.unique(csr_adj_mat[batch[i], :].nonzero()[1])

            # Get only the subset of nodes that are 1-hop away
            new_nodes = np.setdiff1d(new_nodes, batch[i])

        # Initial layer of the adjacency matrix involves pre-computation,
        # but we already store this, so we only need to slice!
        adj_out_mats.append(batch[-1])

        return adj_out_mats[::-1]

    @torch.no_grad()
    def get_subgraphs_unioned_sampling(self, init_batch: typing.List[int], batch_sizes: typing.List[int],
                      csr_adj_mat: sp.csr_matrix) -> list:
        """Here, we sample from the set of current nodes UNIONED with it's 1-hop neighbors"""

        # Create a dummy version of the batch_sizes
        batch_sizes.insert(0, 0)

        # Get the initial batch
        batch = [init_batch]
        adj_out_mats = []

        # Get the next layer of nodes
        new_nodes = np.unique(csr_adj_mat[batch[0], :].nonzero()[1])

        # Loop over the remaining batches
        for i in range(1, len(batch_sizes)):
            # Save denominator to avoid re-computing
            denom = sum(self.samp_probs[new_nodes])

            # Save the probs
            probs = self.samp_probs[new_nodes] / denom

            # Get a batch
            batch.append(np.random.choice(new_nodes, size=batch_sizes[i],
                                          replace=False, p=probs))

            # Create the probability distribution over these nodes
            subgraph_probs = self.samp_probs[batch[i]] / denom

            # Save adjmats
            adj_out_mats.append(csr_to_torch_coo(
                csr_adj_mat[batch[i - 1], :][:, batch[i]].multiply(1. / (subgraph_probs * len(subgraph_probs)))).to(
                self.device))

            # Get the next layer of nodes
            new_nodes = np.unique(csr_adj_mat[batch[i], :].nonzero()[1])

        # Initial layer of the adjacency matrix involves pre-computation,
        # but we already store this, so we only need to slice!
        adj_out_mats.append(batch[-1])

        return adj_out_mats[::-1]

    @torch.no_grad()
    def get_subgraphs_concated_sampling(self, init_batch: typing.List[int], batch_sizes: typing.List[int],
                                       csr_adj_mat: sp.csr_matrix) -> list:
        """Here, we sample from the set of 1-hop neighbors and always include the base nodes"""

        # Create a dummy version of the batch_sizes
        batch_sizes.insert(0, 0)

        # Get the initial batch
        batch = [init_batch]
        adj_out_mats = []

        # Get the next layer of nodes
        all_next_nodes = np.unique(csr_adj_mat[batch[0], :].nonzero()[1])

        # Get only the subset of nodes that are 1-hop away
        new_nodes = np.setdiff1d(all_next_nodes, batch[0])
        old_nodes = np.setdiff1d(all_next_nodes, new_nodes)

        # Loop over the remaining batches
        for i in range(1, len(batch_sizes)):

            # Save denominator to avoid re-computing
            denom = sum(self.samp_probs[new_nodes])

            # Save the probs
            probs = self.samp_probs[new_nodes] / denom

            # Get a batch
            sampled = np.random.choice(new_nodes, size=min(batch_sizes[i], len(probs)),
                                          replace=False, p=probs)

            batch.append(np.concatenate((sampled, old_nodes)))

            # Create the probability distribution over these nodes
            subgraph_probs = np.concatenate((self.samp_probs[sampled] * len(sampled) / denom, np.ones(old_nodes.shape)))

            # Save adjmats
            adj_out_mats.append(csr_to_torch_coo(
                csr_adj_mat[batch[i - 1], :][:, batch[i]].multiply(1. / subgraph_probs)).to(
                self.device))

            # Get the next layer of nodes
            all_next_nodes = np.unique(csr_adj_mat[sampled, :].nonzero()[1])

            # Get only the subset of nodes that are 1-hop away
            new_nodes = np.setdiff1d(all_next_nodes, sampled)
            old_nodes = np.setdiff1d(all_next_nodes, new_nodes)

        # Initial layer of the adjacency matrix involves pre-computation,
        # but we already store this, so we only need to slice!
        adj_out_mats.append(batch[-1])

        return adj_out_mats[::-1]

    @torch.no_grad()
    def get_subgraphs_union_init_batch_sampling(self, init_batch: typing.List[int], batch_sizes: typing.List[int],
                                        csr_adj_mat: sp.csr_matrix) -> list:
        """Here, we sample from the set of 1-hop neighbors and union that with the original training nodes"""

        # Create a dummy version of the batch_sizes
        batch_sizes.insert(0, 0)

        # Get the initial batch
        batch = [init_batch]
        adj_out_mats = []

        # Get the next layer of nodes
        all_next_nodes = np.unique(csr_adj_mat[batch[0], :].nonzero()[1])

        # Get only the subset of nodes that are 1-hop away
        new_nodes = np.setdiff1d(all_next_nodes, batch[0])

        # Loop over the remaining batches
        for i in range(1, len(batch_sizes)):

            # Save denominator to avoid re-computing
            denom = sum(self.samp_probs[new_nodes])

            # Save the probs
            probs = self.samp_probs[new_nodes] / denom

            # Get a batch
            sampled = np.random.choice(new_nodes, size=min(batch_sizes[i], len(probs)),
                                       replace=False, p=probs)

            batch.append(np.concatenate((sampled, batch[0])))

            # Create the probability distribution over these nodes
            subgraph_probs = np.concatenate((self.samp_probs[sampled] * len(sampled) / denom, np.ones(batch[0].shape)))

            # Save adjmats
            adj_out_mats.append(csr_to_torch_coo(
                csr_adj_mat[batch[i - 1], :][:, batch[i]].multiply(1. / subgraph_probs)).to(
                self.device))

            # Get the next layer of nodes
            all_next_nodes = np.unique(csr_adj_mat[batch[i], :].nonzero()[1])

            # Get only the subset of nodes that are 1-hop away
            new_nodes = np.setdiff1d(all_next_nodes, batch[i])

        # Initial layer of the adjacency matrix involves pre-computation,
        # but we already store this, so we only need to slice!
        adj_out_mats.append(batch[-1])

        return adj_out_mats[::-1]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, csr_mat: sp.csr_matrix) -> tuple:
        """Predict the class for a given set of points"""

        # Forward pass without dropout
        x, _ = self.forward(x, csr_mat, drop=False)

        # Find maximum value for class prediction
        pred = torch.argmax(x, dim=1)

        return pred, x

    @torch.no_grad()
    def sample_predict(self, x: torch.Tensor,
                csr_mat: sp.csr_matrix,
                init_batch: typing.List[int],
                batch_sizes: typing.List[int],
                num_inference_times: int = 1) -> tuple:
        """Perform inference using sampled nodes"""

        # Loop over the different attempts
        final_res = 0

        # Do a stochastic forward pass
        for i in range(num_inference_times):

            # Then compute the subgraphs
            batch_adjs = self.get_subgraphs(init_batch=init_batch,
                                            batch_sizes=batch_sizes[1:],
                                            csr_adj_mat=csr_mat)

            # Propagate through the network
            for ind, p in enumerate(self.layers):

                # ALWAYS perform precomputation
                if ind == 0:

                    # Activation
                    out = self.activation(p.precomputed_forward(self.precompute[batch_adjs[ind]]))

                # Final layer
                elif ind == (len(self.layers) - 1):

                    # Softmax
                    out = self.final_activation(p(out, batch_adjs[ind]))

                # Check index:
                else:

                    # Activation
                    out = self.activation(p(out, batch_adjs[ind]))

            # Save final results
            final_res += out

        # Scale
        final_res = final_res / num_inference_times

        # Find maximum value for class prediction
        pred = torch.argmax(final_res, dim=1)

        return pred, final_res, init_batch
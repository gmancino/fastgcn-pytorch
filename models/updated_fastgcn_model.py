#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create the GCN model (w/ sampling abilities)

    In this file, we assume that instead of given an edge list
    to represent the connectivity, we are given a sparse matrix

    Utilize batch normalization from https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
    to ensure competitiveness
"""

# Import files
import os
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
class FastGCNv2(nn.Module):
    """
    FastGCN model from Chen, et. al.
    """

    def __init__(self, input_dim: int, hidden_dims: typing.List[int], output_dim: int,
                 csr_mat: sp.csr_matrix, x: torch.Tensor,
                 use_batch_norms: bool = False,
                 dropout: float = 0.0, samp_probs: np.array = None,
                 device: typing.Optional = torch.device("cpu"),
                 dataset_name: str = "", save_path: str = ""):

        # Declare super
        super().__init__()

        # Save the device
        self.device = device
        self.use_batch_norms = use_batch_norms

        # Check hidden dims
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        else:
            pass

        # Set up the layers
        layer_list = [GCNLayer(in_channels=input_dim, out_channels=hidden_dims[0], device=self.device)]
        batch_norms = [torch.nn.BatchNorm1d(hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layer_list.append(GCNLayer(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1], device=self.device))
            batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[i+1]))
        layer_list.append(GCNLayer(in_channels=hidden_dims[-1], out_channels=output_dim, device=self.device))

        # Create a module list
        self.layers = nn.ModuleList(layer_list).to(self.device)
        self.batch_norms = nn.ModuleList(batch_norms).to(self.device)

        # Set activation functions and dropout
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

        # Save the sampler
        self.samp_probs = samp_probs

        # Save the global adjacency matrix for full batch GCN
        if dataset_name == 'ogbn-products':

            # Check for pre-computation matrix
            try:
                self.precompute = torch.load(f"{save_path}/{dataset_name}_precompute.pt")
                self.full_adj = None # IF YOUR GPU HAS ENOUGH MEMORY, CHANGE TO: csr_to_torch_coo(csr_mat).to(self.device)
            except:
                self.full_adj = csr_to_torch_coo(csr_mat)
                self.precompute = torch.sparse.mm(self.full_adj, x)
                torch.save(self.precompute, f"{save_path}/{dataset_name}_precompute.pt")

        else:
            self.full_adj = csr_to_torch_coo(csr_mat).to(self.device)
            self.precompute = torch.sparse.mm(self.full_adj, x)

        # Save the random generator
        self.rng = np.random.default_rng()

    def forward(self, x: torch.Tensor,
                csr_mat: sp.csr_matrix,
                drop: bool = False,
                stochastic: bool = False,
                batch_sizes: typing.List[int] = None,
                possible_training_nodes: list = None) -> tuple:
        """Forward pass of the model"""

        # One way is to perform full pass
        if stochastic is False:

            # No sampling is performed
            init_batch = None

            # Loop through the parameters
            for ind, p in enumerate(self.layers):

                # Check index:
                if ind == 0:

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p.precomputed_forward(self.precompute)) if self.use_batch_norms else p.precomputed_forward(self.precompute)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p.precomputed_forward(self.precompute)) if self.use_batch_norms else p.precomputed_forward(self.precompute)
                        x = self.activation(x)

                elif 0 < ind < (len(self.layers) - 1):

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p(x, self.full_adj)) if self.use_batch_norms else p(x, self.full_adj)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p(x, self.full_adj)) if self.use_batch_norms else p(x, self.full_adj)
                        x = self.activation(x)
                else:

                    # Softmax
                    x = self.final_activation(p(x, self.full_adj))

        # Another is sampling using FastGCN
        else:

            # First get one batch
            init_batch = self.rng.choice(possible_training_nodes,
                                        size=batch_sizes[0],
                                        replace=False)

            # Then compute the subgraphs
            batch_adjs = self.get_subgraphs_concated_sampling(init_batch=init_batch,
                                                     batch_sizes=batch_sizes[1:],
                                                     csr_adj_mat=csr_mat)

            # Propagate through the network
            for ind, p in enumerate(self.layers):

                # ALWAYS perform precomputation
                if ind == 0:

                    # Only put the current slice on the machine
                    data = self.precompute[batch_adjs[ind]].to(self.device)

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p.precomputed_forward(data)) if self.use_batch_norms else p.precomputed_forward(data)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p.precomputed_forward(data)) if self.use_batch_norms else p.precomputed_forward(data)
                        x = self.activation(x)

                # Final layer
                elif ind == (len(self.layers) - 1):

                    # Softmax
                    x = self.final_activation(p(x, batch_adjs[ind]))

                # Check index:
                else:

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p(x, batch_adjs[ind])) if self.use_batch_norms else p(x, batch_adjs[ind])
                        x = self.drop(self.activation(x))

                    else:
                        x = self.batch_norms[ind](p(x, batch_adjs[ind])) if self.use_batch_norms else p(x, batch_adjs[ind])
                        x = self.activation(x)

        # Return the result
        return x, init_batch

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
        all_next_nodes = np.unique(csr_adj_mat[batch[0], :].indices)

        # Get only the subset of nodes that are 1-hop away
        new_nodes = np.setdiff1d(all_next_nodes, batch[0])
        old_nodes = batch[0].copy()

        # Loop over the remaining batches
        for i in range(1, len(batch_sizes)):

            # Save denominator to avoid re-computing
            denom = sum(self.samp_probs[new_nodes])

            # Save the probs
            probs = self.samp_probs[new_nodes] / denom

            # Get a batch
            sampled = self.rng.choice(new_nodes, size=min(batch_sizes[i], len(probs)),
                                          replace=False, p=probs)

            batch.append(np.concatenate((sampled, old_nodes)))

            # Create the probability distribution over these nodes
            subgraph_probs = np.concatenate((self.samp_probs[sampled] * len(sampled) / self.samp_probs[sampled].sum(), np.ones(old_nodes.shape)))

            # Save adjmats
            adj_out_mats.append(csr_to_torch_coo(
                csr_adj_mat[batch[i - 1], :][:, batch[i]].multiply(1. / subgraph_probs)).to(
                self.device))

            # Get the next layer of nodes
            all_next_nodes = np.unique(csr_adj_mat[sampled, :].indices)

            # Get only the subset of nodes that are 1-hop away
            new_nodes = np.setdiff1d(all_next_nodes, sampled)
            old_nodes = sampled.copy()

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
            batch_adjs = self.get_subgraphs_concated_sampling(init_batch=init_batch,
                                            batch_sizes=batch_sizes[1:],
                                            csr_adj_mat=csr_mat)

            # Propagate through the network
            for ind, p in enumerate(self.layers):

                # ALWAYS perform precomputation
                if ind == 0:

                    # Activation
                    out = self.batch_norms[ind](p.precomputed_forward(self.precompute[batch_adjs[ind]])) if self.use_batch_norms else p.precomputed_forward(
                        self.precompute[batch_adjs[ind]])
                    out = self.activation(out)

                # Final layer
                elif ind == (len(self.layers) - 1):

                    # Softmax
                    out = self.final_activation(p(out, batch_adjs[ind]))

                # Check index:
                else:

                    # Activation
                    out = self.batch_norms[ind](p(out, batch_adjs[ind])) if self.use_batch_norms else p(out, batch_adjs[ind])
                    out = self.activation(out)

            # Save final results
            final_res += out

        # Scale
        final_res = final_res / num_inference_times

        # Find maximum value for class prediction
        pred = torch.argmax(final_res, dim=1)

        return pred, final_res, init_batch
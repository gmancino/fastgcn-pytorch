#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create the GCN model (w/ sampling abilities)

    In this file, we assume that instead of given an edge list
    to represent the connectivity, we are given a sparse matrix
"""

# Import files
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
        self.prob_mask = torch.zeros(self.num_nodes).to(self.device)
        self.mask = torch.zeros(self.num_nodes).to(self.device)

        # Save the global adjacency matrix for full batch GCN
        self.full_adj = None

    def forward(self, x: torch.Tensor,
                edge_list: torch.Tensor,
                csr_mat: sp.csr_matrix,
                drop: bool = True,
                stochastic: bool = False,
                batch_sizes: typing.List[int] = None,
                possible_training_nodes: list = None) -> tuple:
        """Forward pass of the model"""

        # One way is to perform full pass
        if stochastic is False:

            # Create adjacency matrix
            if self.full_adj is None:
                self.full_adj = csr_to_torch_coo(csr_mat).to(self.device)

            # No sampling is performed
            init_batch = None

            # Loop through the parameters
            for ind, p in enumerate(self.layers):

                # Check index:
                if ind < (len(self.layers) - 1):

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
            batch_adjs = self.get_subgraphs(init_batch=init_batch,
                                                     batch_sizes=batch_sizes[1:],
                                                     edge_list=edge_list,
                                                     csr_adj_mat=csr_mat)

            # Propagate through the network
            for ind, p in enumerate(self.layers):

                # ALWAYS perform precomputation
                if ind == 0:

                    # Dropout and activation
                    if drop:
                        x = self.drop(self.activation(p(x, batch_adjs[ind])))

                    else:
                        x = self.activation(p(x, batch_adjs[ind]))

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

    def get_subgraphs(self, init_batch: torch.Tensor, batch_sizes: typing.List[int],
                    edge_list: torch.Tensor, csr_adj_mat: sp.csr_matrix) -> list:
        """Get samples from each layer and create a mask of the nodes"""

        # Create a dummy version of the batch_sizes
        batch_sizes.insert(0, 0)

        # Get the initial batch
        batch = [init_batch]
        adj_out_mats = []

        # # Get the next layer of nodes
        new_nodes, _, _, _ = utils.k_hop_subgraph(node_idx=torch.tensor(batch[0]), num_hops=1, edge_index=edge_list)

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
            curr_adj = csr_adj_mat[batch[i - 1], :]
            curr_adj = curr_adj[:, batch[i]].multiply(1. / (subgraph_probs * len(subgraph_probs)))
            adj_out_mats.append(csr_to_torch_coo(curr_adj).to(self.device))

            # Get the next layer of nodes
            new_nodes, _, _, _ = utils.k_hop_subgraph(node_idx=torch.tensor(batch[i]), num_hops=1, edge_index=edge_list)

        # Initial layer of the adjacency matrix involves pre-computation:
        adj_out_mats.append(csr_to_torch_coo(csr_adj_mat[batch[-1], :]).to(self.device))

        return adj_out_mats[::-1]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_list: torch.Tensor, csr_mat: sp.csr_matrix, ) -> tuple:
        """Predict the class for a given set of points"""

        # Forward pass without dropout
        x, _ = self.forward(x, edge_list, csr_mat, drop=False)

        # Find maximum value for class prediction
        pred = torch.argmax(x, dim=1)

        return pred, x
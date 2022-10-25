#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create custom GCN layers
"""

# Import files
import math
import torch
import typing
import torch.nn as nn


# Create the class
class GCNLayer(nn.Module):
    """
    Custom GCN Layer
    """
    def __init__(self, in_channels: int, out_channels: int, device: typing.Optional = torch.device("cpu")):
        super().__init__()

        # Save the device
        self.device = device

        # Create the linear layer with bias always included
        weights = torch.Tensor(in_channels, out_channels).to(self.device)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(out_channels).to(self.device)
        self.bias = nn.Parameter(bias)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the weights"""

        # Initialize the weights
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_mat: torch.sparse.Tensor) -> torch.Tensor:
        """Do a forward pass of the network"""

        return torch.mm(torch.sparse.mm(adj_mat, x), self.weights) + self.bias

    def precomputed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass of the network, under the assumption that AX has already been computed"""

        return torch.mm(x, self.weights) + self.bias
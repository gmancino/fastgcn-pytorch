#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Helper functions for the memory efficient implementation of FastGCN
"""

# Import files
import torch
import typing
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch_geometric.utils as utils


# Declare some helper functions
def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    Copied from: https://github.com/matenure/FastGCN/blob/b8e6e6412d8cb696e0adb7910832ed1ae7243ac9/utils.py#L136
    """

    # Convert to COO
    if sp.isspmatrix_coo(adj) is False:
        adj = sp.coo_matrix(adj)

    # Get the degree
    rowsum = np.array(adj.sum(1))

    # Compute the normalizing factor
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Return CSR matrix
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()


# Declare some helper functions
def csr_to_torch_coo(adj):
    """
    Convert csr sparse matrix to coo sparse tensor
    """

    # Convert to COO
    if sp.isspmatrix_coo(adj) is False:
        adj = sp.coo_matrix(adj)

    # Get the indices
    inds = np.array([adj.row, adj.col])

    # Return COO tensor matrix
    return torch.sparse_coo_tensor(indices=inds, values=adj.data, size=adj.shape, dtype=torch.float32)

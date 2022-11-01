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

# Try micro F1 score
from sklearn.metrics import f1_score as F1


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
def row_normalize(mat):
    """
    Row normalize the given matrix mat by its row norm
    """

    # Get the row sum
    inv_r_sum = 1. / torch.norm(mat, dim=1, p=2)
    inv_r_sum[inv_r_sum == torch.inf] = 0.

    # Return normalized matrix
    return mat * inv_r_sum.unsqueeze(1)


# Declare some helper functions
def csr_to_torch_coo(adj):
    """
    Convert csr sparse matrix to coo sparse tensor
    """

    # Convert to COO
    if sp.isspmatrix_coo(adj) is False:
        adj = sp.coo_matrix(adj)

    # Return COO tensor matrix
    return torch.sparse_coo_tensor(indices=np.array([adj.row, adj.col]), values=adj.data, size=adj.shape, dtype=torch.float32)


# Declare a training function
def train(model: nn.Module, opt: torch.optim, X: torch.Tensor, y: torch.Tensor,
        csr_mat: sp.csr_matrix,
          training_mask: torch.Tensor, loss_function: nn.Module, stoch: bool, losses: list,
          batch_list: list = None, training_indices: list = None) -> list:

    # Set to train
    model.train()
    opt.zero_grad()

    # Forward pass
    out, init_batch = model(x=X, csr_mat=csr_mat, drop=False, stochastic=stoch,
                            batch_sizes=batch_list, possible_training_nodes=training_indices)

    # Make sure we are only grabbing training nodes computed from the FIRST batch
    if stoch:
        # Make training mask actually feasible
        l = loss_function(out, y[init_batch])
    else:
        l = loss_function(out[training_mask], y[training_mask])

    # Backward pass
    l.backward()
    opt.step()

    # Save the losses
    losses.append(l.item())

    return losses


# Declare a training function
def sample_validation_test(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                            csr_mat: sp.csr_matrix,
                            batch_list: list, num_samp_inf: int, validation_mask: list,
                            loss_function: nn.Module, val_losses: list) -> list:

    # Set to eval
    model.eval()
    v_loss = 0

    c = 0
    for i in range(0, len(validation_mask), batch_list[0]):

        # Get this subset of nodes to perform prediction on
        m = validation_mask[i:i + batch_list[0]]

        # Forward pass
        out, res, init_batch = model.sample_predict(x=X, csr_mat=csr_mat,
                                                    init_batch=m, batch_sizes=batch_list,
                                                    num_inference_times=num_samp_inf)

        # Save loss
        v_loss += loss_function(res, y[init_batch]).detach().item()
        c += 1

    # Save the losses
    val_losses.append(v_loss / c)

    return val_losses


# Declare a training function
def validation_test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, csr_mat: sp.csr_matrix,
          validation_mask: torch.Tensor, loss_function: nn.Module, val_losses: list) -> list:

    # Set to eval
    model.eval()

    # Get the loss
    _, out = model.predict(X, csr_mat)
    l = loss_function(out[validation_mask], y[validation_mask]).detach()

    # Save the losses
    val_losses.append(l.item())

    return val_losses


# Declare a testing function
def sample_test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, csr_mat: sp.csr_matrix,
            batch_list: list, num_samp_inf: int, mask: list, accuracy: list) -> list:

    # Set to test
    model.eval()

    # Loop over all of the indices
    total_out = np.array([])
    total_y = np.array([])
    for i in range(0, len(mask), batch_list[0]):

        # Get this subset of nodes to perform prediction on
        m = mask[i:i + batch_list[0]]

        # Forward pass
        out, res, init_batch = model.sample_predict(x=X, csr_mat=csr_mat,
                            init_batch=m, batch_sizes=batch_list, num_inference_times=num_samp_inf)

        total_out = np.concatenate((total_out, out.cpu().numpy())).flatten()
        total_y = np.concatenate((total_y, y[init_batch].cpu().numpy())).flatten()

    # Save the accuracy
    accuracy.append(F1(total_y, total_out, average='micro') * 100.0)

    return accuracy


# Declare a testing function
def test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, csr_mat: sp.csr_matrix,
          mask: torch.Tensor, accuracy: list) -> list:

    # Set to test
    model.eval()

    # Forward pass
    out, _ = model.predict(X, csr_mat)
    out = out[mask].cpu().numpy()
    y = y[mask].cpu().numpy()
    acc = F1(y, out, average='micro') * 100.0

    # Save the accuracy
    accuracy.append(acc)

    return accuracy

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Test the FastGCN algorithm from:
    https://openreview.net/pdf?id=rytstxWAW
"""

# Import files
import time
import argparse
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid

# Try micro F1 score
from sklearn.metrics import f1_score as F1

# Import custom classes
from models.fastgcn_model import FastGCN
from models.helper_functions import *

# Declare colors
cs = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000"
}


# Declare a training function
def train(model: nn.Module, opt: torch.optim, X: torch.Tensor, y: torch.Tensor, adj_list: torch.Tensor,
          csr_mat: sp.csr_matrix,
          training_mask: torch.Tensor, loss_function: nn.Module, stoch: bool, losses: list,
          batch_list: list = None, training_indices: list = None) -> list:

    # Set to train
    model.train()
    opt.zero_grad()

    # Forward pass
    out, init_batch = model(x=X, edge_list=adj_list, csr_mat=csr_mat, drop=False, stochastic=stoch,
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
def validation_test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, adj_list: torch.Tensor, csr_mat: sp.csr_matrix,
          validation_mask: torch.Tensor, loss_function: nn.Module, val_losses: list) -> list:

    # Set to eval
    model.eval()

    # Get the loss
    _, out = model.predict(X, adj_list, csr_mat)
    l = loss_function(out[validation_mask], y[validation_mask]).detach()

    # Save the losses
    val_losses.append(l.item())

    return val_losses


# Declare a training function
def test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, adj_list: torch.Tensor, csr_mat: sp.csr_matrix,
          mask: torch.Tensor, accuracy: list) -> list:

    # Set to test
    model.eval()

    # Forward pass
    out = model.predict(X, adj_list, csr_mat)
    out = out[0][mask].cpu().numpy()
    y = y[mask].cpu().numpy()
    acc = F1(y, out, average='micro') * 100.0

    # Save the accuracy
    accuracy.append(acc)

    return accuracy


# Run main
if __name__=="__main__":

    # ------------------------------------------------
    # Parse user arguments
    parser = argparse.ArgumentParser(description='Test the FastGCN paper method for node classification.')

    parser.add_argument('--epochs', type=int, default=200, help='Total number of updates rounds.')
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=16, help='Dimension of the hidden layer.')
    parser.add_argument('--sample_size', type=int, default=400, help='Sample size size.')
    parser.add_argument('--init_batch', type=int, default=256, help='Initial batch size.')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--dataset', type=str, default="Cora", choices=["Cora", "PubMed", "CiteSeer", "Reddit"],
                        help='Dataset to use.')
    parser.add_argument('--fast', type=str, default="false", choices=["true", "false"],
                        help='Use FastGCN or regular GCN.')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate.')
    parser.add_argument('--samp_dist', type=str, default='importance', choices=['importance', 'uniform'],
                        help='Which sampling distribution to use.')
    parser.add_argument('--save_results', type=int, default=0, choices=[0, 1],
                        help='Save results or not (0 = do NOT save, 1 = save).')

    args = parser.parse_args()

    # ------------------------------------------------
    # Get the device
    user_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args.fast = True if args.fast == 'true' else False

    # ------------------------------------------------
    # Load the data - ToUndirected ensures that we can scan edge_list[0, :] to get all of the neighbors
    dataset = Reddit(root='data', transform=T.ToSparseTensor(remove_edge_index=False)) \
        if args.dataset == 'Reddit' \
        else Planetoid(root='data', name=args.dataset, transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]

    # Gather the variables and responses
    X = data.x.to(user_device)
    y = data.y.to(user_device)

    # Normalize the rows
    if args.dataset != "Reddit":
        row_norms = torch.norm(X, dim=1, p=2)
        X = X * (1. / row_norms.unsqueeze(1))

    # Account for CiteSeer
    if args.dataset == 'CiteSeer':
        nan_correction = torch.where(torch.isnan(X))[0]
        X[nan_correction] = torch.zeros((nan_correction.shape[0], X.shape[1])).to(user_device)

    # Create the adjacency matrix
    data.edge_index = utils.add_self_loops(data.edge_index)[0]
    numpy_edges = data.edge_index.numpy()
    csr_edge_list = sp.csr_matrix((np.ones(data.edge_index.shape[1]), # data
                                   (numpy_edges[0], numpy_edges[1])), # (row, col)
                                  shape=(X.shape[0], X.shape[0])) # size

    # Normalize the features
    adjmat = normalize_adj(csr_edge_list)

    # ------------------------------------------------
    # Declare the model and optimizer
    model = FastGCN(input_dim=X.shape[1], hidden_dims=args.hidden_dim, output_dim=max(y).item() + 1, dropout=args.drop,
                samp_probs=np.ones((len(y),)) if args.samp_dist == 'uniform' else np.asarray(adjmat.multiply(adjmat).sum(1)).flatten(),
                num_nodes=len(y),
                device=user_device
                )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-4)
    criteria = nn.NLLLoss(reduction='mean')
    batches = [args.init_batch] + ([args.sample_size] * (len(model.layers) - 1))

    # ------------------------------------------------
    # Save meaningful results
    loss_hist = []
    val_hist = []
    test_acc = []

    # ------------------------------------------------
    # Train the model
    print(f"{'=' * 25} STARTING TRAINING {'=' * 25}")
    print(f"TRAINING INFORMATION:")
    print(f"[DATA] {args.dataset} dataset")
    print(f"[FAST] using FastGCN? {args.fast}")
    print(f"[DEV] device: {user_device}")
    print(f"[ITERS] performing {args.epochs} Adam updates")
    print(f"[LR] Adam learning rate: {args.lr}")
    print(f"[BATCH] batch size: {args.init_batch}")
    print(f"[SAMP] layer sample size: {args.sample_size}\n")

    # Set the training masks
    stochastic = args.fast
    training_mask = torch.tensor([True] * len(y))
    training_mask = training_mask * (data.test_mask == False) * (data.val_mask == False)
    training_mask = training_mask.to(user_device)
    training_indices = torch.where(training_mask == True)[0].tolist()

    # Perform the for loop over the iterations
    running_time = 0
    for i in range(args.epochs):
        # Perform training
        t0 = time.time()
        loss_hist = train(model, optimizer, X, y, data.edge_index, adjmat, training_mask, criteria, stochastic, loss_hist,
                        batches, training_indices)
        running_time += time.time() - t0

        # Perform testing
        test_acc = test(model, X, y, data.edge_index, adjmat, data.test_mask, test_acc)

        # Check validation:
        val_hist = validation_test(model, X, y, data.edge_index, adjmat, data.val_mask, criteria, val_hist)

        # Check the validation performance
        count = 0
        for v in val_hist[-10:-1]:
            if val_hist[-1] >= v:
                count += 1

        if count == 9:
            print(f"[STOP] early stopping at iteration: {i}\n")
            break

    # Print some results
    print(f"RESULTS:")
    print(f"[LOSS] minimum loss: {min(loss_hist)}")
    print(f"[ACC] maximum micro F1 testing accuracy: {max(test_acc)} %")
    print(f"[TIME] {round(running_time, 4)} seconds")
    print(f"{'=' * 26} ENDING TRAINING {'=' * 26}\n")

    # ------------------------------------------------
    # Make plots
    # Loss
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    # Make border non-encompassing
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot results_at_avg
    ax.plot(loss_hist, color=cs["blue"], alpha=1.0, lw=4.0, label='train loss')
    ax.plot(val_hist, color=cs["red"], alpha=1.0, lw=4.0, label='val loss')
    # Change axis
    ax.set_xlabel('Iteration', labelpad=5)
    ax.set_ylabel(f'Log(loss)', labelpad=5)
    plt.yscale('log')
    # Set limits
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.show()
    if args.save_results == 1:
        plt.savefig(f"results/train_loss.png", transparent=True, bbox_inches='tight', pad_inches=0)

    # Test accuracy
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    # Make border non-encompassing
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot results_at_avg
    ax.plot(test_acc, color=cs["orange"], alpha=1.0, lw=4.0, label='FastGCN')
    # Change axis
    ax.set_xlabel('Iteration', labelpad=5)
    ax.set_ylabel(f'Test accuracy', labelpad=5)
    # Set limits
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.show()
    if args.save_results == 1:
        plt.savefig(f"results/testing_accuracy.png", transparent=True, bbox_inches='tight', pad_inches=0)

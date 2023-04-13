#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Test the FastGCN algorithm from:
    https://openreview.net/pdf?id=rytstxWAW
"""

# Import files
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.utils as utils
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

# Import custom classes
from models.helper_functions import *
from models.fastgcn_model import FastGCN
from models.updated_fastgcn_model import FastGCNv2

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

# Run main
if __name__=="__main__":

    # ------------------------------------------------
    # Parse user arguments
    parser = argparse.ArgumentParser(description='Test the FastGCN paper method for node classification.')

    # DATA
    parser.add_argument('--dataset', type=str, default="Cora", choices=["Cora", "PubMed", "CiteSeer", "Reddit",
                                                                        "ogbn-arxiv", "ogbn-products"],
                        help='Dataset to use.')
    parser.add_argument('--norm_feat', type=str, default='false', choices=['true', 'false'],
                        help='Normalized features?')
    parser.add_argument('--batch_norm', type=str, default='false', choices=['true', 'false'],
                        help='Use batch normalization?')
    parser.add_argument('--report', type=int, default=1, help='How often to report accuracies; for bigger data'
                                                              'it may be better to take more GD steps first.')

    # METHOD + ARCHITECTURE
    parser.add_argument('--fast', type=str, default="true", choices=["true", "false"],
                        help='Use FastGCN or regular GCN.')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Dimension of the hidden layer.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--init_batch', type=int, default=256, help='Initial batch size.')
    parser.add_argument('--sample_size', type=int, default=400, help='Sample size size.')
    parser.add_argument('--scale_factor', type=float, default=1, help='For deeper networks, we need more samples.')
    parser.add_argument('--samp_dist', type=str, default='importance', choices=['importance', 'uniform'],
                        help='Which sampling distribution to use.')

    # TRAINING
    parser.add_argument('--epochs', type=int, default=200, help='Total number of updates rounds.')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate.')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping term.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (l2 regularization).')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate.')

    # INFERENCE
    parser.add_argument('--samp_inference', type=str, default='false', choices=['true', 'false'],
                        help='Sample during inference phase for testing accuracy?')
    parser.add_argument('--use_val', type=str, default='true', choices=['true', 'false'],
                        help='Use the validation dataset to perform early stopping?')
    parser.add_argument('--num_samp_inference', type=int, default=1,
                        help='Number of times to sample during inference.')
    parser.add_argument('--inference_init_batch', type=int, default=256,
                        help='Initial batch size for inference.')
    parser.add_argument('--inference_sample_size', type=int, default=400, help='Sample size for inference.')

    # EXTRAS
    parser.add_argument('--use_cuda', type=str, default="true", choices=['true', 'false'],
                        help='Number of times to sample during inference.')
    parser.add_argument('--save_results', type=int, default=0, choices=[0, 1],
                        help='Save results or not (0 = do NOT save, 1 = save).')

    args = parser.parse_args()

    # ------------------------------------------------
    # Get the device
    user_device = torch.device("cuda:0") if torch.cuda.is_available() and args.use_cuda == 'true' else torch.device("cpu")
    args.fast = True if args.fast == 'true' else False
    args.batch_norm = True if args.batch_norm == 'true' else False
    args.samp_inference = True if args.samp_inference == 'true' else False
    args.use_val = True if args.use_val == 'true' else False
    args.norm_feat = True if args.norm_feat == 'true' else False
    args.early_stop = args.epochs + 1 if args.early_stop <= 0 else args.early_stop

    # Set the architecture
    args.hidden_dim = [args.hidden_dim] * args.num_layers

    # ------------------------------------------------
    # Load the data - ToUndirected ensures that we can scan edge_list[0, :] to get all of the neighbors
    if args.dataset in ['ogbn-arxiv', 'ogbn-products']:
        dataset = PygNodePropPredDataset(name=args.dataset, root='data', transform=T.ToSparseTensor())
        # Extract the data
        data = dataset[0]

        # Gather the variables and responses
        X = data.x.to(user_device)
        y = data.y.flatten().to(user_device)

        # Normalize the features
        X = row_normalize(X) if args.norm_feat else X

        # Get the adjacency matrix - save to file because to symmetric is slow
        try:
            adjmat = sp.load_npz(f"data/{args.dataset}_adjmat.npz")
        except:
            csr_edge_list = data.adj_t.to_symmetric().to_scipy().tocsr()
            csr_edge_list += sp.identity(X.shape[0], format='csr')
            # Normalize the adjacency matrix
            adjmat = normalize_adj(csr_edge_list)
            sp.save_npz(f"data/{args.dataset}_adjmat.npz", adjmat)

        # Load the fast-gcn probabilities because this calculation too is slow
        try:
            fast_gcn_probs = np.loadtxt(f"data/{args.dataset}_probs.txt")
        except:
            fast_gcn_probs = np.asarray(adjmat.multiply(adjmat).sum(1)).flatten()
            np.savetxt(f"data/{args.dataset}_probs.txt", fast_gcn_probs)

        # Get the training and testing split
        t0 = time.time()
        split_idx = dataset.get_idx_split()
        training_mask = torch.tensor([False] * len(y))
        training_mask[split_idx['train']] = True
        training_mask = training_mask.to(user_device)
        training_indices = split_idx['train'].cpu().numpy()
        testing_indices = split_idx['test'].cpu().numpy()
        validation_indices = split_idx['valid'].cpu().numpy()

        # Standardize data location
        test_mask = torch.tensor([False] * len(y))
        test_mask[testing_indices] = True
        data.test_mask = test_mask

        val_mask = torch.tensor([False] * len(y))
        val_mask[validation_indices] = True
        data.val_mask = val_mask

    else:

        dataset = Reddit(root='data', transform=T.ToSparseTensor(remove_edge_index=False)) \
            if args.dataset == 'Reddit' \
            else Planetoid(root='data', name=args.dataset, transform=T.ToSparseTensor(remove_edge_index=False))
        data = dataset[0]

        # Gather the variables and responses
        X = data.x.to(user_device)
        y = data.y.to(user_device)

        # Normalize the features
        X = row_normalize(X) if args.norm_feat else X

        # Create the adjacency matrix
        numpy_edges = data.edge_index.numpy()
        csr_edge_list = sp.csr_matrix((np.ones(data.edge_index.shape[1]), # data
                                       (numpy_edges[0], numpy_edges[1])), # (row, col)
                                      shape=(X.shape[0], X.shape[0])) # size
        csr_edge_list += sp.identity(X.shape[0], format='csr')

        # Calculate adjacency matrix and probabilities
        adjmat = normalize_adj(csr_edge_list)
        fast_gcn_probs = np.asarray(adjmat.multiply(adjmat).sum(1)).flatten()

        # Get the training and testing masks
        training_mask = torch.tensor([True] * len(y))
        training_mask = training_mask * (data.test_mask == False) * (data.val_mask == False)
        training_mask = training_mask.to(user_device)
        training_indices = torch.where(training_mask == True)[0].cpu().numpy()
        testing_indices = torch.where(data.test_mask == True)[0].cpu().numpy()
        validation_indices = torch.where(data.val_mask == True)[0].cpu().numpy()

    # ------------------------------------------------
    # Declare the model and optimizer
    model = FastGCNv2(input_dim=X.shape[1], hidden_dims=args.hidden_dim, output_dim=max(y).item() + 1, dropout=args.drop,
                    csr_mat=adjmat, x=X,
                    samp_probs=np.ones((len(y),)) if args.samp_dist == 'uniform' else fast_gcn_probs,
                    device=user_device,
                    use_batch_norms=args.batch_norm,
                    dataset_name=args.dataset,
                    save_path='data'
                    )
    print(f"Your model:\n{model}")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    criteria = nn.NLLLoss(reduction='mean')
    batches = [args.init_batch] + [min(X.shape[0], int(args.sample_size * (1 if i == 0 else args.scale_factor))) for i in range(len(model.layers) - 1)]
    inference_batches = [args.inference_init_batch] + [min(X.shape[0], int(args.inference_sample_size * (1 if i == 0 else args.scale_factor))) for i in range(len(model.layers) - 1)]

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
    print(f"[INF] using sampling for inference? {args.samp_inference}")
    print(f"[FEAT] normalized features? {args.norm_feat}")
    print(f"[DEV] device: {user_device}")
    print(f"[ITERS] performing {args.epochs} Adam updates")
    print(f"[LR] Adam learning rate: {args.lr}")

    if args.fast:
        print(f"[BATCH] batch size: {batches}")

    if args.samp_inference:
        print(f"[INF BATCH] batch size: {inference_batches}")

    # Set the training type
    stochastic = args.fast

    # Perform the for loop over the iterations
    max_acc = 0
    running_time = 0
    total_times = []
    for i in range(1, args.epochs + 1):

        # Perform training
        t0 = time.time()
        loss_hist = train(model, optimizer, X, y, adjmat, training_mask, criteria, stochastic, loss_hist,
                        batches, training_indices)
        t1 = time.time()
        running_time += t1 - t0
        if i > 0:
            total_times.append(t1 - t0)

        # Only report every few iterations
        if i % args.report == 0 and i > 1:

            # Perform testing and validation
            if args.samp_inference:

                test_acc = sample_test(model, X, y, adjmat, inference_batches, args.num_samp_inference, testing_indices, test_acc)

                # Save validation time if needed
                if args.use_val:
                    val_hist = sample_validation_test(model, X, y, adjmat, inference_batches, args.num_samp_inference, validation_indices, criteria, val_hist)
                else:
                    pass

            # No sampling
            else:

                test_acc = test(model, X, y, adjmat, data.test_mask, test_acc)

                # Save validation time if needed
                if args.use_val:
                    val_hist = validation_test(model, X, y, adjmat, data.val_mask, criteria, val_hist)
                else:
                    pass

            # Check the validation performance ONLY if we are not performing sampled inference
            max_acc = max(test_acc)
            if len(val_hist) > args.early_stop:

                if val_hist[-(args.early_stop + 1)] <= min(val_hist[-args.early_stop:]):
                    print(f"[STOP] early stopping at iteration: {i}\n")
                    break

    # Print some results
    print(f"RESULTS:")
    print(f"[LOSS] minimum loss: {min(loss_hist)}")
    print(f"[ACC] maximum micro F1 testing accuracy: {max_acc} %")
    print(f"[BATCH TIME] {round(sum(total_times) / len(total_times), 4)} seconds")
    print(f"[TOTAL TIME] {round(running_time, 4)} seconds")
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
        plt.savefig(f"results/{args.dataset}_train_loss.png", transparent=True, bbox_inches='tight', pad_inches=0)

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
        plt.savefig(f"results/{args.dataset}_testing_accuracy.png", transparent=True, bbox_inches='tight', pad_inches=0)

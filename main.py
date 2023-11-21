import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
from models.gnn import GCNModel, GraphSageModel, GATModel
from models.mlp import MLP
from utils import outputWriter, getLoaders
from train import *

# from karateclub import DeepWalk, Node2Vec
from models.hgnn import HGCNModel


def gnnTraining(args, dataset):
    """
    Train a GCN model on the given dataset

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for training
    dataset : torch_geometric.datasets.Planetoid
        Dataset to train on
    """

    # Create dataloader
    indices = torch.randperm(len(dataset[0].x))
    train_indices = indices[: int(0.8 * len(indices))]
    val_indices = indices[int(0.8 * len(indices)) : int(0.9 * len(indices))]
    test_indices = indices[int(0.9 * len(indices)) :]

    dataset[0].train_mask[:] = False
    dataset[0].train_mask[train_indices] = True
    dataset[0].val_mask[:] = False
    dataset[0].val_mask[val_indices] = True
    dataset[0].test_mask[:] = False
    dataset[0].test_mask[test_indices] = True

    print(f"Train: {dataset[0].train_mask.sum()}/ {len(dataset[0].train_mask)}")
    print(f"Val: {dataset[0].val_mask.sum()}/ {len(dataset[0].val_mask)}")
    print(f"Test: {dataset[0].test_mask.sum()}/ {len(dataset[0].test_mask)}")

    dataloader = torch_geometric.loader.DataLoader(dataset)

    input_dimension = dataset.num_features
    output_dimension = dataset.num_classes

    # Initialising csv writer
    output_writer = outputWriter(
        args.file_name, args.output_dir, column_heading="GNN Training"
    )

    # Training loop
    for i in tqdm(range(args.num_runs), desc="Run"):
        print(f"Run: {i}")

        # Initialising model and optimizer
        if args.gcn_flag:
            if args.hyperbolic_flag:
                print("Training Hyperbolic GCN")
                model = HGCNModel(
                    input_dim=input_dimension,
                    hidden_dims=[
                        args.gnn_hidden_dim,
                        args.embedding_dim,
                        args.mlp_hidden_dim,
                    ],
                    output_dim=output_dimension,
                    device=args.device,
                    model=args.hyperbolic_model,
                    dropout=args.dropout,
                    batch_norm=args.batch_norm,
                ).to(args.device)
            else:
                print("Training GCN")
                model = GCNModel(
                    input_dim=input_dimension,
                    hidden_dims=[
                        args.gnn_hidden_dim,
                        args.embedding_dim,
                        args.mlp_hidden_dim,
                    ],
                    output_dim=output_dimension,
                    device=args.device,
                    dropout=args.dropout,
                    batch_norm=args.batch_norm,
                ).to(args.device)
        elif args.graphsage_flag:
            print("Training GraphSage")
            model = GraphSageModel(
                input_dim=input_dimension,
                hidden_dims=[
                    args.gnn_hidden_dim,
                    args.embedding_dim,
                    args.mlp_hidden_dim,
                ],
                output_dim=output_dimension,
                agg_layer=args.aggregate_layer,
                device=args.device,
                dropout=args.dropout,
                batch_norm=args.batch_norm,
            ).to(args.device)
        elif args.gat_flag:
            print("Training GAT")
            model = GATModel(
                input_dim=input_dimension,
                hidden_dims=[
                    args.gnn_hidden_dim,
                    args.embedding_dim,
                    args.mlp_hidden_dim,
                ],
                output_dim=output_dimension,
                num_heads=args.num_heads,
                attention_dim=args.attention_dim,
                device=args.device,
                dropout=args.dropout,
                batch_norm=args.batch_norm,
            ).to(args.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
        )

        # Training Model
        (
            model,
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
        ) = gnnTrainLoop(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            batch_size=args.batch_size,
            patience=args.patience,
            epochs=args.num_epochs,
            verbose=args.verbose,
        )

        # Validating Model
        test_loss, test_accuracy = gnnTestLoop(
            model=model,
            dataloader=dataloader,
            device=args.device,
            verbose=args.verbose,
        )

        # Writing to csv
        output_writer.csv_append(
            [
                i,
                train_losses,
                train_accuracies,
                val_losses,
                val_accuracies,
                test_loss,
                test_accuracy,
            ]
        )

        if args.model_save:
            output_writer.model_save(model, f"{i}")

    # Writing to csv and json
    output_writer.write_csv()
    output_writer.write_json(vars(args))


def randomWalk(args, dataset):
    """
    Build a random walk based embedding on the given dataset
    and train a MLP model on it

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for training
    dataset : torch_geometric.datasets.Planetoid
        Dataset to train on
    """

    output_dimension = dataset.num_classes

    # Get embeddings
    embeddings = getRandomWalkEmbeddings(args, dataset)

    # Create dataloader
    train_loader, val_loader, test_loader = getLoaders(
        embeddings=embeddings,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialising csv writer
    output_writer = outputWriter(
        args.file_name, args.output_dir, column_heading="Random Walk Training"
    )

    # Training loop
    for i in tqdm(range(args.num_runs), desc="Run"):
        # Initialising model and optimizer
        model = MLP(
            input_dim=args.embedding_dim,
            hidden_dim=args.mlp_hidden_dim,
            output_dim=output_dimension,
        ).to(args.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )

        # Training Model
        (
            model,
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
        ) = mlpTrainLoop(
            model=model,
            dataloaders=(train_loader, val_loader),
            optimizer=optimizer,
            scheduler=scheduler,
            patience=args.patience,
            epochs=args.num_epochs,
            device=args.device,
            verbose=True,
        )

        # Validating Model
        test_loss, test_accuracy = mlpValidationLoop(
            model=model, dataloader=test_loader, device=args.device, verbose=True
        )

        # Writing to csv
        output_writer.csv_append(
            [
                i,
                train_losses,
                train_accuracies,
                val_losses,
                val_accuracies,
                test_loss,
                test_accuracy,
            ]
        )

    # Writing to csv and json
    output_writer.write_csv()
    output_writer.write_json(vars(args))

    if args.model_save:
        output_writer.save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run related arguments
    parser.add_argument("--gcn_flag", action="store_true")
    parser.add_argument("--graphsage_flag", action="store_true")
    parser.add_argument("--gat_flag", action="store_true")
    parser.add_argument("--hyperbolic_flag", action="store_true")
    parser.add_argument("--random_walk_flag", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Dataset related arguments
    parser.add_argument("--dataset", type=str, default="Cora")

    # Device related arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=1)

    # Common arguments
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--mlp_hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_norm", action="store_true")

    # GraphSage related arguments
    parser.add_argument(
        "--aggregate_layer", type=str, choices=["mean", "sum", "mlp"], default="mean"
    )

    # GAT related arguments
    parser.add_argument("--attention_dim", type=int, default=16)
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )

    # Random Walk related arguments
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument(
        "--random_walk_model",
        type=str,
        choices=["DeepWalk", "Node2Vec"],
        default="DeepWalk",
    )

    # GNN Model related arguments
    parser.add_argument("--gnn_hidden_dim", type=int, default=16)

    # Hyperbolic GNN related arguments
    parser.add_argument("--hyperbolic_model", type=str, choices=["lorentz", "poincare"])

    # Training related arguments
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    # Output related arguments
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--model_save", action="store_true")

    args = parser.parse_args()

    if args.dataset == "Cora":
        dataset = Planetoid(root="data/Planetoid", name="Cora")
    elif args.dataset == "Citeseer":
        dataset = Planetoid(root="data/Planetoid", name="Citeseer")
    elif args.dataset == "Pubmed":
        dataset = Planetoid(root="data/Planetoid", name="Pubmed")
    else:
        raise ValueError("Dataset not supported")

    if args.gcn_flag or args.graphsage_flag or args.gat_flag:
        gnnTraining(args, dataset)
    elif args.random_walk_flag:
        print("Training Random Walk")
        randomWalk(args, dataset)
    else:
        print("No flag provided")

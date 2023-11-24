import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from karateclub import DeepWalk, Node2Vec


def gnnTrainLoop(
    model,
    dataloader,
    optimizer,
    scheduler,
    epochs,
    patience,
    device,
    batch_size=32,
    verbose=False,
):
    """
    Train loop for GNN

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    dataloader : torch_geometric.data.DataLoader
        DataLoader for the dataset
    optimizer : torch.optim.Optimizer
        Optimizer for the model
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    epochs : int
        Number of epochs to train for
    patience : int
        Number of epochs to wait before early stopping
    device : str
        Device to train on (cpu, cuda, mps)
    batch_size : int
        Batch size for training
    verbose : bool
        Whether to print training progress

    Returns
    -------
    model : torch.nn.Module
        Trained model
    train_losses : list
        List of training losses versus epochs
    train_accuracies : list
        List of training accuracies versus epochs
    val_losses : list
        List of validation losses versus epochs
    val_accuracies : list
        List of validation accuracies versus epochs

    by Divyanshu + Hitesh
    """

    model.train()
    model = model.to(device)
    patience_counter = 0
    best_val = np.inf
    best_model = None

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    train_indices = torch.arange(0, len(dataloader.dataset.x))[
        dataloader.dataset.train_mask
    ]
    train_indices = train_indices.to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        total_training_samples = 0.0

        train_indices = train_indices[torch.randperm(len(train_indices))]

        for batch in torch.split(train_indices, batch_size):
            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)

                out = out[batch]
                targets = data.y[batch]

                loss = F.cross_entropy(out, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                train_loss += loss.item()

                _, pred = torch.max(out, 1)
                correct = (pred == targets).sum().item()
                train_accuracy += correct
                total_training_samples += batch_size

        val_loss, val_accuracy = gnnValidationLoop(
            model=model,
            dataloader=dataloader,
            device=device,
            verbose=False,
        )

        train_losses.append(train_loss / total_training_samples)
        train_accuracies.append(train_accuracy / total_training_samples)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if verbose:
            print(
                f"Epoch: {epoch}| Training Loss: {train_losses[-1]:.4f}| "
                + f"Training Accuracy: {train_accuracies[-1]:.4f}| Validation Loss: {val_losses[-1]:.4f}| "
                + f"Validation Accuracy: {val_accuracies[-1]:.4f}| LR: {scheduler.get_last_lr()[0]:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_model = model

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

        scheduler.step()

    return best_model, train_losses, train_accuracies, val_losses, val_accuracies


def gnnValidationLoop(model, dataloader, device, verbose=False):
    """
    Validation loop for GNN

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate
    dataloader : torch_geometric.data.DataLoader
        DataLoader for the dataset
    device : str
        Device to validate on (cpu, cuda, mps)
    verbose : bool
        Whether to print validation progress

    Returns
    -------
    val_loss : float
        Validation loss
    val_accuracy : float
        Validation accuracy

    by Divyanshu + Hitesh
    """

    model.eval()
    model = model.to(device)

    val_losses = 0.0
    val_accuracies = 0.0
    total_val_samples = 0.0

    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[batch.val_mask], batch.y[batch.val_mask])

        val_losses += loss.item()

        _, pred = torch.max(out[batch.val_mask], 1)
        correct = (pred == batch.y[batch.val_mask]).sum().item()
        val_accuracies += correct
        total_val_samples += batch.val_mask.sum().item()

    if verbose:
        print(
            f"Validation Loss: {val_losses / total_val_samples}| Validation Accuracy: {val_accuracies / total_val_samples}"
        )

    return val_losses / total_val_samples, val_accuracies / total_val_samples


def gnnTestLoop(model, dataloader, device, verbose=False):
    """
    Test loop for GNN

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate
    dataloader : torch_geometric.data.DataLoader
        DataLoader for the dataset
    device : str
        Device to validate on (cpu, cuda, mps)
    verbose : bool
        Whether to print validation progress

    Returns
    -------
    test_loss : float
        Validation loss
    test_accuracy : float
        Validation accuracy

    by Divyanshu + Hitesh
    """

    model.eval()
    model = model.to(device)

    test_loss = 0.0
    test_accuracy = 0.0
    total_val_samples = 0.0

    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[batch.test_mask], batch.y[batch.test_mask])

        test_loss += loss.item()

        _, pred = torch.max(out[batch.test_mask], 1)
        correct = (pred == batch.y[batch.test_mask]).sum().item()
        test_accuracy += correct
        total_val_samples += batch.test_mask.sum().item()

    if verbose:
        print(
            f"Test Loss: {(test_loss / total_val_samples):.4f}| Test Accuracy: {(test_accuracy / total_val_samples):.4f}"
        )

    return test_loss / total_val_samples, test_accuracy / total_val_samples


def mlpTrainLoop(
    model, dataloaders, optimizer, scheduler, epochs, patience, device, verbose=False
):
    """
    Train loop for MLP

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    dataloaders : tuple of torch.utils.data.DataLoader
        Tuple of train and validation dataloaders
    optimizer : torch.optim.Optimizer
        Optimizer for the model
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    epochs : int
        Number of epochs to train for
    patience : int
        Number of epochs to wait before early stopping
    device : str
        Device to train on (cpu, cuda, mps)
    verbose : bool
        Whether to print training progress

    Returns
    -------
    model : torch.nn.Module
        Trained model
    train_losses : list
        List of training losses versus epochs
    train_accuracies : list
        List of training accuracies versus epochs
    val_losses : list
        List of validation losses versus epochs
    val_accuracies : list
        List of validation accuracies versus epochs

    by Tejadhith
    """

    model.train()
    model = model.to(device)
    patience_counter = 0
    best_val = np.inf
    best_model = None

    train_loader, val_loader = dataloaders

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        total_training_samples = 0.0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(out, 1)
            correct = (pred == target).sum().item()
            train_accuracy += correct
            total_training_samples += data.shape[0]

        val_loss, val_accuracy = mlpValidationLoop(
            model=model,
            dataloader=val_loader,
            device=device,
            verbose=False,
        )

        train_losses.append(train_loss / total_training_samples)
        train_accuracies.append(train_accuracy / total_training_samples)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if verbose:
            print(
                f"Epoch: {epoch}| Training Loss: {train_losses[-1]:.4f}| "
                + f"Training Accuracy: {train_accuracies[-1]:.4f}| Validation Loss: {val_losses[-1]:.4f}| "
                + f"Validation Accuracy: {val_accuracies[-1]:.4f}| LR: {scheduler.get_last_lr()[0]:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_model = model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

        scheduler.step()

    return best_model, train_losses, train_accuracies, val_losses, val_accuracies


def mlpValidationLoop(model, dataloader, device, verbose=False):
    """
    Validation loop for MLP

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate
    dataloader : torch.utils.data.DataLoader
        DataLoader for the dataset
    device : str
        Device to validate on (cpu, cuda, mps)
    verbose : bool
        Whether to print validation progress

    Returns
    -------
    val_loss : float
        Validation loss
    val_accuracy : float
        Validation accuracy

    by Tejadhith
    """

    model.eval()
    model = model.to(device)

    val_losses = 0.0
    val_accuracies = 0.0
    total_val_samples = 0.0

    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        loss = F.cross_entropy(out, target)

        val_losses += loss.item()

        _, pred = torch.max(out, 1)
        correct = (pred == target).sum().item()
        val_accuracies += correct
        total_val_samples += data.shape[0]

    if verbose:
        print(
            f"Validation Loss: {val_losses / total_val_samples}| Validation Accuracy: {val_accuracies / total_val_samples}"
        )

    return val_losses / total_val_samples, val_accuracies / total_val_samples


def getRandomWalkEmbeddings(args, dataset):
    """
    Get node embeddings from random walk models

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for training
    dataset : torch_geometric.datasets.Planetoid
        Dataset to train on

    Returns
    -------
    embeddings : torch.tensor
        Node embeddings

    by Tejadhith
    """

    if args.random_walk_model == "Node2Vec":
        model = Node2Vec(
            dimensions=args.embedding_dim,
            walk_length=args.walk_length,
            walk_number=args.num_walks,
            workers=args.num_workers,
        )
    elif args.random_walk_model == "DeepWalk":
        model = DeepWalk(
            dimensions=args.embedding_dim,
            walk_length=args.walk_length,
            walk_number=args.num_walks,
            workers=args.num_workers,
        )

    graph = torch_geometric.utils.to_networkx(dataset[0], to_undirected=True)
    model.fit(graph)
    embeddings = model.get_embedding()
    embeddings = torch.from_numpy(embeddings).float()

    return embeddings

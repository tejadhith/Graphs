import os
import csv
import json
import torch
import torch_geometric


class outputWriter(object):
    """
    Write a list of list to a csv file.
    """

    def __init__(self, file_name, output_dir, column_heading=None):
        self.file_name = file_name
        self.output_dir = os.path.join(output_dir, file_name)
        self.csv_writing = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if column_heading is not None:
            if (
                column_heading == "GNN Training"
                or column_heading == "Random Walk Training"
            ):
                self.csv_writing.append(
                    [
                        "Run",
                        "Train losses",
                        "Train accuracies",
                        "Val losses",
                        "Val accuracies",
                        "Test losses",
                        "Test accuracies",
                    ]
                )
            else:
                self.csv_writing.append(column_heading)

    def csv_append(self, row):
        self.csv_writing.append(row)

    def write_csv(self):
        with open(os.path.join(self.output_dir, self.file_name + ".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.csv_writing)

    def write_json(self, arguments):
        with open(os.path.join(self.output_dir, self.file_name + ".json"), "w") as f:
            json.dump(arguments, f, indent=4, sort_keys=True)

    def model_save(self, model, run_num=""):
        print("Saving model at: ", self.output_dir)
        torch.save(
            model.state_dict(),
            os.path.join(self.output_dir, self.file_name + run_num + ".pt"),
        )


def getLoaders(embeddings, dataset, batch_size, num_workers=1):
    """
    Create dataloaders from embeddings and dataset

    Parameters
    ----------
    embeddings : torch.Tensor
        Embeddings
    dataset : torch_geometric.data.Dataset
        Dataset
    batch_size : int
        Batch size
    num_workers : int, optional
        Number of workers, by default 1

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        Train dataloader
    val_dataloader : torch.utils.data.DataLoader
        Validation dataloader
    test_dataloader : torch.utils.data.DataLoader
        Test dataloader
    """

    # Split into train, val datasetclear
    train_mask = dataset[0].train_mask
    val_mask = dataset[0].val_mask
    test_mask = dataset[0].test_mask

    train_embeddings = embeddings[train_mask]
    val_embeddings = embeddings[val_mask]
    test_embeddings = embeddings[test_mask]

    train_labels = dataset[0].y[train_mask]
    val_labels = dataset[0].y[val_mask]
    test_labels = dataset[0].y[test_mask]

    # Create dataloader
    train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_embeddings, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_embeddings, test_labels)

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader

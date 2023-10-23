from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import csv
import numpy as np

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_name = "GCN_Cora.csv"
csv_writing = []

for embedding_dim in range(1, 33):
    row = []
    print("Embedding dim: ", embedding_dim)
    row.append(str(embedding_dim))
    for i in range(10):
        GCNmodel = GCN(hidden_dims=[64, embedding_dim]).to(device)
        MLPmodel = MLP(input_dim=embedding_dim, hidden_dim=16).to(device)
        data = data.to(device)
        optimizer = torch.optim.AdamW(GCNmodel.parameters(), lr=0.01, weight_decay=5e-4)

        GCNmodel.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = GCNmodel(data.x, data.edge_index)
            out = MLPmodel(out)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        GCNmodel.eval()
        x = GCNmodel(data.x, data.edge_index)
        pred = MLPmodel(x).argmax(dim=-1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        print("Accuracy: {:.4f}".format(acc))
        row.append(str(acc))
    csv_writing.append(",".join(row))

    # Save the embeddings
    if embedding_dim == 2:
        out = GCNmodel(data.x, data.edge_index).detach().cpu().numpy()
        np.save("GCN_Cora_2.npy", out)
        np.save("GCN_Cora_2_label.npy", data.y.detach().cpu().numpy())
    
    if embedding_dim == 32:
        out = GCNmodel(data.x, data.edge_index).detach().cpu().numpy()
        np.save("GCN_Cora_32.npy", out)
        np.save("GCN_Cora_32_label.npy", data.y.detach().cpu().numpy())

with open(file_name, "w") as f:
    f.write("embedding_dim,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7,accuracy8,accuracy9,accuracy10\n")
    for row in csv_writing:
        f.write(row + "\n")
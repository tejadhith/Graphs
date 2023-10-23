import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from karateclub.utils.walker import RandomWalker, BiasedRandomWalker
from karateclub import DeepWalk, Node2Vec
from torch_geometric.utils import to_networkx

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]

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

def randomWalk(start_node, walk_length):
    walk = [start_node]
    for i in range(walk_length):
        neighbors = data.edge_index[1][data.edge_index[0] == walk[-1]]
        walk.append(np.random.choice(neighbors))
    return walk


# walk_number = 10  # Number of walks performed per vertex
# walk_length = 5
# vertices = np.arange(data.num_nodes)
# walks = []
#
# for i in tqdm(range(walk_number), desc="Generating walks"):
#     np.random.shuffle(vertices)
#     for j in vertices:
#         walks.append(randomWalk(j, walk_length))

embedding_dim = 128
print("Embedding dim: ", embedding_dim)

# G = nx.Graph()
# G.add_edges_from(data.edge_index.t().tolist())

G = to_networkx(data, to_undirected=True)

# walker = RandomWalker(walk_length=walk_length, walk_number=walk_number)
# walker.do_walks(G)

# model = Word2Vec(walker.walks, vector_size=embedding_dim, window=10, min_count=1, sg=1, hs=1, workers=2, epochs=10)
# embeddings = model.wv.vectors
model = DeepWalk(dimensions=embedding_dim)
model.fit(G)
embeddings = model.get_embedding()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim=embedding_dim, hidden_dim=64).to(device)
embeddings = torch.tensor(embeddings).to(device)
data = data.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Training MLP...")
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(embeddings)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(embeddings).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
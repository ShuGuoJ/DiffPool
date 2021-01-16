import torch
from torch.nn import functional as F
from torch_geometric import nn
from torch_geometric.utils import to_dense_batch, to_dense_adj


class DiffPool(torch.nn.Module):
    def __init__(self, in_channel, hidden_size):
        super().__init__()
        self.pooling = nn.DenseSAGEConv(in_channel, hidden_size)

    def forward(self, x, adj, mask=None):
        s = self.pooling(x, adj, mask)
        return nn.dense_diff_pool(x, adj, s, mask)


class Net(torch.nn.Module):
    def __init__(self, in_channel: int, hidden_size: int, nc: int, max_nodes: int):
        super().__init__()
        self.gcn_1 = nn.DenseSAGEConv(in_channel, hidden_size)
        self.gcn_2 = nn.DenseSAGEConv(hidden_size, hidden_size)
        self.pooling = DiffPool(hidden_size, int(0.25 * max_nodes))
        self.gcn_3 = nn.DenseSAGEConv(hidden_size, 2*hidden_size)
        self.gcn_4 = nn.DenseSAGEConv(2*hidden_size, 2*hidden_size)
        self.classifier = torch.nn.Linear(2*hidden_size, nc)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        x = F.relu(self.gcn_1(x, adj, mask), True)
        x = F.relu(self.gcn_2(x, adj, mask), True)
        x, adj, l_lp, l_e = self.pooling(x, adj, mask)
        x = F.relu(self.gcn_3(x, adj))
        x = F.relu(self.gcn_4(x, adj))
        x = x.mean(1)
        logits = self.classifier(x)
        return logits, l_lp, l_e

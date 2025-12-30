import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """GCN编码器"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class InnerProductDecoder(nn.Module):
    """内积解码器"""

    def forward(self, z):
        # z: [num_nodes, embedding_dim]
        adj = torch.sigmoid(z @ z.t())
        return adj


class GraphAutoencoder(nn.Module):
    """基础图自编码器（GAE）"""

    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, embedding_dim)
        self.decoder = InnerProductDecoder()
        self.embedding_dim = embedding_dim

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_recon = self.decoder(z)
        return z, adj_recon

    def encode(self, x, edge_index):
        """仅编码，用于推理"""
        with torch.no_grad():
            return self.encoder(x, edge_index)

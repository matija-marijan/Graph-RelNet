import torch
import torch.nn as nn
from torch.nn import Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, GATv2Conv
from torch_geometric.data import Data
import torch.nn.init as init

class RelNet(nn.Module):
    def __init__(self, edge_dim: int = 2049, emb_dim: int = 625, num_output: int = 2):
        super().__init__()

        self.edge_dim = edge_dim

        # [batch_size, num_pairs, edge_dim + 6]

        self.pairwise_relation_network = nn.Sequential(
            nn.Linear(in_features = self.edge_dim + 2 * 3, out_features = emb_dim),     # (x, y, z) for both microphones/nodes
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.2)
        )

        # [batch_size, num_pairs, emb_dim]

        # Sum/Mean all pairwise features
        # [batch_size, 1, emb_dim]

        self.relation_fusion_network = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_output),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch, array = data.x, data.edge_index, data.edge_attr, data.batch, data.array
        
        # print(edge_attr.shape)
        # print(array.shape)
        # print(edge_index.shape)
        # print(edge_index[:, 0])
        # print(edge_index[:, 1])

        # Construct pairwise features by concatenating node coordinates to edge attributes
        # edge_attr: [num_edges, edge_dim]
        # array[edge_index[0], :]: [num_edges, 3] (source node coords)
        # array[edge_index[1], :]: [num_edges, 3] (target node coords)
        edge_attr = torch.cat([
            edge_attr,
            array[edge_index[0], :],  # source node coordinates
            array[edge_index[1], :]   # target node coordinates
        ], dim=1)

        # Edge-wise MLP
        edge_attr = self.pairwise_relation_network(edge_attr)

        # Pool edge features per graph using the source node's batch index
        edge_batch = batch[edge_index[0]]  # [num_edges]
        graph_feat = global_mean_pool(edge_attr, edge_batch)  # [B, emb_dim]

        # Final prediction per graph
        out = self.relation_fusion_network(graph_feat)
        return out
    
if __name__ == "__main__":
    model = RelNet(edge_dim=401)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
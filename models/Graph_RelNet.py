import torch
import torch.nn as nn
from torch.nn import Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, GATv2Conv, GCNConv, GraphConv, GATConv
from torch_geometric.data import Data
from torch_scatter import scatter_add, scatter_mean
import torch.nn.init as init
import torch.nn.functional as F

class Edge2Node(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())

    def forward(self, edge_emb, edge_index, num_nodes):
        src, dst = edge_index
        msg_src = scatter_mean(self.proj(edge_emb), src, dim=0, dim_size=num_nodes)
        msg_dst = scatter_mean(self.proj(edge_emb), dst, dim=0, dim_size=num_nodes)
        return 0.5 * (msg_src + msg_dst)
    
class EdgeEncoder(nn.Module):
    def __init__(self, edge_dim, emb_dim):
        super().__init__()
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(in_features = edge_dim + 4, out_features = emb_dim),     # (x, y, z) for both microphones/nodes
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
            # nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),            
            # nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)         
            # nn.Dropout(0.2)
        )

    def forward(self, edge_attr):
        return self.edge_encoder(edge_attr)

class Graph_RelNet(nn.Module):
    def __init__(self, edge_dim: int = 2049, emb_dim: int = 128, node_hid: int = 128, num_output: int = 2):
        super().__init__()

        self.edge_dim = edge_dim
        self.emb_dim = emb_dim
        self.num_output = num_output

        self.edge_encoder = EdgeEncoder(edge_dim, emb_dim)
        self.edge2node = Edge2Node(emb_dim, node_hid)

        # self.gnn1 = GATv2Conv(node_hid, node_hid, edge_dim=1, heads=2, concat=False)
        # self.gnn2 = GATv2Conv(node_hid, node_hid, edge_dim=1, heads=2, concat=False)
        self.gnn1 = GCNConv(node_hid, node_hid)
        self.gnn2 = GCNConv(node_hid, node_hid)

        self.n_norm1 = nn.LayerNorm(node_hid)
        self.n_norm2 = nn.LayerNorm(node_hid)

        self.head = nn.Sequential(
            nn.Linear(node_hid + emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_output),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch, array = data.x, data.edge_index, data.edge_attr, data.batch, data.array

        source_nodes = array[edge_index[0], :]
        target_nodes = array[edge_index[1], :]
        relative_positions = source_nodes - target_nodes                        # [E,3]
        distances = torch.norm(relative_positions, dim=1, keepdim=True)         # [E,1]
        relative_unit = relative_positions / (distances + 1e-8)                 # [E,3]
        log_dist = torch.log(distances + 1e-6)                                  # [E,1]
        
        edge_attr = torch.cat([edge_attr, relative_unit, log_dist], dim=1)

        # edge_attr = torch.cat([
        #     edge_attr,
        #     source_nodes,
        #     target_nodes,
        #     relative_positions,
        #     distances
        # ], dim=1)

        edge_attr = self.edge_encoder(edge_attr)

        x = self.edge2node(edge_attr, edge_index, x.size(0))

        edge_weight = distances.squeeze(-1)
        
        h0 = x
        x = self.gnn1(x, edge_index, edge_weight)
        x = self.n_norm1(x + h0)
        x = F.gelu(x)

        h1 = x
        x = self.gnn2(x, edge_index, edge_weight)
        x = self.n_norm2(x + h1)

        edge_batch = batch[edge_index[0]]
        node_graph = global_mean_pool(x, batch)
        edge_graph = global_mean_pool(edge_attr, edge_batch) 
        graph = torch.cat([node_graph, edge_graph], dim = -1)

        out = self.head(graph)

        return out
    
if __name__ == "__main__":
    model = Graph_RelNet(edge_dim=401)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
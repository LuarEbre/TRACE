import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GNNVehiclePositionPredictor(torch.nn.Module):

    # defines model structure and layers
    def __init__(self, in_features, hidden_dim):
        super().__init__()

        # Convolution Attention Layer
        self.conv1 = GATConv(in_features, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)

        # Main Sequence Block
        self.pos_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                
                # Output dimension is 2 (target_x, target_y)
                nn.Linear(32,2)
                )

    # defines steps of forward propagation
    def forward(self, data):

        # x = Tensor with features of node
        # edge_index = Tensor describing what nodes are connected
        x, edge_index = data.x, data.edge_index

        # First Conv Attention Layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Seconds Conv Attention Layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Main Sequence Block
        pred_positions = self.pos_head(x)

        return pred_positions

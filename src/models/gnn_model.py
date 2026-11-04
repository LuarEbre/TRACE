import torch
from torch_geometric.nn import GATConv


class GNNTrafficRiskPredictor(torch.nn.Module):

    # defines model structure and layers
    def __init__(self, in_features, hidden_dim):
        super().__init__()

        # Convolution Attention Layer
        self.conv1 = GATConv(in_features, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)

        # Main Sequence Block
        self.risk_head = torch.nn.Sequential(

                # Linear Layer with dynamic input and 32 output features
                torch.nn.Linear(hidden_dim, 32),
                torch.nn.ReLU(),

                # 32 input features, 1 output (continious risk value)
                torch.nn.Linear(32, 1),

                # Scales between 0.0 and 1.0
                torch.nn.Sigmoid()
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
        node_risks = self.risk_head(x)

        return node_risks

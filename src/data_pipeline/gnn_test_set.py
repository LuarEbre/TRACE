import torch
from torch_geometric.data import Data
import random
import os

# converts raw data into PyG Data Object
def create_traffic_graph(car_features, edge_indices, labels=None):
    graph_data = Data(
            x = torch.tensor(car_features, dtype=torch.float32),
            edge_index = torch.tensor(edge_indices, dtype=torch.float32) 
            )
    return graph_data

# generates a test dataset with random feature values and number of cars
def generate_test_dataset(num_samples=200, save_path="data/gnn_test_dataset.pt"):
    dataset = []
    for _ in range(num_samples):
        # random number of cars between 5 and 15
        num_cars = random.randint(5,15)
        
        # random values for 8 features of all cars
        x = torch.randn(num_cars, 8)

        # Create random edges (setup random adjacency matrix)
        edges = []
        for i in range(num_cars):
            for j in range(num_cars):
                # Creates a connection with a chance of 0.3
                if i != j and random.random() < 0.3:
                    edges.append([i, j])
        
        # Creates the edge_index even if by chance no edges were created
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2,0), dtype=torch.long)
        
        # Generate a random risk score
        y = torch.rand(num_cars, 1)

        # Data object with features, graph_adjacency and risk value are appended to dataset
        dataset.append(Data(x=x, edge_index=edge_index, y=y))

    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)
    print(f"Test dataset saved at {save_path}")

    return dataset

if __name__ == "__main__":
    generate_test_dataset()

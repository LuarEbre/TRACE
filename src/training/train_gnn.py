import os
import torch
from torch_geometric.loader import DataLoader

# Imports from own functions
from src.models.gnn_model import GNNTrafficRiskPredictor
from src.data_pipeline.gnn_test_set import generate_test_dataset

def main():
    DATASET_PATH = "data/gnn_test_dataset.pt"
    SAVE_PATH = "trained_models/gnn_trained.pth" 
    EPOCHS = 10

    # load data
    if os.path.exists(DATASET_PATH):
        dataset = torch.load(DATASET_PATH, weights_only=False)
    else:
        dataset = generate_test_dataset(num_samples=200, save_path=SAVE_PATH)

    # Create dataloader (does batching)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create model, optimizer and loss function(MSE)
    model = GNNTrafficRiskPredictor(in_features=8, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training Loop

    # set model to training mode
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        # Train model with data in batches
        for batch in loader:
            # reset gradients to zero
            optimizer.zero_grad()

            # forward pass
            pred_risks = model(batch)

            # Calc loss (pred vs ground truth y)
            loss = criterion(pred_risks, batch.y)

            # Backpropagation and optimize step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    main()

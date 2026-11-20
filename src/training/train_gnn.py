import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader

# Imports model definition
from src.models.gnn_model import GNNVehiclePositionPredictor

def main():
    DATASET_PATH = Path("data/training_data/highD")
    SAVE_PATH = "trained_models/gnn_trained.pth"
    
    EPOCHS = 10
    BATCH_SIZE = 16

    # If gpu is found, then use it otherwise use cpu
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    pth_files = sorted(DATASET_PATH.glob("*.pth"))

    # Create model, optimizer and loss function(MSE)
    model = GNNVehiclePositionPredictor(in_features=16, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training Loop

    # set model to training mode
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_batches = 0
        
        for pth in pth_files:
            scenario = torch.load(pth, weights_only=False)

            loader = DataLoader(scenario, batch_size=BATCH_SIZE, shuffle=True)
            
            # Train model with data in batches
            for batch in loader:
            
                # move data to device
                batch = batch.to(DEVICE)

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
                total_batches += 1
    
        print(f"Epoch {epoch+1} | Loss: {total_loss / total_batches:.4f}")

    # Save trained model
    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    main()


import torch
from torch_geometric.data import Data
import polars as pl
import numpy as np
from pathlib import Path
from typing import List

# Processes a dataframe into gnn ready dataset
def process_data(df: pl.DataFrame, radius) -> List[Data]:
   
    # 1. Change string vector collumns to numeric data
    exprs = []
    for col in ["position", "velocity", "acceleration"]:
        if col in df.columns:
            cleaned = pl.col(col).str.replace_all(r"[\[\]]", "").str.split(", ")
            exprs.extend([
                cleaned.list.get(0).cast(pl.Float64).alias(f"{col}_0"),
                cleaned.list.get(1).cast(pl.Float64).alias(f"{col}_1")
                ])

    df = df.with_columns(exprs)

    # 2. Define target (next frame position)
    df = df.sort(["id", "frame"])
    df = df.with_columns([
        pl.col("position_0").shift(-1).over("id").alias("target_x"),
        pl.col("position_1").shift(-1).over("id").alias("target_y")
        ])
    
    # drop nulls (no next frame)
    df = df.drop_nulls(["target_x", "target_y"])

    # 3. Define feature vector for GNN
    feature_cols = [
            "position_0", "position_1",
            "velocity_0", "velocity_1",
            "acceleration_0", "acceleration_1",
            "heading", "length", "width", "lane_id",
            "dhw", "rel_speed", "lat_dist", "thw", "ttc", "drac"
            ]
    
    # 4. Partition by time frame (graph for each time step)
    frames_grouped = df.partition_by("frame", maintain_order=True)
    
    data_list = []

    for frame in frames_grouped:
        
        # 5. All feature values as numpy, then as torch tensor
        X_np = frame.select(feature_cols).to_numpy().astype(np.float32)
        X = torch.from_numpy(X_np)
        
        # 6. Build graph within radius
        pos = X[:, :2]
        
        if frame.height >= 2:
            dist_matrix = torch.cdist(pos, pos)
            
            mask = (dist_matrix <= radius) & (dist_matrix > 0)

            edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)


        # 7. Define y (target values)
        y_np = frame.select("target_x", "target_y").to_numpy().astype(np.float32)
        y = torch.from_numpy(y_np)
        
        # 8. Create PyG Data object
        data = Data(x=X, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list

if __name__ == "__main__":
    INPUT_DIR = "./data/translated/highD"
    OUTPUT_DIR = "./data/training_data/highD"
    TARGET_SCENARIOS = None
    
    RADIUS = 30.0

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if TARGET_SCENARIOS:
        csv_files = []
        for s in TARGET_SCENARIOS:
            csv_files.append(input_path / f"{s}_processed.csv")
    else:
        csv_files = sorted(input_path.glob("*_processed.csv"))

    for csv in csv_files:
        df = pl.read_csv(csv)
    
        dataset = process_data(df, radius=RADIUS)
    
        output_file = output_path / f"{csv.stem.replace('_processed', '')}_pyg.pth"
    
        torch.save(dataset, output_file)
        print(f"Saved preprocessed dataset at {output_file}")

    print("\nFinished preprocessing all files")

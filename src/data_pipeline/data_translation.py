import os
from pathlib import Path
import polars as pl
import numpy as np
from typing import Optional, List
 
# Loads highD tracking and metadata files, filters by driving direction,
# computes heading angles, and normalizes coordinates/units for MetaDrive compatibility.
def load_and_normalize_highd(tracks_path: str, meta_path: str, direction: int = 2) -> pl.DataFrame:
    
    # Read trajectory tracks data
    tracks = pl.read_csv(tracks_path)

    # Read vehicle metadata (ID, vehicle class, and driving direction)
    meta = pl.read_csv(meta_path, columns=["id", "class", "drivingDirection"])

    # Inner join tracks and metadata, then filter by the target driving direction
    df = tracks.join(meta, on="id", how="inner").filter(pl.col("drivingDirection") == direction)

    # Calculate heading angle from velocity components using arctan2
    heading_array = np.arctan2(-df["yVelocity"].to_numpy(), df["xVelocity"].to_numpy())
    df = df.with_columns(pl.Series("heading", heading_array))
    
    # Normalize and rename columns to match MetaDrive schema expectations
    df_norm = df.select([
        pl.col("frame").cast(pl.Int32),
        pl.col("id").cast(pl.Utf8),
        pl.col("class"),
        
        # Shift coordinate center from bounding box top-left to center
        (pl.col("x") + pl.col("width") / 2.0).round(4).alias("x"),
        
        # Invert y-axis to match standard coordinate systems used in simulation
        (-(pl.col("y") + pl.col("height") / 2.0)).round(4).alias("y"),
        
        pl.col("xVelocity").round(4).alias("vx"),
        (-pl.col("yVelocity")).round(4).alias("vy"),
        pl.col("xAcceleration").round(4).alias("ax"),
        (-pl.col("yAcceleration")).round(4).alias("ay"),
        pl.col("heading").round(4),
        
        # highD 'width' corresponds to vehicle length, and 'height' to vehicle width
        pl.col("width").round(4).alias("length"),
        pl.col("height").round(4).alias("width"),
        pl.col("laneId").cast(pl.Int32).alias("lane_id")
    ])

    # Normalize y-coordinates relative to the maximum y value in the dataset
    y_max = df_norm.select(pl.col("y").max()).item()
    df_norm = df_norm.with_columns(
        (pl.col("y") - y_max).round(4).alias("y")
    )
    
    return df_norm

# Computes DHW (Distance Headway), THW(Time Headway), 
# TTC(Time-To-Collision), DRAC(Deceleration Rate to Avoid Crash) and relative speed
def compute_safety_metrics(df: pl.DataFrame) -> pl.DataFrame:
    
    # Sort by frame, lane, and position to identify the leading vehicle
    df = df.sort(["frame", "lane_id", "x"])

    # Retrieve leading vehicle data in the same lane and frame (shift(-1) = next row in df)
    df = df.with_columns([
        pl.col("x").shift(-1).over(["frame", "lane_id"]).alias("lead_x"),
        pl.col("y").shift(-1).over(["frame", "lane_id"]).alias("lead_y"),
        pl.col("vx").shift(-1).over(["frame", "lane_id"]).alias("lead_vx"),
        pl.col("length").shift(-1).over(["frame", "lane_id"]).alias("lead_length"),
        pl.col("width").shift(-1).over(["frame", "lane_id"]).alias("lead_width"),
    ])

    # Compute distance headway, relative speed, and lateral distance to the leading vehicle
    df = df.with_columns([
        
        # Calc dhw
        pl.max_horizontal(
            pl.lit(0.01), 
            (pl.col("lead_x") - pl.col("x")) - (pl.col("length") + pl.col("lead_length")) / 2.0
        ).alias("dhw"),
        
        # Calc relative speed
        (pl.col("vx") - pl.col("lead_vx")).alias("rel_speed"),
        
        # Calc lat_dist to leading vehicle
        pl.max_horizontal(
            pl.lit(0.0), 
            (pl.col("lead_y") - pl.col("y")).abs() - (pl.col("width") + pl.col("lead_width")) / 2.0
        ).alias("lat_dist")
    ])
    
    # Compute safety parameters (THW, TTC, DRAC)
    df = df.with_columns([
        
        # Calc thw
        pl.when(pl.col("vx") > 0.1)
          .then(pl.min_horizontal(pl.lit(10.0), pl.col("dhw") / pl.col("vx")))
          .otherwise(pl.lit(10.0))
          .alias("thw"),
        
        # Calc ttc
        pl.when(pl.col("rel_speed") > 0.05)
          .then(pl.min_horizontal(pl.lit(10.0), pl.col("dhw") / pl.col("rel_speed")))
          .otherwise(pl.lit(10.0))
          .alias("ttc"),
        
        # Calc drac
        pl.when(pl.col("rel_speed") > 0.05)
          .then((pl.col("rel_speed") ** 2) / (2.0 * pl.col("dhw")))
          .otherwise(pl.lit(0.0))
          .alias("drac")
    ])
    
    # If no lead vehicle exists, change all null values with default values
    df = df.with_columns([
        pl.col("dhw").fill_null(250.0).round(4),
        pl.col("thw").fill_null(10.0).round(4),
        pl.col("rel_speed").fill_null(0.0).round(4),
        pl.col("ttc").fill_null(10.0).round(4),
        pl.col("drac").fill_null(0.0).round(4),
        pl.col("lat_dist").fill_null(3.5).round(4),
        # Pet = Post encroachment time
        pl.lit(10.0).alias("pet")
    ])
    
    # Metadrive has vector attributes as strings in format "[x, y]"
    df = df.with_columns([
        pl.concat_str([pl.lit("["), pl.col("x"), pl.lit(", "), pl.col("y"), pl.lit("]")]).alias("position"),
        pl.concat_str([pl.lit("["), pl.col("vx"), pl.lit(", "), pl.col("vy"), pl.lit("]")]).alias("velocity"),
        pl.concat_str([pl.lit("["), pl.col("ax"), pl.lit(", "), pl.col("ay"), pl.lit("]")]).alias("acceleration")
    ])
    
    # Drop all old collumns (raw vector values)
    df = df.drop([
        "x", "y", "vx", "vy", "ax", "ay", 
        "lead_x", "lead_y", "lead_vx", "lead_length", "lead_width"
    ])
    
    # Return processed dataset sorted chronologically by frame and vehicle id
    return df.sort(["frame", "id"])


# Process highD dataset track files
# scnearios = None [Process all datasets]
# scenarios = [1, 2, 3] [Process datasets of scenario 1, 2, 3]
def process_datasets(input_dir: str, output_dir: str, scenarios: Optional[List[str]] = None):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)
    
    # scenarios = None: Process all *_tracks.csv
    # scenarios != None: Process the *_tracks.csv of given scenarios
    if scenarios is not None:
        # Normalize scenario IDs to strings (e.g., handles ints like 1 -> "01" or keeps "01" as is)
        track_files = []
        for s in scenarios:
            prefix_str = str(s).zfill(2) if str(s).isdigit() and len(str(s)) == 1 else str(s)
            track_file = input_path / f"{prefix_str}_tracks.csv"
            track_files.append(track_file)
    else:
        # Default behavior: grab all track files in directory
        track_files = sorted(input_path.glob("*_tracks.csv"))
    
    # Process files
    for track_file in track_files:

        # Skip if file not found
        if not track_file.exists():
            print(f"Skipping: Track file not found {track_file.name}")
            continue

        # Extract scenario prefix (e.g., "01" from "01_tracks.csv")
        prefix = track_file.name.split("_")[0]

        # Locate corresponding metadata file
        meta_file = input_path / f"{prefix}_tracksMeta.csv"
        
        if not meta_file.exists():
            print(f"Skip scenario {prefix}: No meta file found.")
            continue
            
        print(f"Processing scenario {prefix}...")
        
        # Load and normalize data
        df = load_and_normalize_highd(str(track_file), str(meta_file))
        
        if df.is_empty():
            print(f"Error loading dataframe for scenario {prefix}")
            continue

        # Compute safety metrics
        df_processed = compute_safety_metrics(df)
        
        # Insert dataset_id column as the first column
        df_processed = df_processed.with_columns(pl.lit(prefix).alias("dataset_id"))
        cols = ["dataset_id"] + [c for c in df_processed.columns if c != "dataset_id"]
        df_processed = df_processed.select(cols)
        
        # Save processed dataset
        output_file = out_path / f"{prefix}_processed.csv"
        df_processed.write_csv(str(output_file))
        print(f"Saved scenario {prefix} at {output_file}")

    print("\nFinished processing selected files.")


if __name__ == "__main__":
    INPUT_DIR = "./data/raw/highD"
    OUTPUT_DIR = "./data/translated/highD"
    
    # Configuration options:
    # 1. Set to None to process ALL available datasets in INPUT_DIR (Default)
    # 2. Set to a list of strings/integers to process only specific scenarios (e.g., ["01", "05", "12"])
    TARGET_SCENARIOS = None 

    process_datasets(INPUT_DIR, OUTPUT_DIR, scenarios=TARGET_SCENARIOS)
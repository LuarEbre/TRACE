import os
from pathlib import Path
import polars as pl
import numpy as np
from typing import Optional, List

def translate_data(tracks_path: str, meta_path: str) -> pl.DataFrame:

    tracks = pl.read_csv(tracks_path)
    # reads from meta only id, class and drivingDirection
    meta = pl.read_csv(meta_path, columns=["id", "class", "drivingDirection"])
    
    # inner join, creates dataframe with all vehicles of the scenario
    df = tracks.join(meta, on="id", how="inner")
    
    # calculations
    x = df.get_column("x") + df.get_column("width") / 2.0
    y = -(df.get_column("y") + df.get_column("height") / 2.0)
    vx = df.get_column("xVelocity").to_numpy()
    vy = -(df.get_column("yVelocity").to_numpy())
    heading = np.arctan2(vy, vx)
    ax = df.get_column("xAcceleration")
    ay = -(df.get_column("yAcceleration"))
    length = df.get_column("width")
    width = df.get_column("height")
    lane_id = df.get_column("laneId")
    
    # put calculations into dataframe
    df = df.with_columns([
        pl.Series("x", x),
        pl.Series("y", y),
        pl.Series("vx", vx),
        pl.Series("vy", vy),
        pl.Series("heading", heading),
        pl.Series("ax", ax),
        pl.Series("ay", ay),
        pl.Series("length", length),
        pl.Series("width", width),
        pl.Series("lane_id", lane_id),
    ])
    
    # normalize the dataframe
    df_norm = df.select([
        pl.col("frame").cast(pl.Int32),
        pl.col("id").cast(pl.Int32),
        pl.col("class"),

        pl.col("x").round(4),
        pl.col("y").round(4),
        pl.col("vx").round(4),
        pl.col("vy").round(4),
        pl.col("heading").round(4),
        pl.col("ax").round(4),
        pl.col("ay").round(4),
        pl.col("length").round(4),
        pl.col("height").round(4),
        pl.col("lane_id").cast(pl.Int32),

        pl.col("precedingId").cast(pl.Int32),
    ])
    
    # Sort the frame after frame and id
    df_norm = df_norm.sort(["frame", "id"]) 

    # Translate y, so that biggest y is 0 
    y_max = df_norm.select(pl.col("y").max()).item()
    df_norm = df_norm.with_columns(
        (pl.col("y") - y_max).round(4).alias("y")
    ) 

    return df_norm

# calculate safety metrics
def calc_metric(df: pl.DataFrame) -> pl.DataFrame:
    # Retrieve leading vehicle data
    lead = df.select(
        "frame",
        pl.col("id").alias("lead_id"),
        pl.col("x").alias("lead_x"),
        pl.col("vx").alias("lead_vx"),
        pl.col("length").alias("lead_length"),
    )  
        
    df = df.join(lead, 
                 left_on=["frame", "precedingId"],
                 right_on=["frame", "lead_id"],
                 how="left",
    )
    
    # Calculate DHW, distance headway
    df = df.with_columns(
        pl.when((pl.col("vx") > 0) & pl.col("lead_x").is_not_null())
          .then(pl.col("lead_x") - (pl.col("x") + pl.col("length")))
          .when((pl.col("vx") < 0) & pl.col("lead_x").is_not_null())
          .then(pl.col("x") - (pl.col("lead_x") + pl.col("length")))
          .otherwise(-1)
          .round(4)
          .alias("dhw")
    )
    
    closing_speed = pl.col("vx").abs() - pl.col("lead_vx").abs()
    
    # Calculate THW, time headway
    df = df.with_columns(
        pl.when(pl.col("vx").abs() > 0)
          .then(pl.col("dhw") / pl.col("vx").abs())
          .otherwise(-1)
          .round(4)
          .alias("thw")
    )
    
    # Calculate TTC, time to collision
    df = df.with_columns(
        pl.when((closing_speed > 0) & (pl.col("dhw") >= 0))
          .then(pl.col("dhw") / closing_speed)
          .otherwise(-1)
          .round(4)
          .alias("ttc")
    )
    
    df = df.drop([
        "lead_x", "lead_vx", "lead_length"
    ])

    return(df)

# calls the different processing steps
def process_datasets(in_dir: str, out_dir: str):
    in_path = Path(in_dir)
    out_path = Path(out_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    
    track_files = []
    
    track_files = sorted(in_path.glob("*_tracks.csv"))

    for track_file in track_files:
        prefix = track_file.name.split("_")[0]
        meta_file = in_path / f"{prefix}_tracksMeta.csv"
    
        print(f"Processing scenario {prefix}...")

        df = translate_data(str(track_file), str(meta_file))
        
        df_proc = calc_metric(df)
        
        # Save processed dataset
        out_file = out_path / f"{prefix}_processed.csv"
        df_proc.write_csv(str(out_file))
        print(f"Saved scenario {prefix} at {out_file}")

    print("\n Finished processing files.")

# Entry point, defines directories
if __name__ == "__main__":
    IN_DIR = "./data/raw/highD"
    OUT_DIR = "./data/translated/highD"
    
    process_datasets(IN_DIR, OUT_DIR)

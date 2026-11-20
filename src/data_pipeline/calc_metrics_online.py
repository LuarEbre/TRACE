import polars as pl
import torch

def calc_and_add_metrics(df: pl.DataFrame) -> pl.DataFrame:
    
    if df.is_empty():
        return df

    # Sort by frame, lane, and position to identify the leading vehicle
    df = df.sort(["lane_id", "x"])

    # Retrieve leading vehicle data in the same lane and frame (shift(-1) = next row in df)
    df = df.with_columns([
        pl.col("x").shift(-1).over(["lane_id"]).alias("lead_x"),
        pl.col("y").shift(-1).over(["lane_id"]).alias("lead_y"),
        pl.col("vx").shift(-1).over(["lane_id"]).alias("lead_vx"),
        pl.col("length").shift(-1).over(["lane_id"]).alias("lead_length"),
        pl.col("width").shift(-1).over(["lane_id"]).alias("lead_width"),
    ])

    # Compute distance headway, relative speed, and lateral distance to the leading vehicle
    df = df.with_columns([
        
        # Calc dhw
        pl.max_horizontal(
            pl.lit(0.01), 
            (pl.col("lead_x") - pl.col("x")) - (pl.col("length") + pl.col("lead_length")) * 0.5,
        ).alias("dhw"),
        
        # Calc relative speed
        (pl.col("vx") - pl.col("lead_vx")).alias("rel_speed"),
        
        # Calc lat_dist to leading vehicle
        pl.max_horizontal(
            pl.lit(0.0), 
            (pl.col("lead_y") - pl.col("y")).abs() - (pl.col("width") + pl.col("lead_width")) * 0.5,
        ).alias("lat_dist")
    ])
    
    # Compute safety parameters (THW, TTC, DRAC)
    df = df.with_columns([
        
        # Calc thw
        pl.when(pl.col("vx") > 0.1)
          .then(pl.min_horizontal(pl.lit(10.0), pl.col("dhw") / pl.col("vx")))
          .otherwise(pl.lit(10.0))
          .fill_null(10.0)
          .alias("thw"),
        
        # Calc ttc
        pl.when(pl.col("rel_speed") > 0.05)
          .then(pl.min_horizontal(pl.lit(10.0), pl.col("dhw") / pl.col("rel_speed")))
          .otherwise(pl.lit(10.0))
          .fill_null(10.0)
          .alias("ttc"),
        
        # Calc drac
        pl.when(pl.col("rel_speed") > 0.05)
          .then((pl.col("rel_speed") ** 2) / (2.0 * pl.col("dhw")))
          .otherwise(pl.lit(0.0))
          .fill_null(0.0)
          .alias("drac"),
    ])
    
    # If no lead vehicle exists, change all null values with default values
    df = df.with_columns([
        pl.col("dhw").fill_null(250.0),
        pl.col("thw").fill_null(10.0),
        pl.col("rel_speed").fill_null(0.0),
        pl.col("ttc").fill_null(10.0),
        pl.col("drac").fill_null(0.0),
        pl.col("lat_dist").fill_null(3.5),
        # Pet = Post encroachment time
        pl.lit(10.0).alias("pet")
    ])
    
    df = df.drop(["lead_x", "lead_y", "lead_vx", "lead_vy", "lead_length", "lead_width"])

    # Return processed dataset sorted chronologically by frame and vehicle id
    return df.sort(["frame", "id"])


if __name__ == "__main__":
    
    # 1. Get current data from metadrive
    # includes: id, x, y, vx, vy, ax, ay, heading, length, width, lane_id
    df = pl.DataFrame()

    # 2. Calculate and add new metrics to dataframe
    df = calc_and_add_metrics(df)
    
    #TODO Send to preprocessing values

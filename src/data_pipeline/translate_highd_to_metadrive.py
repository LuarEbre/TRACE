import os
import polars as pl

# Converts highD csv files to metadrive compatible parameters and saves in csv format
def convert_highd_to_metadrive_csv(data_dir: str, recording_id: str, output_dir: str):
   
    prefix = str(recording_id).zfill(2)
    tracks_file = os.path.join(data_dir, f"{prefix}_tracks.csv")
    
    if not os.path.exists(tracks_file):
        print(f"Could not find {prefix} in {tracks_file}")
        return False

    print(f"Processing {prefix}")
    
    # Read as dataframe
    df_tracks = pl.read_csv(tracks_file)
    
    # Prepare expressions
    exprs = [
        (-pl.col("y")).alias("y"),
        (-pl.col("yVelocity")).alias("yVelocity")
    ]
    
    # Invert accelerations
    if "yAcceleration" in df_tracks.columns:
        exprs.append((-pl.col("yAcceleration")).alias("yAcceleration"))
        
    # transform to metadrive compatability
    df_processed = df_tracks.with_columns(exprs)
    
    # Caclulate "Gier"-angle form transformed accelerations
    df_processed = df_processed.with_columns(
        pl.arctan2(pl.col("yVelocity"), pl.col("xVelocity")).alias("heading")
    )

    # Save as new csv
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"highd_recording_{prefix}_translated.csv")
    df_processed.write_csv(output_csv_path)
    
    print(f"Succesfully saved at {output_csv_path}")
    return True

if __name__ == "__main__":
    
    DATA_DIR = "./data/raw/highD"
    OUTPUT_DIR = "./data/translated/highD"

    NUM_SCENARIOS = 1
    
    success_count = 0
   
    # range(1,61) for all 60 highD szenarios
    for rec_id in range(1, NUM_SCENARIOS+1):
        success = convert_highd_to_metadrive_csv(DATA_DIR, rec_id, OUTPUT_DIR)
        if success:
            success_count += 1
            
    print(f"\nConverted {success_count} of {NUM_SCENARIOS} Scenarios")

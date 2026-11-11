import os
import polars as pl

# Converts highD csv files to metadrive compatible parameters and saves in csv format
# data_dir: directory with the highD csv files
# recording_id: the index of the scenario
# output_dir: the directory where the result should be saved at
def convert_highd_to_metadrive_csv(data_dir: str, recording_id: str, output_dir: str):
   
    # prefix: the index of the current szenario. 
    # Different to recording_id as it fills "0" to 2 spots (01, 02, ..., 10, 11)
    prefix = str(recording_id).zfill(2)

    # Name of the file (index + tracks_csv) 
    # HardCoded on the names of the highD csv files
    tracks_file = os.path.join(data_dir, f"{prefix}_tracks.csv")
    
    # exits if file is not found
    if not os.path.exists(tracks_file):
        print(f"Could not find {tracks_file}")
        return False

    print(f"Processing {prefix}")
    
    # Read file as dataframe
    df_tracks = pl.read_csv(tracks_file)
    
    # Prepare expressions
    # The col[y] is replaced by negative col[y] (invert the axis)
    # same with col[yVelocity]
    exprs = [
        (-pl.col("y")).alias("y"),
        (-pl.col("yVelocity")).alias("yVelocity")
    ]
    
    # If yAcceleration already exists, then its also inverted
    # added to exprs 
    if "yAcceleration" in df_tracks.columns:
        exprs.append((-pl.col("yAcceleration")).alias("yAcceleration"))
        
    # the dataframe gets updated with all the queries in exprs
    df_processed = df_tracks.with_columns(exprs)
    
    # The Heading (Ausrichtung) is calculated via arctan2(yVelocity, xVelocity)
    # This replaces the old col[heading]
    df_processed = df_processed.with_columns(
        pl.arctan2(pl.col("yVelocity"), pl.col("xVelocity")).alias("heading")
    )

    # Checks if the directory already exists, if not then creates it
    os.makedirs(output_dir, exist_ok=True)

    # The path to save is the output dir / highd_recording_{prefix}_translated.csv
    output_csv_path = os.path.join(output_dir, f"highd_recording_{prefix}_translated.csv")

    # Writes the csv at the path
    df_processed.write_csv(output_csv_path)
    
    print(f"Succesfully saved at {output_csv_path}")
    return True

if __name__ == "__main__":
    
    DATA_DIR = "./data/raw/highD"
    OUTPUT_DIR = "./data/translated/highD"

    NUM_SCENARIOS = 60
    
    success_count = 0
   
    # range(1,61) for all 60 highD szenarios
    for rec_id in range(1, NUM_SCENARIOS+1):
        success = convert_highd_to_metadrive_csv(DATA_DIR, rec_id, OUTPUT_DIR)
        if success:
            success_count += 1
            
    print(f"\nConverted {success_count} of {NUM_SCENARIOS} Scenarios")

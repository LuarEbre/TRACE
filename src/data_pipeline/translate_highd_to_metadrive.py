import os
import pickle
import numpy as np
import polars as pl

# converts highD csv to metadrive compatible parameters in format .pkl files
def convert_highd_to_metadrive_polars(data_dir: str, recording_id: str, output_path: str): 

    prefix = str(recording_id).zfill(2)
    tracks_file = os.path.join(data_dir, f"{prefix}_tracks.csv")
    tracks_meta_file = os.path.join(data_dir, f"{prefix}_tracksMeta.csv")
    recording_meta_file = os.path.join(data_dir, f"{prefix}_recordingMeta.csv")
    
    # Check if file exists
    if not os.path.exists(tracks_file):
        print(f"File {tracks_file} not found")
        return False

    print(f"Load highD-Data {prefix}...")
    df_tracks = pl.read_csv(tracks_file)
    df_tracks_meta = pl.read_csv(tracks_meta_file)
    df_recording_meta = pl.read_csv(recording_meta_file)
    
    frame_rate = int(df_recording_meta.select(pl.col("frameRate")).item(0, 0)) if "frameRate" in df_recording_meta.columns else 25
    total_frames = int(df_tracks.select(pl.col("frame").max()).item(0, 0))
    
    scenario_dict = {
        "scenario_id": f"highd_{prefix}",
        "dataset": "highd",
        "length": total_frames,
        "metadata": {
            "frame_rate": frame_rate,
            "coordinate_system": "cartesian",
            "sdc_id": None
        },
        "tracks": {},
        "map_features": {
            "type": "straight_highway",
            "lanes": df_recording_meta.get_column("upperLaneIds").to_list() if "upperLaneIds" in df_recording_meta.columns else "N/A"
        }
    }
    
    unique_vehicle_ids = df_tracks.select("id").unique().to_series().to_list()
    print(f"Verarbeite {len(unique_vehicle_ids)} Fahrzeuge über {total_frames} Frames...")
    
    for vid in unique_vehicle_ids:
        v_tracks = df_tracks.filter(pl.col("id") == vid).sort("frame")
        v_meta = df_tracks_meta.filter(pl.col("id") == vid)
        
        if v_meta.is_empty():
            continue

        width = float(v_tracks.select(pl.col("width")).item(0, 0))
        height_val = float(v_tracks.select(pl.col("height")).item(0, 0)) 
        
        veh_length = width
        veh_width = height_val
        veh_height = 1.5
        
        valid_mask = np.zeros(total_frames + 1, dtype=bool)
        positions = np.zeros((total_frames + 1, 3), dtype=np.float32)
        velocities = np.zeros((total_frames + 1, 3), dtype=np.float32)
        headings = np.zeros(total_frames + 1, dtype=np.float32)
        
        frames = v_tracks.get_column("frame").to_numpy()
        x_vals = v_tracks.get_column("x").to_numpy()
        y_vals = v_tracks.get_column("y").to_numpy()
        vx_vals = v_tracks.get_column("xVelocity").to_numpy()
        vy_vals = v_tracks.get_column("yVelocity").to_numpy()
        
        for i, frame in enumerate(frames):
            if frame > total_frames:
                continue
            
            f_idx = int(frame)
            valid_mask[f_idx] = True
            
            # Transform y axis from highD to metadrive format (inverted)
            pos_x = float(x_vals[i])
            pos_y = float(-y_vals[i])
            positions[f_idx] = [pos_x, pos_y, 0.0]
            
            vel_x = float(vx_vals[i])
            vel_y = float(-vy_vals[i])
            velocities[f_idx] = [vel_x, vel_y, 0.0]
            
            if abs(vel_x) > 1e-3 or abs(vel_y) > 1e-3:
                headings[f_idx] = np.arctan2(vel_y, vel_x)
            else:
                headings[f_idx] = 0.0

        scenario_dict["tracks"][str(vid)] = {
            "type": "VEHICLE",
            "shape": [veh_length, veh_width, veh_height],
            "state": {
                "position": positions,
                "velocity": velocities,
                "heading": headings,
                "valid": valid_mask
            }
        }
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scenario_dict, f)
        
    print(f"Successfully converted and saved to: {output_path}\n")
    return True

# --- BATCH-VERARBEITUNG FÜR ALLE 60 RECORDINGS ---
if __name__ == "__main__":

    DATA_DIRECTORY = "./data/raw/highD"
    OUTPUT_DIRECTORY = "./data/translated/highD"
    
    success_count = 0

    # currently range (1,2) for testing. range(1,60) for all of highD
    for rec_id in range(1, 2):
        rec_str = str(rec_id)
        output_file = os.path.join(OUTPUT_DIRECTORY, f"highd_recording_{rec_str.zfill(2)}.pkl")
        
        print(f"Process Szenario {rec_id} of 60")
        success = convert_highd_to_metadrive_polars(DATA_DIRECTORY, rec_str, output_file)
        if success:
            success_count += 1
            
    print(f"Converted {success_count} of 60 Szenarios succesfully")

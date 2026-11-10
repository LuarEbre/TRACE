import os
import torch
import numpy as np
import polars as pl

# converts highD csv to metadrive compatible parameters in format torch.tensors .pt file 
def convert_highd_to_tensors(data_dir: str, recording_id: str, output_path: str):

    prefix = str(recording_id).zfill(2)
    tracks_file = os.path.join(data_dir, f"{prefix}_tracks.csv")
    recording_meta_file = os.path.join(data_dir, f"{prefix}_recordingMeta.csv")
    
    if not os.path.exists(tracks_file):
        print(f"Überspringe Recording {prefix}: Datei nicht gefunden unter {tracks_file}")
        return False

    print(f"Lade highD-Daten für Recording {prefix}...")
    df_tracks = pl.read_csv(tracks_file)
    df_recording_meta = pl.read_csv(recording_meta_file)
    
    frame_rate = int(df_recording_meta.select(pl.col("frameRate")).item(0, 0)) if "frameRate" in df_recording_meta.columns else 25
    unique_vehicle_ids = df_tracks.select("id").unique().to_series().to_list()
    
    agents_data = {}
    
    for vid in unique_vehicle_ids:
        v_tracks = df_tracks.filter(pl.col("id") == vid).sort("frame")
        if v_tracks.is_empty():
            continue
            
        frames = v_tracks.get_column("frame").to_numpy().astype(np.int32)
        x_vals = v_tracks.get_column("x").to_numpy().astype(np.float32)
        
        y_vals = -v_tracks.get_column("y").to_numpy().astype(np.float32)
        
        vx_vals = v_tracks.get_column("xVelocity").to_numpy().astype(np.float32)
        vy_vals = -v_tracks.get_column("yVelocity").to_numpy().astype(np.float32)
        
        width = float(v_tracks.select(pl.col("width")).item(0, 0))
        height_val = float(v_tracks.select(pl.col("height")).item(0, 0))
        
        headings = np.zeros_like(x_vals)
        mask_vel = (np.abs(vx_vals) > 1e-3) | (np.abs(vy_vals) > 1e-3)
        headings[mask_vel] = np.arctan2(vy_vals[mask_vel], vx_vals[mask_vel])
        
        features = np.stack([x_vals, y_vals, vx_vals, vy_vals, headings], axis=-1)
        
        agents_data[str(vid)] = {
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "shape": [width, height_val, 1.5],
            "frames": frames,
            "features": torch.tensor(features, dtype=torch.float32)
        }
        
    scenario_package = {
        "scenario_id": f"highd_{prefix}",
        "dataset": "highd",
        "frame_rate": frame_rate,
        "agents": agents_data
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(scenario_package, output_path)
    print(f"Erfolgreich optimiert gespeichert: {output_path}\n")
    return True

if __name__ == "__main__":

    DATA_DIR = "./data/raw/highD"
    OUTPUT_DIR = "./data/training_data/highD"
     
    success_count = 0

    # range(1,61) for all 60 highD szenarios
    for rec_id in range(1, 61):
        output_file = os.path.join(OUTPUT_DIR, f"highd_recording_{str(rec_id).zfill(2)}.pt")
        
        print(f"Process Szenario {rec_id} of 60")
        success = convert_highd_to_tensors(DATA_DIR, rec_id, output_file)
        if success:
            success_count += 1
            
    print(f"Converted {success_count} of 60 Szenarios succesfully")

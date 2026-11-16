import os
from pathlib import Path
import polars as pl
import numpy as np

#translate highD data to metadrive compliant csv
def load_and_normalize_highd(tracks_path: str, meta_path: str, direction: int = 2) -> pl.DataFrame:
    """
    Liest highD-Daten ein, filtert auf eine Fahrtrichtung (2 = untere Spuren nach rechts),
    korrigiert Mittelpunkt & Achsen und gibt das einheitliche Schema zurück.
    """
    tracks = pl.read_csv(tracks_path)
    meta = pl.read_csv(meta_path, columns=["id", "class", "drivingDirection"])

    # Merge und Filter
    df = tracks.join(meta, on="id", how="inner").filter(pl.col("drivingDirection") == direction)

    # --- FIX: Heading absolut kugelsicher und rasend schnell über Numpy berechnen ---
    heading_array = np.arctan2(-df["yVelocity"].to_numpy(), df["xVelocity"].to_numpy())
    df = df.with_columns(pl.Series("heading", heading_array))

    # Spalten auswählen, berechnen und umbenennen (vektorisiert)
    df_norm = df.select([
        pl.col("frame").cast(pl.Int32),
        pl.col("id").cast(pl.Utf8),
        pl.col("class"),
        (pl.col("x") + pl.col("width") / 2.0).round(4).alias("x"),
        (-(pl.col("y") + pl.col("height") / 2.0)).round(4).alias("y"),
        pl.col("xVelocity").round(4).alias("vx"),
        (-pl.col("yVelocity")).round(4).alias("vy"),
        pl.col("xAcceleration").round(4).alias("ax"),
        (-pl.col("yAcceleration")).round(4).alias("ay"),
        pl.col("heading").round(4),  # <--- Einfach unsere Numpy-Spalte übernehmen
        pl.col("width").round(4).alias("length"),
        pl.col("height").round(4).alias("width"),
        pl.col("laneId").cast(pl.Int32).alias("lane_id")
    ])

    # Y normalisieren
    y_max = df_norm.select(pl.col("y").max()).item()
    df_norm = df_norm.with_columns((pl.col("y") - y_max).round(4).alias("y"))

    return df_norm


def compute_safety_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Berechnet DHW, THW, rel_speed, TTC, DRAC und lat_dist pfeilschnell 
    mit Polars Window-Functions ohne jegliche Loops.
    """
    # WICHTIG: Nach Frame, Lane und X-Position sortieren. 
    # Dadurch ist der Vordermann (größeres X) garantiert immer in der nächsten Zeile!
    df = df.sort(["frame", "lane_id", "x"])

    # "shift(-1)" holt die Daten der nächsten Zeile (des Vordermanns).
    # "over(['frame', 'lane_id'])" verhindert, dass wir versehentlich Autos 
    # aus anderen Frames oder Spuren als Vordermann zuordnen.
    df = df.with_columns([
        pl.col("x").shift(-1).over(["frame", "lane_id"]).alias("lead_x"),
        pl.col("y").shift(-1).over(["frame", "lane_id"]).alias("lead_y"),
        pl.col("vx").shift(-1).over(["frame", "lane_id"]).alias("lead_vx"),
        pl.col("length").shift(-1).over(["frame", "lane_id"]).alias("lead_length"),
        pl.col("width").shift(-1).over(["frame", "lane_id"]).alias("lead_width"),
    ])

    # 1. Basis-Metriken berechnen
    df = df.with_columns([
        pl.max_horizontal(pl.lit(0.01), (pl.col("lead_x") - pl.col("x")) - (pl.col("length") + pl.col("lead_length")) / 2.0).alias("dhw"),
        (pl.col("vx") - pl.col("lead_vx")).alias("rel_speed"),
        pl.max_horizontal(pl.lit(0.0), (pl.col("lead_y") - pl.col("y")).abs() - (pl.col("width") + pl.col("lead_width")) / 2.0).alias("lat_dist")
    ])

    # 2. Bedingte Metriken (THW, TTC, DRAC)
    df = df.with_columns([
        pl.when(pl.col("vx") > 0.1).then(pl.min_horizontal(pl.lit(10.0), pl.col("dhw") / pl.col("vx"))).otherwise(pl.lit(10.0)).alias("thw"),
        
        pl.when(pl.col("rel_speed") > 0.05).then(pl.min_horizontal(pl.lit(10.0), pl.col("dhw") / pl.col("rel_speed"))).otherwise(pl.lit(10.0)).alias("ttc"),
        
        pl.when(pl.col("rel_speed") > 0.05).then((pl.col("rel_speed")**2) / (2.0 * pl.col("dhw"))).otherwise(pl.lit(0.0)).alias("drac")
    ])

    # 3. Leere Werte (Fahrzeuge ohne Vordermann) mit Defaults füllen & runden
    df = df.with_columns([
        pl.col("dhw").fill_null(250.0).round(4),
        pl.col("thw").fill_null(10.0).round(4),
        pl.col("rel_speed").fill_null(0.0).round(4),
        pl.col("ttc").fill_null(10.0).round(4),
        pl.col("drac").fill_null(0.0).round(4),
        pl.col("lat_dist").fill_null(3.5).round(4),
        pl.lit(10.0).alias("pet")
    ])

    # 4. Vektoren direkt als Text-Strings im Format "[x, y]" basteln
    df = df.with_columns([
        pl.concat_str([pl.lit("["), pl.col("x"), pl.lit(", "), pl.col("y"), pl.lit("]")]).alias("position"),
        pl.concat_str([pl.lit("["), pl.col("vx"), pl.lit(", "), pl.col("vy"), pl.lit("]")]).alias("velocity"),
        pl.concat_str([pl.lit("["), pl.col("ax"), pl.lit(", "), pl.col("ay"), pl.lit("]")]).alias("acceleration")
    ])

    df = df.drop(["x", "y", "vx", "vy", "ax", "ay", "lead_x", "lead_y", "lead_vx", "lead_length", "lead_width"])
    
    # Optional: Nach Frame und ID sortieren (für hübschere Ausgabe)
    return df.sort(["frame", "id"])


def process_all_datasets(input_dir: str, output_dir: str):
    """
    Sucht alle Datensätze im Input-Ordner, verarbeitet sie und speichert JEDEN 
    als eigenständige CSV-Datei im Output-Ordner ab.
    """
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    
    # Sicherstellen, dass der Output-Ordner existiert
    out_path.mkdir(parents=True, exist_ok=True)
    
    track_files = sorted(input_path.glob("*_tracks.csv"))
    
    if not track_files:
        print(f"Keine '_tracks.csv' Dateien in {input_dir} gefunden.")
        return

    for track_file in track_files:
        prefix = track_file.name.split("_")[0]
        meta_file = input_path / f"{prefix}_tracksMeta.csv"
        
        if not meta_file.exists():
            print(f"Überspringe {prefix}: Keine passende Meta-Datei gefunden.")
            continue
            
        print(f"Verarbeite Datensatz {prefix}...")
        
        # 1. Einlesen & Normalisieren
        df = load_and_normalize_highd(str(track_file), str(meta_file))
        
        if df.is_empty():
            print(f"  -> Warnung: Datensatz {prefix} ist leer (evtl. keine Fahrzeuge in Direction 2).")
            continue

        # 2. Safety Metrics berechnen 
        df_processed = compute_safety_metrics(df)
        
        # 3. Dataset-ID einfügen
        df_processed = df_processed.with_columns(pl.lit(prefix).alias("dataset_id"))
        
        # Spalte 'dataset_id' nach ganz links schieben
        cols = ["dataset_id"] + [c for c in df_processed.columns if c != "dataset_id"]
        df_processed = df_processed.select(cols)
        
        # 4. Direkt als eigene CSV abspeichern!
        output_file = out_path / f"{prefix}_processed.csv"
        df_processed.write_csv(str(output_file))
        print(f"  -> Gespeichert unter: {output_file}")

    print("\nAlle Datensätze wurden erfolgreich verarbeitet und einzeln gespeichert!")


if __name__ == "__main__":
    BASE_DIR = "C:/Users/liuzz/Documents/Leandro/Uni/Semester 5/Project/TRACE"
    
    INPUT_DIR = f"{BASE_DIR}/data/raw/highD"
    OUTPUT_DIR = f"{BASE_DIR}/data/translated/highD"  # <--- Jetzt ein Ordner statt einer Datei!

    process_all_datasets(INPUT_DIR, OUTPUT_DIR)
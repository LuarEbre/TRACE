from typing import Any, Dict, List
import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from src.inference.ModelStrategy import BaseModelStrategy
from src.data_pipeline.preprocessing.online.gnn_online_preprocessing import GNNPreprocessing

FEATURE_COLS: List[str] = [
    "x", "y", "vx", "vy", "ax", "ay",
    "heading", "length", "width", "lane_id",
    "dhw", "rel_speed", "lat_dist", "thw", "ttc", "drac",
]

class GNNStrategy(BaseModelStrategy):
    def __init__(self, model_name: str="GNN", radius: float = 30.0, frame_step: int = 1):
        super().__init__(model_name=model_name, frame_step=frame_step)
        self.preprocessor = GNNPreprocessing(radius=radius)

    def preprocess(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        dataset = self.preprocessor.preprocess(df)
        graph = dataset[0]
        return {
            "X": graph.x.numpy().astype(np.float32),
            "EDGE_INDEX": graph.edge_index.numpy().astype(np.int64)
            }

    def format_output(self, raw_outputs: Dict[str, np.ndarray]) -> Any:
        arr = next(iter(raw_outputs.values()))
        return arr.tolist()

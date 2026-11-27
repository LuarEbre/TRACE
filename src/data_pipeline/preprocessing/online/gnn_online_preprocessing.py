import polars as pl
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict

from src.data_pipeline.preprocessing.online.base_preprocessing import BasePreprocessor

class GNNPreprocessing(BasePreprocessor):
    def __init__(self, radius: float = 30.0):
        self.radius = radius

    def preprocess(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        feature_cols = [
            "x", "y", "vx", "vy", "ax", "ay",
            "heading", "lentgh", "width", "lane_id",
            "dhw", "rel_speed", "lat_dist", "thw", "ttc", "drac",
            ]

        x = torch.from_numpy(df.select(feature_cols).to_numpy().astype(np.float32))

        if df.height >= 2:
            dist = torch.cdist(x[:, :2], x[:, :2])
            mask = (dist <= self.radius) & (dist > 0)
            edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)

        return{
            "X":data.x.numpy(),
            "EDGE_INDEX" : data.edge_index.numpy().astype(np.int64)
        }

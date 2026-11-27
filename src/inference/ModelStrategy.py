from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import polars as pl

FEATURE_COLS: List[str] = [
    "x", "y", "vx", "vy", "ax", "ay",
    "heading", "length", "width", "lane_id",
    "dhw", "rel_speed", "lat_dist", "thw", "ttc", "drac",
]


class BaseModelStrategy(ABC):
    def __init__(self, model_name: str, frame_step: int = 1) -> None:
        self.model_name = model_name
        self.frame_step = max(1, frame_step)

    def should_run(self, frame: int) -> bool:
        return frame % self.frame_step == 0

    @abstractmethod
    def preprocess(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def format_output(self, raw_outputs: Dict[str, np.ndarray]) -> Any:
        pass

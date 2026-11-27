from abc import ABC, abstractmethod
from typing import Any
import polars as pl

class BasePreprocessor(ABC):

    @abstractmethod
    def preprocess(self, df: pl.DataFrame) -> Any:
        raise NotImplementedError

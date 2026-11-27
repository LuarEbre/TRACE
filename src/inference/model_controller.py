import json
from typing import Any, Dict, List
import numpy as np
import polars as pl
import redis
import tritonclient.http as httpclient

from calc_metrics_online import calc_and_add_metrics
from gnn_strategy import BaseModelStrategy, GNNStrategy


DTYPE_MAP = {
    np.dtype(np.float32): "FP32",
    np.dtype(np.float64): "FP64",
    np.dtype(np.int64): "INT64",
    np.dtype(np.int32): "INT32",
}

STRATEGIES = [
        GNNStrategy(model_name="GNN", radius=30.0, frame_step=1)
        ]

class ModelController:
    def __init__(
        self,
        strategies: List[BaseModelStrategy],
        triton_url: str = "localhost:8000",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        queue_stream: str = "frames:incoming",
        result_key_prefix: str = "model:result",
    ) -> None:
        self.strategies = strategies
        self.triton = httpclient.InferenceServerClient(url=triton_url)
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.queue_stream = queue_stream
        self.result_key_prefix = result_key_prefix
        self._last_id = "$"

    def calc_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        return calc_and_add_metrics(df)

    def preprocess(self, strategy: BaseModelStrategy, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        return strategy.preprocess(df)

    def infer(self, model_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        triton_inputs = []
        for name, arr in inputs.items():
            inp = httpclient.InferInput(name, arr.shape, DTYPE_MAP[arr.dtype])
            inp.set_data_from_numpy(arr)
            triton_inputs.append(inp)

        res = self.triton.infer(model_name=model_name, inputs=triton_inputs)
        return {out.name: res.as_numpy(out.name) for out in res.get_response()["outputs"]}

    def save_to_redis(self, frame: int, model_name: str, output: Any) -> None:
        payload = {"frame": frame, "model_name": model_name, "output": output}
        self.redis.set(f"{self.result_key_prefix}:{model_name}:{frame}", json.dumps(payload))

    def step(self, frame: int, raw_df: pl.DataFrame) -> None:
        metrics_df = self.calc_metrics(raw_df)

        for strategy in self.strategies:
            if not strategy.should_run(frame):
                continue
            inputs = self.preprocess(strategy, metrics_df)
            raw_out = self.infer(strategy.model_name, inputs)
            formatted_out = strategy.format_output(raw_out)
            self.save_to_redis(frame=frame, model_name=strategy.model_name, output=formatted_out)

    def run_live(self, block_ms: int = 1000) -> None:
        while True:
            streams = self.redis.xread({self.queue_stream: self._last_id}, count=1, block=block_ms)
            for _, messages in streams:
                for msg_id, fields in messages:
                    self._last_id = msg_id
                    df = pl.DataFrame(json.loads(fields["rows"]))
                    frame = int(fields.get("frame", df["frame"][0] if "frame" in df.columns else 0))
                    self.step(frame, df)


if __name__ == "__main__":
    controller = ModelController(
        strategies=STRATEGIES
    )
    controller.run_live()

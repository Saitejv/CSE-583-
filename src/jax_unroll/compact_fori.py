"""Load an exported fori tree model with NumPy at inference time."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS

FIELDS = ("trip_count", "num_consts", "num_carry", "num_xs", "num_equations", "eq_reduce",
          "eq_gather_scatter", "eq_dot", "carry_bytes", "xs_bytes", "const_bytes",
          "intermediate_bytes", "max_tensor_bytes", "longest_path", "flops_proxy",
          "max_rank", "num_feed_forward_values", "bytes_moved_proxy")
ARRAYS = ("mean", "scale", "center", "roots", "feature", "threshold", "values", "depth", "left", "right")


def tree_costs(row, *, mean, scale, center, roots, feature, threshold, values, depth, left, right):
    """Traverse the fitted forest using the original ExtraTrees comparison precision."""
    features = np.log1p([float(row[k]) for k in FIELDS])
    features = np.append(features, [row["loop_kind"] == "scan", int(row["trip_count"]) % 8 / 8])
    x = ((features - mean) / scale).astype(np.float32)
    nodes = roots
    for _ in range(depth):
        use_left = x[feature[nodes]] <= threshold[nodes]
        nodes = np.where(use_left, left[nodes], right[nodes])
    logs = (center + values[nodes].mean(axis=0)).reshape(-1, 2)
    logs[1:] += logs[:1]
    return np.exp(logs)


class CompactForiPredictor:
    """Inference-only export of a trained ForiCostPredictor with tree effects."""

    @classmethod
    def export(cls, source, destination, *, compressed=True):
        from jax_unroll.fori_model import ForiCostPredictor

        model = ForiCostPredictor.load(source)
        if model.effect != "trees":
            raise ValueError("compact export requires a fori predictor with joint trees")
        metadata = json.dumps({
            "format": 1, "source_version": model.version, "backend": model.backend,
            "feature_version": model.feature_version, "jax_versions": model.jax_versions,
            "margin": model.margin, "fields": FIELDS, "factors": FACTORS,
        })
        destination = Path(destination)
        temporary = None
        try:
            with NamedTemporaryFile(dir=destination.parent, delete=False) as file:
                temporary = Path(file.name)
                save = np.savez_compressed if compressed else np.savez
                save(file, metadata=metadata, **model._tree_arrays())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        """Read numerical arrays without importing the training stack."""
        data = Path(path).read_bytes()
        model = cls()
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].item())
            if (meta["format"] != 1 or tuple(meta["fields"]) != FIELDS
                    or tuple(meta["factors"]) != FACTORS):
                raise ValueError("incompatible compact fori predictor")
            model._arrays = {key: archive[key] for key in ARRAYS}
        model.backend = meta["backend"]
        model.feature_version = meta["feature_version"]
        model.jax_versions = tuple(meta["jax_versions"])
        model.source_version = meta["source_version"]
        model.margin = meta["margin"]
        model.version = sha256(data).hexdigest()
        return model

    def supports(self, row):
        return (
            row["backend"] == self.backend and row["loop_kind"] == "fori_loop"
            and row["loop_primitive"] == "scan" and 16 <= int(row["trip_count"]) <= 1024
            and int(row["dtype_bits"]) == 32
        )

    def predict_candidates(self, row):
        costs = tree_costs(row, **self._arrays)
        costs[1:, 1] *= np.exp(self.margin)
        return {factor: LogCosts(*np.log(pair)) for factor, pair in zip(FACTORS, costs, strict=True)}

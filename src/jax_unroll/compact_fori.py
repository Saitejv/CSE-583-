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


# Cheap scalar arithmetic plus the index normalization produced by JAX.
_STREAM_PRIMITIVES = frozenset({
    "abs", "add", "sub", "mul", "min", "max", "clamp", "neg", "lt",
    "select_n", "convert_element_type", "dynamic_slice", "squeeze",
})


def scalar_stream_features(loop):
    """Identify short arithmetic bodies with one float32 stream and one scalar carry."""
    args = loop.jaxpr.invars
    if (loop.loop_kind != "fori_loop" or loop.loop_primitive != "scan" or loop.num_consts != 1
            or len(args) != 3 or args[2].aval.shape != () or args[2].aval.dtype != np.float32):
        return {}
    stream = args[0].aval
    if len(stream.shape) != 1 or stream.shape[0] <= 1024 or stream.dtype != np.float32:
        return {}

    def simple(eqns):
        for eq in eqns:
            op = str(eq.primitive)
            if op in ("jit", "pjit") and "jaxpr" in eq.params:
                if not simple(eq.params["jaxpr"].jaxpr.eqns):
                    return False
            elif op not in _STREAM_PRIMITIVES:
                return False
            if any(v.aval.size != 1 for v in eq.outvars):
                return False
        return True

    return {"scalar_stream_length": stream.shape[0] if simple(loop.jaxpr.eqns) else 0}


def is_scalar_stream(row):
    return (row.get("scalar_stream_length") == row["trip_count"]
            and row["eq_gather_scatter"] == 1 and row["num_equations"] <= 16)


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
            model._scalar_trips = np.array([], dtype=int)
            if ("scalar_trips" in archive) != ("scalar_costs" in archive):
                raise ValueError("incomplete scalar-stream cost table")
            if "scalar_trips" in archive:
                trips, costs = archive["scalar_trips"], archive["scalar_costs"]
                if (trips.ndim != 1 or not len(trips) or trips[0] <= 1024 or np.any(np.diff(trips) <= 0)
                        or costs.shape != (len(trips), len(FACTORS), 2)
                        or not np.isfinite(costs).all() or np.any(costs <= 0)):
                    raise ValueError("invalid scalar-stream cost table")
                model._scalar_trips, model._scalar_costs = trips, costs
                model.loop_features = scalar_stream_features
        model.backend = meta["backend"]
        model.feature_version = meta["feature_version"]
        model.jax_versions = tuple(meta["jax_versions"])
        model.source_version = meta["source_version"]
        model.margin = meta["margin"]
        model.version = sha256(data).hexdigest()
        return model

    def _supports_scalar(self, row, trip):
        return (self._scalar_trips.size > 0 and self._scalar_trips[0] <= trip <= self._scalar_trips[-1]
                and is_scalar_stream(row))

    def supports(self, row):
        return (
            row["backend"] == self.backend and row["loop_kind"] == "fori_loop"
            and row["loop_primitive"] == "scan"
            and (16 <= int(row["trip_count"]) <= 1024 or self._supports_scalar(row, int(row["trip_count"])))
            and int(row["dtype_bits"]) == 32
        )

    def predict_candidates(self, row):
        trip = int(row["trip_count"])
        if self._supports_scalar(row, trip):
            upper = int(np.searchsorted(self._scalar_trips, trip))
            lower = max(0, upper - 1)
            span = self._scalar_trips[upper] - self._scalar_trips[lower]
            fraction = (trip - self._scalar_trips[lower]) / span if span else 0.
            costs = (1 - fraction) * self._scalar_costs[lower] + fraction * self._scalar_costs[upper]
            return {f: LogCosts(*np.log(pair)) for f, pair in zip(FACTORS, costs, strict=True)}
        costs = tree_costs(row, **self._arrays)
        costs[1:, 1] *= np.exp(self.margin)
        return {factor: LogCosts(*np.log(pair)) for factor, pair in zip(FACTORS, costs, strict=True)}


def application_engine(*, model_path=None):
    """Create a shared CPU fori_loop engine with deferred loading of the bundled model.

    Pass this engine and the expected invocation count R to auto_fori_loop.
    B, when supplied, bounds predicted cold specialization cost in milliseconds.
    The bundled model includes measured long scalar-stream costs on Apple M2.
    Supply model_path to reproduce an earlier compact artifact.
    """
    from jax_unroll.selector import DecisionEngine, HorizonPolicy

    rule = HorizonPolicy(min_calls=1800, decision_ms=3.2005205, matrix_min_calls=1000,
                         matrix_bytes=2048, max_index_ops=2, matrix_factor=2)
    path = Path(__file__).with_name("data") / "fori_cpu.npz" if model_path is None else model_path
    return DecisionEngine(model_loader=lambda: CompactForiPredictor.load(path), payoff=rule)

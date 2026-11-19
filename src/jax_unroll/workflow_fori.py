"""Optional empirical costs for two-state CPU stream loops."""

import hashlib
import json
from pathlib import Path

import numpy as np

from jax_unroll.compact_fori import (
    CompactForiPredictor,
    _STREAM_PRIMITIVES,
    application_engine,
    scalar_stream_features,
)
from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine

_OPS = _STREAM_PRIMITIVES | {
    "stack",
    "broadcast_in_dim",
    "concatenate",
    "slice",
    "eq",
    "ge",
    "gt",
    "le",
    "reshape",
}


def workflow_features(loop):
    state = [v.aval for v in loop.jaxpr.outvars[1:]]
    if (
        loop.loop_kind != "fori_loop"
        or loop.loop_primitive != "scan"
        or [v.shape for v in state] not in ([(2,)], [(), ()])
        or any(v.dtype != np.float32 for v in state)
    ):
        return scalar_stream_features(loop)
    arrays = [v.aval for v in loop.jaxpr.invars[: loop.num_consts] if v.aval.shape]
    if (
        not arrays
        or len(arrays[0].shape) != 1
        or arrays[0].shape[0] <= 1024
        or any(a.shape != arrays[0].shape or a.dtype != np.float32 for a in arrays)
    ):
        return {}

    def simple(eqns):
        for e in eqns:
            name = str(e.primitive)
            if name in ("jit", "pjit") and "jaxpr" in e.params:
                if not simple(e.params["jaxpr"].jaxpr.eqns):
                    return False
            elif name not in _OPS:
                return False
            if any(v.aval.size > 2 for v in e.outvars):
                return False
        return True

    if not simple(loop.jaxpr.eqns):
        return {}
    return {"workflow_stream_length": arrays[0].shape[0], "workflow_streams": len(arrays)}


class WorkflowPredictor(CompactForiPredictor):
    @classmethod
    def from_profile(cls, profile_path):
        model = cls.load(Path(__file__).with_name("data") / "fori_cpu.npz")
        raw = Path(profile_path).read_bytes()
        profile = json.loads(raw)
        if profile["base"] != model.version:
            raise ValueError("workflow calibration requires its original base model")
        model.max_streams = int(profile["max_streams"])
        model.trips = np.asarray(profile["trips"], dtype=int)
        model.table = np.asarray(profile["costs"], dtype=float)
        if (
            len(model.trips) != 2
            or not 1024 < model.trips[0] < model.trips[1]
            or model.table.shape != (2, len(FACTORS), 2)
            or not np.all(np.isfinite(model.table))
            or np.any(model.table <= 0)
            or model.max_streams not in (0, 1, 4)
        ):
            raise ValueError("invalid workflow profile")
        model.version = hashlib.sha256(raw).hexdigest()
        model.loop_features = workflow_features
        return model

    def matches(self, row):
        t = row["trip_count"]
        return (
            row["backend"] == self.backend
            and self.trips[0] <= t <= self.trips[1]
            and row.get("workflow_stream_length") == t
            and 0 < row["workflow_streams"] <= self.max_streams
        )

    def supports(self, row):
        return self.matches(row) or super().supports(row)

    def predict_candidates(self, row):
        if not self.matches(row):
            return super().predict_candidates(row)
        weight = (row["trip_count"] - self.trips[0]) / (self.trips[1] - self.trips[0])
        costs = (1 - weight) * self.table[0] + weight * self.table[1]
        return {f: LogCosts(*np.log(cost)) for f, cost in zip(FACTORS, costs, strict=True)}


def workflow_engine(*, profile_path=None):
    """Build a deferred CPU engine for calibrated two-state streaming applications."""
    path = Path(__file__).with_name("data") / "fori_workflows.json" if profile_path is None else profile_path
    return DecisionEngine(
        model_loader=lambda: WorkflowPredictor.from_profile(path), payoff=application_engine().payoff
    )

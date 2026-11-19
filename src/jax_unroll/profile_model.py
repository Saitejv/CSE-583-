"""Opt-in CPU predictor for a previously profiled workload and input variant."""
import math
from collections import defaultdict

from jax_unroll.model import LogCosts, group_configs
from jax_unroll.schema import FACTORS


def _key(row):
    # Body IR alone cannot distinguish runtime index distributions.
    return tuple(row[k] for k in (
        "backend", "device", "jax_version", "jaxlib_version", "workload", "variant",
        "loop_kind", "body_hash", "carry_shapes", "const_shapes", "input_shapes",
        "output_shapes", "dtypes",
    ))


class ProfileCostModel:
    """Transfer cold costs and scale warm costs with nearby iteration counts.

    Scope profiles to one machine and compiler configuration. Workload and variant
    identifiers must describe the same input generation and distribution.
    """
    def __init__(self, rows):
        self.profiles = defaultdict(dict)
        for group in group_configs(rows):
            base = group[1]
            trip = int(base["trip_count"])
            if base["backend"] != "cpu" or trip <= 0:
                raise ValueError("profiles require a positive static CPU trip count")
            for row in group.values():
                if str(row.get("correct_all", True)).lower() != "true":
                    raise ValueError("profile contains incorrect measurements")
                if any(row[k] != base[k] for k in ("jax_version", "jaxlib_version")):
                    raise ValueError("mixed compiler versions in profile")
            costs = {}
            for u, row in group.items():
                pair = tuple(float(row[k]) for k in ("cold_median_ms", "warm_median_us"))
                if not all(math.isfinite(v) and v > 0 for v in pair):
                    raise ValueError("invalid profile timing")
                costs[u] = LogCosts(*(math.log(v) for v in pair))
            profiles = self.profiles[_key(base)]
            if trip in profiles:
                raise ValueError("ambiguous profile for loop signature and trip count")
            profiles[trip] = costs

    def predict_candidates(self, row):
        trip = int(row["trip_count"])
        profiles = self.profiles.get(_key(row), {})
        eligible = [n for n in profiles if n % max(FACTORS) == trip % max(FACTORS)
                    and 0.75 * n <= trip <= n]
        if not eligible:
            return None
        reference = min(eligible, key=lambda n: (abs(n-trip), n))
        adjustment = math.log(trip / reference)
        return {u: LogCosts(cost.log_cold_ms, cost.log_warm_us + adjustment)
                for u, cost in profiles[reference].items()}

    def choose(self, fn, args, *, workload, variant, trip_count, loop_kind,
               B=None, R=100, throughput_only=False):
        """Select from a factor-1 trace and named local profile, defaulting to 1."""
        import jax
        import jaxlib

        from jax_unroll.features import extract_features
        from jax_unroll.selector import select_factor

        row = extract_features(fn, args, trip_count=trip_count, factor=1, loop_kind=loop_kind)
        row.update(workload=workload, variant=variant,
                   jax_version=jax.__version__, jaxlib_version=jaxlib.__version__)
        return select_factor(self.predict_candidates(row), B=B, R=R, throughput_only=throughput_only)


class WarmResidualProfile:
    """Adjust a static fori forecast with one earlier four-factor warm profile.

    Give each input distribution its own engine. The profile is used at longer
    trip counts with the same abstract signature and the same unroll remainder.
    """

    def __init__(self, reference, row, warm_us, *, weight=0.5, margin=0.03):
        from hashlib import sha256

        from jax_unroll.selector import LoopSignature

        values = tuple(map(float, warm_us))
        if len(values) != len(FACTORS) or any(not math.isfinite(t) or t <= 0 for t in values):
            raise ValueError("profile requires one finite, positive warm time per factor")
        if not 0 <= weight <= 1 or not math.isfinite(margin) or margin < 0:
            raise ValueError("invalid profile weight or margin")
        if not reference.supports(row):
            raise ValueError("profile is outside the static model's supported signatures")
        estimated = reference.predict_candidates(row)
        self.reference = reference
        self.signature = LoopSignature.from_features(row)
        self.residual = tuple(math.log(t) - estimated[f].log_warm_us for f, t in zip(FACTORS, values, strict=True))
        self.weight, self.margin = weight, margin
        self.backend = reference.backend
        self.feature_version = reference.feature_version
        self.jax_versions = reference.jax_versions
        self.loop_features = getattr(reference, "loop_features", None)
        self.version = sha256(repr((reference.version, self.signature, values, weight, margin)).encode()).hexdigest()

    def supports(self, row):
        return self.reference.supports(row)

    def matches(self, row):
        from dataclasses import replace

        from jax_unroll.selector import LoopSignature

        target = LoopSignature.from_features(row)
        source_trip = self.signature.trip_count
        return (
            source_trip < target.trip_count <= 4.25 * source_trip
            and (target.trip_count - source_trip) % max(FACTORS) == 0
            and replace(target, trip_count=source_trip) == self.signature
        )

    def forecast(self, row):
        costs = self.reference.predict_candidates(row)
        if not self.matches(row):
            return costs
        return {
            factor: LogCosts(costs[factor].log_cold_ms, costs[factor].log_warm_us + self.weight * residual)
            for factor, residual in zip(FACTORS, self.residual, strict=True)
        }

    def predict_candidates(self, row):
        costs = self.forecast(row)
        if self.matches(row):
            for factor in FACTORS[1:]:
                cost = costs[factor]
                costs[factor] = LogCosts(cost.log_cold_ms, cost.log_warm_us + self.margin)
        return costs

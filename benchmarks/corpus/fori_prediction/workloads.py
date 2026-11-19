"""Registry and predeclared family-level partition for the generated loop bodies."""

from dataclasses import dataclass
from typing import Callable

from benchmarks.corpus.fori_prediction import agy_cases, opencode_cases
from benchmarks.corpus.common import built


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    factory: Callable
    atol: float = 1e-4
    rtol: float = 1e-5

    def build(self, trip, factor, variant):
        fn, args = self.factory(variant, trip, factor)
        return built(fn, *args)


CASES = {
    name: Case(name, family, factory)
    for module in (agy_cases, opencode_cases)
    for name, (family, factory) in module.WORKLOADS.items()
}
HOLDOUT_FAMILIES = frozenset({"table_index", "recurrence", "spatial_update", "linear_algebra"})

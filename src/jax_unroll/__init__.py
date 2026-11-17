"""Compilation-budget-aware loop unroll selection for JAX.

Phase status (see TASKS.md):
    Phase 0: package scaffolding.
    Phase 1: workload registry and measurement sweep.
    Phase 2: Jaxpr features, baselines, dataset, and frozen splits.
    Phase 3: cost model and selector.
    Phase 4: opt-in auto_scan / auto_fori_loop integration.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

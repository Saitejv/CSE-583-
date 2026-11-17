"""Cost-model training and prediction.

Planned for Phase 3 (T3.1-T3.2). Models predict log compile cost C(u) and log
warm latency T(u) from static features plus the candidate factor.
"""

from __future__ import annotations

from typing import Any


class CostModel:
    """Placeholder for the Phase 3 cost model."""

    def fit(self, *args: Any, **kwargs: Any) -> "CostModel":
        raise NotImplementedError(
            "Cost-model training is scheduled for Phase 3 (T3.1)."
        )

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Cost-model prediction is scheduled for Phase 3 (T3.1)."
        )

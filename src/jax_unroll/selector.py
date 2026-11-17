"""Budget-aware unroll-factor selection policy.

Planned for Phase 3 (T3.3-T3.5). The policy minimizes
``C_hat(u) + R * T_hat(u)`` over ``u in {1, 2, 4, 8}`` subject to
``C_hat(u) <= B``, with deterministic tie-breaking toward smaller factors and a
fallback to ``u=1``.
"""

from __future__ import annotations

from typing import Any


def select_factor(*args: Any, **kwargs: Any) -> int:
    """Choose an unroll factor from predicted costs (Phase 3)."""
    raise NotImplementedError(
        "Budget-aware factor selection is scheduled for Phase 3 (T3.3)."
    )

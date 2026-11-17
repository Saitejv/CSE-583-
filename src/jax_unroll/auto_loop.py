"""Opt-in automatic loop wrappers.

Planned for Phase 4 (T4.1). ``auto_scan`` and ``auto_fori_loop`` will inspect
the loop body at trace time, select a factor through the Phase 3 policy, and
delegate to ``jax.lax.scan`` / ``jax.lax.fori_loop`` with ``unroll=u``.
"""

from __future__ import annotations

from typing import Any, Callable


def auto_scan(
    f: Callable[..., Any],
    init: Any,
    xs: Any,
    *,
    B: float | None = None,
    R: int = 1,
    force_factor: int | None = None,
    disable: bool = False,
    **scan_kwargs: Any,
) -> Any:
    """Scan with a predicted unroll factor (Phase 4)."""
    raise NotImplementedError(
        "auto_scan is scheduled for Phase 4 (T4.1); use jax.lax.scan with an "
        "explicit unroll factor for now."
    )


def auto_fori_loop(
    lower: Any,
    upper: Any,
    body_fun: Callable[..., Any],
    init_val: Any,
    *,
    B: float | None = None,
    R: int = 1,
    force_factor: int | None = None,
    disable: bool = False,
    **fori_kwargs: Any,
) -> Any:
    """fori_loop with a predicted unroll factor (Phase 4)."""
    raise NotImplementedError(
        "auto_fori_loop is scheduled for Phase 4 (T4.1); use "
        "jax.lax.fori_loop with an explicit unroll factor for now."
    )

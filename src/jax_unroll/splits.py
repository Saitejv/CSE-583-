"""Frozen loop-family splits for Phase 2 onward.

The held-out families are frozen *before* any cost-model work. All model
selection (including leave-one-family-out cross-validation) happens inside
``DEV_FAMILIES``. The held-out families are reserved for frozen Week 3 gates
and final evaluation. ``SPLIT_VERSION`` and ``SPLIT_FROZEN_DATE`` record when the
split was frozen. See ``docs/splits.md``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

__all__ = [
    "ALL_FAMILIES",
    "DEV_FAMILIES",
    "HELD_OUT_FAMILIES",
    "SPLIT_FROZEN_DATE",
    "SPLIT_VERSION",
    "assert_holdout_excluded",
    "lofo_folds",
    "manifest",
    "split_for_family",
]

ALL_FAMILIES: tuple[str, ...] = (
    "recurrence",
    "elementwise",
    "reduction",
    "dense_linear_algebra",
    "indexed_memory",
    "iterative_solver",
)

HELD_OUT_FAMILIES: tuple[str, ...] = (
    "indexed_memory",
    "iterative_solver",
)

DEV_FAMILIES: tuple[str, ...] = tuple(
    family for family in ALL_FAMILIES if family not in HELD_OUT_FAMILIES
)

SPLIT_VERSION = 1
SPLIT_FROZEN_DATE = "2026-11-17"

SPLIT_DEV = "dev"
SPLIT_HOLDOUT = "holdout"

_SPLIT_BY_FAMILY: dict[str, str] = {
    **{family: SPLIT_DEV for family in DEV_FAMILIES},
    **{family: SPLIT_HOLDOUT for family in HELD_OUT_FAMILIES},
}


def split_for_family(family: str) -> str:
    """Return ``"dev"`` or ``"holdout"`` for a loop family."""

    try:
        return _SPLIT_BY_FAMILY[family]
    except KeyError as exc:
        raise ValueError(
            f"unknown family {family!r}; expected one of {list(ALL_FAMILIES)}"
        ) from exc


def lofo_folds(
    families: Iterable[str] = DEV_FAMILIES,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return leave-one-family-out folds over the development families only."""

    training_families = tuple(families)
    if not training_families:
        raise ValueError("at least one development family is required")
    unknown = set(training_families).difference(ALL_FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    assert_holdout_excluded(training_families)

    folds: dict[str, dict[str, tuple[str, ...]]] = {}
    for held_out in training_families:
        train = tuple(
            family for family in training_families if family != held_out
        )
        if not train:
            raise ValueError("LOFO needs at least two development families")
        folds[f"lofo_{held_out}"] = {"train": train, "test": (held_out,)}
    return folds


def assert_holdout_excluded(families: Iterable[str]) -> None:
    """Raise if any training family is in the frozen held-out set."""

    overlap = set(families).intersection(HELD_OUT_FAMILIES)
    if overlap:
        raise ValueError(
            "held-out families must not be used for training or model "
            f"selection: {sorted(overlap)}"
        )


def manifest() -> Mapping[str, object]:
    """JSON-serializable record of the frozen split."""

    folds = lofo_folds()
    return {
        "split_version": SPLIT_VERSION,
        "frozen_date": SPLIT_FROZEN_DATE,
        "all_families": list(ALL_FAMILIES),
        "dev_families": list(DEV_FAMILIES),
        "held_out_families": list(HELD_OUT_FAMILIES),
        "final_dev": list(DEV_FAMILIES),
        "final_holdout": list(HELD_OUT_FAMILIES),
        "lofo_folds": {
            name: {
                "train": list(fold["train"]),
                "test": list(fold["test"]),
            }
            for name, fold in folds.items()
        },
    }

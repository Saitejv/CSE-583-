from __future__ import annotations

import re
from pathlib import Path

import pytest

import jax_unroll
from jax_unroll import auto_loop, features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_and_has_version():
    assert hasattr(jax_unroll, "__version__")
    assert isinstance(jax_unroll.__version__, str)
    assert jax_unroll.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", jax_unroll.__version__)


def test_phase_two_feature_api_is_implemented():
    assert callable(features.extract_features)
    assert callable(features.extract_features_from_workload)
    assert callable(features.extract_loop_body)


def test_auto_scan_is_a_phase_four_stub():
    with pytest.raises(NotImplementedError) as excinfo:
        auto_loop.auto_scan(lambda carry, x: (carry, None), 0.0, [1.0])
    message = str(excinfo.value)
    assert "Phase 4" in message
    assert "auto_scan" in message


def test_auto_fori_loop_is_a_phase_four_stub():
    with pytest.raises(NotImplementedError) as excinfo:
        auto_loop.auto_fori_loop(0, 4, lambda i, carry: carry, 0.0)
    message = str(excinfo.value)
    assert "Phase 4" in message
    assert "auto_fori_loop" in message


@pytest.mark.parametrize(
    "target",
    [
        "test",
        "list",
        "sweep",
        "sweep-shapes",
        "sweep-all",
        "oracle",
        "dataset",
        "quick",
        "train",
        "evaluate",
        "clean-results",
    ],
)
def test_makefile_declares_target(target):
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(rf"(?m)^{re.escape(target)}:", makefile)

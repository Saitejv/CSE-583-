"""Tests for expansion registry integration, manifest, and command-line execution."""

import json
import subprocess
import sys
from pathlib import Path

from benchmarks.corpus.fori_expansion.registry import (
    ALL_FORI_SPECS,
    CONVERTED_WORKLOAD_SPECS,
    HISTORICAL_FORI_SPECS,
    NEW_INDEPENDENT_SPECS,
    build_workload,
    get_spec,
    manifest,
)


def test_registry_composition():
    assert len(ALL_FORI_SPECS) == 136
    assert len(HISTORICAL_FORI_SPECS) == 13
    assert len(CONVERTED_WORKLOAD_SPECS) == 53
    assert len(NEW_INDEPENDENT_SPECS) == 70

    for spec in ALL_FORI_SPECS:
        assert spec.loop_kind == "fori_loop"
        wl = build_workload(spec.name, 64, 1)
        assert wl.loop_kind == "fori_loop"
        assert wl.name == spec.name


def test_manifest_structure():
    m = manifest()
    assert m["total_workloads"] == 136
    assert m["historical_fori"] == 13
    assert m["converted_scans"] == 53
    assert m["new_independent"] == 70
    assert len(m["workloads"]) == 136


def test_fresh_child_execution():
    """Smoke-test that a workload unique to the expansion runs via CLI in a fresh child process."""
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.corpus.fori_expansion.sweep",
        "--workloads",
        "sig_goertzel",
        "--trip-counts",
        "64",
        "--factors",
        "1",
        "--repeats",
        "1",
        "--shapes",
        "primary",
        "--out",
        "results/raw/test_child_smoke.csv",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Child execution failed:\n{res.stderr}\n{res.stdout}"
    assert Path("results/raw/test_child_smoke.csv").exists()
    content = Path("results/raw/test_child_smoke.csv").read_text()
    assert "sig_goertzel" in content
    assert "fori_loop" in content
    # Clean up test file
    Path("results/raw/test_child_smoke.csv").unlink(missing_ok=True)

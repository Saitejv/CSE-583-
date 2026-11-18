from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import jax
import numpy as np
import pytest

from benchmarks.corpus import registry
from benchmarks.corpus.registry import (
    CORPUS_FAMILIES,
    CORPUS_WORKLOADS,
    SPECS,
    body_hashes,
    build_workload,
    get_spec,
    iter_specs,
    manifest,
    related_groups,
    tier_specs,
    validate_structure,
)
from benchmarks.corpus.spec import (
    CORPUS_FACTORS,
    CORPUS_TRIP_COUNTS,
    SMOKE_TRIP_COUNTS,
)
from benchmarks.corpus.unsupported import (
    UNSUPPORTED_CASES,
    UnsupportedWorkloadError,
    check_supported,
)
from benchmarks.sweep import enumerate_configs
from benchmarks.workloads import WORKLOADS
from benchmarks.recovery_workloads import RECOVERY_WORKLOADS
from jax_unroll.schema import FACTORS

MANIFEST_PATH = Path(registry.__file__).with_name("manifest.json")
_CACHE: dict[tuple[str, int, int, str], tuple[object, object]] = {}


def _run(spec, trip: int, factor: int, variant: str | None = None):
    variant = variant or spec.primary_variant
    key = (spec.name, trip, factor, variant)
    if key not in _CACHE:
        workload = spec.build(trip, factor, variant)
        compiled = jax.jit(workload.fn).lower(*workload.args).compile()
        output = compiled(*workload.args)
        jax.block_until_ready(output)
        _CACHE[key] = (workload, output)
    return _CACHE[key]


def _assert_same(got, want, *, atol: float, rtol: float, label: str) -> None:
    got_leaves = jax.tree.leaves(got)
    want_leaves = jax.tree.leaves(want)
    assert len(got_leaves) == len(want_leaves)
    for index, (left, right) in enumerate(zip(got_leaves, want_leaves, strict=True)):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        assert np.all(np.isfinite(left_array)), f"{label} leaf {index} is non-finite"
        np.testing.assert_allclose(
            left_array,
            right_array,
            atol=atol,
            rtol=rtol,
            err_msg=f"{label} leaf {index} differs from factor 1",
        )


def _scan_unroll(workload) -> int:
    jaxpr = jax.make_jaxpr(workload.fn)(*workload.args).jaxpr
    scans = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "scan"]
    assert len(scans) == 1
    return int(scans[0].params["unroll"])


def test_registry_has_target_size_and_family_coverage():
    summary = validate_structure()
    assert summary == {
        "families": 10,
        "bodies": 47,
        "smoke_bodies": 12,
        "related_groups": len(related_groups()),
    }
    assert len(CORPUS_FAMILIES) == 10
    assert 8 <= len(CORPUS_FAMILIES) <= 12
    assert 40 <= len(SPECS) <= 60
    assert set(CORPUS_FAMILIES) == {spec.family for spec in SPECS}
    assert len(CORPUS_WORKLOADS) == len(SPECS)
    assert [spec.name for spec in iter_specs(["stencil"])] == [
        spec.name for spec in SPECS if spec.family == "stencil"
    ]
    with pytest.raises(ValueError, match="unknown corpus families"):
        iter_specs(["bogus"])
    with pytest.raises(KeyError, match="unknown corpus workload"):
        get_spec("bogus")
    with pytest.raises(ValueError, match="unknown tier"):
        tier_specs("bogus")


def test_names_do_not_collide_with_frozen_registries():
    frozen = set(WORKLOADS) | set(RECOVERY_WORKLOADS)
    assert not set(CORPUS_WORKLOADS) & frozen


def test_smoke_tier_covers_every_family_and_is_subset_of_full():
    smoke = tier_specs("smoke")
    full = tier_specs("full")
    assert 8 <= len(smoke) <= 15
    assert {spec.family for spec in smoke} == set(CORPUS_FAMILIES)
    assert {spec.name for spec in smoke} <= {spec.name for spec in full}
    assert len(full) == len(SPECS)


def test_manifest_tier_counts_and_commands():
    data = manifest()
    smoke = data["tiers"]["smoke"]
    full = data["tiers"]["full"]
    expected_smoke = (
        len(tier_specs("smoke"))
        * len(SMOKE_TRIP_COUNTS)
        * len(CORPUS_FACTORS)
        * smoke["process_repeats"]
    )
    expected_full = (
        sum(len(spec.shape_variants) * len(spec.trip_counts) for spec in SPECS)
        * len(CORPUS_FACTORS)
        * full["process_repeats"]
    )
    assert smoke["measurements"] == expected_smoke == 96
    assert full["measurements"] == expected_full == 4560
    assert smoke["process_repeats"] == 1
    assert full["process_repeats"] == 3
    assert smoke["command"] == "make corpus-smoke"
    assert full["command"] == "make corpus-full"
    assert data["factors"] == list(FACTORS) == list(CORPUS_FACTORS)
    assert data["tiers"]["full"]["trip_counts"] == list(CORPUS_TRIP_COUNTS)


def test_manifest_contains_provenance_and_grouping():
    data = manifest()
    assert {
        row["workload"] for row in data["feature_caveats"]
    } == {"scatter_collision_add", "scatter_min_update"}
    assert data["body_count"] == len(SPECS)
    assert data["family_count"] == len(CORPUS_FAMILIES)
    assert {row["workload"] for row in data["workloads"]} == set(CORPUS_WORKLOADS)
    kinds = Counter(row["provenance"]["kind"] for row in data["workloads"])
    assert kinds["original"] >= 20
    assert kinds["public-inspired"] >= 10
    for row in data["workloads"]:
        assert row["related_group"]
        assert row["provenance"]["source"]
        assert row["provenance"]["license"]
        assert row["variants"]
        assert row["primary_variant"] in row["variants"]
    groups = related_groups()
    assert set(groups) == {spec.related_group for spec in SPECS}
    assert {name for names in groups.values() for name in names} == set(CORPUS_WORKLOADS)
    unsupported = {row["case"] for row in data["unsupported"]}
    assert unsupported == {case.name for case in UNSUPPORTED_CASES}


def test_manifest_file_matches_registry_and_hashes():
    assert MANIFEST_PATH.exists(), "run python -m benchmarks.corpus.manifest"
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert data["body_count"] == len(SPECS)
    assert data["family_count"] == len(CORPUS_FAMILIES)
    entries = {row["workload"]: row for row in data["workloads"]}
    assert set(entries) == set(CORPUS_WORKLOADS)
    hashes = body_hashes()
    assert {name: entries[name]["body_hash"] for name in hashes} == hashes
    assert data == json.loads(json.dumps(manifest(hashes)))


def test_body_hashes_are_distinct():
    hashes = body_hashes()
    assert len(hashes) == len(SPECS)
    assert len(set(hashes.values())) == len(SPECS)


def test_body_hashes_reject_frozen_duplicates(monkeypatch):
    duplicate = next(iter(body_hashes().values()))
    monkeypatch.setattr(
        registry, "frozen_body_hashes", lambda: {"fake:frozen": duplicate}
    )
    with pytest.raises(ValueError, match="duplicate frozen Jaxpr bodies"):
        body_hashes()


def test_every_body_is_a_single_static_scan():
    for spec in SPECS:
        workload = spec.build(64, 1)
        check_supported(workload.fn, workload.args)


@pytest.mark.parametrize("name", sorted(CORPUS_WORKLOADS))
def test_variants_change_argument_shapes(name):
    spec = CORPUS_WORKLOADS[name]
    shapes = {}
    for variant in sorted(spec.shape_variants):
        workload = spec.build(64, 1, variant=variant)
        shapes[variant] = tuple(tuple(np.shape(leaf)) for leaf in workload.args)
    assert len(shapes) >= 2
    assert len(set(shapes.values())) == len(shapes)
    assert shapes[spec.primary_variant]


@pytest.mark.parametrize("name", sorted(CORPUS_WORKLOADS))
@pytest.mark.parametrize("factor", (2, 4, 8))
def test_every_body_matches_factor_one_on_divisible_trip(name, factor):
    spec = CORPUS_WORKLOADS[name]
    _, expected = _run(spec, 64, 1)
    workload, output = _run(spec, 64, factor)
    assert workload.divides_trip is True
    assert workload.trip_count == 64
    assert workload.factor == factor
    _assert_same(
        output,
        expected,
        atol=spec.atol,
        rtol=spec.rtol,
        label=f"{name}/T=64/u={factor}",
    )


@pytest.mark.parametrize("name", sorted(CORPUS_WORKLOADS))
def test_every_body_matches_factor_one_on_remainder_trip(name):
    spec = CORPUS_WORKLOADS[name]
    _, expected = _run(spec, 100, 1)
    workload, output = _run(spec, 100, 8)
    assert workload.divides_trip is False
    _assert_same(
        output,
        expected,
        atol=spec.atol,
        rtol=spec.rtol,
        label=f"{name}/T=100/u=8",
    )


@pytest.mark.parametrize("name", sorted(spec.name for spec in tier_specs("smoke")))
@pytest.mark.parametrize("factor", (2, 4, 8))
def test_smoke_bodies_match_factor_one_on_1023_trip(name, factor):
    spec = CORPUS_WORKLOADS[name]
    _, expected = _run(spec, 1023, 1)
    workload, output = _run(spec, 1023, factor)
    assert workload.divides_trip is False
    _assert_same(
        output,
        expected,
        atol=spec.atol,
        rtol=spec.rtol,
        label=f"{name}/T=1023/u={factor}",
    )


@pytest.mark.parametrize(
    "name",
    sorted(
        set(CORPUS_WORKLOADS)
        - {spec.name for spec in tier_specs("smoke")}
    ),
)
def test_non_smoke_bodies_match_factor_one_on_1023_trip(name):
    spec = CORPUS_WORKLOADS[name]
    _, expected = _run(spec, 1023, 1)
    workload, output = _run(spec, 1023, 8)
    assert workload.divides_trip is False
    _assert_same(
        output,
        expected,
        atol=spec.atol,
        rtol=spec.rtol,
        label=f"{name}/T=1023/u=8",
    )


@pytest.mark.parametrize("name", sorted(CORPUS_WORKLOADS))
def test_secondary_variants_match_factor_one(name):
    spec = CORPUS_WORKLOADS[name]
    for variant in sorted(spec.shape_variants):
        if variant == spec.primary_variant:
            continue
        _, expected = _run(spec, 64, 1, variant=variant)
        _, output = _run(spec, 64, 8, variant=variant)
        _assert_same(
            output,
            expected,
            atol=spec.atol,
            rtol=spec.rtol,
            label=f"{name}/{variant}/T=64/u=8",
        )


@pytest.mark.parametrize("name", sorted(CORPUS_WORKLOADS))
@pytest.mark.parametrize("factor", CORPUS_FACTORS)
def test_loop_unroll_matches_requested_factor(name, factor):
    spec = CORPUS_WORKLOADS[name]
    workload = spec.build(64, factor)
    assert _scan_unroll(workload) == factor


@pytest.mark.parametrize("name", sorted(CORPUS_WORKLOADS))
def test_feature_extraction_covers_every_body(name):
    from jax_unroll.features import extract_features_from_workload

    spec = CORPUS_WORKLOADS[name]
    features = extract_features_from_workload(spec.build(64, 1))
    assert features["body_hash"]
    assert features["num_equations"] > 0


def test_rolling_mean_matches_independent_numpy_oracle():
    spec = get_spec("rolling_mean_shift")
    workload = spec.build(37, 1, "w8")
    _, output = _run(spec, 37, 1, variant="w8")
    window0, _, xs = (np.asarray(leaf) for leaf in workload.args)
    expected = np.mean(np.concatenate([window0, xs])[-window0.shape[0] :])
    np.testing.assert_allclose(
        np.asarray(output), expected, atol=spec.atol, rtol=spec.rtol
    )


@pytest.mark.parametrize("name", sorted(CORPUS_WORKLOADS))
def test_secondary_variants_run_finite(name):
    spec = CORPUS_WORKLOADS[name]
    for variant in sorted(spec.shape_variants):
        if variant == spec.primary_variant:
            continue
        _, output = _run(spec, 64, 1, variant=variant)
        assert all(
            np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(output)
        )


def test_smoke_sweep_configuration_uses_existing_harness():
    configs = enumerate_configs(
        specs=tier_specs("smoke"),
        shapes="primary",
        trip_counts=SMOKE_TRIP_COUNTS,
        factors=CORPUS_FACTORS,
        repeats=1,
        seed=583,
    )
    assert len(configs) == 96
    assert {config["family"] for config in configs} == set(CORPUS_FAMILIES)
    assert {config["trip_count"] for config in configs} == set(SMOKE_TRIP_COUNTS)
    assert {config["factor"] for config in configs} == set(CORPUS_FACTORS)
    assert all(config["workload"] in CORPUS_WORKLOADS for config in configs)


@pytest.mark.parametrize("case", UNSUPPORTED_CASES, ids=lambda case: case.name)
def test_unsupported_cases_are_rejected(case):
    assert case.name not in CORPUS_WORKLOADS
    fn, args = case.build()
    with pytest.raises(UnsupportedWorkloadError) as excinfo:
        check_supported(fn, args)
    expected = {
        "dynamic_trip_fori": "while",
        "while_pointer_chase": "while",
        "nested_scan": "nested",
        "nested_cond": "nested",
        "dynamic_scan_length": "static",
        "dynamic_nonzero_shape": "static",
        "debug_print_effect": "effectful",
    }[case.name]
    assert expected in str(excinfo.value).lower()


def test_corpus_parent_launches_corpus_child_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "corpus_child.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
    env["JAX_PLATFORMS"] = "cpu"
    command = [
        sys.executable,
        "-m",
        "benchmarks.corpus.sweep",
        "--tier",
        "smoke",
        "--limit",
        "1",
        "--trip-counts",
        "64",
        "--factors",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--sample-budget-ms",
        "1",
        "--min-samples",
        "1",
        "--max-samples",
        "1",
        "--r-calls",
        "1",
        "--out",
        str(out),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["error"] == ""
    assert rows[0]["correct"].lower() == "true"


def test_build_workload_helper_validates_variants_and_trips():
    workload = build_workload("gather_scalar_table", 64, 2)
    assert workload.family == "indirect_gather"
    with pytest.raises(ValueError, match="unknown variant"):
        build_workload("gather_scalar_table", 64, 2, variant="bogus")
    with pytest.raises(ValueError, match="trip_count must be positive"):
        build_workload("gather_scalar_table", 0, 1)


def test_makefile_declares_corpus_targets():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(
        encoding="utf-8"
    )
    for target in ("corpus-validate", "corpus-manifest", "corpus-smoke", "corpus-full"):
        assert f"\n{target}:" in makefile

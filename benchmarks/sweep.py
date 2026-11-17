"""Factor-sweep harness for fixed-trip-count JAX loops.

One configuration is measured in one fresh Python process. The parent process
enumerates configurations, randomizes order per process repeat, launches one
child per measurement, and appends each row to a CSV. Each child measures:

    tracing (``jax.make_jaxpr``), lowering (``.lower``), backend compilation
    (``.compile``), warm execution with ``block_until_ready``, end-to-end time
    for ``R`` calls, StableHLO/optimized-HLO text sizes, and numerical
    correctness against ``u=1``.

The harness never benchmarks factors at decision time; this is offline data
collection only. See ``docs/data_schema.md`` for column definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from benchmarks.workloads import (
    FAMILIES,
    WORKLOADS,
    Workload,
    build_workload,
    iter_specs,
    registry_table,
)
from jax_unroll.schema import (
    FACTORS,
    SWEEP_COLUMNS,
    TRIP_COUNTS,
    default_sweep_path,
    ensure_parent,
)

ROW_PREFIX = "ROW_JSON:"


def _env_flag(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _leaves(tree: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]


def _max_abs_diff(a: Any, b: Any) -> float:
    diffs = [
        float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        for x, y in zip(_leaves(a), _leaves(b), strict=True)
    ]
    return max(diffs) if diffs else 0.0


def _allclose(a: Any, b: Any, *, atol: float, rtol: float) -> bool:
    pairs = zip(_leaves(a), _leaves(b), strict=True)
    return all(
        bool(np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True))
        for x, y in pairs
    )


def _warm_backend() -> None:
    trivial = jax.jit(lambda x: x + jnp.float32(1.0))
    lowered = trivial.lower(jnp.float32(0.0))
    lowered.compile()(jnp.float32(0.0)).block_until_ready()
    jax.clear_caches()


def _run_once(compiled: Any, args: Sequence[Any]) -> Any:
    out = compiled(*args)
    jax.block_until_ready(out)
    return out


def _adaptive_sample_count(
    compiled: Any,
    args: Sequence[Any],
    *,
    sample_budget_s: float,
    min_samples: int,
    max_samples: int,
) -> tuple[int, float]:
    pilot_times = []
    for _ in range(3):
        start = time.perf_counter()
        _run_once(compiled, args)
        pilot_times.append(time.perf_counter() - start)
    per_call = statistics.median(pilot_times)
    if per_call <= 0:
        return max_samples, per_call
    count = int(sample_budget_s / per_call)
    return max(min_samples, min(max_samples, count)), per_call


def measure_config(
    config: dict[str, Any],
    *,
    warmup: int,
    sample_budget_s: float,
    min_samples: int,
    max_samples: int,
    r_calls: int,
) -> dict[str, Any]:
    workload: Workload = build_workload(
        config["workload"],
        int(config["trip_count"]),
        int(config["factor"]),
        config["variant"],
    )
    row = _empty_row(config)
    row.update(workload.signature())
    notes: list[str] = []

    _warm_backend()

    fn = workload.fn
    args = workload.args
    jitted = jax.jit(fn)

    start = time.perf_counter()
    jax.make_jaxpr(fn)(*args)
    trace_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    lowered = jitted.lower(*args)
    lower_wall_ms = (time.perf_counter() - start) * 1e3
    lower_ms = max(lower_wall_ms - trace_ms, 0.0)

    stablehlo_text = lowered.as_text()
    stablehlo_bytes = len(stablehlo_text.encode("utf-8"))

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = (time.perf_counter() - start) * 1e3

    hlo_bytes: int | None
    try:
        hlo_bytes = len(compiled.as_text().encode("utf-8"))
    except Exception as exc:
        hlo_bytes = None
        notes.append(f"optimized HLO text unavailable: {type(exc).__name__}")

    cold_total_ms = trace_ms + lower_ms + compile_ms

    _run_once(compiled, args)
    for _ in range(max(0, warmup)):
        _run_once(compiled, args)

    sample_count, pilot_us = _adaptive_sample_count(
        compiled,
        args,
        sample_budget_s=sample_budget_s,
        min_samples=min_samples,
        max_samples=max_samples,
    )
    warm_times_us: list[float] = []
    last_output = None
    for _ in range(sample_count):
        start = time.perf_counter()
        last_output = _run_once(compiled, args)
        warm_times_us.append((time.perf_counter() - start) * 1e6)

    warm_median_us = statistics.median(warm_times_us)
    warm_p95_us = float(np.percentile(warm_times_us, 95))
    warm_min_us = min(warm_times_us)
    warm_std_us = statistics.pstdev(warm_times_us) if len(warm_times_us) > 1 else 0.0

    start = time.perf_counter()
    for _ in range(r_calls):
        last_output = _run_once(compiled, args)
    e2e_calls_ms = (time.perf_counter() - start) * 1e3
    end_to_end_ms = cold_total_ms + e2e_calls_ms

    if workload.factor == 1:
        correct = True
        max_diff = 0.0
    else:
        reference = build_workload(
            config["workload"],
            int(config["trip_count"]),
            1,
            config["variant"],
        )
        reference_compiled = jax.jit(reference.fn).lower(*reference.args).compile()
        reference_output = _run_once(reference_compiled, reference.args)
        correct = _allclose(
            last_output,
            reference_output,
            atol=reference.atol,
            rtol=reference.rtol,
        )
        max_diff = _max_abs_diff(last_output, reference_output)

    notes.append(f"pilot_call_us={pilot_us * 1e6:.3f}")
    notes.append(f"adaptive_samples={sample_count}")
    row.update(
        {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "trip_count": workload.trip_count,
            "shape": workload.shape,
            "dtype": workload.dtype,
            "factor": workload.factor,
            "divides_trip": workload.divides_trip,
            "trace_ms": round(trace_ms, 6),
            "lower_ms": round(lower_ms, 6),
            "compile_ms": round(compile_ms, 6),
            "cold_total_ms": round(cold_total_ms, 6),
            "warm_median_us": round(warm_median_us, 3),
            "warm_p95_us": round(warm_p95_us, 3),
            "warm_min_us": round(warm_min_us, 3),
            "warm_std_us": round(warm_std_us, 3),
            "warm_samples": sample_count,
            "end_to_end_ms_R": round(end_to_end_ms, 6),
            "r_calls": r_calls,
            "stablehlo_bytes": stablehlo_bytes,
            "hlo_bytes": hlo_bytes if hlo_bytes is not None else "",
            "correct": correct,
            "max_abs_diff": round(max_diff, 10),
            "error": "",
            "notes": "; ".join(notes),
        }
    )
    return row


def _empty_row(config: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in SWEEP_COLUMNS}
    for key in (
        "workload",
        "family",
        "variant",
        "loop_kind",
        "trip_count",
        "factor",
        "process_repeat",
    ):
        if key in config:
            row[key] = config[key]
    return row


def enumerate_configs(
    *,
    families: Iterable[str] | None = None,
    shapes: str = "primary",
    trip_counts: Sequence[int] = TRIP_COUNTS,
    factors: Sequence[int] = FACTORS,
    repeats: int = 3,
    seed: int = 583,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if shapes not in {"primary", "secondary", "all"}:
        raise ValueError("shapes must be 'primary', 'secondary', or 'all'")

    base: list[dict[str, Any]] = []
    for spec in iter_specs(families):
        if shapes == "primary":
            variants = [spec.primary_variant]
        elif shapes == "secondary":
            variants = [
                variant
                for variant in sorted(spec.shape_variants)
                if variant != spec.primary_variant
            ]
        else:
            variants = sorted(spec.shape_variants)
        for variant in variants:
            for trip_count in trip_counts:
                for factor in factors:
                    base.append(
                        {
                            "workload": spec.name,
                            "family": spec.family,
                            "variant": variant,
                            "loop_kind": spec.loop_kind,
                            "trip_count": int(trip_count),
                            "factor": int(factor),
                        }
                    )

    ordered: list[dict[str, Any]] = []
    for repeat in range(repeats):
        batch = list(base)
        random.Random(seed + repeat).shuffle(batch)
        for config in batch:
            row = dict(config)
            row["process_repeat"] = repeat
            ordered.append(row)
    if limit is not None:
        ordered = ordered[: max(0, limit)]
    return ordered


def _parse_int_list(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return tuple(int(part) for part in parts)


def _parse_families(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    families = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(families).difference(FAMILIES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown families {sorted(unknown)}; available: {list(FAMILIES)}"
        )
    return families


def _child(command_config: dict[str, Any], args: argparse.Namespace) -> int:
    try:
        with jax.disable_jit(False):
            row = measure_config(
                command_config,
                warmup=args.warmup,
                sample_budget_s=args.sample_budget_ms / 1e3,
                min_samples=args.min_samples,
                max_samples=args.max_samples,
                r_calls=args.r_calls,
            )
    except Exception as exc:
        row = _empty_row(command_config)
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["notes"] = "measurement failed in child process"
    print(ROW_PREFIX + json.dumps(row, default=str))
    return 0 if not row.get("error") else 1


def _parse_row(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(ROW_PREFIX):
            return json.loads(line[len(ROW_PREFIX) :])
    return None


def _launch(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.sweep",
        "--single-json",
        json.dumps(config),
        "--warmup",
        str(args.warmup),
        "--sample-budget-ms",
        str(args.sample_budget_ms),
        "--min-samples",
        str(args.min_samples),
        "--max-samples",
        str(args.max_samples),
        "--r-calls",
        str(args.r_calls),
    ]
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    source_root = str(Path(project_root) / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [project_root, source_root, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    if args.backend in {"cpu", "gpu", "tpu"}:
        env["JAX_PLATFORMS"] = args.backend

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        row = _empty_row(config)
        row["error"] = f"child timeout after {args.timeout_s:.0f}s"
        row["notes"] = str(exc)
        return row, elapsed
    elapsed = time.perf_counter() - start
    row = _parse_row(completed.stdout)
    if row is None:
        row = _empty_row(config)
        stderr_tail = completed.stderr.strip().splitlines()[-8:]
        row["error"] = (
            f"child exit={completed.returncode}; no row emitted; "
            + " | ".join(stderr_tail)
        )
        row["notes"] = "harness failure"
    return row, elapsed


def _write_header(writer: csv.DictWriter, path: Path) -> None:
    writer.writeheader()
    path.parent.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    if args.list:
        print(json.dumps(registry_table(), indent=2))
        return 0

    families = _parse_families(args.families)
    trip_counts = _parse_int_list(args.trip_counts)
    factors = _parse_int_list(args.factors)
    configs = enumerate_configs(
        families=families,
        shapes=args.shapes,
        trip_counts=trip_counts,
        factors=factors,
        repeats=args.repeats,
        seed=args.seed,
        limit=args.limit,
    )
    out_path = Path(args.out)
    ensure_parent(out_path)
    append = args.append and out_path.exists()
    mode = "a" if append else "w"
    path_for_write = out_path
    total = len(configs)
    print(
        f"sweep: {total} measurements, backend={args.backend}, "
        f"repeats={args.repeats}, out={out_path}"
    )
    if total == 0:
        print("sweep: no configurations selected")
        return 0

    errors = 0
    correct = 0
    started = time.perf_counter()
    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SWEEP_COLUMNS, extrasaction="ignore"
        )
        if not append:
            _write_header(writer, path_for_write)
        elif handle.tell() == 0:
            _write_header(writer, path_for_write)
        for index, config in enumerate(configs, start=1):
            row, elapsed = _launch(config, args)
            writer.writerow(row)
            handle.flush()
            errors += int(bool(row.get("error")))
            correct += int(str(row.get("correct", "")).lower() == "true")
            warm = row.get("warm_median_us", "")
            warm_text = f"{warm} us" if warm != "" else "n/a"
            print(
                f"[{index}/{total}] r{config['process_repeat']} "
                f"{config['workload']}/{config['variant']} "
                f"T={config['trip_count']} u={config['factor']} "
                f"warm={warm_text} child={elapsed:.1f}s"
                + (f" ERROR={row['error']}" if row.get("error") else "")
            )
    wall_s = time.perf_counter() - started
    print(
        f"sweep: wrote {total} rows to {out_path} in {wall_s / 60:.1f} min "
        f"({errors} errors, {correct} correctness passes)"
    )
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=_env_flag("JAX_PLATFORMS", "cpu"))
    parser.add_argument(
        "--out",
        default=str(default_sweep_path(_env_flag("JAX_PLATFORMS", "cpu"), date.today().isoformat())),
        help="raw CSV output path",
    )
    parser.add_argument("--families", default=None, help="comma-separated family names")
    parser.add_argument("--shapes", choices=("primary", "secondary", "all"), default="primary")
    parser.add_argument(
        "--trip-counts",
        default=",".join(str(value) for value in TRIP_COUNTS),
    )
    parser.add_argument("--factors", default=",".join(str(value) for value in FACTORS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=583)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-budget-ms", type=float, default=250.0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--r-calls", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the workload registry")
    parser.add_argument("--single-json", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.single_json is not None:
        return _child(json.loads(args.single_json), args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

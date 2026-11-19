"""Score the committed recovery model once, or verify its recorded evidence."""

import argparse
import json
import os
import platform
import subprocess
import time

import jax
import numpy as np

from benchmarks.recovery_data import (
    DATA, MODEL, PROTOCOL, RESULTS, config_key, dump, joined_rows, prepare, read_csv, require_cpu, sha, write_csv,
)
from benchmarks.recovery_metrics import calibration, confirmation_summary, paired_rows
from benchmarks.workloads import build_workload
from jax_unroll.model import CostModel
from jax_unroll.recovery_split import CONFIRMATION, TRIPS, VARIANTS
from jax_unroll.schema import utc_timestamp
from jax_unroll.selector import DecisionEngine
from jax_unroll.validation import forecast

RAW = DATA / "confirmation_raw.csv"
JOINED = DATA / "confirmation.csv"
OVERHEAD = RESULTS / "confirmation_overhead.json"
PAIRS = RESULTS / "confirmation_pairs.csv"
SUMMARY = RESULTS / "confirmation_report.json"
HLO = RESULTS / "confirmation_hlo.json"
LOG = RESULTS / "confirmation_gate.jsonl"
REPEATS = 31


def _git(*args):
    return subprocess.check_output(["git", *args])


def require_frozen(*, check_sources=True):
    commit = _git("log", "-1", "--format=%H", "HEAD", "--", str(MODEL)).decode().strip()
    if not commit:
        raise ValueError("commit the selected model before accessing confirmation labels")
    artifact = json.loads(MODEL.read_bytes())
    provenance = artifact["dev_provenance"]
    comparisons = artifact["comparisons"]
    paths = {
        MODEL: sha(MODEL), PROTOCOL: artifact["protocol_sha256"],
        DATA / "dev_raw.csv": provenance["raw_sha256"], DATA / "dev.csv": provenance["joined_sha256"],
        RESULTS / "dev_report.json": comparisons["dev_report_sha256"],
        RESULTS / "dev_pairs.csv": comparisons["dev_pairs_sha256"],
    }
    frozen_paths = [*paths, DATA / "dev.meta.json"]
    if check_sources:
        frozen_paths.extend([
            "benchmarks/recovery_confirm.py", "benchmarks/recovery_sweep.py", "benchmarks/recovery_workloads.py",
            "benchmarks/recovery_data.py", "benchmarks/recovery_metrics.py", "benchmarks/recovery_train.py",
            "benchmarks/dataset.py", "benchmarks/sweep.py", "benchmarks/workloads.py", "benchmarks/oracle.py",
            "src/jax_unroll",
        ])
    if subprocess.run(["git", "diff", "--quiet", commit, "--", *map(str, frozen_paths)], check=False).returncode != 0:
        raise ValueError("confirmation requires unchanged committed protocol, inputs, model and runner")
    if any(sha(path) != expected for path, expected in paths.items()):
        raise ValueError("frozen recovery provenance mismatch")
    return commit


def _append(handle, record):
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1e6


def _stats(samples):
    return {"median": float(np.median(samples)), "p95": float(np.quantile(samples, 0.95))}


def measure_overhead(model, cases):
    pooled = {stage: [] for stage in ("miss_ms", "trace_then_hit_ms", "known_signature_hit_ms")}
    groups = []
    for case in cases:
        name, variant, trip = config_key(case.features)
        samples = {stage: [] for stage in pooled}
        for repeat in range(REPEATS + 1):
            workload = build_workload(name, trip, 1, variant)
            engine = DecisionEngine(model)
            settings = {"trip_count": trip, "loop_kind": workload.loop_kind, "R": 100_000, "backend": "cpu"}
            decision, miss = _timed(lambda: engine.decide(workload.fn, workload.args, **settings))
            retraced, trace_hit = _timed(lambda: engine.decide(workload.fn, workload.args, **settings))
            cached, hit = _timed(lambda: engine.decide(signature=decision.signature, **settings))
            if decision.cache_hit or decision.fallback_reason or not (cached.cache_hit and retraced.cache_hit):
                raise RuntimeError("overhead benchmark did not exercise the intended decision/cache paths")
            if repeat:
                for values, elapsed in zip(samples.values(), (miss, trace_hit, hit), strict=True):
                    values.append(elapsed)
        for stage, values in samples.items():
            pooled[stage].extend(values)
        groups.append({
            "workload": name, "variant": variant, "trip_count": trip, "factor_R100000_unlimited": decision.factor,
            **{stage: _stats(values) for stage, values in samples.items()},
        })
    return {
        "model_version": model.version, "artifact_sha256": sha(MODEL), "backend": "cpu",
        "architecture": platform.machine(), "processor": platform.processor(), "repeats_per_configuration": REPEATS,
        "timings_per_configuration": groups, "pooled": {stage: _stats(values) for stage, values in pooled.items()},
    }


def inspect_hlo(rows):
    winning = [row for row in rows if row["paired_win"]]
    factors = {}
    for row in winning:
        factors.setdefault(config_key(row), set()).update(row[f"{policy}_factor"] for policy in ("selector", "factor_1", "llvm_size"))
    comparisons = []
    for (name, variant, trip), choices in sorted(factors.items()):
        for factor in sorted(choices):
            workload = build_workload(name, trip, factor, variant)
            hlo = jax.jit(workload.fn).lower(*workload.args).compile().as_text()
            comparison = {"workload": name, "variant": variant, "trip_count": trip, "factor": factor}
            comparison["optimized_hlo_bytes"] = len(hlo.encode())
            for opcode in ("while", "fusion", "dynamic-slice", "dynamic-update-slice", "scatter", "gather", "reduce"):
                comparison[f"{opcode}_ops"] = hlo.count(f" {opcode}(")
            comparisons.append(comparison)
    return comparisons


def score():
    require_cpu()
    freeze = require_frozen()
    model = CostModel.load(MODEL)
    provenance = {
        "freeze_commit": freeze, "artifact_sha256": sha(MODEL), "model_version": model.version,
        "protocol_sha256": sha(PROTOCOL), "raw_sha256": sha(RAW), "started_utc": utc_timestamp(),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("x", encoding="utf-8") as ledger:
        _append(ledger, {**provenance, "status": "started"})
        metadata = prepare(RAW, "confirmation", JOINED)
        if metadata["dropped"] or metadata["configurations"] != len(CONFIRMATION) * len(TRIPS) * len(VARIANTS):
            _append(ledger, {**provenance, "status": "inconclusive", "joined_meta_sha256": sha(JOINED.with_suffix(".meta.json")), "finished_utc": utc_timestamp()})
            print("Confirmation inconclusive: incomplete correctness-checked measurement matrix")
            return
        rows, _ = joined_rows(JOINED, "confirmation")
        cases = forecast(model, rows)
        overhead = measure_overhead(model, cases)
        charges = {config_key(row): row["miss_ms"]["median"] for row in overhead["timings_per_configuration"]}
        runtime_choices = {config_key(row): row["factor_R100000_unlimited"] for row in overhead["timings_per_configuration"]}
        pairs = paired_rows(cases, charges)
        if any(row["selector_factor"] != runtime_choices[config_key(row)] for row in pairs if row["R"] == 100_000 and row["B_ms"] == "unlimited"):
            raise RuntimeError("offline forecast and decision engine disagree")
        summary = confirmation_summary(pairs)
        write_csv(PAIRS, pairs, pairs[0])
        dump(OVERHEAD, overhead)
        dump(HLO, inspect_hlo(pairs))
        dump(SUMMARY, {**summary, "calibration": calibration(cases), "provenance": provenance})
        status = "passed" if summary["gate_passed"] else "failed"
        _append(ledger, {
            **provenance, "status": status, "finished_utc": utc_timestamp(),
            "evidence_sha256": {str(path): sha(path) for path in (JOINED, JOINED.with_suffix(".meta.json"), PAIRS, OVERHEAD, HLO, SUMMARY)},
            "feasible_long_wins": summary["strata"]["feasible_long"]["paired"]["win_settings"],
        })
    print(f"Fresh confirmation gate: {status.upper()}; {summary['strata']['feasible_long']['paired']['win_settings']} paired long-horizon wins")


def _parse_value(value):
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def verify():
    records = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines()]
    start, finish = records
    if (
        start["status"] != "started" or any(start[key] != finish[key] for key in start if key != "status")
        or start["artifact_sha256"] != sha(MODEL) or start["raw_sha256"] != sha(RAW)
        or start["protocol_sha256"] != sha(PROTOCOL) or start["freeze_commit"] != require_frozen(check_sources=False)
    ):
        raise ValueError("gate ledger disagrees with frozen inputs")
    if finish["status"] == "inconclusive":
        if finish["joined_meta_sha256"] != sha(JOINED.with_suffix(".meta.json")):
            raise ValueError("inconclusive gate metadata hash mismatch")
    else:
        if finish["evidence_sha256"] != {str(path): sha(path) for path in (JOINED, JOINED.with_suffix(".meta.json"), PAIRS, OVERHEAD, HLO, SUMMARY)}:
            raise ValueError("recorded confirmation evidence hash mismatch")
        rows = [{key: _parse_value(value) for key, value in row.items()} for row in read_csv(PAIRS)]
        summary = confirmation_summary(rows)
        record = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if (
            any(record[key] != value for key, value in summary.items())
            or finish["status"] != ("passed" if summary["gate_passed"] else "failed")
            or finish["feasible_long_wins"] != summary["strata"]["feasible_long"]["paired"]["win_settings"]
        ):
            raise ValueError("recorded confirmation outcome is not reproducible from its paired evidence")
    print(f"Verified the one-shot confirmation record: {finish['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("score", "verify"))
    args = parser.parse_args()
    if args.action == "verify":
        verify()
    else:
        if LOG.exists() or subprocess.run(["git", "cat-file", "-e", f"HEAD:{LOG}"], capture_output=True).returncode == 0:
            parser.error("confirmation has already started; use verify, not score")
        score()


if __name__ == "__main__":
    main()

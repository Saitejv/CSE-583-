"""Rebuild and replay the archived Phase 3 artifacts without opening a new gate."""

import csv
import io
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks import dataset, evaluate
from benchmarks.recovery_data import LEGACY_DATA, read_csv, require_cpu, sha
from jax_unroll.model import CostModel, dev_rows_digest, read_rows
from jax_unroll.splits import HELD_OUT_FAMILIES
from jax_unroll.validation import cv_metrics, forecast, lofo_forecasts

BASE = "c3a4ce1fa4bd0024d0321c4a9234ca04f6dbc1df"
VOLATILE_META = {"created_utc", "git_commit", "extraction_ms", "extraction_ms_mean", "extraction_ms_median", "extraction_ms_p95"}


def _git(*args):
    return subprocess.check_output(["git", *args])


def verify_archives():
    paths = _git("ls-tree", "-r", "--name-only", BASE, "data", "models", "results").decode().splitlines()
    paths += ["docs/splits.md", "docs/week3_results.md", "src/jax_unroll/splits.py"]
    if any(Path(path).read_bytes() != _git("show", f"{BASE}:{path}") for path in paths):
        raise ValueError("archived artifacts differ from the Phase 3 endpoint")
    return len(paths)


def verify_dataset():
    meta = json.loads(LEGACY_DATA.with_suffix(".meta.json").read_bytes())
    raw = [Path(record["path"]) for record in meta["raw_files"]]
    for path, record in zip(raw, meta["raw_files"], strict=True):
        if sha(path) != record["sha256"]:
            raise ValueError(f"historical raw sweep hash mismatch: {path}")
    rebuilt, rebuilt_meta = dataset.build_dataset(raw, backend="cpu")
    archive = read_csv(LEGACY_DATA)
    reproduced = list(csv.DictReader(io.StringIO(rebuilt.to_csv(index=False))))
    if len(reproduced) != len(archive) or any(
        dict(row, extraction_ms=prior["extraction_ms"]) != prior
        for row, prior in zip(reproduced, archive, strict=True)
    ):
        raise ValueError("historical feature/label rows did not reproduce")
    if any(rebuilt_meta[field] != original for field, original in meta.items() if field not in VOLATILE_META):
        raise ValueError("historical dataset metadata did not reproduce")
    print(f"Rebuilt {len(archive)} original rows; only extraction timing and acquisition metadata are volatile")


def verify_gates():
    log = [json.loads(line) for line in evaluate.GATE_LOG.read_text(encoding="utf-8").splitlines()]
    if [row["status"] for row in log] != ["started", "failed", "started", "failed"]:
        raise ValueError("expected the two archived failures")
    dev = read_rows(LEGACY_DATA, split="dev")
    holdout = read_rows(LEGACY_DATA, split="holdout")
    with TemporaryDirectory(prefix="jax-unroll-week3-") as directory:
        for start, finish, model_id, overhead_id in zip(
            log[::2], log[1::2], ("cost_cpu_gate1", "cost_cpu"), ("gate1_overhead", "overhead"), strict=True,
        ):
            model_path = Path("models") / f"{model_id}.json"
            overhead = json.loads((Path("results") / f"selector_cpu_2026-11-17_{overhead_id}.json").read_bytes())
            artifact = json.loads(model_path.read_bytes())
            model = CostModel.from_dict(artifact)
            miss_ms = overhead["pooled"]["miss_ms"]["median"]
            if (
                any(finish[key] != value for key, value in start.items() if key != "status")
                or start["artifact_sha256"] != sha(model_path) or start["dataset_sha256"] != sha(LEGACY_DATA)
                or model.version != start["model_version"] or miss_ms != start["decision_overhead_ms"]
                or dev_rows_digest(dev) != artifact["model"]["dev_rows_sha256"]
            ):
                raise ValueError("archived gate provenance mismatch")
            folds = lofo_forecasts(dev, warm_terms=model.warm_terms, warm_alpha=model.warm_alpha)
            held_out = forecast(model, holdout)
            groups = {
                **folds, "dev_pooled": [case for cases in folds.values() for case in cases], "holdout": held_out,
                **{f"holdout_{family}": [case for case in held_out if case.features["family"] == family] for family in HELD_OUT_FAMILIES},
            }
            output = Path(directory) / start["output"]
            passed = evaluate._write_results(output, model, groups, cv_metrics(folds), miss_ms=miss_ms,
                                             bootstraps=evaluate.DEFAULT_BOOTSTRAPS, gate_count=start["run"])
            if passed or finish["configurations"] != len(held_out):
                raise ValueError("archived holdout outcome did not reproduce")
            for suffix in (".csv", ".md"):
                reproduced = output.with_suffix(suffix)
                original = Path("results") / reproduced.name
                if reproduced.read_bytes() != original.read_bytes():
                    raise ValueError(f"archived report did not reproduce byte for byte: {original}")
            print(f"Replayed Week 3 gate {start['run']}: archived CSV/Markdown byte-identical, zero wins")


def main():
    require_cpu()
    print(f"Verified {verify_archives()} untouched commit-{BASE[:7]} paths")
    verify_dataset()
    verify_gates()


if __name__ == "__main__":
    main()

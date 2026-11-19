"""Time all held-out workflows through the public wrapper in fresh CPU processes."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from functools import partial
from itertools import product

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_horizon.replay import curves, total
from benchmarks.corpus.fori_regret.acquire import REPEATS, host
from benchmarks.corpus.fori_scalar.fit import BUNDLED
from benchmarks.corpus.fori_scalar.validate import ARMS, worker
from benchmarks.corpus.fori_workflows.cases import settings
from benchmarks.corpus.fori_workflows.dataset import CALLS, OUT, freeze, load
from jax_unroll.schema import FACTORS
from jax_unroll.workflow_fori import workflow_engine


def write_oracle():
    frame, labels = load("test")
    table = {}
    for r, cost in zip(frame.itertuples(), labels, strict=True):
        table[f"{r.workload}/{r.trip_count}"] = {
            h: FACTORS[total(cost[:, 0], cost[:, 1], h).argmin()] for h in CALLS
        }
    (OUT / "oracle.json").write_text(json.dumps(table, indent=2) + "\n")


def report(blocks):
    left = set(product(ARMS, CALLS, range(REPEATS)))
    expected = {(s.name, t) for s, _, t in settings("test")}
    choices = json.loads((OUT / "oracle.json").read_text())
    opportunity = {}
    for r, _, w in curves(OUT / "test_runs.json"):
        ratios = w[:, :1] / w
        opportunity[r["workload"], r["trip_count"]] = bool(
            np.any((np.median(ratios, axis=0) >= 1.05) & (ratios > 1).all(axis=0))
        )
    records = []
    for b in blocks:
        arm, h, k = b["arm"], b["calls"], b["repeat"]
        left.remove((arm, h, k))
        if len(b["rows"]) != len(expected) or {(r["workload"], r["trip"]) for r in b["rows"]} != expected:
            raise ValueError("incomplete live block")
        for r in b["rows"]:
            name, t, f = r["workload"], r["trip"], r["factor"]
            if (
                f not in FACTORS
                or (arm == "native" and f != 1)
                or (arm == "oracle" and f != choices[f"{name}/{t}"][str(h)])
            ):
                raise ValueError("unexpected factor")
            records.append(
                dict(
                    workload=name,
                    trip=t,
                    arm=arm,
                    calls=h,
                    repeat=k,
                    factor=f,
                    opportunity=opportunity[name, t],
                    seconds=(r["direct_ms"] + b["setup_ms"] / len(expected)) / 1000,
                )
            )
    if left:
        raise ValueError("missing repeats")
    frame = pd.DataFrame(records)
    keys = ["workload", "trip", "opportunity", "calls"]
    pairs = frame.pivot(index=keys + ["repeat"], columns="arm", values="seconds")
    summary = pairs.groupby(keys).mean()
    factors = frame[frame.arm == "candidate"].groupby(keys).factor
    if factors.nunique().max() != 1:
        raise ValueError("inconsistent selection")
    summary["factor"] = factors.first()
    summary["saved_native_s"] = summary.native - summary.candidate
    summary["saved_incumbent_s"] = summary.incumbent - summary.candidate
    summary["median_ratio"] = (pairs.candidate / pairs.native).groupby(keys).median()
    summary["native_wins_for_candidate"] = (pairs.candidate < pairs.native).groupby(keys).sum()
    summary.to_csv(OUT / "cases.csv")
    groups = pairs.reset_index()
    totals = []
    for name, group in (
        ("all", groups),
        ("opportunity", groups[groups.opportunity]),
        ("gauge", groups[groups.workload == "muskingum"]),
    ):
        for h, g in group.groupby("calls"):
            pooled = g.groupby("repeat")[list(ARMS)].sum()
            s = pooled.mean()
            totals.append(
                dict(
                    subset=name,
                    calls=h,
                    **s.to_dict(),
                    saved_native_s=s.native - s.candidate,
                    saved_incumbent_s=s.incumbent - s.candidate,
                    native_wins=int((pooled.candidate < pooled.native).sum()),
                    incumbent_wins=int((pooled.candidate < pooled.incumbent).sum()),
                    workload_weighted_s=g.groupby("workload").candidate.mean().mean() * len(g) / REPEATS,
                    recovery=(s.native - s.candidate) / (s.native - s.oracle)
                    if s.native > s.oracle
                    else None,
                )
            )
    pd.DataFrame(totals).to_csv(OUT / "totals.csv", index=False)
    print(pd.DataFrame(totals).round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=int)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--calls", choices=CALLS, type=int)
    args = p.parse_args()
    if args.worker is not None:
        print(
            json.dumps(
                worker(
                    args.arm,
                    args.calls,
                    args.worker,
                    batch=settings("test"),
                    out_dir=OUT,
                    candidate_factory=partial(workflow_engine, profile_path=OUT / "profile.json"),
                    incumbent_path=BUNDLED,
                )
            )
        )
        return
    freeze()
    write_oracle()
    stamps = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (
            BUNDLED,
            OUT / "profile.json",
            OUT / "split.json",
            OUT / "test_runs.json",
            OUT / "oracle.json",
        )
    }
    path = OUT / "live.json"
    data = json.loads(path.read_text()) if path.exists() else dict(host=host(), stamps=stamps, blocks=[])
    if data["host"] != host() or data["stamps"] != stamps:
        raise ValueError("incompatible saved process blocks")
    done = {(b["arm"], b["calls"], b["repeat"]) for b in data["blocks"]}
    for repeat in range(REPEATS):
        order = list(product(ARMS, CALLS))
        random.Random(583253 + repeat).shuffle(order)
        for arm, h in order:
            if (arm, h, repeat) in done:
                continue
            print(repeat + 1, arm, h, file=sys.stderr, flush=True)
            cmd = [
                sys.executable,
                "-m",
                __spec__.name,
                "--worker",
                str(repeat),
                "--arm",
                arm,
                "--calls",
                str(h),
            ]
            block = subprocess.check_output(cmd, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
            data["blocks"].append(json.loads(block))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n")
            tmp.replace(path)
    report(data["blocks"])


if __name__ == "__main__":
    main()

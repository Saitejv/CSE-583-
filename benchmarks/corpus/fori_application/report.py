"""Regenerate paired direct comparisons; never substitute sweep replay for live totals."""

import argparse
import json
from hashlib import sha256
from itertools import product

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_application.live import key, specs
from benchmarks.corpus.fori_application.protocol import ARMS, HORIZONS, OUT, REPEATS
from jax_unroll.schema import FACTORS


def detail(cohort):
    raw = json.loads((OUT / f"{cohort}_live.json").read_text())
    for filename, expected_hash in raw["freeze"].items():
        if sha256((OUT / filename).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"{filename} changed")
    oracle_file = "transfer_oracle.json" if cohort == "transfer" else "oracle.json"
    sweep = {key(r): r for r in json.loads((OUT / oracle_file).read_text()) if r["cohort"] == cohort}
    expected = {(s.name, v, t) for s, v, t in specs(cohort)}
    remaining = set(product(ARMS, HORIZONS, range(REPEATS)))
    rows = []
    for block in raw["blocks"]:
        arm, h, repeat = block["arm"], block["calls"], block["repeat"]
        remaining.remove((arm, h, repeat))
        if len(block["rows"]) != len(expected) or {key(r) for r in block["rows"]} != expected:
            raise ValueError("incomplete process")
        for r in block["rows"]:
            identity = key(r)
            if (
                r["factor"] not in FACTORS
                or arm == "native"
                and r["factor"] != 1
                or arm == "oracle"
                and r["factor"] != sweep[identity]["oracle"][str(h)]
            ):
                raise ValueError("unexpected factor")
            elapsed = (r["direct_ms"] + block["setup_ms"] / len(expected)) / 1000
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("invalid direct measurement")
            rows.append(
                dict(
                    cohort=cohort,
                    workload=identity[0],
                    variant=identity[1],
                    trip_count=identity[2],
                    calls=h,
                    arm=arm,
                    repeat=repeat,
                    factor=r["factor"],
                    direct_s=elapsed,
                    opportunity=sweep[identity]["opportunity"],
                )
            )
    if remaining:
        raise ValueError(f"incomplete {cohort} repeats")
    return pd.DataFrame(rows)


def summarize(frame):
    cols = ["cohort", "workload", "variant", "trip_count", "calls", "opportunity"]
    index = cols + ["repeat"]
    paired = frame.pivot(index=index, columns="arm", values="direct_s").reset_index()
    per_case, totals = [], []
    for arm in ARMS:
        timing = paired.merge(frame[frame.arm == arm][index + ["factor"]], on=index, validate="one_to_one")
        if timing.groupby(cols).factor.nunique().max() != 1:
            raise ValueError("factor changed across repeats")
        for identifier, g in timing.groupby(cols):
            per_case.append(
                dict(
                    **dict(zip(cols, identifier)),
                    arm=arm,
                    factor=g.factor.iloc[0],
                    mean_s=g[arm].mean(),
                    saved_native_s=(g.native - g[arm]).mean(),
                    saved_previous_s=(g.previous - g[arm]).mean(),
                    median_ratio=(g[arm] / g.native).median(),
                    wins_native=int((g[arm] < g.native).sum()),
                )
            )
        for (cohort, h), g in timing.groupby(["cohort", "calls"]):
            for subset, group in (("all", g), ("opportunity", g[g.opportunity])):
                if group.empty:
                    continue
                repeat = group.groupby("repeat")[list(ARMS)].sum()
                seconds = repeat.mean()
                ratios = (group[arm] / group.native).groupby([group[c] for c in cols]).median()
                headroom = seconds.native - seconds.oracle
                totals.append(
                    dict(
                        cohort=cohort,
                        subset=subset,
                        calls=h,
                        arm=arm,
                        cases=len(ratios),
                        seconds=seconds[arm],
                        saved_native_s=seconds.native - seconds[arm],
                        saved_previous_s=seconds.previous - seconds[arm],
                        workload_weighted_s=len(ratios) * group.groupby("workload")[arm].mean().mean(),
                        wins_native=int((repeat[arm] < repeat.native).sum()),
                        wins_previous=int((repeat[arm] < repeat.previous).sum()),
                        wins_5pct=int((ratios <= 1 / 1.05).sum()),
                        losses_3pct=int((ratios > 1.03).sum()),
                        worst_ratio=ratios.max(),
                        recovery=(seconds.native - seconds[arm]) / headroom if headroom > 0 else None,
                    )
                )
    return pd.DataFrame(per_case), pd.DataFrame(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("applications", "transfer", "holdout"))
    opts = parser.parse_args()
    cases, totals = summarize(detail(opts.cohort))
    cases.to_csv(OUT / f"{opts.cohort}_cases.csv", index=False)
    totals.to_csv(OUT / f"{opts.cohort}_totals.csv", index=False)
    print(totals[totals.subset == "all"].pivot(index="calls", columns="arm", values="seconds").round(3))


if __name__ == "__main__":
    main()

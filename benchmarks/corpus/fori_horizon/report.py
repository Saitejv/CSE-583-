"""Rebuild direct totals and fixed-factor replay from the retained measurements."""

import argparse
from hashlib import sha256
from itertools import product
import json

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_horizon import coverage
from benchmarks.corpus.fori_horizon.cohort import HORIZONS, OUT
from benchmarks.corpus.fori_horizon.live import ARMS, key
from benchmarks.corpus.fori_regret.acquire import REPEATS
from jax_unroll.schema import FACTORS

IDS = ["workload", "variant", "trip_count"]


def aggregate(blocks, sweep, horizons=HORIZONS):
    expected = {key(row) for row in sweep}
    cases = {key(row): row for row in sweep}
    processes = set(product(ARMS, horizons, range(REPEATS)))
    rows = []
    for b in blocks:
        processes.remove((b["arm"], b["calls"], b["repeat"]))
        if len(b["rows"]) != len(expected) or {key(r) for r in b["rows"]} != expected:
            raise ValueError("incomplete or duplicate process measurements")
        inline = sum(r["inline_load_ms"] for r in b["rows"])
        share = (b["setup_ms"] + inline) / len(expected)
        for r in b["rows"]:
            c = cases[key(r)]
            factor = r["factor"]
            h = b["calls"]
            if (r["calls"] != h or factor not in FACTORS or r["direct_ms"] <= 0
                    or (b["arm"] == "oracle" and factor != c["oracle"][str(h)])):
                raise ValueError("invalid measurement")
            rows.append(dict(**{k:r[k] for k in IDS}, calls=h, repeat=b["repeat"], arm=b["arm"],
                             cohort=c["cohort"], opportunity=c["opportunity"], factor=factor,
                             direct_s=(r["direct_ms"] - r["inline_load_ms"] + share) / 1000,
                             replay_s=c["replay_ms"][str(h)][FACTORS.index(factor)] / 1000))
    if processes:
        raise ValueError("missing live processes")
    frame = pd.DataFrame(rows)
    if frame.groupby(IDS + ["arm","calls"]).factor.nunique().max() != 1:
        raise ValueError("factor changed between repeats")
    return frame


def recover(base, oracle, measured):
    return (base - measured) / (base - oracle) if base > oracle else None


def summarize(frame):
    processes = frame.pivot(index=IDS + ["calls","repeat","cohort","opportunity"], columns="arm",
                            values="direct_s").reset_index()
    totals, details = [], []
    for (h, arm), group in frame.groupby(["calls","arm"]):
        paired = group.merge(processes, on=IDS + ["calls","repeat","cohort","opportunity"], validate="one_to_one")
        for _, c in paired.groupby(IDS):
            details.append(dict(**{k:c.iloc[0][k] for k in IDS + ["cohort","opportunity"]},
                                calls=h, arm=arm, factor=c.factor.iloc[0], mean_s=c.direct_s.mean(),
                                saved_vs_native_s=(c.native-c.direct_s).mean(),
                                saved_vs_incumbent_s=(c.incumbent-c.direct_s).mean(),
                                median_ratio=(c.direct_s/c.native).median(),
                                repeats_faster_native=int((c.direct_s < c.native).sum())))
        for cohort, subset in (
                ("all_fresh", paired[paired.cohort != "historical"]),
                ("fresh_opportunity", paired[(paired.cohort != "historical") & paired.opportunity]),
                ("coverage", paired[paired.cohort == "coverage"]),
                ("mixed", paired[paired.cohort == "mixed"]),
                ("mixed_opportunity", paired[(paired.cohort == "mixed") & paired.opportunity]),
                ("transfer", paired[paired.cohort == "transfer"]),
                ("historical", paired[paired.cohort == "historical"])):
            if subset.empty:
                continue
            n = len(subset) // REPEATS
            native, inc, oracle, elapsed = subset[["native","incumbent","oracle","direct_s"]].sum() / REPEATS
            ratios = (subset.direct_s/subset.native).groupby([subset[k] for k in IDS]).median()
            per_repeat = subset.groupby("repeat")[["direct_s","native","incumbent","crossover"]].sum()
            sub = frame[(frame.calls == h) & frame.workload.isin(subset.workload)]
            # Restrict at configuration granularity for opportunity subsets.
            sub = sub.merge(subset[IDS].drop_duplicates(), on=IDS, validate="many_to_one")
            replay = sub.groupby("arm").replay_s.sum()/REPEATS
            totals.append(dict(cohort=cohort, calls=h, arm=arm, cases=n, pooled_s=elapsed,
                               workload_weighted_s=n*subset.groupby("workload").direct_s.mean().mean(),
                               saved_native_s=native-elapsed, saved_incumbent_s=inc-elapsed,
                               direct_recovery=recover(native,oracle,elapsed),
                               factor_replay_s=replay[arm],
                               replay_recovery=recover(replay.native,replay.oracle,replay[arm]),
                               worst_ratio=ratios.max(), geo_speedup=np.exp(-np.log(ratios).mean()),
                               gains_5pct=(ratios<=1/1.05).sum(), losses_3pct=(ratios>1.03).sum(),
                               repeat_wins_native=(per_repeat.direct_s < per_repeat.native).sum(),
                               repeat_wins_incumbent=(per_repeat.direct_s < per_repeat.incumbent).sum(),
                               repeat_wins_crossover=(per_repeat.direct_s < per_repeat.crossover).sum()))
    return pd.DataFrame(totals), pd.DataFrame(details)


def check_live_choices(directory, frame):
    from benchmarks.corpus.fori_horizon.replay import curves
    from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
    from jax_unroll.compact_fori import CompactForiPredictor
    from jax_unroll.selector import HorizonPolicy, select_factor

    model = CompactForiPredictor.load(UNCOMPRESSED)
    config = json.loads((OUT / "policy.json").read_text())
    if config["model_version"] != model.version:
        raise ValueError("compact model changed")
    rules = {arm: HorizonPolicy(**config[arm]) for arm in ("crossover", "candidate")}
    indexed = frame.groupby(IDS + ["arm", "calls"]).factor.first()
    for row, _, _ in curves(directory / "sweep.json"):
        features = row["features"]
        prices = model.predict_candidates(features)
        for h in frame.calls.unique():
            expected = {arm: rule.choose(prices, R=int(h), B=config["budget_ms"], features=features)
                        for arm, rule in rules.items()}
            expected.update(native=1, incumbent=select_factor(prices, throughput_only=True))
            for arm, factor in expected.items():
                if indexed[(*key(row), arm, h)] != factor:
                    raise ValueError(f"sweep/live factor mismatch for {key(row)}, {arm}, {h}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    directory, horizons = (coverage.DIRECTORY, coverage.DIRECT) if args.coverage else (OUT, HORIZONS)
    raw = json.loads((directory / "live.json").read_text())
    sweep = json.loads((directory / "oracle_replay.json").read_text())
    for path, expected in raw["freeze"].items():
        if sha256((directory / path).read_bytes()).hexdigest() != expected:
            raise ValueError("source artifact changed: " + path)
    frame = aggregate(raw["blocks"], sweep, horizons)
    check_live_choices(directory, frame)
    totals, cases = summarize(frame)
    totals.to_csv(directory / "totals.csv", index=False)
    cases.to_csv(directory / "cases.csv", index=False)


if __name__ == "__main__":
    main()

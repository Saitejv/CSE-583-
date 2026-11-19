"""Offline replay of saved native factor curves, excluding model overhead."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_horizon.cohort import HORIZONS, OUT
from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
from benchmarks.corpus.fori_iterations.validate import read_rule
from benchmarks.corpus.fori_regret.development import archive
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.schema import FACTORS
from jax_unroll.selector import select_factor


def curves(path):
    rows = pd.DataFrame(r for run in json.loads(path.read_text())["runs"] for r in run)
    for _, g in rows.groupby(["workload", "variant", "trip_count"]):
        if sorted(g.repeat) != [0, 1, 2] or len({r["body_hash"] for r in g.features}) != 1:
            raise ValueError("missing repeat or inconsistent body")
        row = g.iloc[0]
        cold = np.array(list(g.cold_ms))
        warm = np.median(np.array(list(g.warm_samples_us)), axis=2)
        if any(x.shape != (3, 4) or not np.all(np.isfinite(x) & (x > 0)) for x in (cold, warm)):
            raise ValueError("invalid native sweep")
        yield row.to_dict(), cold, warm


def history():
    frame, timing = archive()
    for row, cost in zip(frame.to_dict("records"), timing, strict=True):
        yield row["source"], row, row, cost[:, 0], cost[:, 1]
    paths = {"regret_train": "fori_regret/train_runs", "regret_holdout": "fori_regret/holdout_runs",
             "proposal": "fori_proposal/native_runs", "iterations": "fori_iterations/confirmation_native"}
    for cohort, stem in paths.items():
        for row, cold, warm in curves(Path(f"results/{stem}.json")):
            yield cohort, row, row["features"], np.median(cold, axis=0), np.median(warm, axis=0)


def total(cold, warm, h):
    return cold + (h - 1) * warm / 1000


def main():
    model = CompactForiPredictor.load(UNCOMPRESSED)
    _, rule = read_rule("savings")
    rows = []
    for cohort, row, features, cold, warm in history():
        pred = model.predict_candidates(features)
        inc = FACTORS.index(select_factor(pred, throughput_only=True))
        for h in HORIZONS[1:]:
            times = total(cold, warm, h)
            best = np.argmin(times)
            pay = FACTORS.index(rule.choose(pred, R=h, B=125))
            rows.append(dict(cohort=cohort, **{k: row[k] for k in ("family","workload","variant","trip_count")},
                             calls=h, incumbent=FACTORS[inc], oracle=FACTORS[best], payoff=FACTORS[pay],
                             native_ms=times[0], incumbent_ms=times[inc], oracle_ms=times[best],
                             regret_ms=times[inc]-times[best], payoff_regret_ms=times[pay]-times[best],
                             incumbent_result=("optimal" if inc == best else "unrecovered_setup" if best == 0
                                               else "missed_unroll" if inc == 0 else "wrong_factor"),
                             payoff_veto_ms=(times[pay]-times[inc] if pay == 0 and times[inc] < times[0] else 0.),
                             warm_win_without_payback=bool(best == 0 and np.argmin(warm) != 0)))
    ranks = pd.DataFrame(rows).sort_values(["calls","regret_ms"], ascending=[True, False])
    ranks.to_csv(OUT / "development_regret.csv", index=False)
    for group in ("iterations", "targeted", "proposal"):
        for h, g in ranks[ranks.cohort == group].groupby("calls"):
            print(group,h,g[["native_ms","incumbent_ms","oracle_ms","payoff_veto_ms"]].sum().round().to_dict(),
                  g.iloc[:3][["workload","variant","regret_ms"]].round().values.tolist())


if __name__ == "__main__":
    main()

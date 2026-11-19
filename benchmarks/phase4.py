"""Run the Phase 4 demo or collect fresh CPU deployment measurements."""
from __future__ import annotations

import argparse
from functools import partial
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter_ns

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.baselines import llvm_style_heuristic
from jax_unroll.features import extract_features
from jax_unroll.library_kernels import (
    SEGMENTED_FACTOR, SEGMENTED_SHAPES, SUBTRACT_FACTOR, SUBTRACT_SHAPES,
    add_reduceat_rows, subtract_reduce_rows,
)
from jax_unroll.model import CostModel
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/cost_cpu_recovery_v1.json"
OUT = ROOT / "results/raw/phase4"
REDUCTIONS = {
    "residual_128x8": (128, 8, 0),
    "residual_1024x8": (1024, 8, 0),
    "energy_256x1_4": (256, 1, 4),
    "energy_1024x1_16": (1024, 1, 16),
    "residual_513x8_native": (513, 8, 0),
}
LEARNED = {"weighted_blend": 511, "clipped_ema": 1024, "debounced_counter": 257}
CONTEXTS = {"25ms": 25, "100ms": 100, "unlimited": None}
HORIZON = 100_000
CALL_COUNTS = (1, 10, 100, 1_000, 10_000, HORIZON)


@jax.jit
def native_subtract(x):
    return jnp.subtract.reduce(x, axis=0)


@jax.jit
def native_segmented(x, starts):
    return jnp.add.reduceat(x, starts, axis=0)


def reduction_case(name, rng):
    rows, width, segments = REDUCTIONS[name]
    if segments:
        def caller(reduce):
            def energy(x, starts):
                return jnp.sqrt(reduce(jnp.square(x), starts))
            return energy

        native, deployed = native_segmented, add_reduceat_rows
        factor = SEGMENTED_FACTOR if (rows, segments) in SEGMENTED_SHAPES else 1
    else:
        def caller(reduce):
            def residual(x, scale):
                return jnp.abs(reduce(x * scale))
            return residual

        native, deployed = native_subtract, subtract_reduce_rows
        factor = SUBTRACT_FACTOR if (rows, width) in SUBTRACT_SHAPES else 1

    samples = []
    for sample in range(4):
        x = rng.normal(size=(rows, width)).astype(np.float32)
        if segments:
            starts = rng.choice(rows, size=segments, replace=False).astype(np.int32)
            if sample % 2 == 0:
                starts.sort()
            if sample == 3:
                starts[1] = starts[0]
            samples.append((x, starts))
        else:
            scale = rng.choice(np.array([0.25, 0.5, 1., 2.], dtype=np.float32), size=width)
            samples.append((x, scale))
    return caller(native), caller(deployed), factor, samples


def learned_case(name, rng):
    trip = LEARNED[name]

    def build(**options):
        def caller(stream):
            if name == "weighted_blend":
                def body(i, carry):
                    value, weight = stream[i]
                    return (carry + value) * weight

                init = jnp.float32(0)
            elif name == "clipped_ema":
                def body(i, carry):
                    incoming = jnp.clip(stream[i], -1., 1.)
                    return 0.75 * carry + 0.25 * incoming

                init = jnp.zeros(stream.shape[1:], dtype=stream.dtype)
            else:
                def body(i, carry):
                    run, hits = carry
                    run = jnp.where(stream[i] > 0., run + 1, 0)
                    return run, hits + (run == 3)

                init = (jnp.int32(0), jnp.int32(0))
            return auto_fori_loop(0, trip, body, init, **options)
        return caller

    samples = []
    shape = {"weighted_blend": (trip, 2), "clipped_ema": (trip, 16), "debounced_counter": (trip,)}[name]
    for _ in range(4):
        stream = rng.normal(size=shape).astype(np.float32)
        if name == "weighted_blend":
            stream[:, 1] = rng.uniform(0.25, 0.875, size=trip)
        samples.append((stream,))
    return build, samples


def first_call(kernel, args):
    start = perf_counter_ns()
    result = jax.block_until_ready(kernel(*args))
    return result, (perf_counter_ns() - start) / 1e6


def check_outputs(kernels, inputs, *, exact):
    close = partial(np.testing.assert_allclose, rtol=2e-6, atol=2e-7, equal_nan=False)

    def compare(actual, expected):
        if exact or not jnp.issubdtype(expected.dtype, jnp.inexact):
            np.testing.assert_array_equal(actual, expected)
        else:
            close(actual, expected)

    for args in inputs:
        reference = jax.block_until_ready(kernels[0](*args))
        for kernel in kernels[1:]:
            result = jax.block_until_ready(kernel(*args))
            jax.tree.map(compare, result, reference)


def warm_latency(kernel, inputs, calls):
    start = perf_counter_ns()
    for i in range(calls):
        jax.block_until_ready(kernel(*inputs[i % len(inputs)]))
    return (perf_counter_ns() - start) / calls / 1e3


def paired_warm(kernels, inputs, rng):
    calls = int(np.clip(2500 / warm_latency(kernels[0], inputs, 16), 16, 256))
    blocks = np.empty((15, len(kernels)))
    for block in blocks:
        for i in rng.permutation(len(kernels)):
            block[i] = warm_latency(kernels[i], inputs, calls)
    return np.median(blocks, axis=0).tolist()


def audit(native, variant, args, factor):
    original = str(jax.make_jaxpr(native)(*args))
    expected = original.replace("unroll=1\n", f"unroll={factor}\n")
    actual = str(jax.make_jaxpr(variant)(*args))
    # JAX keeps the nested jit function's name in the printed graph.
    actual = actual.replace("name=subtract_reduce_rows", "name=native_subtract")
    actual = actual.replace("name=add_reduceat_rows", "name=native_segmented")
    if original.count("unroll=1\n") != 1 or actual != expected:
        raise AssertionError("Expected the full caller Jaxpr to change only in unroll")


def hlo_summary(kernel, args):
    ir = kernel.lower(*args).compile().as_text()
    opcodes = ("while", "fusion", "dynamic-slice", "dynamic-update-slice", "gather", "scatter", "reduce")
    return {
        "bytes": len(ir.encode()),
        "ops": {op: len(re.findall(rf" = [^\n]*? {op}\(", ir)) for op in opcodes},
    }


class TimedEngine(DecisionEngine):
    def decide(self, *args, **kwargs):
        start = perf_counter_ns()
        self.decision = super().decide(*args, **kwargs)
        self.decision_ms = (perf_counter_ns() - start) / 1e6
        return self.decision


def worker(name, variant, repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    rng = np.random.default_rng(491_583 + repeat)
    load_ms = 0.
    engine = None
    if name in REDUCTIONS:
        native, deployed, factor, samples = reduction_case(name, rng)
        fns = {"native": native, "deployed": deployed}
        chosen = variant
    else:
        build, samples = learned_case(name, rng)
        fns = {str(factor): build() if factor == 1 else build(force_factor=factor) for factor in FACTORS}
        chosen = variant
        if variant in CONTEXTS:
            start = perf_counter_ns()
            model = CostModel.load(MODEL)
            load_ms = (perf_counter_ns() - start) / 1e6
            engine = TimedEngine(model)
            fns[variant] = build(engine=engine, B=CONTEXTS[variant], R=HORIZON)

    inputs = jax.block_until_ready(jax.device_put(samples))
    args = inputs[0]
    kernels = {key: jax.jit(fn) for key, fn in fns.items()}
    _, first_ms = first_call(kernels[chosen], args)
    row = dict(case=name, variant=variant, repeat=repeat, first_ms=first_ms)
    if name in REDUCTIONS and variant == "native":
        check_outputs(list(kernels.values()), inputs, exact=True)
        row.update(factor=factor, warm_us=paired_warm(list(kernels.values()), inputs, rng))
        if repeat == 0:
            audit(native, deployed, args, factor)
            row["hlo"] = {key: hlo_summary(kernel, args) for key, kernel in kernels.items()}
    if engine is not None:
        chosen_factor = engine.decision.factor
        row.update(factor=chosen_factor, fallback=engine.decision.fallback_reason,
                   decision_ms=engine.decision_ms, model_load_ms=load_ms)
        check_outputs(list(kernels.values()), inputs, exact=False)
        row["warm_us"] = dict(zip(kernels, paired_warm(list(kernels.values()), inputs, rng), strict=True))
        features = extract_features(fns["1"], args, trip_count=LEARNED[name], factor=1, loop_kind="fori_loop")
        predictions = model.predict_candidates(features)
        row.update(llvm_factor=llvm_style_heuristic(features),
                   predicted_cold_ms=predictions[chosen_factor].linear()[0])
        if repeat == 0:
            audit(fns["1"], fns[variant], args, chosen_factor)
            row["hlo"] = {key: hlo_summary(kernel, args) for key, kernel in kernels.items()}
    return row


def elapsed(first_ms, warm_us, calls):
    return first_ms + (calls - 1) * warm_us / 1000


def break_even(native_ms, deployed_ms, native_us, deployed_us):
    """First integer invocation count with a strict, durable time advantage."""
    setup = deployed_ms - native_ms
    saving = (native_us - deployed_us) / 1000
    if saving <= 0:
        return 1 if saving == 0 and setup < 0 else None
    return 1 if setup < 0 else math.floor(setup / saving) + 2


def analyze(rows):
    cold = {(r["case"], r["repeat"], r["variant"]): r["first_ms"] for r in rows}
    reductions, learned, hlo = [], [], []
    for row in rows:
        name, rep, variant = (row[key] for key in ("case", "repeat", "variant"))
        if "hlo" in row:
            hlo.append(dict(case=name, variant=variant, optimized_hlo=row["hlo"]))
        if name in REDUCTIONS and variant == "native":
            native_ms = row["first_ms"]
            deployed_ms = cold[name, rep, "deployed"]
            native_us, deployed_us = row["warm_us"]
            reductions.append(dict(case=name, repeat=rep, factor=row["factor"],
                                   native_first_ms=native_ms, deployed_first_ms=deployed_ms,
                                   native_us=native_us, deployed_us=deployed_us,
                                   warm_speedup=native_us / deployed_us,
                                   break_even_calls=break_even(native_ms, deployed_ms, native_us, deployed_us),
                                   speedup_by_calls={n: elapsed(native_ms, native_us, n)
                                                     / elapsed(deployed_ms, deployed_us, n)
                                                     for n in CALL_COUNTS}))
        if name in LEARNED and variant in CONTEXTS:
            budget = CONTEXTS[variant]
            candidate_ms = {f: cold[name, rep, str(f)] for f in FACTORS}
            warm = row["warm_us"]
            totals = {f: elapsed(candidate_ms[f], warm[str(f)], HORIZON) for f in FACTORS}
            feasible = [f for f in FACTORS if budget is None or candidate_ms[f] <= budget]
            oracle = min(feasible, key=totals.get) if feasible else None
            measured_ms = row["first_ms"] + row["model_load_ms"]
            chosen_ms = elapsed(measured_ms, warm[variant], HORIZON)
            llvm = row["llvm_factor"]
            learned.append({**{k: v for k, v in row.items() if k != "hlo"},
                            "budget_ms": budget, "calls": HORIZON, "measured_cold_ms": measured_ms,
                            "predicted_violation": budget is not None and row["predicted_cold_ms"] > budget,
                            "measured_violation": budget is not None and measured_ms > budget,
                            "llvm_violation": budget is not None and candidate_ms[llvm] > budget,
                            "native_violation": budget is not None and candidate_ms[1] > budget,
                            "oracle_factor": oracle, "oracle_agreement": None if oracle is None else row["factor"] == oracle,
                            "oracle_regret": None if oracle is None else chosen_ms / totals[oracle] - 1,
                            "speedup_vs_native": totals[1] / chosen_ms, "speedup_vs_llvm": totals[llvm] / chosen_ms})
    return dict(reductions=reductions, learned=learned, hlo=hlo)


def demo():
    for name in REDUCTIONS:
        native, deployed, factor, samples = reduction_case(name, np.random.default_rng(583))
        inputs = jax.block_until_ready(jax.device_put(samples))
        audit(native, deployed, inputs[0], factor)
        trace_count = [0, 0]
        kernels = []
        for i, fn in enumerate((native, deployed)):
            def traced(*args, fn=fn, i=i):
                trace_count[i] += 1
                return fn(*args)

            kernels.append(jax.jit(traced))
        first = [first_call(kernel, inputs[0])[1] for kernel in kernels]
        check_outputs(kernels, inputs, exact=True)
        warm = paired_warm(kernels, inputs, np.random.default_rng(583))
        if trace_count != [1, 1]:
            raise AssertionError(f"Repeated tracing for one signature: {trace_count}")
        print(f"{name:24s} factor {factor}  native/deployed first {first[0]:6.2f}/{first[1]:6.2f} ms"
              f"  warm {warm[0]:5.2f}/{warm[1]:5.2f} µs  exact parity, one trace each")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(out, reduction_repeats, learned_repeats):
    out.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__), MODEL, *(ROOT / "src/jax_unroll" / filename for filename in (
        "library_kernels.py", "auto_loop.py", "selector.py", "features.py", "model.py", "baselines.py", "schema.py",
    )))
    hardware = (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                if platform.system() == "Darwin" else platform.processor() or platform.machine())
    freeze = {
        "hardware": hardware, "platform": platform.platform(), "backend": jax.default_backend(),
        "python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "reduction_repeats": reduction_repeats, "learned_repeats": learned_repeats,
        "sha256_before_measurement": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                      for path in sources},
    }
    write_json(out / "freeze.json", freeze)
    jobs = [(name, variant, r) for name in REDUCTIONS for variant in ("native", "deployed")
            for r in range(reduction_repeats)]
    jobs += [(name, variant, r) for name in LEARNED for variant in (*map(str, FACTORS), *CONTEXTS)
             for r in range(learned_repeats)]
    np.random.default_rng(583).shuffle(jobs)
    results = []
    for i, (name, variant, repeat) in enumerate(jobs, 1):
        command = [sys.executable, "-m", "benchmarks.phase4", "--worker", name, variant, str(repeat)]
        child = subprocess.run(command, capture_output=True, text=True,
                               env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
        if child.returncode:
            raise RuntimeError(f"{name}/{variant}/{repeat}: {child.stderr}")
        results.append(json.loads(child.stdout))
        if i % 20 == 0 or i == len(jobs):
            print(f"Measured {i}/{len(jobs)} isolated first calls", flush=True)
    if any(sha256(path.read_bytes()).hexdigest() != freeze["sha256_before_measurement"][str(path.relative_to(ROOT))]
           for path in sources):
        raise RuntimeError("Policy or protocol changed during measurement")
    write_json(out / "runs.json", results)
    write_json(out / "analysis.json", analyze(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--reduction-repeats", type=int, default=7)
    parser.add_argument("--learned-repeats", type=int, default=3)
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if jax.default_backend() != "cpu" or (jax.__version__, jaxlib.__version__) != ("0.12.0.1", "0.12.0.1"):
        parser.error("use JAX_PLATFORMS=cpu with the pinned JAX/jaxlib environment")
    if args.worker:
        name, variant, repeat = args.worker
        print(json.dumps(worker(name, variant, int(repeat)), allow_nan=False))
    elif args.demo:
        demo()
    else:
        if min(args.reduction_repeats, args.learned_repeats) < 1:
            parser.error("repeat counts must be positive")
        evaluate(args.out, args.reduction_repeats, args.learned_repeats)


if __name__ == "__main__":
    main()

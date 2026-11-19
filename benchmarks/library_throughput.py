"""External-validity check on actual pinned JAX public functions, in isolated processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.extend import core
import numpy as np
import pandas as pd
from scipy.linalg import schur

from jax_unroll.autotune import _times, tune_fori_kernel
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.selector import DecisionEngine

OUT = Path("results/library_throughput")
MODEL = Path("results/fori_predictor/model.joblib")
ROUTINES = ("reduceat", "lexsort", "sph_harm_y", "rsf2csf")


def application(name, size, seed):
    rng = np.random.default_rng(seed)
    if name == "reduceat":
        n = (128, 512, 1024)[size]
        args = (rng.normal(size=(n, 8)).astype("float32"), np.arange(0,n,8,dtype="int32"))
        fn = jnp.add.reduceat
        trip = n
    elif name == "lexsort":
        n = (256, 512, 1024)[size]
        args = (rng.integers(0, 16, size=(n, 48), dtype=np.int32),)
        fn = jnp.lexsort
        trip = n // 16
    elif name == "sph_harm_y":
        n = (18, 24, 32)[size]
        degrees = rng.integers(0, n+1, 24, dtype=np.int32)
        orders = (rng.random(24) * (degrees+1)).astype("int32")
        args = (degrees, orders, rng.uniform(.1, 3., 24).astype("float32"),
                rng.uniform(-3., 3., 24).astype("float32"))
        def fn(deg, order, theta, phi):
            return jsp.special.sph_harm_y(deg, order, theta, phi, n_max=n)
        trip = n-1
    else:
        n = (20, 32, 48)[size]
        args = schur(rng.normal(size=(n,n)).astype("float32"), output="real")
        fn = jsp.linalg.rsf2csf
        trip = n-1
    return fn, tuple(jax.block_until_ready(jax.device_put(x)) for x in args), trip


def factor_factory(fn, args, trip, *, rewrite_default=False):
    # These four library functions call a single outer static fori_loop. Retain
    # surrounding computation and result pytrees; specialize only its unroll.
    traced = None
    def build(factor):
        nonlocal traced
        if factor == 1 and not rewrite_default:
            return fn
        if traced is None:
            traced = jax.make_jaxpr(fn, return_shape=True)(*args)
        closed, shape = traced
        matches = []
        def visit(graph):
            equations = []
            for eq in graph.jaxpr.eqns:
                params = dict(eq.params)
                if eq.primitive.name in ("jit", "pjit") and "jaxpr" in params:
                    params["jaxpr"] = visit(params["jaxpr"])
                elif eq.primitive.name == "scan":
                    matches.append(params["length"])
                    params["unroll"] = factor
                equations.append(eq.replace(params=params))
            return graph.replace(jaxpr=graph.jaxpr.replace(eqns=equations))
        specialized = visit(closed)
        if matches != [trip]:
            raise ValueError(f"expected one library fori_loop of length {trip}, observed {matches}")
        evaluate = core.jaxpr_as_fun(specialized)
        tree = jax.tree.structure(shape)
        return lambda *xs: jax.tree.unflatten(tree, evaluate(*xs))
    return build


def child(name, size, repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    fn, args, trip = application(name, size, 583 + repeat)
    _, held, _ = application(name, size, 5830 + repeat)
    # Rotate cold-start order once per repeat. Clearing JAX's caches cannot
    # reset every process-global compiler initialization.
    order = ("native", "model", "tuned")
    for policy in order[repeat:] + order[:repeat]:
        jax.clear_caches()
        start = perf_counter_ns()
        if policy == "native":
            native = jax.jit(fn)
            jax.block_until_ready(native(*args))
            native_ms = (perf_counter_ns()-start)/1e6
        elif policy == "model":
            engine = DecisionEngine(ForiCostPredictor.load(MODEL))
            load_ms = (perf_counter_ns()-start)/1e6
            start = perf_counter_ns()
            decision = engine.decide(fn, args, trip_count=trip, loop_kind="fori_loop", throughput_only=True)
            predicted = jax.jit(factor_factory(fn,args,trip)(decision.factor))
            jax.block_until_ready(predicted(*args))
            model_ms = (perf_counter_ns()-start)/1e6
        else:
            build = factor_factory(fn,args,trip)
            tuned = tune_fori_kernel(build,args,seed=583+repeat)
    # Materialize all three kernels after cache clearing and compare actual
    # public-call outputs on an input that the tuner never timed.
    expected = jax.block_until_ready(native(*held))
    diagnostic_build = factor_factory(fn, args, trip, rewrite_default=True)
    kernels = [native, predicted, tuned.kernel] + [jax.jit(diagnostic_build(factor)) for factor in (1, 2, 4, 8)]
    for kernel in kernels:
        output = jax.block_until_ready(kernel(*held))
        if jax.tree.structure(output) != jax.tree.structure(expected):
            raise ValueError("incorrect output structure")
        for actual, wanted in zip(jax.tree.leaves(output), jax.tree.leaves(expected), strict=True):
            if not np.isfinite(actual).all() or not np.isfinite(wanted).all():
                raise ValueError("nonfinite output")
            np.testing.assert_allclose(actual, wanted, rtol=2e-4, atol=2e-5)
    calls = int(np.clip(2500/max(1,_times(native,held,3)), 3, 128))
    rng = np.random.default_rng(41+repeat)
    samples = np.empty((11,len(kernels)))
    for block in samples:
        for i in rng.permutation(len(kernels)):
            block[i] = _times(kernels[i],held,calls)
    us = np.median(samples,axis=0)
    return dict(routine=name,size=size,trip=trip,repeat=repeat,model_factor=decision.factor,
                fallback=decision.fallback_reason or "",tuned_factor=tuned.factor,
                native_us=us[0],model_us=us[1],tuned_us=us[2],native_setup_ms=native_ms,
                model_setup_ms=model_ms,model_load_ms=load_ms,tuned_setup_ms=tuned.setup_ms,
                **{f"rewrite_{factor}_us": duration for factor,duration in zip((1,2,4,8),us[3:],strict=True)})


def summarize(rows):
    results = []
    for policy in ("model", "tuned"):
        # When factor 1 was selected both policies run exactly the native callable.
        warm = rows[f"{policy}_us"].where(rows[f"{policy}_factor"] != 1, rows.native_us)
        setup = rows[f"{policy}_setup_ms"]
        if policy == "model":
            setup = setup + rows.model_load_ms
        for horizon in (1, 100, 1_000, 10_000, 100_000, None):
            if horizon is None:
                base, chosen = rows.native_us, warm
            else:
                base = rows.native_setup_ms + (horizon-1)*rows.native_us/1000
                chosen = setup + (horizon-1)*warm/1000
            medians = rows.assign(base=base, chosen=chosen).groupby(
                ["routine", "size"])[["base", "chosen"]].median()
            results.append(dict(policy=policy, calls=horizon or "warm",
                                equal_signature_speedup=np.exp(np.log(medians.base/medians.chosen).mean()),
                                pooled_speedup=medians.base.sum()/medians.chosen.sum()))
    pd.DataFrame(results).to_csv(OUT/"summary.csv",index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=3)
    args = parser.parse_args()
    if args.child:
        name,size,repeat = args.child
        print(json.dumps(child(name,int(size),int(repeat))))
        return
    OUT.mkdir(parents=True,exist_ok=True)
    jobs = [(n,s,r) for r in range(3) for n in ROUTINES for s in range(3)]
    np.random.default_rng(583).shuffle(jobs)
    rows = []
    for n,s,r in jobs:
        proc = subprocess.run([sys.executable,"-m","benchmarks.library_throughput","--child",n,str(s),str(r)],
                              env={**os.environ,"JAX_PLATFORMS":"cpu","OPENBLAS_NUM_THREADS":"1"},
                              capture_output=True,text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"runs.csv",index=False)
    summarize(frame)


if __name__ == "__main__":
    main()

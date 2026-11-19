"""Independent warm-latency confirmation. Compile once per variant and shape."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from benchmarks.library_throughput import application, factor_factory
from jax_unroll.library_kernels import add_reduceat_rows, subtract_reduce_rows

OUT = Path("results/library_specialization")
SHAPES = ((128,8), (257,1), (512,8), (769,16), (1024,8))
SEGMENTS = ((256, 4), (513, 7), (769, 13), (1024, 1), (1024, 16))
REPEATS = 7


def measure(kernel, inputs, count):
    start = perf_counter_ns()
    for i in range(count):
        jax.block_until_ready(kernel(*inputs[i % len(inputs)]))
    return (perf_counter_ns()-start) / count / 1000


def confirm(repeat):
    rng = np.random.default_rng(79001 + repeat)
    cases = [("subtract_reduce", i) for i in range(len(SHAPES))] + [("lexsort", 2)]
    cases += [("add_reduceat", i) for i in range(len(SEGMENTS))]
    rng.shuffle(cases)
    rows = []
    for name, size in cases:
        segments = 0
        if name == "subtract_reduce":
            n, width = SHAPES[size]
            inputs = [(jax.device_put(rng.normal(size=(n,width)).astype("float32")),) for _ in range(4)]
            fn = jnp.subtract.reduce
            trip = n - 1
            optimized = subtract_reduce_rows
            factor = 8
        elif name == "add_reduceat":
            n, segments = SEGMENTS[size]
            width = 1
            inputs = []
            for sample in range(4):
                starts = rng.choice(n, size=segments, replace=False).astype("int32")
                if sample % 2 == 0:
                    starts.sort()
                if sample == 3 and segments > 1:
                    starts[1] = starts[0]
                array = rng.normal(size=(n, width)).astype("float32")
                inputs.append((jax.device_put(array), jax.device_put(starts)))
            fn, optimized, trip, factor = jnp.add.reduceat, add_reduceat_rows, n, 4
        else:
            inputs = [application(name, size, 91000+100*repeat+i)[1] for i in range(4)]
            fn, _, trip = application(name, size, 91000+100*repeat)
            n,width = inputs[0][0].shape
            factor = 4
            optimized = factor_factory(fn,inputs[0],trip)(factor)
        jax.block_until_ready(inputs)
        # IR controls isolate factor changes in the exact original library loop.
        build = factor_factory(fn,inputs[0],trip,rewrite_default=True)
        functions = (fn, optimized, build(1), build(factor))
        kernels = [jax.jit(f) for f in functions]
        if name != "lexsort":
            kernels[1] = optimized  # The shipped entry point is already jitted.
        setup = np.empty(len(kernels))
        for i in rng.permutation(len(kernels)):
            start = perf_counter_ns()
            jax.block_until_ready(kernels[i](*inputs[0]))
            setup[i] = (perf_counter_ns() - start)/1e6
        for args in inputs:
            reference = np.asarray(kernels[0](*args))
            for k in kernels[1:]:
                np.testing.assert_array_equal(k(*args), reference)
        count = int(np.clip(3000 / measure(kernels[0],inputs,8), 4, 512))
        times = np.empty((15,len(kernels)))
        for block in times:
            for i in rng.permutation(len(kernels)):
                block[i] = measure(kernels[i],inputs,count)
        times = np.median(times,axis=0)
        rows.append(dict(routine=name,rows=n,width=width,segments=segments,repeat=repeat,factor=factor,
                         native_us=times[0],optimized_us=times[1],control1_us=times[2],
                         control_unrolled_us=times[3],native_setup_ms=setup[0],optimized_setup_ms=setup[1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    if args.repeat is not None:
        jax.config.update("jax_enable_compilation_cache", False)
        print(json.dumps(confirm(args.repeat)))
        return
    rows = []
    for rep in range(REPEATS):
        run = subprocess.run([sys.executable, "-m", "benchmarks.library_confirmation","--repeat",str(rep)],
                             check=True, capture_output=True,text=True,
                             env={**os.environ,"JAX_PLATFORMS":"cpu","OPENBLAS_NUM_THREADS":"1"})
        rows.extend(json.loads(run.stdout))
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(OUT/"confirmation.csv",index=False)
    data["speedup"] = data.native_us / data.optimized_us
    data["control_speedup"] = data.control1_us / data.control_unrolled_us
    print(data.groupby(["routine","rows","width","segments"])[["native_us","optimized_us","speedup","control_speedup"]].median())


if __name__ == "__main__":
    main()

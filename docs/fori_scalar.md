# Training on the supplied recurrence scenarios

All four 583 scenarios now enter the incremental training set. The loader retains the first-call milliseconds and warm microseconds for **each** native factor. Using first-call cost plus the remaining warm calls, the scalar recurrence's label is 4 at 300 calls and 8 at both 5,000 and 20,000 calls. The sine labels are 1 throughout. The measured matrix winner at the longer horizons is 4, but the original per-factor warm times differ by less than 0.1%. The fit keeps the incumbent on these in-range controls. [All training examples and per-horizon labels](../results/fori_scalar/training_labels.csv)

Two further training families, cumulative sums with a reset and a leaky running minimum, supply observations at 2,048, 4,095 and 8,192 steps. There are ten training configurations in total. The six test configurations come from three different families at 3,073 and 6,144 steps. Only training measurements enter the fit. The original scalar's result below is an in-training replication. [Split and input source hashes](../results/fori_scalar/split.json)

## What the engine learned

The packaged model adds a small table of four-factor costs to the existing compact forest. Seven scalar training cases set the table. At each measured length the fit takes the maximum across the families' median first-call costs for each factor. Warm estimates use the median per-factor ratio to native times the median native latency. The engine interpolates those costs between measured lengths. It minimizes predicted first-call plus remaining warm time among factors that meet the cold budget after the decision allowance.

The new support region covers one float32 scalar carry, one indexed float32 input stream of the trip length, and at most 16 simple equations. Transcendentals, reductions and multi-element intermediate results fall outside this region. The existing short-call bypass still makes the 300-call choice factor 1. The model uses only the loop trace to choose a factor, and its artifact digest versions the decision cache. [Fitted costs, model and base digests](../results/fori_scalar/fit.json)

## Direct Apple M2 timings

The three repeats at each horizon used a fresh CPU process for each arm, with a shared engine inside the process. Times include engine creation, lazy loading, trace-time selection, compilation and every synchronized call. The native baseline uses factor 1. The oracle executes a factor chosen by a separate four-factor sweep. Search cost is excluded. Each held-out family contributes two configurations, so equal-family and pooled seconds coincide.

| Partition | Calls per configuration | Native | Previous engine | Updated engine | Direct oracle | Seconds saved vs native |
|---|---:|---:|---:|---:|---:|---:|
| Original scalar, training control | 300 | .0200 | .0202 | .0203 | .0188 | −.0003 |
| Original scalar, training control | 5,000 | .1768 | .1795 | .1024 | .1001 | .0744 |
| Original scalar, training control | 20,000 | .6724 | .6846 | .3448 | .3532 | **.3275** |
| Six held-out configurations | 300 | .1625 | .1613 | .1621 | .1579 | .0003 |
| Six held-out configurations | 5,000 | 1.5764 | 1.5764 | 1.3932 | 1.2914 | **.1833** |
| Six held-out configurations | 20,000 | 6.0284 | 6.1425 | 5.2955 | 4.8042 | **.7328** |

On the original scalar, the updated engine and the long-horizon oracle both chose 8. Its measured runtime fell 48.7% versus native at 20,000 calls. On the six held-out configurations it saved 11.6% at 5,000 calls and 12.2% at 20,000, winning all three pooled repeats at both horizons. At 20,000 calls inventory saved .418 s and tracking error saved .345 s across their two lengths. The nonlinear filter retained native factor 1 and averaged .031 s slower with the wrapper; its directly measured oracle saved .431 s with factor 8. The engine recovered 59.9% of the held-out native-to-oracle savings. [Every configuration](../results/fori_scalar/cases.csv) · [Pooled timings](../results/fori_scalar/totals.csv) · [Raw measurements and decision reasons](../results/fori_scalar/live.json)

## Reproduction

Run from the repository root with the pinned CPU environment.

```sh
export PYTHONPATH=src:. JAX_PLATFORMS=cpu
py=.venv/bin/python
m=benchmarks.corpus.fori_scalar
$py -m "$m.dataset" freeze
$py -m "$m.dataset" train
$py -m "$m.fit"
$py -m "$m.dataset" test
$py -m "$m.validate"
$py -m pytest -q
```

The fit updates the bundled NPZ from the historical compact artifact and writes the training table and fit summary. Validation resumes complete process blocks with matching input digests. To acquire fresh measurements, use a separate checkout with the saved live file moved aside.

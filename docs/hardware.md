# Target hardware decision

Status: decided 2026-11-17 (Phase 0). Owner for the GPU question: TBD.

## Decision

**Commit to CPU-only results for now.** All Phase 1 sweep, oracle, and
risk-gate outputs are to be produced on the CPU backend of the machine described
below; no GPU measurements are part of the current dataset. Do not merge CPU and
GPU measurements into one table or one oracle, and do not make GPU speedup
claims from CPU data.

If a documented GPU is not secured by the end of week 1 (2026-11-23), the
report will state that results are CPU-only and that GPU behavior is future
work.

## Open question and deadline

- Open question: can the group secure access to a documented GPU machine?
- Deadline: end of week 1, **2026-11-23**.
- If yes: install and pin the matching CUDA/jaxlib variant on that machine,
  record the detailed device (for example `cuda:0` / GPU model) and exact
  versions in every row, then rerun `make sweep`, `make sweep-shapes`, and
  `make oracle` with `BACKEND=gpu`. Keep the CPU and GPU result sets separate.
- If no: finish the CPU-only evaluation and state the limitation explicitly.

`make BACKEND=gpu ...` labels the output files `..._gpu_...` and sets
`JAX_PLATFORMS=gpu` in the measurement children. On the current machine those
children would fail to initialize a GPU, producing error rows; do not run it
here.

## Recorded facts

Captured from `.venv` on 2026-11-17:

| Fact | Value |
|---|---|
| Machine / CPU | Apple M2 (`arm64`) |
| OS | macOS 26.3.1 (a), build `25D771280a` |
| Python | 3.13.2 |
| JAX | 0.12.0.1 |
| jaxlib | 0.12.0.1 |
| Backend | `cpu` (`jax.default_backend()`) |
| Devices | `[CpuDevice(id=0)]` from `jax.devices()`; the raw CSV `device` field stores the string form, `cpu:0` |

Verification commands and output:

```bash
$ source .venv/bin/activate
$ python -m jax_unroll
jax-unroll 0.1.0
jax 0.12.0.1 backend=cpu devices=[CpuDevice(id=0)]

$ python -c "import platform; print(platform.machine(), platform.mac_ver()[0])"
arm64 26.3.1
```

The sweep records `backend`, `device`, `jax_version`, and `jaxlib_version` on
every row, so the provenance of each measurement is in the raw CSV. See
[docs/data_schema.md](data_schema.md) for the column definitions.

## Reproduction

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
python -m jax_unroll
```

`requirements.txt` pins `jax==0.12.0.1` and `jaxlib==0.12.0.1` (the CPU
distribution used here), plus NumPy, pandas, scikit-learn, matplotlib, and
pytest. There is no CUDA pin because no GPU target has been approved.

`requirements.txt` ends with `-e .`, so the project is installed in editable
mode by `uv pip install -r requirements.txt`; `python -m jax_unroll` works from a
fresh checkout. The Makefile and `conftest.py` also add `src/` and the repo root
to the path automatically.

## Implications for results

- Warm and cold timings reflect a laptop-class M2 CPU under normal system load,
  not a dedicated benchmark host. Medians over fresh-process repeats are used
  to reduce noise; the raw spread is retained in `warm_p95_us` and
  `warm_std_us`.
- Unroll behavior, code generation, and compile cost on CPU say nothing about
  GPU behavior; XLA backends may differ substantially.
- If a GPU is added later, every table and plot must be regenerated separately
  for that backend. The oracle and risk gate should be run once per backend and
  never concatenated across backends.

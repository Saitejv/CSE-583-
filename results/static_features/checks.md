# Verification record

Run from the project root using pinned CPU JAX/jaxlib 0.12.0.1.

| Command | Outcome | Log |
|---|---|---|
| `.venv/bin/python -m pytest -q` with `JAX_PLATFORMS=cpu PYTHONPATH=src:.` | 935 passed in 96.01s | pytest.log |
| `make quick DATE=2026-11-17_static_features` | 8/8 correct, zero errors | quick.log |
| `make corpus-pilot-report` | Reproduced baseline MAEs and 3/8 refit wins | corpus_pilot_report_check.log |
| `make recovery-verify` | Verified existing gate and reproduced dataset/metadata byte for byte; no new score | recovery_verify.log |
| `make verify-week3` | Rebuilt 220 rows; both archived gate outputs byte-identical | verify_week3.log |
| `python -m benchmarks.corpus.static_experiment --check-pairs` | All 20 trip/factor pairs have identical complete Jaxprs and deterministic features | identical_pairs.json |
| Matched fresh-process acquisition | 120/120 correct, zero errors | acquisition_1023.log, acquisition_1024.log |
| `shasum -a 256 -c premeasurement.sha256` and `-c source.sha256` from root with full receipt path | All OK | receipts retained |

The quick target used a distinct DATE suffix to preserve the preexisting quick
CSV. The normal full pytest includes all new semantic/grouping/backward tests.

Independent reviewer checked the merged implementation and actual outputs:
recomputed all MAEs, solved each ridge equation independently (all predictions
agree within 1e-9), confirmed all eight train/test body partitions, and recounted
all policy budget violations in both matrices. No blocking issues. The primary
fails; no feature/hyperparameter tuning or new confirmation followed.

`git diff --exit-code` is empty for the frozen features/model/selector,
auto_loop stubs and frozen model artifact. Preexisting modified files remain
as found; no reset, stash, commit, or unrelated edit was made.

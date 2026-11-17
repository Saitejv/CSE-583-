SHELL := /bin/bash
ROOT := $(CURDIR)
DATE := $(shell date +%Y-%m-%d)
BACKEND ?= cpu
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

export PYTHONPATH := $(ROOT)/src:$(ROOT)

SWEEP_OUT ?= results/raw/sweep_$(BACKEND)_$(DATE).csv
SHAPE_OUT ?= results/raw/sweep_$(BACKEND)_$(DATE)_shapes.csv
ORACLE_OUT ?= results/oracle_$(BACKEND)_$(DATE).csv
ORACLE_MD ?= results/oracle_$(BACKEND)_$(DATE).md
GATE_OUT ?= results/risk_gate_$(BACKEND)_$(DATE).md
SWEEP_ARGS ?= --repeats 3
SHAPE_ARGS ?= --shapes secondary --trip-counts 256,1000 --repeats 3
ORACLE_ARGS ?= --r-calls 100
DATASET_OUT ?= data/features_$(BACKEND)_$(DATE).csv
MODEL_DATA ?= data/features_cpu_2026-11-17.csv
MODEL_DIR ?= models
MODEL_OUT := $(MODEL_DIR)/cost_cpu.json
OVERHEAD_OUT ?= results/selector_cpu_$(DATE)_overhead.json
RAW_FILES ?= $(filter-out %_quick.csv,$(wildcard results/raw/sweep_$(BACKEND)_*.csv))

.PHONY: help test list sweep sweep-shapes sweep-all oracle dataset quick train decision-overhead evaluate clean-results

help:
	@echo "make test         run pytest"
	@echo "make list         print the workload registry"
	@echo "make sweep        primary-shape factor sweep (T1.1-T1.5)"
	@echo "make sweep-shapes secondary shape variants at T=256,1000"
	@echo "make sweep-all    primary and secondary shape sweeps"
	@echo "make oracle       aggregate raw rows into oracle table + risk gate"
	@echo "make dataset      join static features with sweep labels (Phase 2)"
	@echo "make quick        small smoke sweep (use for harness checks)"
	@echo "make train        train and freeze a dev-only cost model"
	@echo "make decision-overhead   measure the static selector on CPU"
	@echo "make evaluate     evaluate dev LOFO policies (holdout is an explicit gate)"

test:
	$(PYTHON) -m pytest -q

list:
	$(PYTHON) -m benchmarks.sweep --list

sweep:
	$(PYTHON) -m benchmarks.sweep --backend $(BACKEND) --out $(SWEEP_OUT) $(SWEEP_ARGS)

sweep-shapes:
	$(PYTHON) -m benchmarks.sweep --backend $(BACKEND) --out $(SHAPE_OUT) $(SHAPE_ARGS)

sweep-all: sweep sweep-shapes

oracle:
	@RAW="$(SWEEP_OUT)"; \
	if [ -f "$(SHAPE_OUT)" ]; then RAW="$$RAW $(SHAPE_OUT)"; fi; \
	$(PYTHON) -m benchmarks.oracle --raw $$RAW $(ORACLE_ARGS) \
		--out $(ORACLE_OUT) --md-out $(ORACLE_MD) --gate-out $(GATE_OUT)

dataset:
	$(PYTHON) -m benchmarks.dataset --raw $(RAW_FILES) --out $(DATASET_OUT)

quick:
	$(PYTHON) -m benchmarks.sweep --backend $(BACKEND) \
		--out results/raw/sweep_$(BACKEND)_$(DATE)_quick.csv \
		--families recurrence,elementwise --trip-counts 16,100 --factors 1,8 \
		--repeats 1 --limit 12 --sample-budget-ms 50 --r-calls 5

train:
	$(PYTHON) -m jax_unroll.train --data $(MODEL_DATA) --out $(MODEL_DIR)

decision-overhead:
	$(PYTHON) -m benchmarks.decision_overhead --model $(MODEL_OUT) --out $(OVERHEAD_OUT)

$(OVERHEAD_OUT): $(MODEL_OUT)
	$(PYTHON) -m benchmarks.decision_overhead --model $(MODEL_OUT) --out $@

evaluate: $(OVERHEAD_OUT)
	$(PYTHON) -m benchmarks.evaluate --data $(MODEL_DATA) --model $(MODEL_OUT) --overhead $(OVERHEAD_OUT)

clean-results:
	rm -f results/raw/sweep_$(BACKEND)_$(DATE)*.csv

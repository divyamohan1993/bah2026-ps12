# FrameFlow Makefile — convenience targets for the monorepo.
# `make demo` is the headline end-to-end target: it runs the WHOLE pipeline on synthetic
# data (data -> models -> train -> infer -> validate -> viz/precompute) and produces a
# validated artifacts/<scene>/manifest.json plus a RESULTS.md metrics table.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

# Cap BLAS/OMP/torch thread pools so the demo/tests never oversubscribe the CPU.
THREAD_CAPS = OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
              NUMEXPR_NUM_THREADS=2 PYTORCH_NUM_THREADS=2

.DEFAULT_GOAL := help
.PHONY: help install install-all synth train-demo interpolate-demo validate-demo \
        precompute-demo web-install web-build demo test lint clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with the `core` extra (light foundation install).
	$(PIP) install -e ".[core]"

install-all:  ## Install with ALL extras (core, ml, geo, viz, serve, dev).
	$(PIP) install -e ".[core,ml,geo,viz,serve,dev]"

synth:  ## Generate the synthetic moving-cloud cube + demo .nc triplet.
	$(PYTHON) -m frameflow.synthetic

train-demo:  ## Tiny 1-epoch training run on the synthetic cube (Team TRAIN).
	$(PYTHON) -m frameflow.cli train configs/config.yaml train.epochs=1 || \
	  echo "train-demo skipped (frameflow.train not implemented yet)."

interpolate-demo:  ## Interpolate the middle frame of the demo triplet (Team INFER).
	$(PYTHON) -m frameflow.cli interpolate \
	  data/demo_nc/synthetic_20250620T000000Z_observed.nc \
	  data/demo_nc/synthetic_20250620T002000Z_observed.nc \
	  --t 0.5 --out-nc out/interp_nc/mid.nc || \
	  echo "interpolate-demo skipped (frameflow.infer not implemented yet)."

validate-demo:  ## Validate interpolated vs ground-truth frames (fixed-K data_range; P1).
	$(PYTHON) -m frameflow.cli validate \
	  --pred-dir out/interp_nc --truth-dir data/demo_nc --out out/validation || \
	  echo "validate-demo skipped (frameflow.validate not implemented yet)."

precompute-demo:  ## Precompute per-frame web artifacts from the synthetic cube (P2).
	$(PYTHON) -m frameflow.cli precompute \
	  --cube data/cubes/synthetic.zarr --out out/web || \
	  echo "precompute-demo skipped (frameflow.precompute not implemented yet)."

web-install:  ## Install the web dashboard's node dependencies (Team WEB).
	cd web && npm install || echo "web-install skipped (web/ not present yet)."

web-build:  ## Build the web dashboard static bundle (Team WEB).
	cd web && npm run build || echo "web-build skipped (web/ not present yet)."

demo:  ## Run the full REAL end-to-end demo (train IFNet -> infer -> validate -> precompute).
	$(THREAD_CAPS) $(PYTHON) scripts/demo.py

test:  ## Run the test suite (pytest).
	$(THREAD_CAPS) $(PYTHON) -m pytest

lint:  ## Lint with ruff.
	$(PYTHON) -m ruff check frameflow scripts tests

clean:  ## Remove generated data/outputs and Python caches.
	rm -rf data out runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true

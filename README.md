<div align="center">

# FrameFlow

### Fill in the Frames Seamlessly — AI/ML optical-flow temporal interpolation of geostationary satellite imagery

**ISRO Bharatiya Antariksh Hackathon (BAH) 2026 · Problem Statement 12**

[![CI](https://github.com//actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Web: Vite + React + TS](https://img.shields.io/badge/web-Vite%20%2B%20React%20%2B%20TS-646cff.svg)](web/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](pyproject.toml)
[![Tests: 173 passed](https://img.shields.io/badge/tests-173%20passed%20·%201%20skipped-success.svg)](tests/)

*Train on the world's densest geostationary TIR feed (GOES-19, 10 min) → deploy on INSAT-3DS/3DR (30 min).*

[Quickstart](#quickstart) · [Architecture](ARCHITECTURE.md) · [Contracts](CONTRACTS.md) · [Model Card](MODEL_CARD.md) · [Results](RESULTS.md) · [Research](research/)

</div>

---

## The problem

Geostationary weather satellites trade temporal resolution for permanence: **INSAT-3DS/3DR**
deliver a full-disk Thermal-IR (TIR1, 10.8 µm) frame only every **~30 minutes**, yet cyclone
eyewalls, convective initiation, squall lines, wildfire fronts and flash floods evolve
meaningfully *inside* that gap. Launching more satellites is expensive; **synthesizing the
missing frames with AI is not.**

FrameFlow takes two consecutive brightness-temperature frames (e.g. 00:00 and 00:20) and
synthesizes the intermediate frame(s) (e.g. 00:10) with an **optical-flow video-interpolation
model**, densifying cadence **30 → 15 → 7.5 min** without new hardware — then *proves* the
result is physically faithful with a rigorous, multi-satellite validation suite and an
interactive ground-truth-vs-interpolated mission-control dashboard.

## The workflow: train on GOES, deploy on INSAT

```
   DENSE training feed                 LEARN motion + in-betweening            OPERATIONAL target
 ┌──────────────────────┐            ┌───────────────────────────┐          ┌─────────────────────┐
 │ GOES-19 ABI C13 10min│  ───────►  │  RIFE / IFNet (intermediate│ ──fine── │ INSAT-3DS/3DR TIR1   │
 │ Himawari-9 B13 10min │   triplets │  optical flow, arbitrary t)│   tune   │ 30min → 15 → 7.5 min │
 │ GK-2A AMI IR105 10min│            │  + 40-method validation    │          │ 15-min animations    │
 └──────────────────────┘            └───────────────────────────┘          └─────────────────────┘
```

The primary engine is **RIFE / Practical-RIFE (IFNet)** — fast, MIT-licensed, natively
arbitrary-time, ~10 M params — fine-tuned on single-channel brightness temperature, then
deployed and domain-adapted on INSAT TIR1. Comparators (EMA-VFI, IFRNet, Super-SloMo, FILM)
and classical/persistence baselines are all in-repo.

---

## Quickstart

```bash
# 1) Install everything (core + ml + geo + viz + serve + dev).
make install-all

# 2) Run the FULL real end-to-end demo on synthetic data:
#    synthetic Zarr cube → TripletDataset → train IFNet (CPU) → interpolate (.nc) →
#    validate vs withheld truth (fixed-K metrics) → 40-method cross-val →
#    precompute web artifacts (manifest + per-frame WebP + tiles + videos + flow), and
#    assert the manifest passes validate_manifest.  ~45 s on CPU.
make demo

# 3) Launch the dashboard (it ships a REAL precomputed scene under web/public/data/demo-0001).
cd web && npm install && npm run dev          # → http://localhost:5173
```

To embed a fresh real scene from your own `make demo` run into the dashboard:

```bash
make demo && python scripts/embed_demo_scene.py   # writes web/public/data/demo-0001/
```

Lighter paths:

```bash
make synth     # just materialize the synthetic cube + demo .nc triplet
make test      # pytest (173 passed / 1 skipped, fully offline)
make lint      # ruff
cd web && npm run mock   # regenerate the synthetic MOCK dashboard scenes (no Python)
```

> **Threading note.** Python/pytest runs cap BLAS/OMP/torch thread pools
> (`OMP_NUM_THREADS=2 …`) so the demo/tests never oversubscribe the CPU; the Makefile and CI
> already set this.

---

## Real demo metrics

Reproducible via `make demo` — a tiny IFNet trained briefly on **synthetic** data on CPU,
validated on **withheld true middle frames** vs classical baselines. All full-reference
metrics use the **fixed Kelvin `data_range`** (P1, 180–320 K); per-image min/max is forbidden.
These are integration/sanity numbers (see the honest caveat below), not the production result.

| method | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | FSIM ↑ | BT-RMSE (K) ↓ |
|---|---|---|---|---|---|
| **trained IFNet** | **26.39** | **0.929** | **0.912** | **0.961** | **6.80** |
| linear blend | 26.19 | 0.889 | 0.872 | 0.954 | 6.97 |
| TV-L1 + warp | 25.53 | 0.889 | 0.872 | 0.951 | 7.50 |

*Cross-validation:* 18 / 40 methods execute on the synthetic cube; the rest need real
multi-satellite data and register as `skipped`. Full table + reproduce steps in
[`RESULTS.md`](RESULTS.md); model details/limitations in [`MODEL_CARD.md`](MODEL_CARD.md).

> **Honest scope.** The demo trains a ~0.05 M IFNet for a few hundred CPU steps on synthetic
> data purely to prove the whole pipeline wires together and emits *real numbers* end-to-end —
> it is **not** expected to beat the prior-art bar (Vandal & Nemani 2021: sub-K RMSE, SSIM ≥
> 0.933). Production quality needs **GPU training on multi-day real satellite sequences**.

---

## Architecture

Six-stage monorepo — **data engineering → models → training → inference → validation →
serving/web** — glued by authoritative in-code contracts. Full design, citations to the six
research reports, and diagrams are in **[ARCHITECTURE.md](ARCHITECTURE.md)**; the exact
interfaces each stage implements are in **[CONTRACTS.md](CONTRACTS.md)**.

```
raw .nc/.h5 (GOES / Himawari / GK-2A / INSAT)
   └─► VirtualiZarr / kerchunk (zero-copy refs) ─► regrid (cached pyresample KDTree) + BT + normalize
        └─► Zarr v3 cube (time,y,x) bt float32 K ─► TripletDataset ─► RIFE / IFNet (intermediate flow)
             └─► interpolate ─► CF NetCDF ─► validate (fixed-K metrics, 40 methods) ─► precompute
                  └─► per-frame tiles + PMTiles + all-intra video + manifest.json ─► web dashboard
```

Three correctness guarantees are baked into the contracts:

1. **Fixed physical metric range (P1)** — every full-reference metric uses the shared Kelvin
   `data_range` `BT_DATA_RANGE_K = 140 K` (180–320 K). Per-image min/max is forbidden — it
   hides radiometric (warm/cold-bias, contrast) errors.
2. **Per-frame tile sources (P2)** — a single raster PMTiles archive is addressed only by
   `z/x/y` and cannot select a timestamp, so **every frame has its own tile pyramid / PMTiles
   archive**; the timeline slider switches the active frame's source.
3. **Batched `t` ONNX export** — the timestep input `t` is shaped `[B,1]` and listed in
   `dynamic_axes` so Triton dynamic batching works.

---

## Multi-satellite data strategy

Grounded in [`research/02_satellite_data.md`](research/02_satellite_data.md) — **38 catalogued
datasets / access methods**. The world's GEO satellites form a *ring* whose limbs overlap, and
polar orbiters cut across all of them — enabling train-on-densest / test-on-every-other /
truth-from-overpasses.

| Phase | Dataset(s) | Band / λ | Cadence | Access |
|---|---|---|---|---|
| **Primary training** | GOES-19 ABI **C13** | 10.3 µm | 10 min | anon S3 `noaa-goes19` (`.nc`) |
| **Secondary / diversity** | Himawari-9 AHI **B13**, GK-2A AMI **IR105** | 10.4 / 10.5 µm | 10 min | anon S3 (`.nc`) |
| **Deployment target** | **INSAT-3DS / 3DR TIR1** | 10.8 µm | ~30 min | MOSDAC / EUMETSAT (`.h5`) |
| **Independent truth** | MODIS B31, VIIRS M15, Sentinel-3 SLSTR S8; Meteosat IODC, FY-4B, Electro-L | ~10.8–11 µm | overpass / various | S3 / data stores |

Fastest access: anonymous S3 via `s3fs`/`fsspec` straight into `xarray` — no download, no
credentials, lazy chunked reads — virtualized with kerchunk/VirtualiZarr for O(1) chunk reads.
GOES/Himawari (`.nc`) and INSAT (`.h5`) both satisfy the PS's NetCDF/HDF5 I/O requirement.

---

## Validation — 40 cross-validation methods

Far beyond "SSIM/MSE/PSNR/FSIM" — see [`research/05_validation.md`](research/05_validation.md)
and [`RESULTS.md`](RESULTS.md). A layered suite (pixel/structure, motion/temporal, nowcasting
skill, TIR-domain) plus an explicit **40-method** robustness framework:
leave-the-middle-out on dense GOES-19/Himawari, cross-satellite verification, polar-orbiter
independent truth, triple collocation, baseline/ablation comparisons, stratified breakdowns,
and dual-implementation metric agreement. Headline set: PSNR, SSIM, MS-SSIM, FSIM,
**BT-RMSE (K)**, CSI@235 K, FSS@scale, optical-flow EPE — all on the fixed Kelvin range (P1).

---

## Web dashboard

Vite + React + TypeScript, **MapLibre GL JS + deck.gl**, raster **PMTiles**, uPlot — see
[`research/04_web_viz.md`](research/04_web_viz.md). A dark mission-control UI with synced
**ground-truth-vs-interpolated** swipe/side-by-side compare, a unified scrubbable timeline,
an optical-flow vector overlay, an error-heatmap toggle, and a live per-frame metric HUD.

The dashboard ships a **genuine precomputed scene** (`web/public/data/demo-0001/`, real IFNet
output from `make demo`) and loads it by default — manifest + per-frame WebP + all-intra MP4s
+ flow overlays. It reads the manifest, builds the timeline from `frames[]`, and switches the
active raster source per slider tick (P2); when a frame has no tile pyramid it falls back to a
deck.gl BitmapLayer over the frame image (the path the embedded scene uses).

---

## Repository layout

```
frameflow/
  constants.py       # shared constants: band specs, FIXED metric range (P1), grid, tiles, S3 templates
  contracts.py       # AUTHORITATIVE contracts: GridSpec, CubeSchema, Sample, .nc schema, Manifest, validate_manifest
  config.py          # typed Data/Model/Train/Infer/Validate/Serve configs (Hydra-backed)
  synthetic.py       # working synthetic moving-cloud BT generator (numpy+xarray+zarr)
  cli.py             # Typer CLI: synth / ingest / train / interpolate / validate / precompute / serve / demo
  data/              # access + ingest (Zarr cube, kerchunk refs) + preprocess (BT, regrid) + TripletDataset
  models/            # VFIModel base + RIFE/IFNet, EMA-VFI, IFRNet, Super-SloMo, FILM + classical baselines
  train/             # losses (Charbonnier/census/MS-SSIM/gradient) + Lightning trainer
  infer/             # interpolate_pair / interpolate_recursive (.nc out) + ONNX/TensorRT export
  validate/          # fixed-K metrics + the 40-method CrossValSuite
  viz/  serve/  precompute.py   # render/tiles/video/flow + FastAPI app + the O(1) precompute → manifest
configs/             # Hydra YAMLs (config.yaml + data/model/train/infer/validate/serve groups)
scripts/
  demo.py            # the `make demo` end-to-end target
  embed_demo_scene.py# copy a compact, browser-valid REAL scene into web/public/data/
tests/               # conftest + test_<area>.py  (173 passed / 1 skipped, fully offline)
web/                 # Vite + React + TS dashboard (MapLibre + deck.gl + uPlot)
research/            # the six deep-research reports grounding every decision (01…06)
ARCHITECTURE.md  CONTRACTS.md  MODEL_CARD.md  RESULTS.md  pyproject.toml  Makefile
```

Module ownership and directory boundaries are tabulated in [CONTRACTS.md §2](CONTRACTS.md).

---

## Tech stack

| Layer | Tools |
|---|---|
| **Data** | xarray · Zarr v3 · netCDF4 / h5py · kerchunk / VirtualiZarr · s3fs / fsspec · pyresample · rioxarray / rasterio · dask |
| **Model / Train** | PyTorch · Lightning · torchmetrics · piq · Hydra / OmegaConf |
| **Inference / Serve** | ONNX (opset 17) → TensorRT FP16 · FastAPI · uvicorn · (Triton dynamic batching) |
| **Validation** | NumPy / SciPy · piq · OpenCV (TV-L1 / Farnebäck) · the 40-method suite |
| **Viz / Web** | Vite · React · TypeScript · MapLibre GL JS · deck.gl · raster PMTiles · uPlot · Tailwind |
| **Tooling** | ruff · pytest · GitHub Actions CI (Python + web) |

---

## Deployment

The serving substrate is **O(1) by construction**: all expensive work happens **once**,
offline, in `precompute`, persisting immutable, directly-addressable static artifacts
(per-frame XYZ tiles + **PMTiles** + all-intra video + a validated `manifest.json`). The web
app then only issues constant-time range GETs against immutable files.

- **Static frontend** → **Cloudflare Pages** / **Vercel** (`cd web && npm run build`, deploy
  `web/dist/`). The embedded `demo-0001` scene makes the deployed dashboard self-contained.
- **Artifacts / tiles** → **Cloudflare R2** (or any object store + CDN). **PMTiles** serve a
  whole per-frame pyramid from a single immutable object via HTTP range requests — **O(1)**,
  no tile server.
- **Optional on-demand interpolation** → the FastAPI app (`frameflow serve`) exposes a
  content-addressed `/interpolate` endpoint backed by the ONNX/TensorRT model (FP16; never
  INT8) with Triton dynamic batching — used only when a frame isn't already precomputed.

CI (`.github/workflows/ci.yml`) runs two jobs on every push/PR: **python** (ruff + the full
offline pytest suite on CPU-only torch) and **web** (`npm ci && npm run build`).

---

## Status

The foundation and all six module areas (data, models, train, infer, validate, serve/viz,
web) are implemented and verified: `make demo` runs the whole pipeline end-to-end and writes a
`validate_manifest`-clean scene; the test suite is **173 passed / 1 skipped** and **ruff
clean**; the dashboard builds green and ships a real precomputed scene. Real quality is a
matter of GPU training on real multi-day data — every hook (GOES-19 / Himawari-9 / GK-2A /
INSAT-3DS) runs through the same pipeline as the synthetic demo.

---

<div align="center">

*License: MIT (see [`pyproject.toml`](pyproject.toml)). Built for ISRO BAH 2026 PS-12.*
**[ARCHITECTURE.md](ARCHITECTURE.md) · [CONTRACTS.md](CONTRACTS.md) · [MODEL_CARD.md](MODEL_CARD.md) · [RESULTS.md](RESULTS.md) · [research/](research/)**

</div>

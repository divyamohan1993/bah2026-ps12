# FrameFlow

**Fill in the Frames Seamlessly — AI/ML optical-flow temporal interpolation of geostationary satellite imagery.**

*ISRO Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement 12.*

---

Geostationary weather satellites trade temporal resolution for permanence: **INSAT-3DS/3DR**
deliver a full-disk Thermal-IR (TIR1, 10.8 µm) frame only every **~30 minutes**, yet
cyclone eyewalls, convective initiation, squall lines, wildfire fronts and flash floods
evolve meaningfully *inside* that gap. Launching more satellites is expensive; **synthesizing
the missing frames with AI is not.** FrameFlow takes two consecutive brightness-temperature
frames (e.g. 00:00 and 00:20) and synthesizes the intermediate frame(s) (e.g. 00:10),
densifying cadence **30 min → 15 min → 7.5 min** without new hardware — and proves the
result is physically faithful with a rigorous, multi-satellite validation suite and an
interactive ground-truth-vs-interpolated dashboard.

The primary engine is **RIFE / Practical-RIFE (IFNet)** — fast, MIT-licensed, natively
arbitrary-time, ~10 M params — fine-tuned on single-channel brightness temperature from
**GOES-19 ABI C13** (10.3 µm) and **Himawari-9 AHI B13** (10.4 µm), then deployed on
**INSAT-3DS/3DR TIR1**.

---

## PS-12 mapping

| PS-12 deliverable (`idea.md`) | FrameFlow |
|---|---|
| Optical-flow model for motion between consecutive frames | RIFE/IFNet (internal intermediate-flow estimation); RAFT/SEA-RAFT for flow-vector viz & EPE |
| Generate synthetic intermediate frames via DL interpolation | RIFE primary + EMA-VFI / IFRNet / Super SloMo / FILM comparators + classical & persistence baselines |
| Improve temporal resolution 30 → 15 → 7.5 min | Arbitrary-`t` interpolation + recursion (`interpolation_factor` 2, 4, …) |
| Validate against higher-cadence GOES-19 / Himawari with SSIM/MSE/PSNR/FSIM/… | 40-method cross-validation framework; headline PSNR/SSIM/MS-SSIM/FSIM/BT-RMSE(K)/CSI/FSS/EPE |
| `.nc` input **and** output | CF-style NetCDF in/out (`InferenceNetCDFSchema`) |
| Web dashboard: original vs interpolated time-lapse + metric report | MapLibre + deck.gl dashboard, per-frame PMTiles/tiles, uPlot metric strip |
| Apply best model on INSAT-3DS, produce 15-min animations | INSAT fine-tune/deploy path; the same precompute → manifest → dashboard pipeline |

---

## Architecture

Six-stage monorepo: **data engineering → models → training → inference → validation →
serving/web**, glued by authoritative in-code contracts. Full design, citations to the six
research reports, and diagrams are in **[ARCHITECTURE.md](ARCHITECTURE.md)**. The exact
interfaces each team implements are in **[CONTRACTS.md](CONTRACTS.md)**.

```
raw .nc/.h5 (GOES/Himawari/INSAT)
   └─► VirtualiZarr/Icechunk (zero-copy) ─► regrid (cached KDTree) + BT + normalize
        └─► Zarr v3 cube (time,y,x) bt float32 K ─► TripletDataset ─► RIFE/IFNet
             └─► interpolate ─► .nc ─► validate (fixed-K metrics) ─► precompute
                  └─► per-frame tiles + PMTiles + video + manifest.json ─► web dashboard
```

Three correctness guarantees are baked into the contracts (see below): a **fixed physical
metric range**, **per-frame tile sources**, and a **batched `t` ONNX export**.

---

## Quickstart

```bash
# 1) install everything (core + ml + geo + viz + serve + dev)
make install-all

# 2) run the full end-to-end demo (synthetic data → train → interpolate → validate → web)
make demo
```

`make demo` is robust: it always generates a physically-plausible synthetic moving-cloud
dataset and then runs each downstream stage, gracefully **skipping** any stage whose team
module isn't landed yet. To just materialize the synthetic data and run the tests:

```bash
make synth        # writes data/cubes/synthetic.zarr + data/demo_nc/*.nc
make test         # pytest (foundation smoke tests pass today)
make lint         # ruff
```

A lighter install (foundation only — constants, contracts, synthetic generator, CLI):

```bash
make install      # pip install -e ".[core]"
python -m frameflow.synthetic         # generate the demo cube + .nc triplet
python -m frameflow.cli --help        # explore the CLI
```

---

## Repository layout

```
frameflow/
  __init__.py        # package + __version__ (no submodule imports at top level)
  constants.py       # REAL shared constants: band specs, FIXED metric range, grid, tiles, S3 templates
  contracts.py       # AUTHORITATIVE contracts: GridSpec, CubeSchema, Sample, .nc schema, Manifest, validate_manifest
  config.py          # typed Data/Model/Train/Infer/Validate/Serve configs (Hydra-backed)
  synthetic.py       # WORKING synthetic moving-cloud BT generator (numpy+xarray+zarr)
  cli.py             # Typer CLI (lazy imports): synth/ingest/train/interpolate/validate/precompute/serve/demo
  data/      models/  train/  infer/  validate/  serve/  viz/  precompute.py   # built by the six teams
configs/             # Hydra YAMLs (config.yaml + data/model/train/infer/validate/serve groups)
scripts/demo.py      # end-to-end demo (the `make demo` target)
tests/               # conftest.py + test_foundation.py (+ per-team test_<area>.py)
web/                 # Vite+React+TS dashboard (Team WEB)
research/            # the six deep-research reports grounding every decision
ARCHITECTURE.md  CONTRACTS.md  pyproject.toml  Makefile
```

Module ownership and directory boundaries (so teams never conflict) are tabulated in
[CONTRACTS.md §2](CONTRACTS.md).

---

## Data sources

Grounded in [`research/02_satellite_data.md`](research/02_satellite_data.md) (38 catalogued
datasets/access methods):

- **Primary training:** GOES-19 ABI C13 (10.3 µm, 10-min) — anonymous S3 `noaa-goes19`.
- **Secondary / Asia-Pacific:** Himawari-9 AHI B13 (10.4 µm), GK-2A AMI IR105 (10.5 µm).
- **Deployment target:** INSAT-3DS/3DR TIR1 (10.8 µm) — MOSDAC `mdapi` / EUMETSAT `eumdac`.
- **Cross-validation ring:** Meteosat IR10.8 (IODC overlaps INSAT), FY-4B, Electro-L (76°E),
  plus polar-orbiter truth (MODIS B31, VIIRS M15, Sentinel-3 SLSTR S8) at overpass times.

Fast access pattern: anonymous S3 via `s3fs`/`fsspec` straight into `xarray`, virtualized
with VirtualiZarr/kerchunk for O(1) chunk reads. INSAT (`.h5`) and GOES/Himawari (`.nc`)
both satisfy the PS's NetCDF/HDF5 I/O requirement.

---

## Model

RIFE/Practical-RIFE (IFNet) as primary — see
[`research/01_vfi_models.md`](research/01_vfi_models.md). Single-channel adaptation
(replicate-3 or averaged conv1), BF16 fine-tuning with Charbonnier + census + MS-SSIM +
gradient losses (+ IFRNet privileged flow-distillation / geometry-consistency). The
scientific bar to beat is Vandal & Nemani (IEEE TNNLS 2021): on GOES-R Band 13 at 10-min,
sub-Kelvin RMSE and SSIM ≥ 0.93.

---

## Validation (40 cross-validation methods)

Far beyond "SSIM/MSE/PSNR/FSIM" — see
[`research/05_validation.md`](research/05_validation.md). Layered suite (pixel/structure,
motion/temporal, nowcasting skill, TIR-domain) plus an explicit **40-method** robustness
framework: leave-the-middle-out on dense GOES-19/Himawari, cross-satellite verification,
polar-orbiter independent truth, triple collocation, baseline/ablation comparisons,
stratified breakdowns, and dual-implementation metric agreement.

> **Fixed metric range (P1):** every full-reference metric uses the shared physical Kelvin
> range `BT_DATA_RANGE_K = 140 K` (180–320 K). Per-image min/max is forbidden — it hides
> radiometric (warm/cold-bias, contrast) errors.

---

## Web dashboard

Vite + React + TypeScript, MapLibre GL JS + deck.gl, raster **PMTiles**, uPlot — see
[`research/04_web_viz.md`](research/04_web_viz.md). Synced ground-truth-vs-interpolated
swipe/side-by-side compare, a unified scrubbable timeline, optical-flow vector overlay,
error-heatmap toggle, and a live metric HUD, in a dark mission-control theme.

> **Per-frame tile sources (P2):** a single raster PMTiles archive is addressed only by
> `z/x/y` and cannot select a timestamp, so **every frame has its own tile pyramid /
> PMTiles archive**; the timeline slider switches the active frame's source. The manifest
> encodes per-frame `tiles_url_template` and `pmtiles`.

---

## Status

This is an **actively-built, robust framework**, not a toy. The foundation is complete and
verified: shared constants, the authoritative interface contracts, typed configs, the
Hydra config tree, the CLI, and a **working synthetic generator** that produces a
physically-plausible moving-cloud brightness-temperature Zarr cube and a schema-correct
NetCDF demo triplet — all importable and runnable today (`make synth`, `make demo`,
`make test`). The six module subpackages (data, models, train, infer, validate, serve/viz,
web) are built in parallel by independent teams against the contracts in
[CONTRACTS.md](CONTRACTS.md), with **real multi-satellite data hooks** (GOES-19 / Himawari-9
/ GK-2A / INSAT-3DS) wired through the same pipeline as the synthetic demo.

---

*License: MIT (see `pyproject.toml`). Built for ISRO BAH 2026 PS-12.*

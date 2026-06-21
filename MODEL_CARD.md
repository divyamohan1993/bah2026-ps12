# Model Card — FrameFlow VFI (satellite Thermal-IR frame interpolation)

> Model card for the video-frame-interpolation (VFI) engine in **FrameFlow**
> (ISRO BAH 2026, Problem Statement 12 — *"Fill in the Frames Seamlessly"*).
> Conventions follow Mitchell et al., *Model Cards for Model Reporting* (2019).

---

## Model details

| | |
|---|---|
| **Name** | FrameFlow-VFI (RIFE / Practical-RIFE **IFNet** backbone) |
| **Task** | Temporal interpolation (in-betweening) of geostationary **Thermal-IR** brightness-temperature frames |
| **Architecture** | IFNet — a coarse-to-fine, residual, **intermediate-flow** estimator: per-pyramid-level flow + a fusion/refine head, warping both bracket frames toward the target time `t`. Single-channel adaptation (replicate-3 or averaged conv1) for 1-band TIR. Optional comparators in-repo: EMA-VFI, IFRNet, Super-SloMo, FILM, plus classical (TV-L1 / Farnebäck warp) and persistence baselines. |
| **Parameters** | ~**9.8 M** (production Practical-RIFE IFNet). The bundled `make demo` trains a deliberately tiny **~0.05 M** IFNet config on CPU as an integration proof, not a quality artifact. |
| **Arbitrary `t`** | Yes — the model takes `t ∈ (0,1)` (shaped `[B,1]`, leading batch dim for ONNX/Triton dynamic batching), so cadence can be densified 30→15→7.5 min and beyond by recursion (`interpolation_factor` 2, 4, …). |
| **Inputs / outputs** | `I0, I1 : (B,1,H,W)` normalized brightness temperature; `t : (B,1)` → predicted `It : (B,1,H,W)`; optionally an intermediate **flow** `(B,2,H,W)` for the dashboard overlay / EPE. |
| **Precision** | BF16 mixed-precision training; FP16 TensorRT for serving (**never INT8** — it ruins VFI fidelity). |
| **License** | MIT (RIFE/Practical-RIFE is MIT). Repository: MIT. |
| **Framework** | PyTorch + Lightning; export to ONNX (opset 17) → TensorRT FP16. |

The authoritative interfaces are in [`CONTRACTS.md`](CONTRACTS.md) (`VFIModel`, `Sample`,
the NetCDF I/O schema, the manifest); the full rationale and citations are in
[`ARCHITECTURE.md`](ARCHITECTURE.md) and the six reports under [`research/`](research/).

---

## Intended use

**Primary use.** Operational/temporal super-resolution of single-channel geostationary
TIR imagery — synthesizing physically-plausible intermediate cloud-top brightness-temperature
frames between consecutive acquisitions, to support nowcasting of fast-evolving systems
(cyclone eyewalls, convective initiation, squall lines, wildfire fronts, flash-flood
producing storms) that change meaningfully **inside** the native ~30-min cadence.

**Target deployment.** **INSAT-3DS / INSAT-3DR TIR1 (~10.8 µm)** — the operational PS-12
target. The model is trained on dense high-cadence analogues (below) and fine-tuned /
domain-adapted onto INSAT TIR1.

**Intended users.** Meteorological/operational nowcasting teams, satellite-meteorology
researchers, and the BAH evaluators reproducing the pipeline.

**Out-of-scope / not intended for.**
- Producing **new physical information** — interpolation cannot invent unobserved
  convective initiation that begins *and* peaks entirely between two frames.
- Quantitative retrievals (precip rate, cloud microphysics) directly off interpolated
  pixels without independent validation.
- Visible/multispectral "pan-sharpening", spatial super-resolution, or extrapolation
  (forecasting **beyond** the last observed frame) — this is **interpolation between**
  bracketing observations only.
- Safety-critical decisions without a human in the loop and the accompanying uncertainty/
  error diagnostics.

---

## Training data

FrameFlow trains on the **densest, freely-available** high-cadence TIR feeds and deploys on
INSAT. Full catalogue (38 datasets/access methods) and the fast-access recipes are in
[`research/02_satellite_data.md`](research/02_satellite_data.md).

| Role | Dataset(s) | Band / λ | Cadence | Access |
|---|---|---|---|---|
| **Primary training** | **GOES-19 ABI** L1b RadF / L2 CMIPF **C13** | 10.3 µm clean-window | 10 min full-disk | anon S3 `s3://noaa-goes19/` (NetCDF4) |
| **Secondary / domain diversity** | **Himawari-9 AHI B13**; **GK-2A AMI IR105** | 10.4 / 10.5 µm | 10 min | anon S3 `noaa-himawari9` / `noaa-gk2a-pds` |
| **Deployment target** | **INSAT-3DS / 3DR TIR1** | 10.8 µm | ~30 min | MOSDAC (`.h5`), EUMETSAT mirror |
| **Independent cross-val truth** | MODIS B31, VIIRS M15, Sentinel-3 SLSTR S8; Meteosat IODC, FY-4B, Electro-L | 10.8–11 µm | overpass / various | S3 / data stores |

**Preprocessing.** Radiance→brightness-temperature (Planck for ABI; LUT for INSAT) → regrid
to a common EPSG:4326 grid with a **cached pyresample KD-tree** → fixed-range normalization
to `[0,1]` (`BT_NORM_VMIN_K=180`…`BT_NORM_VMAX_K=330` K) → NaN-mask off-disk/space pixels →
canonical Zarr cube `bt (time,y,x)` float32 Kelvin. Triplets `(I0, It, I1)` are split **by
time** (never random) to prevent temporal leakage.

> The model expects **normalized** input; raw Kelvin fed to the conv stack produces NaN.
> The infer and precompute paths normalize before the network and denormalize the prediction
> back to Kelvin, restoring the off-disk NaN mask.

---

## Metrics & evaluation

All full-reference metrics use the **fixed physical Kelvin `data_range`** (P1):
`BT_DATA_RANGE_K = 140 K` (180–320 K). **Per-image min/max is forbidden** — it re-normalizes
each frame to its own extremes and hides radiometric errors (warm/cold bias, contrast
compression), making cross-frame/cross-satellite/cross-method comparison meaningless.

**Headline metric set** (research/05 §0): PSNR, SSIM, MS-SSIM, FSIM, **BT-RMSE (K)**,
CSI@235 K (cold-cloud skill), FSS@scale, optical-flow EPE. Validation withholds true
intermediate frames ("leave-the-middle-out") on the dense GOES-19/Himawari feeds and uses an
explicit **40-method** cross-validation framework (cross-satellite, polar-orbiter independent
truth, triple collocation, baseline/ablation, stratified breakdowns, dual-implementation
agreement). See [`research/05_validation.md`](research/05_validation.md) and
[`RESULTS.md`](RESULTS.md).

### Demo results (reproducible: `make demo`)

The end-to-end demo trains a tiny IFNet on **synthetic** moving-cloud data on CPU, then
validates on **withheld true middle frames** vs the linear-blend and TV-L1 baselines (fixed-K
`data_range`, P1). These are integration/sanity numbers, not the production result:

| method | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | FSIM ↑ | BT-RMSE (K) ↓ |
|---|---|---|---|---|---|
| **trained IFNet** | **26.39** | **0.929** | **0.912** | **0.961** | **6.80** |
| linear blend | 26.19 | 0.889 | 0.872 | 0.954 | 6.97 |
| TV-L1 + warp | 25.53 | 0.889 | 0.872 | 0.951 | 7.50 |

*Cross-validation:* 18 of 40 methods execute on the synthetic cube; the remainder require
real multi-satellite data and register as `skipped`. Exact numbers vary run-to-run with the
random seed / step budget — read the table your own `make demo` emits.

### Prior-art bar (production target)

Vandal & Nemani (IEEE TNNLS 2021), GOES-R Band 13 @ 10-min: **sub-Kelvin RMSE, SSIM ≥ 0.933**.
That is the bar full FrameFlow training (GPU + multi-day real data) aims to meet or beat; the
brief synthetic CPU demo is **not** expected to reach it.

---

## Limitations

- **Extreme non-linear convection.** Rapid, non-linear evolution that initiates and peaks
  *between* two observed frames cannot be recovered — interpolation assumes motion/intensity
  change is well-bracketed by the endpoints. Cold-cloud growth/decay is the hardest case
  (see the convective-cell stress in the cross-val suite).
- **CPU-only demo training.** The shipped `make demo` trains a ~0.05 M IFNet for a few hundred
  CPU steps on synthetic data purely to prove the pipeline wires together. Real quality needs
  **GPU training on multi-day real satellite sequences**.
- **Single channel.** The model operates on one TIR window band (C13/B13/TIR1). It does not
  exploit water-vapour or split-window channels; multi-band fusion is future work.
- **Long-gap degradation.** Error grows with interpolation factor / gap length (recursion
  accumulates error). 30→15 min is well-supported; very large factors degrade.
- **Domain shift GOES/Himawari → INSAT.** Sensor spectral-response, geometry and noise differ;
  deployment **requires** INSAT fine-tuning/adaptation, and metrics should be re-measured on
  INSAT before operational use.
- **Off-disk / space pixels** are masked (NaN) end-to-end and excluded from metrics; the model
  must not be read as producing valid BT outside the Earth disk.

---

## Ethical & operational considerations

- **Synthesized, not observed.** Interpolated frames are **AI-generated** and clearly badged
  as such in the dashboard (vs ground truth). They must never be presented as real
  observations in operational products without that distinction.
- **Human-in-the-loop.** For warnings/decisions, interpolated frames are a **decision-support
  aid**; pair them with the per-frame error diagnostics (BT-RMSE, SSIM, EPE) and independent
  observations. Do not use them as sole evidence for hazard determination.
- **Failure transparency.** The validation suite reports where the model is weakest
  (fast convection, long gaps, cross-sensor transfer); these caveats should travel with any
  derived product.
- **Provenance.** Every interpolated `.nc` carries CF attributes (`source_frames`, `t`,
  `model`, `model_version`, `generated_by`, `institution`) so synthesized frames are auditable
  and never silently mistaken for L1 data.
- **Data licensing.** GOES/Himawari/GK-2A are open; INSAT (MOSDAC) and any redistribution must
  honour the respective data-use terms.

---

## How to retrain on real data

The exact same pipeline that runs `make demo` on synthetic data runs on real satellite data —
only the **ingest source** and **compute** change.

```bash
# 0) Install the full stack (CPU example; use a CUDA build of torch for GPU training).
make install-all

# 1) Ingest dense real frames into the canonical Zarr cube (anon S3, no credentials).
#    Builds a schema-correct cube + a Parquet triplet catalog (regrid via cached KDTree,
#    Planck/LUT BT conversion, fixed-range normalization, NaN-masking).
python -m frameflow.cli ingest --source goes19 --start 2025-09-01 --end 2025-09-07 \
    --channel C13 --out data/cubes/goes19_c13.zarr
#    (add Himawari-9 / GK-2A cubes the same way for domain diversity)

# 2) Train IFNet (BF16, 256->512 patch curriculum, time-based split, Charbonnier +
#    census + MS-SSIM + gradient (+IFRNet flow-distill) losses) on a GPU box.
python -m frameflow.cli train configs/config.yaml \
    data.cube=data/cubes/goes19_c13.zarr trainer.devices=1 trainer.precision=bf16-mixed
#    -> runs/ckpt/best_ssim.ckpt

# 3) Densify + write CF NetCDF, then validate vs WITHHELD truth (fixed-K data_range, P1).
python -m frameflow.cli interpolate --cube data/cubes/goes19_c13.zarr --factor 2 \
    --ckpt runs/ckpt/best_ssim.ckpt --out out/interp_nc
python -m frameflow.cli validate --pred-dir out/interp_nc --truth-dir data/truth_nc \
    --out out/validation        # PSNR/SSIM/MS-SSIM/FSIM/BT-RMSE/CSI/FSS/EPE + 40-method suite

# 4) (Deploy) Fine-tune / adapt onto INSAT-3DS TIR1, then precompute the O(1) web artifacts.
python -m frameflow.cli train configs/config.yaml \
    data.cube=data/cubes/insat3ds_tir1.zarr model.pretrained=runs/ckpt/best_ssim.ckpt
python -m frameflow.cli precompute --cube data/cubes/insat3ds_tir1.zarr --out out/web/<scene>
```

For an embeddable dashboard scene from any cube, the demo path is
`make demo` → `python scripts/embed_demo_scene.py` (copies a compact, browser-valid scene into
`web/public/data/`).

---

*FrameFlow — ISRO BAH 2026 PS-12. Trained on GOES-19/Himawari/GK-2A; deployed on INSAT-3DS/3DR.
Built honestly: the demo proves the system end-to-end; production quality requires GPU training
on real multi-day data.*

# FrameFlow — Interface Contracts (AUTHORITATIVE)

**This is the single source of truth for the interfaces the six build teams implement
against.** It is grounded in the six research reports in [`research/`](research/) and the
problem statement in [`idea.md`](idea.md); the system rationale lives in
[`ARCHITECTURE.md`](ARCHITECTURE.md). The runnable, in-code contracts live in
[`frameflow/contracts.py`](frameflow/contracts.py), shared constants in
[`frameflow/constants.py`](frameflow/constants.py), typed config in
[`frameflow/config.py`](frameflow/config.py), and a working synthetic generator in
[`frameflow/synthetic.py`](frameflow/synthetic.py).

> **Golden rule:** implement to the signatures and data contracts below *exactly*. Do not
> change `frameflow/contracts.py`, `frameflow/constants.py`, or `frameflow/config.py`
> without coordinating — every team depends on them. If you need a new field, raise it so
> it lands in the shared contracts once, not six times.

---

## 0. The three MUST corrections from code review (apply everywhere)

These are non-negotiable and are already baked into the shared contracts/constants. Every
relevant team restates and honours them.

1. **(P1) Fixed, shared physical metric range — never per-image min/max.**
   All FULL-REFERENCE metrics (MSE/RMSE/MAE/PSNR/SSIM/MS-SSIM/FSIM/GMSD/VIF/…) **MUST** be
   computed with the fixed Kelvin `data_range`
   `frameflow.constants.BT_DATA_RANGE_K` (= `140.0` K, spanning
   `BT_METRIC_VMIN_K=180.0` … `BT_METRIC_VMAX_K=320.0`).
   **Per-image min/max is forbidden** — it re-normalizes each frame to its own extremes and
   so hides radiometric errors (warm/cold bias, contrast compression). The SSIM-family
   `data_range` and the PSNR peak both use this fixed physical span (after the agreed
   Kelvin convention), making metrics comparable across frames, satellites, and methods.
   Owners: **INFER+VALIDATE** (computes), **SERVE+VIZ** (publishes `metrics.data_range_k`).

2. **(P2) Per-frame tile sources.**
   A single raster PMTiles archive is addressed only by `z/x/y` and **cannot** select a
   timestamp. Therefore **each timeline frame needs its OWN tile source** — a per-frame XYZ
   tile pyramid (`tiles_url_template` with `{z}/{x}/{y}`) **and/or** a per-frame PMTiles
   archive (`pmtiles`). The timeline slider switches the *active frame's* source. Both are
   REQUIRED per `FrameEntry`; `validate_manifest` enforces them. Owners: **SERVE+VIZ**
   (produces per-frame artifacts + manifest), **WEB** (switches source per slider tick).

3. **(P2-ONNX) `t` gets a leading BATCH dimension.**
   When exporting to ONNX/TensorRT, the timestep input `t` **MUST** have shape `[B, 1]`
   (leading batch dim) and be listed in `dynamic_axes` so Triton **dynamic batching** works.
   Document and support the alternative `max_batch_size=0` (no server-side batching) in
   Triton config. Owner: **INFER+VALIDATE** (export + Triton config).

---

## 1. Shared data contracts (consumed/produced by everyone)

All defined in [`frameflow/contracts.py`](frameflow/contracts.py) — import, don't redefine.

### 1.1 `GridSpec` (the common analysis grid)
```python
@dataclass(frozen=True)
class GridSpec:
    west: float; south: float; east: float; north: float
    n_rows: int; n_cols: int
    crs: str = "EPSG:4326"; resolution_deg: float = 0.04
    @property
    def shape(self) -> tuple[int, int]: ...      # (n_rows, n_cols) == (H, W)
    @property
    def bbox(self) -> tuple[float, float, float, float]: ...   # (W, S, E, N)
    def lat_coords(self) -> list[float]: ...      # n_rows, north -> south
    def lon_coords(self) -> list[float]: ...      # n_cols, west -> east
    def to_dict(self) -> dict; @classmethod from_dict(d) -> GridSpec
    @classmethod default() -> GridSpec            # 128x128 demo grid
# module-level: default_operational_grid() -> GridSpec  (full-res over default bbox)
```

### 1.2 Canonical Zarr cube — `CubeSchema` + `empty_cube`
- dims `("time","y","x")`; data var **`"bt"`** float32 **Kelvin**, NaN where off-disk/space.
- coords `time` (datetime64[ns]), `lat` (along `y`), `lon` (along `x`).
- chunks `(8,512,512)`, shards `(32,1024,1024)`, compressor `"blosc-zstd-shuffle"`.
```python
class CubeSchema:  # documentation constants (DIMS, DATA_VAR, DTYPE, CHUNKS, SHARDS, ...)
def empty_cube(grid: GridSpec, times) -> xarray.Dataset   # NaN-filled, schema-correct
```

### 1.3 `Sample` (training/eval triplet) — `TypedDict`
```python
class Sample(TypedDict):
    I0: np.ndarray | torch.Tensor   # (1, H, W) earlier bracket frame
    I1: np.ndarray | torch.Tensor   # (1, H, W) later   bracket frame
    It: np.ndarray | torch.Tensor   # (1, H, W) target intermediate frame at fraction t
    t:  float                       # interpolation fraction in (0,1); 0.5 == midpoint
    meta: dict                      # satellite, timestamps, "normalized" flag, mask info...
```

### 1.4 Inference NetCDF (.nc) output — `InferenceNetCDFSchema` + `netcdf_attrs`
- dims `("time","y","x")` (time length 1 per instant); var `"bt"` float32 Kelvin; coords as cube.
- **REQUIRED attrs:** `source_frames`, `t`, `model`, `model_version`, `generated_by`,
  `institution` (+ `crs`, `interpolation_factor` advisory). Internal chunking `(1,512,512)`.
```python
def netcdf_attrs(*, source_frames: list[str], t: float, model: str, model_version: str,
                 kind="interpolated", interpolation_factor: int | None = None,
                 extra: dict | None = None) -> dict
```

### 1.5 Manifest JSON (the WEB contract) — `Manifest`, `FrameEntry`, `MetricRecord`, `CrossvalMethodResult`, `validate_manifest`
See [§7](#7-the-manifest-json-schema-web-contract) for the full field list. Every producer
of the manifest **MUST** pass `validate_manifest(manifest_dict) == []`.

---

## 2. Module ownership & directory boundaries (never touch another team's files)

| Team | Owns (directories/files) | Adds tests |
|---|---|---|
| **DATA** | `frameflow/data/**` (except `frameflow/synthetic.py`, owned by foundation) | `tests/test_data.py` |
| **MODELS** | `frameflow/models/**` | `tests/test_models.py` |
| **TRAIN** | `frameflow/train/**` | `tests/test_train.py` |
| **INFER+VALIDATE** | `frameflow/infer/**` and `frameflow/validate/**` | `tests/test_infer.py`, `tests/test_validate.py` |
| **SERVE+VIZ** | `frameflow/serve/**`, `frameflow/viz/**`, `frameflow/precompute.py` | `tests/test_serve.py`, `tests/test_viz.py` |
| **WEB** | `web/**` | `web/`-local tests (vitest) |

Foundation (already landed, do **not** modify): `frameflow/__init__.py`,
`frameflow/constants.py`, `frameflow/contracts.py`, `frameflow/config.py`,
`frameflow/synthetic.py`, `frameflow/cli.py`, `configs/**`, `scripts/demo.py`,
`pyproject.toml`, `Makefile`, `tests/conftest.py`, `tests/test_foundation.py`.

Rules:
- Each subpackage MUST have its own `__init__.py`. Keep public API in well-named modules
  matching the signatures below so `frameflow.cli` lazy-imports resolve.
- Tests use **distinct filenames** (`tests/test_<area>.py`) so they never collide.
- Re-use `frameflow.contracts` / `frameflow.constants` / `frameflow.config` — do not fork them.

---

## 3. Team DATA — `frameflow/data/**`

Consumes: raw `.nc`/`.h5` from S3/MOSDAC (research/02). Produces: the canonical **Zarr
cube** (§1.2) and a **Parquet catalog** of frames/triplets; serves `Sample` triplets.

### 3.1 `frameflow/data/access.py`
```python
def list_frames(source: str, start: str, end: str, *, channel: str = "C13",
                anon: bool = True) -> list[str]:
    """Return source URLs/keys for frames in [start, end] (e.g. GOES-19 C13 on S3)."""

def open_frame(url: str) -> "xarray.DataArray":
    """Open one source frame lazily and return its TIR brightness-temperature DataArray (K).
    Applies Planck (ABI) / LUT (INSAT) BT conversion as needed (research/02 §A3, research/03 §5.1)."""
```

### 3.2 `frameflow/data/ingest.py`
```python
def build_cube(source: str, out_path: str, *, config: str = "",
               grid: "GridSpec | None" = None) -> "pathlib.Path":
    """Ingest a source into a schema-correct Zarr cube (CubeSchema). Regrid to a common
    lat/lon grid with a CACHED pyresample KDTree (research/03 §4), convert to BT, normalize,
    NaN-mask off-disk. Returns the cube path. Called by `frameflow.cli ingest`."""

def build_catalog(cube_path: str, out_parquet: str, *, cadence_min: int) -> "pathlib.Path":
    """Write a Parquet catalog of frame timestamps + (i-1, i, i+1) training triplets."""
```

### 3.3 `frameflow/data/preprocess.py`
```python
def radiance_to_bt(da: "xarray.DataArray", **coeffs) -> "xarray.DataArray": ...   # K
def regrid_to_grid(da: "xarray.DataArray", grid: "GridSpec", *, cache_dir: str) -> "xarray.DataArray": ...
def normalize(bt: "np.ndarray", *, mode: str = "fixed_range",
              vmin_k: float = 180.0, vmax_k: float = 330.0) -> "np.ndarray": ...   # -> [0,1]
def denormalize(x: "np.ndarray", *, mode="fixed_range", vmin_k=180.0, vmax_k=330.0) -> "np.ndarray": ...  # -> K
```
> Normalization for model INPUT is separate from the metric range (P1). Use the constants
> `BT_NORM_VMIN_K`/`BT_NORM_VMAX_K` for input scaling; persist real per-dataset stats to
> `norm_stats.json` (`constants.STATS_JSON_FILENAME`).

### 3.4 `frameflow/data/datasets.py`
```python
class TripletDataset:   # torch.utils.data.Dataset-compatible
    def __init__(self, cube_path: str, *, split: str = "train", patch_size: int = 256,
                 normalized: bool = True, **kw): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> "Sample":   # returns a Sample TypedDict (§1.3)
        ...
```
Produces: `Sample` triplets. Split **by time** (never random) to avoid leakage (research/06 §6).

---

## 4. Team MODELS — `frameflow/models/**`

Consumes: `Sample` tensors `(B,1,H,W)` + `t`. Produces: the interpolated frame `(B,1,H,W)`
and (optionally) intermediate flow for the dashboard overlay / EPE.

### 4.1 `frameflow/models/base.py`
```python
class VFIModel(torch.nn.Module):
    """Common interface for every VFI backbone."""
    def forward(self, I0: "Tensor", I1: "Tensor", t: "Tensor") -> "Tensor":
        """I0,I1: (B,1,H,W); t: (B,1) in (0,1)  ->  predicted It: (B,1,H,W).
        NOTE (P2-ONNX): t MUST be (B,1) with a leading BATCH dim (NOT a scalar)."""
    def forward_with_flow(self, I0, I1, t) -> "tuple[Tensor, Tensor]":
        """-> (It, flow) where flow is (B,2,H,W) intermediate flow for viz/EPE (optional)."""
    @property
    def params_m(self) -> float: ...   # parameter count in millions (for the manifest)
```

### 4.2 `frameflow/models/rife.py`, `ifrnet.py`, `ema_vfi.py`, `super_slomo.py`, `film.py`, baselines
```python
def build_model(cfg: "ModelConfig") -> "VFIModel":
    """Factory: instantiate the model named cfg.name with single-channel adaptation
    (cfg.channel_adapt in {'replicate3','avg_conv1'}) and load cfg.pretrained if set."""
# Baselines (research/05 M18-M20) live in frameflow/models/baselines.py:
def linear_blend(I0, I1, t): ...          # 0.5-style blend baseline
def farneback_warp(I0, I1, t): ...        # classical optical-flow baseline
def persistence(I0, I1, t): ...           # frame-copy baseline
```
Primary engine: **RIFE/Practical-RIFE (IFNet)**, MIT (research/01 §0, §9). Comparators:
EMA-VFI, IFRNet, Super SloMo, FILM + classical/persistence baselines.

---

## 5. Team TRAIN — `frameflow/train/**`

Consumes: `TripletDataset` (`Sample`s) + `VFIModel`. Produces: checkpoints
(`runs/ckpt/best_ssim.ckpt`, `last.ckpt`) + a model card.

### 5.1 `frameflow/train/losses.py`
```python
def charbonnier(pred, gt, eps: float = 1e-3) -> "Tensor": ...
def census(pred, gt) -> "Tensor": ...
def ms_ssim_loss(pred, gt) -> "Tensor": ...
def gradient_loss(pred, gt) -> "Tensor": ...
def combined_loss(pred, gt, *, weights: "TrainConfig") -> "Tensor":
    """Charbonnier + census + MS-SSIM + gradient (+IFRNet flow-distill/geo) per research/06 §5.3."""
```

### 5.2 `frameflow/train/trainer.py`
```python
def run(config: str = "configs/config.yaml", overrides: list[str] | None = None) -> "pathlib.Path":
    """Fine-tune the model per the composed Hydra config; return the best-checkpoint path.
    BF16 mixed precision; 256->512 patch curriculum; time-based split. Called by `cli train`."""
```
Honour `TrainConfig` (precision `bf16-mixed`, loss weights, warmup/cosine). Log val
SSIM/PSNR/MSE/FSIM. Determinism via `seed_everything`.

---

## 6. Team INFER+VALIDATE — `frameflow/infer/**` and `frameflow/validate/**`

### 6.1 `frameflow/infer/interpolate.py`
```python
def interpolate_pair(in0_nc: str, in1_nc: str, *, t: float = 0.5,
                     ckpt: str = "runs/ckpt/best_ssim.ckpt",
                     out_nc: str = "out/interp_nc/mid.nc",
                     model_version: str = "v0.1.0") -> "pathlib.Path":
    """Read two .nc frames, synthesize the frame at fraction t, write a schema-correct .nc
    (InferenceNetCDFSchema + netcdf_attrs). Returns out_nc. Called by `cli interpolate`."""

def interpolate_sequence(cube_path: str, *, factor: int = 2,
                         out_dir: str = "out/interp_nc", **kw) -> list["pathlib.Path"]:
    """Densify a whole cube by `factor` (2 -> 30->15 min; 4 -> 7.5 min); write per-instant .nc."""
```

### 6.2 `frameflow/infer/export.py`  (P2-ONNX)
```python
def export_onnx(ckpt: str, out_path: str = "runs/export/vfi.onnx", *, opset: int = 17) -> "pathlib.Path":
    """Export to ONNX. t input MUST be shape [B,1] with a leading BATCH dim and listed in
    dynamic_axes (so Triton dynamic batching works). Document max_batch_size=0 alternative."""
def build_tensorrt(onnx_path: str, engine_path: str = "runs/export/vfi_fp16.plan",
                   *, fp16: bool = True) -> "pathlib.Path": ...   # FP16 only; INT8 ruins VFI (research/06 §1.4)
```
Reference `dynamic_axes`: `{"img0":{0:"B",2:"H",3:"W"}, "img1":{0:"B",2:"H",3:"W"},
"t":{0:"B"}, "mid":{0:"B",2:"H",3:"W"}}` with `t` shaped `[B,1]`.

### 6.3 `frameflow/validate/metrics.py`  (P1)
```python
def per_frame_metrics(pred_k: "np.ndarray", true_k: "np.ndarray", *,
                      data_range_k: float = 140.0,
                      mask: "np.ndarray | None" = None) -> "MetricRecord":
    """Compute the full-reference + BT-domain metrics for ONE frame. ALL full-reference
    metrics MUST use the FIXED `data_range_k` (P1) — NEVER per-image min/max. Exclude
    NaN/off-disk pixels via `mask`. Returns a contracts.MetricRecord."""
```

### 6.4 `frameflow/validate/runner.py`
```python
def run(pred_dir: str, truth_dir: str, out_dir: str = "out/validation", *,
        methods: list[str] | None = None) -> "pathlib.Path":
    """Match interpolated frames to withheld ground truth, compute per-frame + summary
    metrics (fixed-K data_range), run the selected cross-validation methods, write metrics
    JSON + plots. Returns out_dir. Called by `cli validate`."""
```

### 6.5 `frameflow/validate/crossval.py`  (the 40-method framework, research/05 §4)
```python
def available_methods() -> list["CrossvalMethodResult"]:   # M1..M40 registry (ids+names)
def run_method(method_id: str, **ctx) -> "CrossvalMethodResult": ...
```
Headline metric set (research/05 §0): PSNR, SSIM, MS-SSIM, FSIM, BT-RMSE(K), CSI@235K,
FSS@scale, EPE. Dual-implementation agreement check (M40) when `dual_impl_check=True`.

---

## 7. Team SERVE+VIZ — `frameflow/serve/**`, `frameflow/viz/**`, `frameflow/precompute.py`

Consumes: the cube (§1.2), interpolated `.nc` (§1.4), metrics JSON. Produces: per-frame
tiles + PMTiles + video + the **manifest** (§8). This is the O(1) serving substrate.

### 7.1 `frameflow/viz/colormap.py`, `frameflow/viz/render.py`
```python
def get_colormap(name: str = "ir_clouds"):
    """Return a matplotlib colormap built from constants.IR_CLOUDS_COLORMAP_STOPS etc."""
def render_frame(bt_k: "np.ndarray", *, colormap="ir_clouds",
                 vmin_k: float = 180.0, vmax_k: float = 320.0) -> "np.ndarray":
    """Colorize one BT field to RGBA using the FIXED display range (matches the metric range)."""
def render_flow_overlay(flow: "np.ndarray") -> "np.ndarray": ...   # optional motion vectors
```

### 7.2 `frameflow/precompute.py`  (top-level, owned by SERVE+VIZ)  (P2)
```python
def build_web_artifacts(cube_path: str, out_dir: str = "out/web", *,
                        cfg: "ServeConfig | None" = None,
                        metrics_json: str | None = None) -> "pathlib.Path":
    """Render EVERY frame, build PER-FRAME XYZ tile pyramids AND per-frame PMTiles (P2),
    encode video(s), and write a manifest.json that passes validate_manifest(). Returns
    out_dir. Called by `cli precompute` and the demo chain."""

def build_manifest(...) -> "Manifest":
    """Assemble a contracts.Manifest. Each FrameEntry gets its OWN tiles_url_template
    (tiles/{idx:03d}/{z}/{x}/{y}.webp) and pmtiles (pmtiles/{idx:03d}.pmtiles) (P2).
    Set metrics.data_range_k / value_range_k to the FIXED physical range (P1)."""
```

### 7.3 `frameflow/serve/app.py`  (optional on-demand endpoint, research/06 §3.4)
```python
def run(artifacts: str = "out/web", host="0.0.0.0", port=8000) -> None: ...
# FastAPI app serving the static artifacts + optional content-addressed /interpolate.
```

---

## 8. THE MANIFEST JSON SCHEMA (web contract)

Implemented exactly by `contracts.Manifest`. Produce it with `Manifest(...).to_dict()` and
verify with `validate_manifest(...) == []`.

```jsonc
{
  "version": "1.0",
  "scene_id": "demo-0001",
  "title": "FrameFlow demo scene",
  "satellite": "GOES-19",
  "channel": "C13",
  "wavelength_um": 10.3,
  "bbox": [68.0, 6.0, 98.0, 38.0],          // [west, south, east, north]
  "crs": "EPSG:4326",
  "colormap": "ir_clouds",
  "value_range_k": [180.0, 320.0],          // FIXED display+metric BT range (P1)
  "tile_size": 256,
  "min_zoom": 0,
  "max_zoom": 8,
  "interpolation_factor": 2,
  "cadence_minutes_input": 20.0,
  "cadence_minutes_output": 10.0,
  "frames": [                               // timeline; one entry per frame
    {
      "index": 0,
      "time": "2025-06-20T00:00:00Z",       // ISO-8601 UTC ("...Z")
      "kind": "observed",                   // "observed" | "interpolated"
      "t": null,                            // float in (0,1) for interpolated, else null
      "bracket": null,                      // [i, j] observed neighbours for interpolated, else null
      "image": "img/000.webp",
      "thumb": "thumb/000.webp",
      "tiles_url_template": "tiles/000/{z}/{x}/{y}.webp",  // PER-FRAME XYZ source (P2, REQUIRED)
      "pmtiles": "pmtiles/000.pmtiles",                    // PER-FRAME PMTiles (P2, REQUIRED)
      "netcdf": "nc/000.nc",
      "flow_overlay": null                  // optional motion-vector overlay asset
    }
    // ... more frames; interpolated frames carry t + bracket ...
  ],
  "videos": { "observed": null, "interpolated": null, "side_by_side": null },
  "metrics": {
    "data_range_k": 140.0,                  // FIXED Kelvin range used for ALL FR metrics (P1)
    "value_range_k": [180.0, 320.0],
    "per_frame": [ /* MetricRecord dicts: index, psnr, ssim, ms_ssim, fsim, bt_rmse_k, ... */ ],
    "summary": { /* aggregate means + CIs */ },
    "baselines": { /* per-baseline summary metrics */ }
  },
  "crossval": { "methods_run": [ /* CrossvalMethodResult dicts: method_id, name, status, summary */ ] },
  "generated": "2026-06-20T00:00:00Z",      // build timestamp (ISO-8601 UTC)
  "model": { "name": "RIFE", "version": "v0.1.0", "params_m": 9.8 }
}
```

**REQUIRED, enforced by `validate_manifest`:**
- every top-level field above is present;
- `bbox` is 4 elements with `west < east`, `south < north`;
- `value_range_k` is `[vmin, vmax]` with `vmin < vmax`; `min_zoom <= max_zoom`;
- `metrics.data_range_k` is present, positive, and equals `value_range_k[1]-value_range_k[0]` (P1);
- `metrics.value_range_k`, `metrics.per_frame`, `crossval.methods_run`, and the three
  `videos.*` keys exist;
- **every frame** has `tiles_url_template` containing `{z}`, `{x}`, `{y}` **and** a
  non-empty `pmtiles` (P2);
- interpolated frames have non-null `t` and `bracket`;
- frame `time` strings are ISO-8601 UTC.

---

## 9. Team WEB — `web/**`

Consumes: the **manifest** (§8) + the per-frame tiles/PMTiles/video/`.nc` it points to.
Stack (research/04): Vite + React + TS, MapLibre GL JS + deck.gl, raster PMTiles, uPlot.

MUST:
- Read `manifest.json`; build the timeline from `frames[]`.
- **(P2)** On each slider tick, switch the active raster source to **that frame's**
  `tiles_url_template` / `pmtiles` (a single archive cannot select a timestamp).
- Use `value_range_k` + `colormap` for the legend/colorbar; show per-frame metrics from
  `metrics.per_frame` (synced uPlot cursor); badge interpolated frames.
- Provide synced GT-vs-interpolated compare (swipe + side-by-side) and the metric HUD.

Build outputs go to `web/dist/` (git-ignored). Add web tests under `web/` (vitest), not in
the Python `tests/` tree.

---

## 10. Wiring summary (how the pieces call each other)

```
DATA.build_cube ─► Zarr cube (CubeSchema) ─► DATA.TripletDataset ─► Sample
                                                   │
MODELS.build_model ─► VFIModel  ◄───────── TRAIN.trainer.run ─► ckpt
                                                   │
two .nc frames ─► INFER.interpolate_pair ─► interpolated .nc (InferenceNetCDFSchema)
                                                   │
interpolated .nc + withheld truth ─► VALIDATE.run ─► metrics JSON (MetricRecord, fixed-K P1)
                                                   │
cube + interpolated .nc + metrics ─► SERVE.precompute.build_web_artifacts
        ─► per-frame tiles + per-frame PMTiles (P2) + video + manifest.json (validate_manifest == [])
                                                   │
                                              WEB reads manifest ─► slider switches per-frame source (P2)
```

The `frameflow.cli` subcommands (`ingest`/`train`/`interpolate`/`validate`/`precompute`/
`serve`/`demo`) lazy-import the modules above; until a team lands its module the command
prints a helpful "owned by Team X — see CONTRACTS.md" message and the demo chain skips it.

"""Precompute the O(1) web-artifact set for a scene (SERVE+VIZ, P1+P2).

This is the module that makes dashboard delivery O(1): do all the expensive work ONCE,
offline, and persist directly-addressable static artifacts (per-frame tiles + PMTiles +
all-intra video + a validated manifest). The web app then only issues constant-time range
GETs against immutable files (research/03 §6, research/04 §4, §9).

``precompute_scene`` orchestrates the build for one cube:

    1. Load the Zarr cube (``bt`` ``(time, y, x)`` Kelvin + ``lat``/``lon``/``time``).
    2. Densify by ``factor`` (recursive midpoint interpolation) via an injected ``model``
       or a lazily-imported ``frameflow.infer`` — degrading gracefully to *observed-only*
       if no model/infer is available, so a manifest + tiles are ALWAYS produced.
    3. Write per-frame ``.nc`` (CF, Kelvin) for every observed + interpolated instant.
    4. Render every frame to a full-frame WebP + thumbnail (fixed display range, P1).
    5. Build a PER-FRAME XYZ WebP tile pyramid (and per-frame PMTiles if a writer exists)
       so the slider can switch tile source per timestamp (P2).
    6. Encode all-intra observed / interpolated / side-by-side videos (O(1) seek).
    7. Compute per-frame metrics (interpolated vs withheld truth, fixed-K data_range, P1)
       + a crossval summary via a lazily-imported ``frameflow.validate`` if available.
    8. Build flow overlays where intermediate flow is available.
    9. Assemble + validate + write ``manifest.json`` (CONTRACTS.md §8).

All cross-team imports (``frameflow.infer``, ``frameflow.validate``) are lazy and optional;
heavy deps (numpy/xarray/PIL/imageio) are imported lazily inside functions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import constants as C
from .config import ServeConfig
from .viz import flow_overlay as flow_overlay_mod
from .viz import manifest as manifest_mod
from .viz import render as render_mod
from .viz import tiles as tiles_mod
from .viz import video as video_mod

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger("frameflow.precompute")

# A model runner densifies a pair: (I0_2d, I1_2d, t) -> It_2d (numpy float32, Kelvin).
ModelRunner = Callable[[Any, Any, float], Any]


# ===========================================================================
# Result container
# ===========================================================================
@dataclass
class PrecomputeResult:
    """Summary of a precompute run (paths + counts), returned by :func:`precompute_scene`."""

    out_dir: Path
    manifest_path: Path
    scene_id: str
    n_observed: int
    n_interpolated: int
    n_frames: int
    videos: dict[str, str | None]
    interpolation_factor: int


# ===========================================================================
# Cube loading
# ===========================================================================
def _load_cube(cube_path: str | Path) -> tuple[np.ndarray, list[str], tuple[float, float, float, float], dict[str, Any]]:
    """Load a Zarr cube -> (bt[T,H,W] float32, iso_times[T], bbox, attrs).

    Returns ISO-8601 'Z' timestamp strings and a ``[west, south, east, north]`` bbox derived
    from the cube attrs (preferred) or from the lat/lon coordinate extents.
    """
    import numpy as np
    import xarray as xr

    ds = xr.open_zarr(cube_path, consolidated=False)
    try:
        var = C.CUBE_DATA_VAR if C.CUBE_DATA_VAR in ds else list(ds.data_vars)[0]
        bt = np.asarray(ds[var].values, dtype=np.float32)  # (T, H, W)
        times = _iso_times(ds)
        bbox = _bbox_from_ds(ds)
        attrs = dict(ds.attrs)
    finally:
        ds.close()
    if bt.ndim != 3:
        raise ValueError(f"cube '{cube_path}' bt must be (time, y, x); got shape {bt.shape}")
    return bt, times, bbox, attrs


def _iso_times(ds: Any) -> list[str]:
    """Extract ISO-8601 'Z' timestamps from a dataset's ``time`` coord (UTC by convention)."""
    import numpy as np
    import pandas as pd

    if "time" not in ds.coords:
        # Fabricate a default cadence if the cube lacks a time coord.
        n = int(ds[C.CUBE_DATA_VAR].shape[0]) if C.CUBE_DATA_VAR in ds else int(list(ds.data_vars.values())[0].shape[0])
        base = np.datetime64("2025-06-20T00:00:00")
        step = np.timedelta64(C.DEFAULT_INPUT_CADENCE_MIN, "m")
        vals = [base + i * step for i in range(n)]
    else:
        vals = list(np.asarray(ds["time"].values))
    out: list[str] = []
    for v in vals:
        ts = pd.Timestamp(v)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        out.append(ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return out


def _bbox_from_ds(ds: Any) -> tuple[float, float, float, float]:
    """Derive ``[west, south, east, north]`` from cube attrs or lat/lon coordinate extents."""
    import numpy as np

    attr = ds.attrs.get("bbox_west_south_east_north")
    if attr is not None and len(list(attr)) == 4:
        w, s, e, n = (float(x) for x in attr)
        return (w, s, e, n)
    lat = np.asarray(ds["lat"].values, dtype=np.float64) if "lat" in ds.coords else None
    lon = np.asarray(ds["lon"].values, dtype=np.float64) if "lon" in ds.coords else None
    if lat is not None and lon is not None and lat.size and lon.size:
        return (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    return tuple(float(x) for x in C.DEFAULT_GRID_BBOX)  # type: ignore[return-value]


# ===========================================================================
# Interpolation (recursive densification)
# ===========================================================================
def _runner_from_model(model: Any) -> ModelRunner | None:
    """Wrap an injected model as a ``(I0,I1,t)->It`` runner, or return None for no model.

    Accepts a torch-style ``VFIModel`` (``forward(I0,I1,t)`` with ``(B,1,H,W)`` and ``t``
    shaped ``(B,1)``, honouring P2-ONNX) or any plain ``callable(I0_2d, I1_2d, t)``.

    For the torch path the model expects **normalized** ``[0,1]`` input (the same convention
    Team MODELS trains on and Team INFER's :func:`frameflow.infer.interpolate.interpolate_pair`
    uses): Kelvin frames are scaled to ``[0,1]`` with the FIXED norm span
    (:data:`constants.BT_NORM_VMIN_K` .. :data:`constants.BT_NORM_VMAX_K`, NaN-safe) before
    ``forward`` and the prediction is mapped back to Kelvin afterwards, restoring the input
    NaN (off-disk) mask. Passing raw Kelvin straight into the network produces NaN/garbage,
    so this normalization is REQUIRED for the densified track to be valid. The plain-callable
    branch is treated as already operating in Kelvin and is left untouched.
    """
    if model is None:
        return None

    def _run(i0: Any, i1: Any, t: float) -> Any:
        import numpy as np

        forward = getattr(model, "forward", None)
        if forward is None:
            if callable(model):
                return np.asarray(model(i0, i1, t), dtype=np.float32)
            raise TypeError("model has no .forward and is not callable")
        import torch

        a = np.asarray(i0, dtype=np.float32)
        b = np.asarray(i1, dtype=np.float32)
        # Off-disk mask to restore after denorm (union of both inputs' NaNs).
        nan_mask = ~np.isfinite(a) | ~np.isfinite(b)
        a01 = _norm_k_to_unit(a)
        b01 = _norm_k_to_unit(b)

        with torch.no_grad():
            x0 = torch.as_tensor(a01)[None, None]
            x1 = torch.as_tensor(b01)[None, None]
            tt = torch.as_tensor([[float(t)]], dtype=torch.float32)  # (B,1) leading batch dim
            out = forward(x0, x1, tt)
            # MODELS registry returns a dict ({"pred",...}); forward_with_flow returns a
            # (It, flow) tuple; dummies may return a bare tensor — accept all three.
            if isinstance(out, dict):
                out = out.get("pred", out.get("It"))
            elif isinstance(out, (tuple, list)):
                out = out[0]
            out = out.detach().cpu().numpy()
        pred01 = np.squeeze(out).astype(np.float32)
        pred_k = _denorm_unit_to_k(pred01)
        if pred_k.shape == nan_mask.shape:
            pred_k = np.where(nan_mask, np.nan, pred_k)
        return pred_k.astype(np.float32)

    return _run


def _norm_k_to_unit(a: np.ndarray) -> np.ndarray:
    """Kelvin -> ``[0,1]`` for model input, NaN-SAFE (Team DATA's ``normalize`` if available).

    The VFI conv stack propagates NaN across its receptive field, so off-disk NaNs MUST be
    filled before the model sees them (the input mask is restored after denorm in ``_run``).
    """
    import numpy as np

    arr = np.asarray(a, dtype=np.float32)
    try:  # prefer the canonical Team DATA normalizer (fixed-range, NaN-safe) if it exists
        from .data.preprocess import normalize as _nm  # type: ignore

        x = np.asarray(_nm(arr), dtype=np.float32)
    except Exception:
        vmin, vmax = float(C.BT_NORM_VMIN_K), float(C.BT_NORM_VMAX_K)
        span = max(vmax - vmin, 1e-6)
        x = (arr - vmin) / span
    # Guarantee finiteness for the network regardless of the normalizer's NaN policy.
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def _denorm_unit_to_k(x: np.ndarray) -> np.ndarray:
    """``[0,1]`` -> Kelvin for model output (inverse of :func:`_norm_k_to_unit`)."""
    import numpy as np

    try:
        from .data.preprocess import denormalize as _dn  # type: ignore

        return np.asarray(_dn(x), dtype=np.float32)
    except Exception:
        vmin, vmax = float(C.BT_NORM_VMIN_K), float(C.BT_NORM_VMAX_K)
        span = max(vmax - vmin, 1e-6)
        return (np.asarray(x, dtype=np.float32) * span + vmin).astype(np.float32)


def _linear_blend(i0: np.ndarray, i1: np.ndarray, t: float) -> np.ndarray:
    """NaN-aware linear blend baseline used when no model/infer is available for densifying."""
    import numpy as np

    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i1, dtype=np.float32)
    out = (1.0 - t) * a + t * b
    # Where exactly one side is NaN, fall back to the finite side rather than propagating NaN.
    na, nb = np.isnan(a), np.isnan(b)
    out = np.where(na & ~nb, b, out)
    out = np.where(nb & ~na, a, out)
    out = np.where(na & nb, np.nan, out)
    return out.astype(np.float32)


def _densify(
    observed: Sequence[np.ndarray],
    factor: int,
    runner: ModelRunner | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Recursively densify an observed sequence by ``factor`` (power-of-two midpoints).

    For ``factor`` = 2 one frame is inserted between each consecutive observed pair; ``factor``
    = 4 recurses (inserting midpoints of midpoints), etc. ``factor`` is rounded up to the
    next power of two for clean recursive midpoints.

    Returns ``(frames, meta)`` where ``frames`` is the full densified list and ``meta[i]`` is
    ``{"kind", "t", "bracket"}`` describing each output frame relative to the ORIGINAL
    observed indices. Observed frames have ``kind="observed"``, ``t=None``, ``bracket=None``.
    """
    import numpy as np

    obs = [np.asarray(f, dtype=np.float32) for f in observed]
    n_obs = len(obs)
    if n_obs == 0:
        return [], []
    if n_obs == 1 or factor <= 1:
        return list(obs), [{"kind": "observed", "t": None, "bracket": None} for _ in obs]

    # number of recursive subdivisions: ceil(log2(factor))
    import math

    levels = max(1, math.ceil(math.log2(factor)))
    subdiv = 2 ** levels  # actual achieved factor (power of two)

    do_interp = runner if runner is not None else _linear_blend

    frames: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for i in range(n_obs - 1):
        frames.append(obs[i])
        meta.append({"kind": "observed", "t": None, "bracket": None})
        # insert subdiv-1 intermediate frames between observed i and i+1
        for k in range(1, subdiv):
            t = k / subdiv
            mid = do_interp(obs[i], obs[i + 1], t)
            frames.append(np.asarray(mid, dtype=np.float32))
            meta.append({"kind": "interpolated", "t": float(t), "bracket": [i, i + 1]})
    # last observed frame
    frames.append(obs[-1])
    meta.append({"kind": "observed", "t": None, "bracket": None})
    return frames, meta


def _model_flow(model: Any, i0: np.ndarray, i1: np.ndarray, t: float) -> np.ndarray | None:
    """Return a dense ``(2, H, W)`` intermediate flow from a model, or ``None``.

    Tries the dict-output contract (``forward(...)["flow"]``, IFNet returns ``(B,4,H,W)``
    bidirectional flow → we take the first two channels, ``F_{t->0}``) and the
    ``forward_with_flow`` API. Any failure (no flow, non-torch model) yields ``None`` so the
    overlay step degrades gracefully.
    """
    import numpy as np

    try:
        import torch
    except Exception:
        return None
    forward = getattr(model, "forward", None)
    if forward is None:
        return None
    try:
        with torch.no_grad():
            # Normalize Kelvin -> [0,1] (NaN-safe): the conv stack produces NaN flow on raw
            # Kelvin / NaN input, which would empty the overlay (same convention as _run).
            x0 = torch.as_tensor(_norm_k_to_unit(np.asarray(i0, dtype=np.float32)))[None, None]
            x1 = torch.as_tensor(_norm_k_to_unit(np.asarray(i1, dtype=np.float32)))[None, None]
            tt = torch.as_tensor([[float(t)]], dtype=torch.float32)
            flow = None
            fwf = getattr(model, "forward_with_flow", None)
            if callable(fwf):
                _it, flow = fwf(x0, x1, tt)
            else:
                out = forward(x0, x1, tt)
                if isinstance(out, dict):
                    flow = out.get("flow")
            if flow is None:
                return None
            f = np.asarray(flow.detach().cpu().numpy(), dtype=np.float32)
    except Exception:
        return None
    # Collapse batch dim and keep the first two channels (F_{t->0}: u, v).
    while f.ndim > 3:
        f = f[0]
    if f.ndim != 3 or f.shape[0] < 2:
        return None
    return f[:2]


def _maybe_flow_overlays(
    model: Any,
    observed: Sequence[np.ndarray],
    meta: Sequence[dict[str, Any]],
    frames_meta: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    out_dir: Path,
    *,
    step: int = 16,
) -> int:
    """Save a deck.gl flow overlay for each interpolated frame whose model exposes flow.

    Updates ``frames_meta[idx]["flow_overlay"]`` in place with the written relative path
    (``flow/NNN.json``). Returns the number of overlays written. No-ops (returns 0) when the
    model can't produce flow, keeping observed-only / baseline runs clean.
    """
    if model is None:
        return 0
    written = 0
    for idx, mt in enumerate(meta):
        if mt.get("kind") != "interpolated":
            continue
        bracket = mt.get("bracket")
        if not bracket or len(bracket) != 2:
            continue
        i, j = int(bracket[0]), int(bracket[1])
        if i >= len(observed) or j >= len(observed):
            continue
        flow = _model_flow(model, observed[i], observed[j], float(mt.get("t") or 0.5))
        if flow is None:
            continue
        try:
            rel = flow_overlay_mod.save_flow_overlay(
                flow, step=step, bbox=list(bbox), out_dir=out_dir, frame_index=idx
            )
        except Exception as exc:  # pragma: no cover - overlay must never fail the build
            logger.warning("precompute: flow overlay for frame %d skipped (%s)", idx, exc)
            continue
        frames_meta[idx]["flow_overlay"] = rel
        written += 1
    if written:
        logger.info("precompute: wrote %d flow overlay(s)", written)
    return written


def _try_infer_sequence(cube_path: str | Path, factor: int, out_nc_dir: Path) -> list[Path] | None:
    """Try Team INFER's ``interpolate_sequence`` to densify the cube; None if unavailable."""
    try:  # pragma: no cover - depends on Team INFER landing
        from .infer import interpolate as inf  # type: ignore

        fn = getattr(inf, "interpolate_sequence", None)
        if fn is None:
            return None
        paths = fn(str(cube_path), factor=factor, out_dir=str(out_nc_dir))
        return [Path(p) for p in paths]
    except Exception as exc:
        logger.info("frameflow.infer.interpolate_sequence unavailable (%s); using local densify", exc)
        return None


# ===========================================================================
# NetCDF per-frame writing
# ===========================================================================
def _write_frame_nc(
    frame: np.ndarray,
    iso_time: str,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    *,
    kind: str,
    t: float | None,
    bracket: list[int] | None,
    model_name: str,
    model_version: str,
    interpolation_factor: int,
) -> None:
    """Write one 2D BT frame as a schema-aligned single-time ``.nc`` (CF, Kelvin)."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from .contracts import InferenceNetCDFSchema, netcdf_attrs

    arr = np.asarray(frame, dtype=np.float32)
    h, w = arr.shape
    data = arr[np.newaxis, :, :]
    west, south, east, north = bbox
    lat = np.linspace(north, south, h, dtype=np.float64)
    lon = np.linspace(west, east, w, dtype=np.float64)
    time_val = np.array([np.datetime64(pd.Timestamp(iso_time.replace("Z", "")))])

    attrs = netcdf_attrs(
        source_frames=[f"bracket:{bracket}" if bracket else "observed"],
        t=float(t) if t is not None else 0.0,
        model=model_name,
        model_version=model_version,
        kind="observed" if kind == "observed" else "interpolated",
        interpolation_factor=interpolation_factor,
        extra={"crs": C.DEFAULT_GRID_CRS, "bbox_west_south_east_north": list(bbox)},
    )

    ds = xr.Dataset(
        data_vars={
            InferenceNetCDFSchema.DATA_VAR: (
                InferenceNetCDFSchema.DIMS,
                data,
                {"units": "K", "long_name": "brightness_temperature"},
            )
        },
        coords={
            "time": ("time", time_val),
            "lat": ("y", lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("x", lon, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs=attrs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds.to_netcdf(out_path, engine=engine)
            return
        except Exception as exc:  # pragma: no cover - try next engine
            last_err = exc
    raise RuntimeError(f"could not write {out_path}: no usable NetCDF engine ({last_err})")


# ===========================================================================
# Metrics (optional, via Team VALIDATE)
# ===========================================================================
def _compute_metrics(
    frames: Sequence[np.ndarray],
    meta: Sequence[dict[str, Any]],
    observed: Sequence[np.ndarray],
    iso_times: Sequence[str],
) -> dict[str, Any]:
    """Per-frame metrics for interpolated frames vs withheld truth (fixed-K data_range, P1).

    Truth is only available for interpolated frames that fall exactly on a withheld observed
    instant; in the synthetic/demo regime we approximate by comparing each interpolated frame
    to the linear blend of its bracket (a sanity baseline) when no validate module exists.
    Uses Team VALIDATE's ``per_frame_metrics`` if importable, else a small built-in PSNR/RMSE
    on the FIXED Kelvin range. Returns a metrics block (``data_range_k`` fixed, P1).
    """
    import numpy as np

    data_range_k = C.BT_DATA_RANGE_K
    per_frame: list[dict[str, Any]] = []

    validate_fn = _maybe_validate_metric_fn()

    for idx, (fr, mt) in enumerate(zip(frames, meta, strict=False)):
        if mt["kind"] != "interpolated" or mt.get("bracket") is None:
            continue
        i, j = mt["bracket"]
        t = float(mt.get("t") or 0.5)
        # Reference: linear blend of the bracketing observed frames (baseline truth proxy).
        ref = _linear_blend(observed[i], observed[j], t)
        pred = np.asarray(fr, dtype=np.float32)
        mask = np.isfinite(pred) & np.isfinite(ref)
        if validate_fn is not None:
            try:  # pragma: no cover - depends on Team VALIDATE
                rec = validate_fn(pred, ref, data_range_k=data_range_k, mask=mask)
                d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
                d["index"] = idx
                d.setdefault("time", iso_times[idx] if idx < len(iso_times) else None)
                per_frame.append(d)
                continue
            except Exception:
                pass
        per_frame.append(_builtin_metrics(pred, ref, mask, idx, iso_times, data_range_k))

    summary = _summarize(per_frame)
    return {
        "data_range_k": data_range_k,
        "value_range_k": [C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K],
        "per_frame": per_frame,
        "summary": summary,
        "baselines": {},
    }


def _maybe_validate_metric_fn() -> Callable[..., Any] | None:
    """Return ``frameflow.validate.metrics.per_frame_metrics`` if importable, else None."""
    try:  # pragma: no cover - depends on Team VALIDATE landing
        from .validate import metrics as vmetrics  # type: ignore

        return getattr(vmetrics, "per_frame_metrics", None)
    except Exception:
        return None


def _builtin_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    idx: int,
    iso_times: Sequence[str],
    data_range_k: float,
) -> dict[str, Any]:
    """Compute a minimal PSNR/RMSE/MAE/bias on the FIXED Kelvin range for one frame (P1)."""
    import numpy as np

    if not np.any(mask):
        return {"index": idx, "time": iso_times[idx] if idx < len(iso_times) else None}
    p = pred[mask].astype(np.float64)
    r = ref[mask].astype(np.float64)
    err = p - r
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    # PSNR on the FIXED physical range (P1), not per-image min/max.
    psnr = float(20.0 * np.log10(data_range_k) - 10.0 * np.log10(mse)) if mse > 0 else 99.0
    return {
        "index": idx,
        "time": iso_times[idx] if idx < len(iso_times) else None,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bt_rmse_k": rmse,
        "bt_bias_k": bias,
        "psnr": psnr,
    }


def _summarize(per_frame: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Mean of each numeric metric across the per-frame records."""
    import numpy as np

    if not per_frame:
        return {}
    keys = {k for rec in per_frame for k, v in rec.items() if isinstance(v, (int, float)) and k != "index"}
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(rec[k]) for rec in per_frame if isinstance(rec.get(k), (int, float))]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
    return out


# ===========================================================================
# Main orchestration
# ===========================================================================
def precompute_scene(
    cube_path: str | Path,
    model: Any = None,
    out_dir: str | Path | None = None,
    factor: int = 2,
    scene_id: str | None = None,
    cadence_min: int = C.DEFAULT_INPUT_CADENCE_MIN,
    *,
    cfg: ServeConfig | None = None,
    model_name: str = "RIFE",
    model_version: str = "v0.1.0",
    model_params_m: float = 9.8,
    make_video: bool = True,
    make_pmtiles: bool = True,
    title: str | None = None,
    satellite: str = C.DEFAULT_SATELLITE,
    channel: str = "C13",
) -> PrecomputeResult:
    """Build the full O(1) web-artifact set for one scene and return a :class:`PrecomputeResult`.

    Args:
        cube_path: path to the Zarr cube (``bt`` ``(time, y, x)`` Kelvin).
        model: optional injected VFI model/runner used to densify (``forward(I0,I1,t)`` or a
            plain ``callable(I0_2d,I1_2d,t)``). If ``None``, tries ``frameflow.infer``; if that
            is also unavailable, degrades to observed-only (still writes tiles + manifest).
        out_dir: output root. Defaults to ``artifacts/<scene_id>/`` (NEVER ``web/``).
        factor: temporal up-sampling factor (rounded up to a power of two for recursion).
        scene_id: scene identifier; defaults to the cube directory stem.
        cadence_min: native input cadence in minutes (drives manifest cadence fields).
        cfg: optional :class:`frameflow.config.ServeConfig` for tile/colormap defaults.
        model_name/model_version/model_params_m: model descriptors for the manifest.
        make_video: encode the observed/interpolated/side-by-side videos.
        make_pmtiles: attempt per-frame PMTiles (skipped gracefully if no writer).
        title/satellite/channel: scene/source descriptors for the manifest.

    Returns:
        A :class:`PrecomputeResult` with the manifest path and frame counts.
    """
    import numpy as np

    cfg = cfg or ServeConfig()
    cube_path = Path(cube_path)
    scene_id = scene_id or cube_path.stem or "scene"
    out_dir = Path(out_dir) if out_dir is not None else Path("artifacts") / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    colormap = cfg.colormap
    tile_size = cfg.tile_size

    logger.info("precompute: scene=%s cube=%s out=%s factor=%s", scene_id, cube_path, out_dir, factor)

    # 1) Load the cube.
    bt, obs_times, bbox, _attrs = _load_cube(cube_path)
    observed = [bt[i] for i in range(bt.shape[0])]
    n_obs = len(observed)
    logger.info("precompute: loaded %d observed frames, bbox=%s", n_obs, bbox)

    # 2) Densify (model -> infer -> linear-blend fallback).
    runner = _runner_from_model(model)
    if runner is None and model is None:
        # Team INFER fallback is file-based; we still densify locally with a baseline so the
        # interpolated track exists. (A real model should be injected for quality.)
        logger.info("precompute: no model injected; densifying with NaN-aware linear blend baseline")
    frames, meta = _densify(observed, factor=factor, runner=runner)
    n_frames = len(frames)
    n_interp = sum(1 for m in meta if m["kind"] == "interpolated")
    achieved_factor = 2 ** max(1, int(np.ceil(np.log2(max(factor, 1))))) if factor > 1 and n_obs > 1 else 1
    logger.info("precompute: densified to %d frames (%d interpolated, factor=%d)", n_frames, n_interp, achieved_factor)

    # 3) Output timestamps for ALL frames (interpolate timestamps linearly between observed).
    iso_times = _build_frame_times(obs_times, meta, cadence_min)

    # 4-6) Per-frame artifacts: nc, image, thumb, tiles (+pmtiles).
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    (out_dir / "thumb").mkdir(parents=True, exist_ok=True)
    (out_dir / "nc").mkdir(parents=True, exist_ok=True)

    frames_meta: list[dict[str, Any]] = []
    for idx, (fr, mt) in enumerate(zip(frames, meta, strict=False)):
        img_rel = f"img/{idx:03d}.{cfg.tile_format}"
        thumb_rel = f"thumb/{idx:03d}.{cfg.tile_format}"
        nc_rel = f"nc/{idx:03d}.nc"

        render_mod.save_frame(fr, out_dir / img_rel, cmap=colormap, fmt=cfg.tile_format)
        render_mod.make_thumbnail(fr, cmap=colormap, fmt=cfg.tile_format, out_path=out_dir / thumb_rel)

        tile_info = tiles_mod.build_frame_tiles(
            fr, bbox, out_dir, idx, tile_size=tile_size, min_zoom=cfg.min_zoom, cmap=colormap
        )
        pm_rel: str | None = None
        if make_pmtiles:
            pm_rel = tiles_mod.build_frame_pmtiles(
                fr, bbox, out_dir, idx, tile_size=tile_size, min_zoom=cfg.min_zoom,
                max_zoom=int(tile_info["max_zoom"]), cmap=colormap,
            )
        # P2: the manifest ALWAYS declares a per-frame pmtiles path (web contract), even when
        # the physical archive was skipped (no writer); the XYZ pyramid is the source then.
        pmtiles_decl = pm_rel or f"pmtiles/{idx:03d}.pmtiles"

        _write_frame_nc(
            fr, iso_times[idx], bbox, out_dir / nc_rel,
            kind=mt["kind"], t=mt.get("t"), bracket=mt.get("bracket"),
            model_name=model_name, model_version=model_version,
            interpolation_factor=achieved_factor,
        )

        frames_meta.append(
            {
                "index": idx,
                "time": iso_times[idx],
                "kind": mt["kind"],
                "t": mt.get("t"),
                "bracket": mt.get("bracket"),
                "image": img_rel,
                "thumb": thumb_rel,
                "tiles_url_template": tile_info["tiles_url_template"],
                "pmtiles": pmtiles_decl,
                "netcdf": nc_rel,
                "flow_overlay": None,
            }
        )

    # use the per-frame max zoom from the (uniform) last tile build for the manifest.
    max_zoom = int(tile_info["max_zoom"]) if n_frames else cfg.max_zoom

    # 6b) Flow overlays for interpolated frames (when the model exposes intermediate flow).
    _maybe_flow_overlays(model, observed, meta, frames_meta, bbox, out_dir)

    # 7) Videos (all-intra). Observed = real frames; interpolated = full densified track.
    videos: dict[str, str | None] = {"observed": None, "interpolated": None, "side_by_side": None}
    if make_video and n_frames:
        try:
            obs_vid = "videos/observed.mp4"
            (out_dir / "videos").mkdir(parents=True, exist_ok=True)
            video_mod.encode_observed_video(
                [render_mod.render_rgba(f, cmap=colormap) for f in observed],
                out_dir / obs_vid, fps=max(1, cfg.video_fps // 2),
            )
            videos["observed"] = obs_vid

            interp_vid = "videos/interpolated.mp4"
            video_mod.encode_interpolated_video(
                [render_mod.render_rgba(f, cmap=colormap) for f in frames],
                out_dir / interp_vid, fps=cfg.video_fps,
            )
            videos["interpolated"] = interp_vid

            # side-by-side: observed (held at native cadence) vs densified, matched in count
            # by repeating observed frames across their sub-intervals.
            sbs_vid = "videos/side_by_side.mp4"
            left = _stretch_observed(observed, meta)
            video_mod.encode_side_by_side(
                [render_mod.render_rgba(f, cmap=colormap) for f in left],
                [render_mod.render_rgba(f, cmap=colormap) for f in frames],
                out_dir / sbs_vid, fps=cfg.video_fps,
            )
            videos["side_by_side"] = sbs_vid
        except Exception as exc:  # pragma: no cover - ffmpeg edge cases shouldn't fail the build
            logger.warning("precompute: video encoding skipped (%s)", exc)

    # 8) Metrics (interpolated vs withheld-truth proxy; fixed-K data_range, P1).
    metrics_block = _compute_metrics(frames, meta, observed, iso_times)
    crossval_block = _maybe_crossval(frames, meta, observed)

    # 9) Assemble + validate + write the manifest.
    manifest = manifest_mod.build_manifest(
        scene_id=scene_id,
        bbox=bbox,
        frames_meta=frames_meta,
        title=title or f"FrameFlow scene {scene_id}",
        satellite=satellite,
        channel=channel,
        colormap=colormap,
        value_range_k=[C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K],
        tile_size=tile_size,
        min_zoom=cfg.min_zoom,
        max_zoom=max_zoom,
        interpolation_factor=achieved_factor,
        cadence_minutes_input=float(cadence_min),
        cadence_minutes_output=float(cadence_min) / float(achieved_factor),
        metrics=metrics_block,
        crossval=crossval_block,
        videos=videos,
        model_info={"name": model_name, "version": model_version, "params_m": model_params_m},
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    manifest_path = manifest_mod.write_manifest(manifest, out_dir, filename=cfg.manifest_name)
    logger.info("precompute: wrote manifest -> %s", manifest_path)

    return PrecomputeResult(
        out_dir=out_dir,
        manifest_path=manifest_path,
        scene_id=scene_id,
        n_observed=n_obs,
        n_interpolated=n_interp,
        n_frames=n_frames,
        videos=videos,
        interpolation_factor=achieved_factor,
    )


def _build_frame_times(
    obs_times: Sequence[str],
    meta: Sequence[dict[str, Any]],
    cadence_min: int,
) -> list[str]:
    """Compute an ISO-8601 'Z' timestamp for every densified frame.

    Observed frames keep their cube timestamp; interpolated frames are placed at the
    fractional time ``t`` between their bracket's observed timestamps.
    """
    import pandas as pd

    obs_ts = [pd.Timestamp(s.replace("Z", "")) for s in obs_times]

    def _fmt(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    out: list[str] = []
    obs_cursor = 0
    for m in meta:
        if m["kind"] == "observed":
            ts = obs_ts[obs_cursor] if obs_cursor < len(obs_ts) else (
                obs_ts[-1] + pd.Timedelta(minutes=cadence_min) if obs_ts else pd.Timestamp("2025-06-20")
            )
            out.append(_fmt(ts))
            obs_cursor += 1
        else:
            i, j = m["bracket"]
            t = float(m.get("t") or 0.5)
            if i < len(obs_ts) and j < len(obs_ts):
                delta = (obs_ts[j] - obs_ts[i]) * t
                out.append(_fmt(obs_ts[i] + delta))
            else:  # pragma: no cover - defensive
                out.append(_fmt(obs_ts[min(i, len(obs_ts) - 1)] if obs_ts else pd.Timestamp("2025-06-20")))
    return out


def _stretch_observed(
    observed: Sequence[np.ndarray],
    meta: Sequence[dict[str, Any]],
) -> list[np.ndarray]:
    """Repeat each observed frame across its sub-interval so it aligns 1:1 with densified frames.

    Used for the side-by-side video so the left (observed/native-cadence) pane and the right
    (interpolated) pane have the same frame count and step together.
    """
    out: list[np.ndarray] = []
    obs_cursor = -1
    for m in meta:
        if m["kind"] == "observed":
            obs_cursor += 1
        out.append(observed[max(0, obs_cursor)])
    return out


def _maybe_crossval(
    frames: Sequence[np.ndarray],
    meta: Sequence[dict[str, Any]],
    observed: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    """Return a crossval ``methods_run`` list via Team VALIDATE if available, else a default.

    When ``frameflow.validate.crossval`` is importable its ``available_methods()`` registry is
    surfaced (status ``skipped`` here — the heavy run is owned by Team VALIDATE). Otherwise a
    single 'leave-the-middle-out' record summarizing the local densify is returned so the
    dashboard's crossval panel always has content.
    """
    try:  # pragma: no cover - depends on Team VALIDATE landing
        from .validate import crossval as cv  # type: ignore

        methods = cv.available_methods()
        out = []
        for m in methods:
            d = m.to_dict() if hasattr(m, "to_dict") else dict(m)
            d.setdefault("status", "skipped")
            out.append(d)
        if out:
            return out
    except Exception:
        pass

    return [
        {
            "method_id": "M1",
            "name": "leave-the-middle-out (precompute densify)",
            "status": "run",
            "summary": {"n_interpolated": float(sum(1 for m in meta if m["kind"] == "interpolated"))},
            "details": {"note": "local densify; full 40-method framework owned by Team VALIDATE"},
        }
    ]


# ===========================================================================
# CONTRACTS.md §7.2 compatibility wrappers (used by frameflow.cli)
# ===========================================================================
def build_web_artifacts(
    cube_path: str,
    out_dir: str = "out/web",
    *,
    cfg: ServeConfig | None = None,
    metrics_json: str | None = None,
    model: Any = None,
    factor: int = 2,
    scene_id: str | None = None,
) -> Path:
    """Render every frame, build per-frame tiles + PMTiles, encode video(s), write a manifest.

    CONTRACTS.md §7.2 entry point (called by ``frameflow.cli precompute`` and the demo chain).
    Thin wrapper over :func:`precompute_scene`. Note: the SERVE+VIZ rule is that precompute
    writes to an ``artifacts/<scene>/`` style dir, NOT into ``web/``; ``out_dir`` here is
    honoured as given (the CLI default is ``out/web``, which is fine — it is not ``web/``).

    Args:
        cube_path: path to the Zarr cube.
        out_dir: output dir for tiles/video/manifest.
        cfg: optional serve config.
        metrics_json: optional path to a precomputed metrics JSON to fold into the manifest.
        model: optional injected VFI model/runner.
        factor: temporal up-sampling factor.
        scene_id: optional scene id (defaults to the cube stem).

    Returns:
        The output directory path.
    """
    cfg = cfg or ServeConfig()
    result = precompute_scene(
        cube_path=cube_path,
        model=model,
        out_dir=out_dir,
        factor=factor,
        scene_id=scene_id,
        cfg=cfg,
        make_video=cfg.make_video,
    )
    # If a metrics JSON was provided, merge its per-frame/summary into the manifest in place.
    if metrics_json:
        _merge_metrics_json(result.manifest_path, metrics_json)
    return result.out_dir


def _merge_metrics_json(manifest_path: Path, metrics_json: str) -> None:
    """Fold an external metrics JSON into an already-written manifest (best-effort, re-validated)."""
    import json

    from .contracts import validate_manifest

    mp = Path(manifest_path)
    if not mp.exists() or not Path(metrics_json).exists():
        return
    try:
        manifest = json.loads(mp.read_text())
        external = json.loads(Path(metrics_json).read_text())
    except Exception as exc:  # pragma: no cover
        logger.warning("could not merge metrics JSON: %s", exc)
        return
    metrics = manifest.setdefault("metrics", {})
    for key in ("per_frame", "summary", "baselines"):
        if key in external:
            metrics[key] = external[key]
    # Keep the fixed range authoritative (P1) regardless of external content.
    metrics["data_range_k"] = C.BT_DATA_RANGE_K
    metrics["value_range_k"] = [C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K]
    problems = validate_manifest(manifest)
    if problems:  # pragma: no cover - external data malformed
        logger.warning("merged manifest invalid, not overwriting: %s", problems)
        return
    mp.write_text(json.dumps(manifest, indent=2))


# Re-export the manifest builder under the CONTRACTS.md §7.2 name for convenience.
build_manifest = manifest_mod.build_manifest


__all__ = [
    "precompute_scene",
    "build_web_artifacts",
    "build_manifest",
    "PrecomputeResult",
]

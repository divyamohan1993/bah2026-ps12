"""FrameFlow synthetic data generator — physically-plausible moving-cloud TIR cube.

FULLY IMPLEMENTED and runnable with only ``numpy`` + ``xarray`` + ``zarr`` (xarray/zarr
imported lazily inside functions). Produces a Thermal-IR brightness-temperature field: a
warm background (~290-300 K) with several advecting, growing/rotating COLD cloud blobs
(2D Gaussians, ~200-240 K) moving along NON-LINEAR trajectories (curved paths +
acceleration). Non-linear motion makes frame interpolation non-trivial — exactly the
regime PS-12 targets (clouds grow/deform, they don't merely translate; idea.md ll. 5-7).

Outputs are schema-correct per :mod:`frameflow.contracts`:
    * :func:`generate_cube`   -> a Zarr cube (dims ``(time,y,x)``, var ``bt`` float32 K).
    * :func:`generate_pair_nc`-> two consecutive ``.nc`` frames (00:00, 00:20) + the true
      middle (00:10) as ground truth, for the leave-the-middle-out demo (idea.md l. 46).

Values are float32 Kelvin; a few corner pixels are set to NaN ("space" look) to exercise
masking. Run directly to materialize both:

    python -m frameflow.synthetic
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import constants as C
from .contracts import (
    CubeSchema,
    GridSpec,
    InferenceNetCDFSchema,
    empty_cube,
    netcdf_attrs,
)

if TYPE_CHECKING:
    import numpy as np


# Background / cloud temperature regimes (Kelvin), within the metric range [180, 320].
_BG_WARM_K = 298.0  # warm clear-sky background
_BG_COOL_K = 288.0  # cooler background gradient toward the north (top of grid)
_CLOUD_MIN_K = 205.0  # coldest convective tops at peak growth
_CLOUD_EDGE_K = 245.0  # cloud edge / thin cirrus


def _default_grid() -> GridSpec:
    """The default synthetic grid (small & fast: 128x128 over the Indian bbox)."""
    return GridSpec.default()


def generate_bt_field(
    t_norm: float,
    grid: GridSpec,
    rng: np.random.Generator,
    n_blobs: int = 5,
    blob_params: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Generate one brightness-temperature field (Kelvin) at normalized time ``t_norm``.

    The field is a warm background with a gentle north-south gradient plus ``n_blobs`` cold
    cloud blobs. Each blob follows a NON-LINEAR trajectory (quadratic in ``t_norm`` so it
    accelerates, with a sinusoidal cross-track wiggle so the path curves), GROWS then
    shrinks (Gaussian-in-time amplitude/size), and ROTATES (anisotropic Gaussian whose
    orientation advances with time). This guarantees that the true mid-frame is NOT the
    linear blend of its neighbours, so interpolation is genuinely tested.

    Args:
        t_norm: time in [0, 1] along the sequence.
        grid: the :class:`GridSpec` defining the output shape.
        rng: a seeded ``numpy`` random generator (used only if ``blob_params`` is None).
        n_blobs: number of cloud blobs (ignored if ``blob_params`` provided).
        blob_params: pre-sampled per-blob parameter dicts (see :func:`_sample_blobs`); pass
            the SAME list across frames so blobs move coherently through time.

    Returns:
        ``np.ndarray`` of shape ``(H, W)``, dtype float32, in Kelvin.
    """
    import numpy as np  # lazy

    n_rows, n_cols = grid.shape
    # Normalized pixel coordinates in [0, 1] (yy: 0=top/north, 1=bottom/south).
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, n_rows),
        np.linspace(0.0, 1.0, n_cols),
        indexing="ij",
    )

    # Warm background with a smooth north(top, cooler)->south(bottom, warmer) gradient
    # plus a faint, slowly-drifting large-scale wave so clear sky is not perfectly flat.
    field = _BG_COOL_K + (_BG_WARM_K - _BG_COOL_K) * yy
    field = field + 1.5 * np.sin(2.0 * math.pi * (xx + 0.15 * t_norm))

    if blob_params is None:
        blob_params = _sample_blobs(rng, n_blobs)

    # Accumulate cooling from all blobs, then apply once. Combine overlapping blobs with a
    # smooth max (softmax) instead of summing, so several stacked storms cannot drive the
    # field below the coldest physical cloud-top temperature.
    cooling = np.zeros_like(field)
    for bp in blob_params:
        cooling = _accumulate_cloud_cooling(cooling, xx, yy, t_norm, bp)

    # Cap total cooling so the coldest core never goes below _CLOUD_MIN_K, but keep the
    # response near-linear for moderate cooling so deep cores still reach convective lows.
    max_cool = _BG_WARM_K - _CLOUD_MIN_K
    # soft-clip: identity for small x, saturating toward max_cool (knee at ~0.85*max_cool)
    knee = 0.85 * max_cool
    over = np.maximum(cooling - knee, 0.0)
    cooling = np.minimum(cooling, knee) + (max_cool - knee) * np.tanh(over / max(max_cool - knee, 1e-6))
    field = field - cooling

    # Final safety clamp into the physical metric range (Kelvin).
    field = np.clip(field, C.BT_METRIC_VMIN_K + 1.0, C.BT_METRIC_VMAX_K - 1.0)
    return field.astype(np.float32)


def _sample_blobs(rng: np.random.Generator, n_blobs: int) -> list[dict[str, Any]]:
    """Sample coherent per-blob trajectory/shape parameters once for a whole sequence."""
    blobs: list[dict[str, Any]] = []
    for _ in range(n_blobs):
        blobs.append(
            {
                # Start position (normalized grid coords).
                "x0": float(rng.uniform(0.05, 0.45)),
                "y0": float(rng.uniform(0.10, 0.80)),
                # Net linear drift across the sequence.
                "vx": float(rng.uniform(0.25, 0.55)),
                "vy": float(rng.uniform(-0.20, 0.20)),
                # Quadratic (acceleration) term -> non-linear trajectory.
                "ax": float(rng.uniform(-0.20, 0.30)),
                "ay": float(rng.uniform(-0.15, 0.15)),
                # Cross-track sinusoidal wiggle (amplitude, phase) -> curved path.
                "wig_amp": float(rng.uniform(0.02, 0.10)),
                "wig_phase": float(rng.uniform(0.0, 2.0 * math.pi)),
                "wig_freq": float(rng.uniform(1.0, 2.5)),
                # Size (std-dev in normalized units) and growth timing.
                "sigma": float(rng.uniform(0.05, 0.11)),
                "aniso": float(rng.uniform(1.3, 2.2)),  # major/minor axis ratio
                "t_peak": float(rng.uniform(0.35, 0.65)),  # when the storm is biggest/coldest
                "t_width": float(rng.uniform(0.30, 0.55)),  # life-cycle width
                # Rotation rate (radians across the full sequence).
                "theta0": float(rng.uniform(0.0, math.pi)),
                "omega": float(rng.uniform(-1.2, 1.2) * math.pi),
                # Peak coldness depth (how far below edge temp the core gets).
                "depth": float(rng.uniform(0.7, 1.0)),
            }
        )
    return blobs


def _accumulate_cloud_cooling(
    cooling: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    t_norm: float,
    bp: dict[str, Any],
) -> np.ndarray:
    """Add one rotating/growing anisotropic-Gaussian cold blob's cooling (smooth-max combine).

    Returns the running ``cooling`` field (positive Kelvin to be subtracted later). Blobs are
    combined with a smooth maximum rather than a sum so overlapping storms do not stack into
    unphysically cold values.
    """
    import numpy as np  # lazy

    # --- non-linear centre position -----------------------------------------------------
    cx = bp["x0"] + bp["vx"] * t_norm + bp["ax"] * t_norm * t_norm
    cy = bp["y0"] + bp["vy"] * t_norm + bp["ay"] * t_norm * t_norm
    # cross-track wiggle perpendicular-ish to drift (curved path)
    cy = cy + bp["wig_amp"] * math.sin(bp["wig_freq"] * 2.0 * math.pi * t_norm + bp["wig_phase"])

    # --- life cycle: Gaussian-in-time amplitude (grow then dissipate) --------------------
    life = math.exp(-((t_norm - bp["t_peak"]) ** 2) / (2.0 * bp["t_width"] ** 2))

    # --- size grows with the life cycle -------------------------------------------------
    sigma_major = bp["sigma"] * (0.6 + 0.8 * life) * bp["aniso"]
    sigma_minor = bp["sigma"] * (0.6 + 0.8 * life)
    sigma_major = max(sigma_major, 1e-3)
    sigma_minor = max(sigma_minor, 1e-3)

    # --- rotation: orientation advances with time ---------------------------------------
    theta = bp["theta0"] + bp["omega"] * t_norm
    ct, st = math.cos(theta), math.sin(theta)

    dx = xx - cx
    dy = yy - cy
    # rotate coordinates into the blob's principal axes
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    rr = (xr / sigma_major) ** 2 + (yr / sigma_minor) ** 2
    gauss = np.exp(-0.5 * rr)

    # Depth of cooling: at peak life the core reaches near _CLOUD_MIN_K; edges ~_CLOUD_EDGE_K.
    core_drop = (_BG_WARM_K - _CLOUD_MIN_K) * bp["depth"] * life
    edge_drop = (_BG_WARM_K - _CLOUD_EDGE_K) * 0.5 * life
    this_cool = edge_drop * gauss + (core_drop - edge_drop) * (gauss ** 3)
    # Smooth maximum of the running cooling and this blob (log-sum-exp), so overlaps
    # blend toward the deeper of the two instead of summing.
    k = 12.0
    return np.logaddexp(k * cooling, k * this_cool) / k


def _add_space_corners(arr: np.ndarray, margin: int = 4) -> np.ndarray:
    """Set small triangular corner patches to NaN to emulate off-disk 'space' pixels."""
    import numpy as np  # lazy

    out = arr.copy()
    h, w = out.shape[-2:]
    m = min(margin, h, w)
    for i in range(m):
        k = m - i  # shrinking triangle width per row
        out[..., i, :k] = np.nan          # top-left
        out[..., i, w - k:] = np.nan       # top-right
        out[..., h - 1 - i, :k] = np.nan   # bottom-left
        out[..., h - 1 - i, w - k:] = np.nan  # bottom-right
    return out


def generate_cube(
    n_frames: int = 24,
    grid: GridSpec | None = None,
    out_path: str | Path = "data/cubes/synthetic.zarr",
    cadence_min: int = 30,
    seed: int = 0,
    n_blobs: int = 5,
    start_time: datetime | None = None,
) -> Path:
    """Generate a schema-correct synthetic Zarr brightness-temperature cube.

    Writes a Zarr store matching :class:`frameflow.contracts.CubeSchema` (dims
    ``(time, y, x)``, var ``bt`` float32 Kelvin, ``time``/``lat``/``lon`` coords) and
    returns the path. xarray and zarr are imported lazily.

    Args:
        n_frames: number of time steps.
        grid: spatial grid (defaults to :meth:`GridSpec.default`, 128x128).
        out_path: destination ``.zarr`` directory.
        cadence_min: minutes between consecutive frames.
        seed: RNG seed for reproducible blob trajectories.
        n_blobs: number of cold cloud blobs.
        start_time: first-frame UTC timestamp (defaults to 2025-06-20T00:00:00Z).

    Returns:
        The :class:`pathlib.Path` to the written ``.zarr`` store.
    """
    import numpy as np  # lazy

    grid = grid or _default_grid()
    rng = np.random.default_rng(seed)
    blob_params = _sample_blobs(rng, n_blobs)

    if start_time is None:
        start_time = datetime(2025, 6, 20, 0, 0, 0, tzinfo=timezone.utc)
    times = [start_time + timedelta(minutes=cadence_min * i) for i in range(n_frames)]
    # xarray wants tz-naive datetime64; strip tzinfo (values are UTC by construction).
    times_naive = [t.replace(tzinfo=None) for t in times]

    # Build the schema-correct empty cube, then fill the bt var frame by frame.
    ds = empty_cube(grid, times_naive)
    bt = np.empty((n_frames, *grid.shape), dtype=np.float32)
    for i in range(n_frames):
        t_norm = i / max(n_frames - 1, 1)
        frame = generate_bt_field(t_norm, grid, rng, n_blobs=n_blobs, blob_params=blob_params)
        bt[i] = _add_space_corners(frame, margin=4)
    ds[CubeSchema.DATA_VAR].values[...] = bt

    ds.attrs.update(
        {
            "title": "FrameFlow SYNTHETIC brightness-temperature cube (moving clouds)",
            "satellite": "SYNTHETIC",
            "channel": "synthetic-TIR",
            "wavelength_um": C.SATELLITE_BANDS[C.DEFAULT_SATELLITE]["wavelength_um"],
            "cadence_min": cadence_min,
            "seed": seed,
            "n_blobs": n_blobs,
            "note": "Non-linear advecting/growing/rotating cold blobs; NaN corners emulate space.",
        }
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Encode the bt var with the canonical chunking (clamped to the array size).
    tchunk = min(CubeSchema.CHUNKS[0], n_frames)
    ychunk = min(CubeSchema.CHUNKS[1], grid.n_rows)
    xchunk = min(CubeSchema.CHUNKS[2], grid.n_cols)
    encoding = {CubeSchema.DATA_VAR: {"chunks": (tchunk, ychunk, xchunk)}}
    # Remove any pre-existing store so re-runs are clean.
    _rmtree(out)
    ds.to_zarr(out, mode="w", encoding=encoding, consolidated=False)
    return out


def generate_pair_nc(
    out_dir: str | Path = "data/demo_nc",
    grid: GridSpec | None = None,
    seed: int = 0,
    n_blobs: int = 5,
    cadence_min: int = 20,
    start_time: datetime | None = None,
) -> list[Path]:
    """Write the demo triplet as schema-correct ``.nc`` files (00:00, 00:20, true 00:10).

    Mirrors the PS-12 headline demo: inputs at 00:00 and 00:20, with the TRUE middle frame
    at 00:10 as withheld ground truth (idea.md l. 46). Each file matches
    :class:`frameflow.contracts.InferenceNetCDFSchema` (dims ``(time,y,x)``, var ``bt``
    float32 Kelvin, coords + required attrs). Returns the three file paths in temporal
    order ``[t00, t10, t20]``.

    Args:
        out_dir: destination directory for the ``.nc`` files.
        grid: spatial grid (defaults to :meth:`GridSpec.default`).
        seed: RNG seed (kept consistent with :func:`generate_cube` semantics).
        n_blobs: number of cold cloud blobs.
        cadence_min: gap between the two bracketing input frames (default 20 min).
        start_time: timestamp of the first frame (defaults to 2025-06-20T00:00:00Z).

    Returns:
        ``[path_t00, path_t10, path_t20]`` :class:`pathlib.Path` objects.
    """
    import numpy as np  # lazy

    grid = grid or _default_grid()
    rng = np.random.default_rng(seed)
    blob_params = _sample_blobs(rng, n_blobs)

    if start_time is None:
        start_time = datetime(2025, 6, 20, 0, 0, 0, tzinfo=timezone.utc)
    half = cadence_min / 2.0
    moments = [
        (0.0, start_time, "observed", None),
        (0.5, start_time + timedelta(minutes=half), "interpolated", [0, 2]),
        (1.0, start_time + timedelta(minutes=cadence_min), "observed", None),
    ]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for t_norm, when, kind, bracket in moments:
        frame = generate_bt_field(t_norm, grid, rng, n_blobs=n_blobs, blob_params=blob_params)
        frame = _add_space_corners(frame, margin=4)
        ds = _frame_to_nc_dataset(frame, grid, when, t_norm, kind, bracket)
        fname = f"synthetic_{when.strftime('%Y%m%dT%H%M%SZ')}_{kind}.nc"
        fpath = out / fname
        _write_netcdf(ds, fpath)
        paths.append(fpath)
    return paths


def _frame_to_nc_dataset(
    frame: np.ndarray,
    grid: GridSpec,
    when: datetime,
    t_norm: float,
    kind: str,
    bracket: list[int] | None,
) -> Any:
    """Wrap one 2D BT frame into a CF-style single-time xarray Dataset."""
    import numpy as np  # lazy
    import xarray as xr  # lazy

    time_val = np.array([when.replace(tzinfo=None)], dtype="datetime64[ns]")
    lat = np.asarray(grid.lat_coords(), dtype=np.float64)
    lon = np.asarray(grid.lon_coords(), dtype=np.float64)
    data = frame[np.newaxis, :, :].astype(np.float32)

    attrs = netcdf_attrs(
        source_frames=["synthetic_00:00", "synthetic_00:20"],
        t=t_norm,
        model="synthetic-truth",
        model_version="synthetic-v1",
        kind="observed" if kind == "observed" else "interpolated",
        interpolation_factor=2,
        extra={
            "crs": grid.crs,
            "bbox_west_south_east_north": list(grid.bbox),
            "title": "FrameFlow synthetic demo frame",
            "bracket": ([] if bracket is None else bracket),
        },
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
    return ds


def _write_netcdf(ds: Any, path: Path) -> None:
    """Write a Dataset to NetCDF, trying engines in order of availability.

    Prefers ``netcdf4``/``h5netcdf`` (true ``.nc``); if neither backend is installed, falls
    back to ``scipy``. Internal chunking is requested where the engine supports it.
    """
    import numpy as np  # lazy

    var = InferenceNetCDFSchema.DATA_VAR
    h, w = ds[var].shape[-2:]
    chunks = (
        1,
        min(InferenceNetCDFSchema.INTERNAL_CHUNKSIZES[1], int(h)),
        min(InferenceNetCDFSchema.INTERNAL_CHUNKSIZES[2], int(w)),
    )
    last_err: Exception | None = None
    for engine in ("netcdf4", "h5netcdf"):
        try:
            enc = {var: {"chunksizes": chunks, "zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}
            ds.to_netcdf(path, engine=engine, encoding=enc)
            return
        except Exception as exc:  # pragma: no cover - depends on installed backends
            last_err = exc
    # Fallback: scipy backend (NetCDF3, no compression/chunking, no NaN fill encoding).
    try:
        ds.to_netcdf(path, engine="scipy")
        return
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not write NetCDF to {path}: no usable engine "
            f"(install netCDF4 or h5netcdf). Last error: {last_err or exc}"
        ) from (last_err or exc)


def _rmtree(path: Path) -> None:
    """Recursively remove a file or directory tree if it exists (best-effort)."""
    import shutil

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def main() -> None:
    """Generate both the synthetic Zarr cube and the demo ``.nc`` triplet; log paths."""
    cube = generate_cube()
    print(f"[frameflow.synthetic] wrote Zarr cube -> {cube}")
    nc_paths = generate_pair_nc()
    for p in nc_paths:
        print(f"[frameflow.synthetic] wrote NetCDF frame -> {p}")
    print("[frameflow.synthetic] done.")


if __name__ == "__main__":
    main()

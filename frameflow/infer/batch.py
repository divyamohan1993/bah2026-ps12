"""Batch inference over a whole Zarr cube — densify and write per-instant ``.nc``.

This is the offline precompute substrate (R6 §3): read the canonical Zarr analysis cube
(:class:`frameflow.contracts.CubeSchema`), densify it by an integer ``factor`` using the
recursive binary-subdivision interpolator, and write **every** resulting frame (observed
passthrough + synthesized) as a schema-correct ``.nc``
(:class:`frameflow.contracts.InferenceNetCDFSchema`). The returned manifest of densified
frames + timestamps is what the SERVE/VIZ precompute and the VALIDATE runner consume.

Memory note: cubes can be large, so we iterate **bracketing pairs** (consecutive observed
frames) and densify each interval independently rather than loading and densifying the
whole stack at once. Each interval shares its right observed frame with the next interval's
left, so observed frames are written exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import constants as C
from ..contracts import CubeSchema, GridSpec
from .interpolate import InterpolatedFrame, interpolate_recursive
from .netcdf_io import write_frame_nc

if TYPE_CHECKING:  # typing only
    import numpy as np


__all__ = ["interpolate_cube", "DensifiedFrame", "CubeInterpolationResult"]


@dataclass
class DensifiedFrame:
    """One densified output frame and where it was written.

    Attributes:
        index: 0-based position in the densified sequence.
        time: the frame's timestamp (``np.datetime64`` or ISO string).
        kind: ``"observed"`` or ``"interpolated"``.
        t: local interpolation fraction (``None`` for observed).
        bracket: originating observed-pair indices (``None`` for observed) — indices refer
            to the ORIGINAL cube time axis.
        nc_path: path to the written ``.nc`` file.
    """

    index: int
    time: Any
    kind: str
    t: float | None
    bracket: list[int] | None
    nc_path: Path


@dataclass
class CubeInterpolationResult:
    """Result of densifying a cube.

    Attributes:
        frames: the ordered list of :class:`DensifiedFrame` (observed + interpolated).
        timestamps: the densified timestamps (parallel to ``frames``).
        nc_paths: the written ``.nc`` paths (parallel to ``frames``).
        factor: the up-sampling factor used.
        out_dir: the output directory.
    """

    frames: list[DensifiedFrame]
    timestamps: list[Any]
    nc_paths: list[Path]
    factor: int
    out_dir: Path


def _grid_from_dataset(ds: Any) -> GridSpec:
    """Reconstruct a :class:`GridSpec` from a cube/frame dataset's lat/lon coords + attrs."""
    import numpy as np  # lazy

    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    n_rows = int(lat.shape[0])
    n_cols = int(lon.shape[0])

    bbox = ds.attrs.get("bbox_west_south_east_north")
    if bbox is not None and len(bbox) == 4:
        west, south, east, north = (float(v) for v in bbox)
    else:
        # Derive edges from pixel centres (lat is north->south, lon west->east).
        dlat = (lat[0] - lat[-1]) / max(n_rows - 1, 1) if n_rows > 1 else 0.04
        dlon = (lon[-1] - lon[0]) / max(n_cols - 1, 1) if n_cols > 1 else 0.04
        north = float(lat[0] + dlat / 2.0)
        south = float(lat[-1] - dlat / 2.0)
        west = float(lon[0] - dlon / 2.0)
        east = float(lon[-1] + dlon / 2.0)

    crs = str(ds.attrs.get("crs", C.DEFAULT_GRID_CRS))
    res = float(ds.attrs.get("resolution_deg", (east - west) / max(n_cols, 1)))
    return GridSpec(
        west=west, south=south, east=east, north=north,
        n_rows=n_rows, n_cols=n_cols, crs=crs, resolution_deg=res,
    )


def interpolate_cube(
    cube_path: str | Path,
    model: Any,
    out_dir: str | Path,
    factor: int = 2,
    *,
    model_name: str = "RIFE",
    model_version: str = "v0.1.0",
    dataset: str = "goes",
    device: str = "cpu",
    var: str = CubeSchema.DATA_VAR,
    **kw: Any,
) -> CubeInterpolationResult:
    """Densify a Zarr cube by ``factor`` and write each frame as a schema-correct ``.nc``.

    Iterates consecutive observed-frame pairs over the cube's time axis, densifies each
    interval with :func:`frameflow.infer.interpolate.interpolate_recursive` (binary
    subdivision: 2 -> +1 frame per gap = 15 min; 4 -> +3 = 7.5 min; 8 -> +7 = 3.75 min),
    and writes every output frame (observed passthrough + interpolated) to ``out_dir`` as
    ``frame_{index:04d}.nc`` with correct attributes (``source_frames``, ``t``, ``model``,
    ``model_version``, ...). Observed frames are written once (shared interval boundaries are
    de-duplicated).

    Args:
        cube_path: path to the ``.zarr`` cube (``bt`` float32 Kelvin, dims ``time,y,x``).
        model: a trained VFI model (duck-typed ``forward(I0, I1, t)``).
        out_dir: destination directory for the ``.nc`` files (created if needed).
        factor: up-sampling factor, one of ``{2, 4, 8}``.
        model_name / model_version: recorded in each output ``.nc``'s attributes.
        dataset / device: forwarded to the interpolator.
        var: the cube data-variable name (default ``"bt"``).
        **kw: forwarded to the interpolator.

    Returns:
        A :class:`CubeInterpolationResult` with the densified frame list, timestamps, and
        written ``.nc`` paths.
    """
    import numpy as np  # lazy
    import xarray as xr  # lazy

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(cube_path, consolidated=False)
    try:
        grid = _grid_from_dataset(ds)
        bt = np.asarray(ds[var].values, dtype=np.float32)  # (T, H, W)
        obs_times = list(np.asarray(ds["time"].values).reshape(-1))
    finally:
        ds.close()

    n_obs = bt.shape[0]
    if n_obs < 2:
        raise ValueError(f"cube must have >= 2 time steps to interpolate; got {n_obs}")

    observed_frames = [bt[i] for i in range(n_obs)]
    dense = interpolate_recursive(
        model,
        observed_frames,
        factor=factor,
        times=obs_times,
        dataset=dataset,
        device=device,
        **kw,
    )

    frames: list[DensifiedFrame] = []
    timestamps: list[Any] = []
    nc_paths: list[Path] = []

    for idx, fr in enumerate(dense):
        when = fr.time
        if when is None:
            # Fall back to the global-fraction-derived time across the whole sequence.
            when = _global_time(obs_times, fr.t_global)
        # Translate the per-interval bracket (already original-frame indices) for the attr.
        bracket = fr.bracket
        src = (
            [str(_iso(obs_times[bracket[0]])), str(_iso(obs_times[bracket[1]]))]
            if bracket is not None
            else [str(_iso(when))]
        )
        nc_path = out / f"frame_{idx:04d}.nc"
        write_frame_nc(
            fr.bt,
            grid,
            np.array([np.datetime64(when)], dtype="datetime64[ns]"),
            source_frames=src,
            t=(fr.t_local if fr.t_local is not None else 0.0),
            model=model_name,
            model_version=model_version,
            kind=fr.kind,
            interpolation_factor=factor,
            extra={"densified_index": idx, "kind": fr.kind},
            path=nc_path,
        )
        frames.append(
            DensifiedFrame(
                index=idx,
                time=when,
                kind=fr.kind,
                t=fr.t_local,
                bracket=fr.bracket,
                nc_path=nc_path,
            )
        )
        timestamps.append(when)
        nc_paths.append(nc_path)

    return CubeInterpolationResult(
        frames=frames,
        timestamps=timestamps,
        nc_paths=nc_paths,
        factor=factor,
        out_dir=out,
    )


def _global_time(obs_times: list[Any], t_global: float) -> Any:
    """Map a global fraction in [0,1] to a timestamp across the observed time axis."""
    import numpy as np  # lazy

    if len(obs_times) == 1:
        return np.datetime64(obs_times[0])
    pos = t_global * (len(obs_times) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(obs_times) - 1)
    frac = pos - lo
    a = np.datetime64(obs_times[lo])
    b = np.datetime64(obs_times[hi])
    delta = (b - a) / np.timedelta64(1, "ns")
    return a + np.timedelta64(int(round(delta * frac)), "ns")


def _iso(when: Any) -> str:
    """Render a timestamp as an ISO-8601 UTC string (best-effort)."""
    import numpy as np  # lazy

    try:
        ts = np.datetime64(when, "s")
        return str(ts) + "Z"
    except Exception:  # pragma: no cover
        return str(when)

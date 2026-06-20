"""Inference NetCDF (.nc) I/O — write/read schema-correct interpolated-frame files.

PS-12 explicitly requires the interpolated frames to be delivered as ``.nc`` (idea.md
l. 35). This module is the single, authoritative implementation of that I/O. It produces
and reads NetCDF4 files matching :class:`frameflow.contracts.InferenceNetCDFSchema`:

    * dims:   ``("time", "y", "x")`` (``time`` length 1 per interpolated instant);
    * var:    ``"bt"`` — float32, **Kelvin**, NaN where off-disk / space;
    * coords: ``time`` (datetime64[ns]), ``lat`` (along ``y``), ``lon`` (along ``x``);
    * attrs (REQUIRED): ``source_frames``, ``t``, ``model``, ``model_version``,
      ``generated_by``, ``institution`` (+ advisory ``crs`` / ``interpolation_factor``).

Internal NetCDF chunking is set to ``(1, 512, 512)`` (clamped to the array) so downstream
virtual (kerchunk / VirtualiZarr) access stays coarse-grained (R3 §1.4). NaN is written as
the ``_FillValue`` so masked space pixels round-trip exactly.

Heavy/optional backends (``netCDF4``, ``h5netcdf``, ``scipy``) are imported lazily by
xarray; this module imports nothing heavy at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts import GridSpec, InferenceNetCDFSchema, netcdf_attrs

if TYPE_CHECKING:  # typing only — never imported at runtime
    import numpy as np
    import xarray as xr


__all__ = ["write_netcdf", "read_netcdf"]


def _coerce_2d_or_3d(bt: Any) -> "np.ndarray":
    """Return ``bt`` as a float32 ``(time, y, x)`` array (adds a leading time axis if 2D).

    Accepts a 2D ``(y, x)`` frame (the common single-instant case) or an already-3D
    ``(time, y, x)`` array. A channel axis of size 1 (``(1, y, x)`` interpreted as
    channel-first single frame) is treated as the time axis, which is the desired layout.
    """
    import numpy as np  # lazy

    arr = np.asarray(bt, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(
            f"bt must be 2D (y,x) or 3D (time,y,x); got shape {arr.shape!r}"
        )
    return arr


def _as_datetime64(time: Any, n_t: int) -> "np.ndarray":
    """Coerce ``time`` to a ``datetime64[ns]`` array of length ``n_t``.

    Accepts a scalar (ISO string / ``datetime`` / ``np.datetime64``) — broadcast to length
    ``n_t`` — or a sequence already of that length. ``datetime`` objects carrying tzinfo are
    accepted (their wall-clock value is used; producers are expected to pass UTC).
    """
    import numpy as np  # lazy

    # Treat a 1-D array / list / tuple (but NOT a string) as an explicit per-step sequence.
    # numpy arrays are deliberately handled here because they are NOT collections.abc.Sequence
    # — a 1-element datetime64 array (the common ``np.array([when])`` case) would otherwise be
    # mis-routed into the scalar branch and fail to convert.
    is_str = isinstance(time, (str, bytes))
    is_seq_like = (not is_str) and (
        isinstance(time, (list, tuple))
        or (isinstance(time, np.ndarray) and time.ndim >= 1)
    )
    if is_seq_like:
        arr = np.asarray(time, dtype="datetime64[ns]")
        if arr.shape[0] != n_t:
            raise ValueError(
                f"time has length {arr.shape[0]} but bt has {n_t} time step(s)"
            )
        return arr

    # Single timestamp (string / datetime / np.datetime64 / 0-d array) -> broadcast.
    try:
        scalar = np.datetime64(time)  # type: ignore[arg-type]
    except Exception:
        # datetime with tzinfo: strip it (value is taken as-is / UTC by convention).
        import datetime as _dt

        if isinstance(time, _dt.datetime) and time.tzinfo is not None:
            scalar = np.datetime64(time.replace(tzinfo=None))
        else:
            raise
    return np.full((n_t,), scalar, dtype="datetime64[ns]")


def _coords_from_grid_or_arrays(
    lat: Any,
    lon: Any,
    n_rows: int,
    n_cols: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Resolve 1-D ``lat`` (len ``n_rows``) and ``lon`` (len ``n_cols``) coordinate arrays.

    ``lat`` / ``lon`` may be 1-D coordinate vectors, or 2-D meshgrids (the first column /
    first row is taken, matching a regular lat/lon grid), validated against the data shape.
    """
    import numpy as np  # lazy

    lat_a = np.asarray(lat, dtype=np.float64)
    lon_a = np.asarray(lon, dtype=np.float64)
    if lat_a.ndim == 2:
        lat_a = lat_a[:, 0]
    if lon_a.ndim == 2:
        lon_a = lon_a[0, :]
    if lat_a.ndim != 1 or lon_a.ndim != 1:
        raise ValueError("lat/lon must be 1-D coordinate vectors (or 2-D meshgrids)")
    if lat_a.shape[0] != n_rows:
        raise ValueError(f"lat length {lat_a.shape[0]} != n_rows {n_rows}")
    if lon_a.shape[0] != n_cols:
        raise ValueError(f"lon length {lon_a.shape[0]} != n_cols {n_cols}")
    return lat_a, lon_a


def write_netcdf(
    bt: Any,
    lat: Any,
    lon: Any,
    time: Any,
    attrs: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write a brightness-temperature frame to a schema-correct interpolated ``.nc`` file.

    Produces a NetCDF4 file matching :class:`frameflow.contracts.InferenceNetCDFSchema`:
    dims ``(time, y, x)``, a float32 Kelvin ``bt`` data variable (NaN-filled where
    off-disk), and ``time`` / ``lat`` / ``lon`` coordinates. The REQUIRED global attributes
    (``source_frames``, ``t``, ``model``, ``model_version``, ``generated_by``,
    ``institution``) MUST be present in ``attrs`` — build them with
    :func:`frameflow.contracts.netcdf_attrs`. ``write_netcdf`` validates their presence and
    raises if any is missing, so a malformed file is never produced.

    Args:
        bt: the BT field. 2D ``(y, x)`` (single instant) or 3D ``(time, y, x)``. Kelvin.
        lat: 1-D latitude vector (len ``n_rows``) or a 2-D lat meshgrid.
        lon: 1-D longitude vector (len ``n_cols``) or a 2-D lon meshgrid.
        time: a scalar timestamp (ISO string / ``datetime`` / ``np.datetime64``) broadcast
            over the time axis, or a sequence of length ``n_t``.
        attrs: global attributes (must include every
            :data:`InferenceNetCDFSchema.REQUIRED_ATTRS`). Lists/None are JSON-encoded as
            needed so NetCDF's attribute typing accepts them.
        path: destination ``.nc`` path (parent dirs are created).

    Returns:
        The :class:`pathlib.Path` written.

    Raises:
        ValueError: if a required attribute is missing or shapes are inconsistent.
        RuntimeError: if no usable NetCDF backend is installed.
    """
    import numpy as np  # lazy
    import xarray as xr  # lazy

    data = _coerce_2d_or_3d(bt)
    n_t, n_rows, n_cols = data.shape
    lat_a, lon_a = _coords_from_grid_or_arrays(lat, lon, n_rows, n_cols)
    time_a = _as_datetime64(time, n_t)

    # --- enforce the REQUIRED-attrs contract before writing anything ----------------------
    missing = [k for k in InferenceNetCDFSchema.REQUIRED_ATTRS if k not in attrs]
    if missing:
        raise ValueError(
            "write_netcdf: missing REQUIRED .nc attribute(s) "
            f"{missing} — build attrs with frameflow.contracts.netcdf_attrs(...)"
        )

    safe_attrs = _netcdf_safe_attrs(attrs)

    ds = xr.Dataset(
        data_vars={
            InferenceNetCDFSchema.DATA_VAR: (
                InferenceNetCDFSchema.DIMS,
                data,
                {
                    "units": "K",
                    "long_name": "brightness_temperature",
                    "standard_name": "toa_brightness_temperature",
                },
            )
        },
        coords={
            "time": ("time", time_a),
            "lat": ("y", lat_a, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("x", lon_a, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs=safe_attrs,
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    var = InferenceNetCDFSchema.DATA_VAR
    chunks = (
        1,
        min(InferenceNetCDFSchema.INTERNAL_CHUNKSIZES[1], int(n_rows)),
        min(InferenceNetCDFSchema.INTERNAL_CHUNKSIZES[2], int(n_cols)),
    )

    last_err: Exception | None = None
    for engine in ("netcdf4", "h5netcdf"):
        try:
            enc = {
                var: {
                    "chunksizes": chunks,
                    "zlib": True,
                    "complevel": 4,
                    "_FillValue": np.float32(np.nan),
                }
            }
            ds.to_netcdf(out, engine=engine, encoding=enc)
            return out
        except Exception as exc:  # pragma: no cover - depends on installed backends
            last_err = exc

    # Fallback: scipy backend (NetCDF3; no chunking/compression, no NaN-fill encoding).
    try:
        ds.to_netcdf(out, engine="scipy")
        return out
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not write NetCDF to {out}: no usable engine "
            f"(install netCDF4 or h5netcdf). Last error: {last_err or exc}"
        ) from (last_err or exc)


def read_netcdf(path: str | Path) -> "xr.Dataset":
    """Read an interpolated/observed ``.nc`` frame into an :class:`xarray.Dataset`.

    The returned dataset is fully loaded into memory (``.load()``) so the file handle is
    released and the values are immediately usable (the files are single-frame and small).
    Its ``bt`` variable is Kelvin float32 with NaN at off-disk pixels; ``time`` / ``lat`` /
    ``lon`` coordinates and the global attributes round-trip from :func:`write_netcdf`.

    Args:
        path: path to a ``.nc`` file.

    Returns:
        The loaded :class:`xarray.Dataset`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        RuntimeError: if no usable NetCDF backend can open the file.
    """
    import xarray as xr  # lazy

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NetCDF file not found: {p}")

    last_err: Exception | None = None
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            with xr.open_dataset(p, engine=engine) as ds:
                return ds.load()
        except Exception as exc:  # pragma: no cover - depends on installed backends
            last_err = exc
    raise RuntimeError(  # pragma: no cover
        f"Could not open NetCDF {p}: no usable engine. Last error: {last_err}"
    )


def write_frame_nc(
    bt: Any,
    grid: GridSpec,
    time: Any,
    *,
    source_frames: list[str],
    t: float,
    model: str,
    model_version: str,
    kind: str = "interpolated",
    interpolation_factor: int | None = None,
    extra: dict[str, Any] | None = None,
    path: str | Path,
) -> Path:
    """Convenience wrapper: write one frame given a :class:`GridSpec` (builds coords + attrs).

    This composes :func:`frameflow.contracts.netcdf_attrs` with :func:`write_netcdf` so
    callers that already hold a :class:`GridSpec` need not assemble lat/lon/attrs by hand.

    Args:
        bt: 2D ``(y, x)`` or 3D ``(time, y, x)`` Kelvin field.
        grid: the spatial grid (supplies lat/lon and ``crs``/``bbox`` advisory attrs).
        time: timestamp(s) for the time axis (see :func:`write_netcdf`).
        source_frames: identifiers of the bracketing frames.
        t: interpolation fraction in (0, 1) (0 / 1 for passthrough observed frames).
        model: model name. model_version: checkpoint/version string.
        kind: ``"interpolated"`` or ``"observed"``.
        interpolation_factor: optional up-sampling factor (advisory attr).
        extra: extra attributes to merge.
        path: destination ``.nc`` path.

    Returns:
        The written :class:`pathlib.Path`.
    """
    merged_extra = {
        "crs": grid.crs,
        "bbox_west_south_east_north": list(grid.bbox),
        "resolution_deg": grid.resolution_deg,
    }
    if extra:
        merged_extra.update(extra)
    attrs = netcdf_attrs(
        source_frames=source_frames,
        t=t,
        model=model,
        model_version=model_version,
        kind="observed" if kind == "observed" else "interpolated",
        interpolation_factor=interpolation_factor,
        extra=merged_extra,
    )
    return write_netcdf(
        bt=bt,
        lat=grid.lat_coords(),
        lon=grid.lon_coords(),
        time=time,
        attrs=attrs,
        path=path,
    )


def _netcdf_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Coerce attribute values to NetCDF-friendly scalars/arrays.

    NetCDF attribute values must be strings or numeric scalars/sequences — ``None`` and
    nested/heterogeneous containers are not allowed. We JSON-encode such values (and empty
    lists) so they survive the round-trip as strings; lists of homogeneous numbers/strings
    are passed through (xarray/netCDF4 handles them). ``bool`` is mapped to int (NetCDF has
    no native bool attribute).
    """
    import json

    safe: dict[str, Any] = {}
    for key, val in attrs.items():
        if val is None:
            safe[key] = "null"
        elif isinstance(val, bool):
            safe[key] = int(val)
        elif isinstance(val, (str, int, float)):
            safe[key] = val
        elif isinstance(val, (list, tuple)):
            seq = list(val)
            if len(seq) == 0:
                safe[key] = "[]"
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seq):
                safe[key] = seq  # homogeneous numeric list is a valid attr
            elif all(isinstance(v, str) for v in seq):
                # Join string lists; NetCDF attrs don't store ragged string arrays portably.
                safe[key] = json.dumps(seq)
            else:
                safe[key] = json.dumps(seq)
        else:
            safe[key] = json.dumps(val, default=str)
    return safe

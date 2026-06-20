"""Assemble a canonical :class:`~frameflow.contracts.CubeSchema` Zarr v3 cube from frames.

This is the offline "do the expensive work once" step (research/03 §0, §5, §8): read a list
of source ``.nc``/``.h5`` frames, convert each to brightness temperature in **Kelvin** (via
:mod:`frameflow.data.preprocess.radiance` when the file ships radiance/counts, or pass an
already-BT variable straight through), **regrid** every frame onto one common lat/lon
:class:`~frameflow.contracts.GridSpec` with a *cached* pyresample kernel
(:func:`frameflow.data.preprocess.regrid.regrid_to_grid`), stack them along ``time``, and
write a schema-correct Zarr v3 store:

    * dims        ``("time", "y", "x")``
    * data var    ``"bt"`` — float32, Kelvin, NaN where off-disk/space
    * coords      ``time`` (datetime64[ns]), ``lat`` (along ``y``), ``lon`` (along ``x``)
    * chunks      ``(8, 512, 512)`` (clamped to the array; CONTRACTS.md / constants)
    * shards      ``(32, 1024, 1024)`` when the array is large enough (Zarr v3 sharding)
    * compressor  ``blosc-zstd-shuffle``
    * fill value  ``NaN``

A read ``cube["bt"][t0:t1, y0:y1, x0:x1]`` then maps by integer arithmetic to exactly the
chunk(s) needed -> O(1) chunk address (research/03 §1.1). The cube the synthetic ``.nc``
triplet produces here is byte-for-byte readable by xarray.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ... import constants as C
from ...contracts import CubeSchema, GridSpec, empty_cube

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import xarray as xr


__all__ = ["build_cube", "read_bt_frame"]


# Candidate names for an already-brightness-temperature variable across sensors. Checked in
# order so the cleanest BT-ready representation wins before we fall back to a Planck inverse.
_BT_VAR_CANDIDATES: tuple[str, ...] = (
    "bt",
    "CMI",
    "brightness_temperature",
    "tbb_13",
    "tbb",
    "Band13",
    "band13",
    "TBB",
    "tb",
    "IMG_TIR1_TB",
)


def build_cube(
    files: Sequence[str | Path],
    grid: GridSpec | None = None,
    out_path: str | Path = "data/cubes/cube.zarr",
    source_meta: dict[str, Any] | None = None,
) -> Path:
    """Build a schema-correct Zarr v3 brightness-temperature cube from source frames.

    Each file in ``files`` is read to a BT :class:`xarray.DataArray` (Kelvin), regridded onto
    the common ``grid`` (cached pyresample kernel), and stacked along ``time`` in file order.
    The result is written as a :class:`~frameflow.contracts.CubeSchema` Zarr v3 store and the
    path is returned. Works directly on the synthetic ``.nc`` triplet from
    :func:`frameflow.synthetic.generate_pair_nc`.

    Args:
        files: ordered source frame paths/URIs (``.nc`` / ``.h5``). Order defines the
            ``time`` axis order; per-file timestamps are read from each file when available,
            else synthesized from ``source_meta`` cadence (or a 0..N index).
        grid: the target common :class:`~frameflow.contracts.GridSpec`. Defaults to
            :meth:`GridSpec.default` (128x128 demo grid).
        out_path: destination ``.zarr`` directory (overwritten if it exists).
        source_meta: optional metadata dict. Recognized keys:

            * ``"source"`` / ``"satellite"`` — satellite key; when set, the matching
              :class:`~frameflow.data.access.SatelliteSource` driver reads each frame.
            * ``"channel"`` — channel/band identifier recorded on the cube.
            * ``"cadence_min"`` — minutes between frames (used to synthesize timestamps when
              files carry none).
            * ``"start_time"`` — first-frame UTC :class:`datetime`/ISO string for synthesized
              timestamps.
            * ``"cache_dir"`` — regridder kernel cache directory (defaults next to
              ``out_path``).
            * arbitrary extra keys are merged into the cube's global attributes.

    Returns:
        The :class:`pathlib.Path` to the written ``.zarr`` store.

    Raises:
        ValueError: if ``files`` is empty or a frame has no recognizable BT/radiance variable.
    """
    import numpy as np  # lazy

    from ..preprocess.regrid import regrid_to_grid

    file_list = [str(f) for f in files]
    if not file_list:
        raise ValueError("build_cube: `files` is empty — provide at least one source frame.")

    grid = grid or GridSpec.default()
    meta = dict(source_meta or {})
    out = Path(out_path)
    cache_dir = meta.get("cache_dir") or str(out.parent / "_regrid_cache")

    source_key = meta.get("source") or meta.get("satellite")
    driver = None
    if source_key:
        try:
            from ..access import get_source  # lazy; avoids cycle at import time

            driver = get_source(str(source_key), anon=bool(meta.get("anon", True)))
        except Exception:  # pragma: no cover - unknown/edge source keys fall back to xarray
            driver = None

    n_rows, n_cols = grid.shape
    n_t = len(file_list)
    bt = np.full((n_t, n_rows, n_cols), np.nan, dtype=np.float32)
    times: list[Any] = []

    for i, fpath in enumerate(file_list):
        da, when = read_bt_frame(fpath, driver=driver)
        regridded = regrid_to_grid(da, grid, cache_dir=cache_dir)
        frame2d = _to_2d(regridded.values)
        bt[i] = np.asarray(frame2d, dtype=np.float32)
        times.append(when)

    time_index = _resolve_times(times, meta, n_t)

    ds = empty_cube(grid, time_index)
    ds[CubeSchema.DATA_VAR].values[...] = bt

    wavelength = meta.get("wavelength_um")
    if wavelength is None and source_key and str(source_key) in C.SATELLITE_BANDS:
        wavelength = C.SATELLITE_BANDS[str(source_key)]["wavelength_um"]
    ds.attrs.update(
        {
            "title": "FrameFlow brightness-temperature analysis cube",
            "satellite": str(source_key) if source_key else meta.get("satellite", "UNKNOWN"),
            "channel": str(meta.get("channel", "")),
            "source_files": file_list,
            "n_frames": n_t,
        }
    )
    if wavelength is not None:
        ds.attrs["wavelength_um"] = float(wavelength)
    # Merge any caller-supplied extra attrs (skip the structured keys handled above).
    _reserved = {
        "source", "satellite", "channel", "cadence_min", "start_time", "cache_dir",
        "anon", "wavelength_um",
    }
    for k, v in meta.items():
        if k not in _reserved and _is_attr_safe(v):
            ds.attrs.setdefault(k, v)

    _write_zarr_cube(ds, out, grid, n_t)
    return out


def read_bt_frame(
    path: str | Path,
    *,
    driver: Any | None = None,
) -> "tuple[xr.DataArray, Any]":
    """Read one source frame as a BT :class:`xarray.DataArray` (Kelvin) + its timestamp.

    Dispatch order:

    1. If a :class:`~frameflow.data.access.SatelliteSource` ``driver`` is given, delegate to
       its ``read_frame`` (handles radiance/count -> BT and lat/lon for that sensor).
    2. Otherwise open the file with xarray and use the first recognized already-BT variable
       (``bt``/``CMI``/``tbb_13``/...); if only ``Rad`` + Planck coefficients are present,
       apply the inverse Planck via :func:`frameflow.data.preprocess.radiance.radiance_to_bt`.

    Args:
        path: a ``.nc`` / ``.h5`` frame path or ``s3://`` URI.
        driver: optional sensor access driver to use instead of the generic reader.

    Returns:
        ``(da, when)`` — the BT DataArray (named ``"bt"``, Kelvin, with ``lat``/``lon`` coords
        and a leading length-1 ``time`` dim) and the frame's timestamp (a numpy datetime64,
        a :class:`datetime`, or ``None`` if the file carries none).

    Raises:
        ValueError: if no BT or radiance variable can be found.
    """
    import numpy as np  # lazy
    import xarray as xr  # lazy

    if driver is not None:
        da = driver.read_frame(path)
        return _ensure_time_dim(da), None

    ds = xr.open_dataset(str(path))
    try:
        when = _extract_time(ds)
        var = next((v for v in _BT_VAR_CANDIDATES if v in ds.data_vars), None)
        if var is not None:
            da = ds[var]
            da = _attach_latlon(da, ds)
            out = xr.DataArray(
                np.asarray(da.values, dtype="float32"),
                dims=da.dims,
                coords=da.coords,
                attrs={**da.attrs, "units": "K", "long_name": "brightness_temperature"},
                name="bt",
            )
            return _ensure_time_dim(out), when

        if "Rad" in ds.data_vars:
            from ..preprocess.radiance import radiance_to_bt

            rad = ds["Rad"]
            fk1 = float(ds["planck_fk1"].values)
            fk2 = float(ds["planck_fk2"].values)
            bc1 = float(ds["planck_bc1"].values)
            bc2 = float(ds["planck_bc2"].values)
            bt_vals = radiance_to_bt(rad.values, fk1, fk2, bc1, bc2).astype("float32")
            da = xr.DataArray(bt_vals, dims=rad.dims, coords=rad.coords, name="bt")
            da = _attach_latlon(da, ds)
            da.attrs.update({"units": "K", "long_name": "brightness_temperature"})
            return _ensure_time_dim(da), when

        raise ValueError(
            f"build_cube: {path} has no recognized BT variable ({_BT_VAR_CANDIDATES}) and no "
            f"'Rad' (+Planck) variable; got data_vars={list(ds.data_vars)}. Pass a "
            f"source_meta['source'] so the right sensor driver reads it."
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_2d(arr: "np.ndarray") -> "np.ndarray":
    """Squeeze a leading length-1 time/band axis so we have a clean ``(y, x)`` field."""
    import numpy as np  # lazy

    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    if a.ndim == 2:
        return a
    return np.squeeze(a)


def _ensure_time_dim(da: "xr.DataArray") -> "xr.DataArray":
    """Return a DataArray with a leading ``time`` dim (regrid_to_grid preserves it)."""
    if "time" in da.dims:
        return da
    return da.expand_dims("time")


def _attach_latlon(da: "xr.DataArray", ds: "xr.Dataset") -> "xr.DataArray":
    """Ensure ``lat``/``lon`` coords are on ``da``, copying from the parent dataset if needed."""
    import numpy as np  # lazy

    if "lat" in da.coords and "lon" in da.coords:
        return da
    lat_name = _first_coord(ds, ("lat", "latitude"))
    lon_name = _first_coord(ds, ("lon", "longitude"))
    if lat_name is None or lon_name is None:
        return da
    latv = np.asarray(ds[lat_name].values, dtype="float64")
    lonv = np.asarray(ds[lon_name].values, dtype="float64")
    spatial = [d for d in da.dims if d != "time"]
    if latv.ndim == 1 and lonv.ndim == 1 and len(spatial) >= 2:
        return da.assign_coords(
            {"lat": (spatial[-2], latv), "lon": (spatial[-1], lonv)}
        )
    if latv.ndim == 2 and lonv.ndim == 2 and len(spatial) >= 2:
        return da.assign_coords(
            {"lat": (tuple(spatial[-2:]), latv), "lon": (tuple(spatial[-2:]), lonv)}
        )
    return da


def _first_coord(ds: "xr.Dataset", names: "tuple[str, ...]") -> str | None:
    for n in names:
        if n in ds.coords or n in ds.variables:
            return n
    return None


def _extract_time(ds: "xr.Dataset") -> Any:
    """Pull a single timestamp from a dataset's ``time`` coordinate, else ``None``."""
    import numpy as np  # lazy

    if "time" in ds.coords:
        vals = np.asarray(ds["time"].values).reshape(-1)
        if vals.size:
            return vals[0]
    return None


def _resolve_times(times: list[Any], meta: dict[str, Any], n_t: int) -> "np.ndarray":
    """Build a complete ``datetime64[ns]`` time index, synthesizing any missing stamps."""
    import numpy as np  # lazy

    have_all = all(t is not None for t in times) and len(times) == n_t
    if have_all:
        return np.asarray(times, dtype="datetime64[ns]")

    # Synthesize from start_time + cadence (fall back to a unit-minute index).
    start = meta.get("start_time")
    if start is None:
        start = datetime(2025, 6, 20, 0, 0, 0)
    elif isinstance(start, str):
        start = _parse_iso(start)
    cadence = int(meta.get("cadence_min", C.DEFAULT_INPUT_CADENCE_MIN))
    base = np.datetime64(start.replace(tzinfo=None), "ns")
    step = np.timedelta64(cadence, "m").astype("timedelta64[ns]")
    synth = np.array([base + i * step for i in range(n_t)], dtype="datetime64[ns]")

    # Prefer any real per-file stamps we *did* read; fill gaps with the synthesized series.
    out = synth.copy()
    for i, t in enumerate(times):
        if t is not None:
            try:
                out[i] = np.datetime64(t, "ns")
            except Exception:  # pragma: no cover - defensive against odd time dtypes
                pass
    return out


def _parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 (optionally ``Z``-suffixed) string into a tz-naive datetime."""
    txt = s.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    return dt.replace(tzinfo=None)


def _is_attr_safe(v: Any) -> bool:
    """True if ``v`` is a Zarr/NetCDF-attribute-safe scalar/sequence of scalars."""
    if isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, (list, tuple)):
        return all(isinstance(x, (str, int, float, bool)) for x in v)
    return False


def _write_zarr_cube(ds: "xr.Dataset", out: Path, grid: GridSpec, n_t: int) -> None:
    """Write ``ds`` to a Zarr v3 store with the canonical chunks/shards/compressor."""
    from zarr.codecs import BloscCodec, BloscShuffle  # lazy

    out.parent.mkdir(parents=True, exist_ok=True)
    _rmtree(out)

    n_rows, n_cols = grid.shape
    # Inner chunks: the canonical (8,512,512), clamped to the array so small cubes still work.
    tchunk = min(C.CUBE_CHUNKS[0], n_t)
    ychunk = min(C.CUBE_CHUNKS[1], n_rows)
    xchunk = min(C.CUBE_CHUNKS[2], n_cols)
    chunks = (tchunk, ychunk, xchunk)

    encoding: dict[str, Any] = {
        CubeSchema.DATA_VAR: {
            "chunks": chunks,
            "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
        }
    }
    # Add Zarr v3 sharding only when the array is strictly larger than one chunk along an
    # axis AND the canonical shard is a clean multiple of the (clamped) chunk — otherwise a
    # shard equal to the chunk is pointless and an indivisible shard is rejected by zarr.
    shards = _maybe_shards(chunks, (n_t, n_rows, n_cols))
    if shards is not None:
        encoding[CubeSchema.DATA_VAR]["shards"] = shards

    ds.to_zarr(out, mode="w", encoding=encoding, zarr_format=3, consolidated=False)


def _maybe_shards(
    chunks: "tuple[int, int, int]", shape: "tuple[int, int, int]"
) -> "tuple[int, int, int] | None":
    """Return canonical shards clamped to the array, or None when sharding adds nothing.

    Each shard dim must be a positive integer multiple of the chunk dim and not exceed the
    array dim. If the result equals ``chunks`` exactly (one chunk per shard) we skip sharding.
    """
    shard_out = []
    for sh, ch, n in zip(C.CUBE_SHARDS, chunks, shape):
        if ch <= 0:
            return None
        # Largest multiple of ch that is <= min(sh, n).
        cap = min(int(sh), int(n))
        mult = max(cap // ch, 1)
        shard_out.append(ch * mult)
    shards = tuple(shard_out)  # type: ignore[assignment]
    if shards == chunks:
        return None
    return shards  # type: ignore[return-value]


def _rmtree(path: Path) -> None:
    """Recursively remove a file or directory tree if it exists (best-effort)."""
    import shutil

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:  # pragma: no cover - defensive
            pass

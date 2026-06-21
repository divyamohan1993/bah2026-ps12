"""Regridding to the common analysis grid with a CACHED, reusable resampling kernel.

The geostationary case is special (research/03 §4): the satellite and its fixed grid do
not move, so the source->target mapping is *identical for every time step*. We therefore
build the expensive nearest-neighbour KDTree exactly once, cache the neighbour indices to
disk, and turn every subsequent frame's regrid into a cheap gather.

Two backends:

* **pyresample** (preferred) — ``kd_tree.get_neighbour_info`` builds the neighbour table
  once; ``kd_tree.get_sample_from_neighbour_info`` applies it per frame. The table is
  cached in an ``.npz`` keyed by the source/target geometry so repeated runs and repeated
  frames reuse it (this is exactly the "cache_dir" trick satpy documents for geostationary
  data).
* **scipy / xarray fallback** — when pyresample area/swath defs cannot be constructed (or
  pyresample is unavailable), fall back to :func:`scipy.interpolate.griddata` (irregular
  source) or a fast :func:`xarray.DataArray.interp` (already-regular lat/lon source). The
  fallback still caches nothing expensive because the regular-grid path is itself cheap.

Public surface:

* :class:`Regridder` — holds the cached kernel + target grid; ``__call__`` regrids a 2-D
  field.
* :func:`make_regridder` — build/load a :class:`Regridder` for a (src, dst) pair.
* :func:`regrid` — apply a regridder to a field (the contract's free function).
* :func:`regrid_to_grid` — convenience: regrid an :class:`xarray.DataArray` straight onto a
  :class:`~frameflow.contracts.GridSpec` (the flat ``preprocess.regrid_to_grid`` entry).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import xarray as xr

    from ...contracts import GridSpec


__all__ = [
    "Regridder",
    "make_regridder",
    "regrid",
    "regrid_to_grid",
    "grid_to_area_def",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def grid_to_area_def(grid: "GridSpec", area_id: str = "frameflow_target") -> "Any":
    """Build a pyresample :class:`AreaDefinition` for a lat/lon :class:`GridSpec`.

    Only meaningful for a geographic (EPSG:4326) target grid, which is what the FrameFlow
    cube uses. Raises if pyresample is missing so callers can fall back.

    Args:
        grid: the target :class:`~frameflow.contracts.GridSpec`.
        area_id: identifier stored on the area definition.

    Returns:
        A :class:`pyresample.geometry.AreaDefinition`.
    """
    from pyresample import geometry  # lazy; may raise ImportError

    # extent is (x_ll, y_ll, x_ur, y_ur) = (west, south, east, north) for lon/lat.
    area_extent = (grid.west, grid.south, grid.east, grid.north)
    proj_dict = {"proj": "longlat", "datum": "WGS84"} if "4326" in str(grid.crs) else {"init": grid.crs}
    return geometry.AreaDefinition(
        area_id,
        "FrameFlow common analysis grid",
        "frameflow",
        proj_dict,
        grid.n_cols,
        grid.n_rows,
        area_extent,
    )


def _coords_key(src_lons: "np.ndarray", src_lats: "np.ndarray", grid: "GridSpec") -> str:
    """Stable hash of the source coordinate arrays + target grid (for kernel caching)."""
    import numpy as np  # lazy

    h = hashlib.sha1()
    for a in (np.ascontiguousarray(src_lons, dtype=np.float64),
              np.ascontiguousarray(src_lats, dtype=np.float64)):
        h.update(str(a.shape).encode())
        # Hash a strided sample + the corners so huge grids stay cheap to key but distinct.
        flat = a.ravel()
        sample = flat[:: max(flat.size // 4096, 1)]
        h.update(sample.tobytes())
        h.update(flat[:1].tobytes())
        h.update(flat[-1:].tobytes())
    h.update(json.dumps(grid.to_dict(), sort_keys=True).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Regridder
# ---------------------------------------------------------------------------
@dataclass
class Regridder:
    """A reusable resampling kernel mapping a fixed source grid onto a target lat/lon grid.

    Instances are cheap to apply: :meth:`__call__` performs a gather (pyresample backend) or
    a cached interpolation (fallback backend) and returns a ``(n_rows, n_cols)`` field. The
    same :class:`Regridder` is reused for every frame of a geostationary sequence, which is
    where the speed-up comes from (research/03 §4).

    Attributes:
        grid: the target :class:`~frameflow.contracts.GridSpec`.
        backend: ``"pyresample"`` or ``"scipy"``.
        radius_of_influence: NN search radius in metres (pyresample backend).
    """

    grid: "GridSpec"
    backend: str = "pyresample"
    radius_of_influence: float = 50000.0
    # pyresample neighbour-info (set for the pyresample backend)
    _valid_input: Any = field(default=None, repr=False)
    _valid_output: Any = field(default=None, repr=False)
    _index_array: Any = field(default=None, repr=False)
    _src_shape: tuple[int, int] | None = field(default=None, repr=False)
    # fallback (scipy/xarray) state
    _src_lons: Any = field(default=None, repr=False)
    _src_lats: Any = field(default=None, repr=False)
    _src_is_regular: bool = field(default=False, repr=False)

    # -- application -------------------------------------------------------------------
    def __call__(self, data: "np.ndarray | Any") -> "np.ndarray":
        """Regrid one 2-D source field onto the target grid.

        Args:
            data: a 2-D source array (matching the geometry the regridder was built for).

        Returns:
            ``numpy.ndarray`` of shape ``grid.shape`` (float32), NaN outside coverage.
        """
        import numpy as np  # lazy

        arr = np.asarray(data, dtype=np.float64)
        if self.backend == "pyresample":
            return self._apply_pyresample(arr)
        return self._apply_fallback(arr)

    def _apply_pyresample(self, arr: "np.ndarray") -> "np.ndarray":
        import numpy as np  # lazy
        from pyresample import kd_tree  # lazy

        if self._src_shape is not None and arr.shape != self._src_shape:
            raise ValueError(
                f"Regridder source shape mismatch: kernel built for {self._src_shape}, "
                f"got {arr.shape}."
            )
        out = kd_tree.get_sample_from_neighbour_info(
            "nn",
            self.grid.shape,
            arr,
            self._valid_input,
            self._valid_output,
            self._index_array,
            fill_value=np.nan,
        )
        return np.asarray(out, dtype=np.float32)

    def _apply_fallback(self, arr: "np.ndarray") -> "np.ndarray":
        import numpy as np  # lazy

        tgt_lon = np.asarray(self.grid.lon_coords(), dtype=np.float64)
        tgt_lat = np.asarray(self.grid.lat_coords(), dtype=np.float64)

        if self._src_is_regular:
            # Source is a regular lat/lon grid -> use fast xarray linear interpolation.
            import xarray as xr  # lazy

            src_lat = np.asarray(self._src_lats, dtype=np.float64)
            src_lon = np.asarray(self._src_lons, dtype=np.float64)
            da = xr.DataArray(arr, dims=("y", "x"), coords={"lat": ("y", src_lat), "lon": ("x", src_lon)})
            da = da.swap_dims({"y": "lat", "x": "lon"})
            interp = da.interp(lat=tgt_lat, lon=tgt_lon, method="linear")
            return np.asarray(interp.values, dtype=np.float32)

        # Irregular source -> scipy griddata (nearest, robust to NaNs/holes).
        from scipy.interpolate import griddata  # lazy

        src_lon = np.asarray(self._src_lons, dtype=np.float64).ravel()
        src_lat = np.asarray(self._src_lats, dtype=np.float64).ravel()
        vals = arr.ravel()
        finite = np.isfinite(vals) & np.isfinite(src_lon) & np.isfinite(src_lat)
        mlon, mlat = np.meshgrid(tgt_lon, tgt_lat)
        out = griddata(
            (src_lon[finite], src_lat[finite]),
            vals[finite],
            (mlon, mlat),
            method="nearest",
        )
        return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_regridder(
    src_area: "Any",
    dst_area: "GridSpec | Any",
    cache_dir: str | Path | None = None,
    *,
    radius_of_influence: float = 50000.0,
) -> Regridder:
    """Build (or load from cache) a reusable :class:`Regridder` from ``src_area`` -> grid.

    The expensive pyresample KDTree neighbour search runs at most once per (source, target)
    pair: the neighbour table is cached as an ``.npz`` in ``cache_dir`` and reloaded on
    subsequent calls and runs (research/03 §4).

    Args:
        src_area: the source geometry. Accepts a pyresample ``AreaDefinition`` /
            ``SwathDefinition``, or a mapping/object exposing ``lons``/``lats`` 2-D arrays
            (e.g. ``{"lons": ..., "lats": ...}``), or an :class:`xarray.DataArray` with
            ``lat``/``lon`` coords.
        dst_area: the target :class:`~frameflow.contracts.GridSpec` (preferred) or a
            pyresample area definition for the same lat/lon grid.
        cache_dir: directory for the cached neighbour table (created if needed). If None, the
            kernel is built in-memory and not persisted.
        radius_of_influence: NN search radius in metres.

    Returns:
        A ready-to-apply :class:`Regridder`.
    """
    import numpy as np  # lazy

    from ...contracts import GridSpec  # lazy to avoid import cycle at module load

    grid = dst_area if isinstance(dst_area, GridSpec) else _area_to_gridspec(dst_area)
    src_lons, src_lats = _extract_lonlat(src_area)

    # --- try pyresample (cached neighbour info) ---------------------------------------
    try:
        return _make_pyresample_regridder(
            src_lons, src_lats, grid, cache_dir, radius_of_influence
        )
    except Exception:  # pragma: no cover - exercised only when pyresample unavailable
        # Fall back to scipy/xarray; decide regular vs irregular from the coord shapes.
        rg = Regridder(grid=grid, backend="scipy", radius_of_influence=radius_of_influence)
        is_regular = src_lons.ndim == 1 and src_lats.ndim == 1
        rg._src_is_regular = bool(is_regular)
        rg._src_lons = src_lons
        rg._src_lats = src_lats
        return rg


def _make_pyresample_regridder(
    src_lons: "np.ndarray",
    src_lats: "np.ndarray",
    grid: "GridSpec",
    cache_dir: str | Path | None,
    radius_of_influence: float,
) -> Regridder:
    import numpy as np  # lazy
    from pyresample import geometry, kd_tree  # lazy; may raise ImportError

    lons2d = src_lons if src_lons.ndim == 2 else np.meshgrid(src_lons, src_lats)[0]
    lats2d = src_lats if src_lats.ndim == 2 else np.meshgrid(src_lons, src_lats)[1]
    src_shape = (int(lons2d.shape[0]), int(lons2d.shape[1]))

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        key = _coords_key(lons2d, lats2d, grid)
        cache_path = cache_path / f"regrid_kernel_{key}.npz"

    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        rg = Regridder(grid=grid, backend="pyresample", radius_of_influence=radius_of_influence)
        rg._valid_input = cached["valid_input"]
        rg._valid_output = cached["valid_output"]
        rg._index_array = cached["index_array"]
        rg._src_shape = (int(cached["src_rows"]), int(cached["src_cols"]))
        return rg

    src_def = geometry.SwathDefinition(lons=lons2d, lats=lats2d)
    tgt_def = grid_to_area_def(grid)
    valid_input, valid_output, index_array, _dist = kd_tree.get_neighbour_info(
        src_def, tgt_def, radius_of_influence=radius_of_influence, neighbours=1
    )

    if cache_path is not None:
        np.savez(
            cache_path,
            valid_input=valid_input,
            valid_output=valid_output,
            index_array=index_array,
            src_rows=src_shape[0],
            src_cols=src_shape[1],
        )

    rg = Regridder(grid=grid, backend="pyresample", radius_of_influence=radius_of_influence)
    rg._valid_input = valid_input
    rg._valid_output = valid_output
    rg._index_array = index_array
    rg._src_shape = src_shape
    return rg


def regrid(data: "np.ndarray | Any", regridder: Regridder) -> "np.ndarray":
    """Apply a :class:`Regridder` to a 2-D field (the contract's free function).

    Args:
        data: a 2-D source array matching the regridder's source geometry.
        regridder: a :class:`Regridder` from :func:`make_regridder`.

    Returns:
        The regridded ``(n_rows, n_cols)`` field (float32, NaN outside coverage).
    """
    return regridder(data)


def regrid_to_grid(
    da: "xr.DataArray",
    grid: "GridSpec",
    *,
    cache_dir: str | Path | None = None,
) -> "xr.DataArray":
    """Regrid an :class:`xarray.DataArray` (with lat/lon coords) onto a :class:`GridSpec`.

    Convenience wrapper matching the flat ``preprocess.regrid_to_grid(da, grid, cache_dir=)``
    contract entry. Builds a (cached) regridder from the DataArray's own coordinates and
    applies it, returning a new DataArray on the target ``(y, x)`` grid with ``lat``/``lon``
    coords. A leading length-1 ``time`` dim is preserved if present.

    Args:
        da: source brightness-temperature DataArray (2-D, or 3-D with a length-1 time dim).
        grid: target :class:`~frameflow.contracts.GridSpec`.
        cache_dir: optional kernel cache directory.

    Returns:
        Regridded :class:`xarray.DataArray` with dims ``(y, x)`` (or ``(time, y, x)``).
    """
    import numpy as np  # lazy
    import xarray as xr  # lazy

    arr = da
    has_time = "time" in arr.dims
    if has_time:
        arr2d = arr.isel(time=0)
    else:
        arr2d = arr

    src_lons, src_lats = _extract_lonlat(arr2d)
    rg = make_regridder({"lons": src_lons, "lats": src_lats}, grid, cache_dir)
    out = rg(np.asarray(arr2d.values, dtype=np.float64))

    lat = np.asarray(grid.lat_coords(), dtype=np.float64)
    lon = np.asarray(grid.lon_coords(), dtype=np.float64)
    coords = {"lat": ("y", lat), "lon": ("x", lon)}
    dims: tuple[str, ...] = ("y", "x")
    data = out
    if has_time:
        data = out[np.newaxis, :, :]
        dims = ("time", "y", "x")
        coords["time"] = ("time", np.asarray(arr["time"].values).reshape(-1)[:1])

    return xr.DataArray(
        data.astype("float32"),
        dims=dims,
        coords=coords,
        attrs={**da.attrs, "units": "K", "long_name": "brightness_temperature"},
        name="bt",
    )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _extract_lonlat(src: "Any") -> "tuple[np.ndarray, np.ndarray]":
    """Pull 1-D-or-2-D lon/lat arrays from the many shapes ``src_area`` can take."""
    import numpy as np  # lazy

    # pyresample area/swath def
    if hasattr(src, "get_lonlats"):
        lons, lats = src.get_lonlats()
        return np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)
    # mapping {"lons":..,"lats":..}
    if isinstance(src, dict) and "lons" in src and "lats" in src:
        return np.asarray(src["lons"], dtype=np.float64), np.asarray(src["lats"], dtype=np.float64)
    # object with .lons/.lats attributes
    if hasattr(src, "lons") and hasattr(src, "lats"):
        return np.asarray(src.lons, dtype=np.float64), np.asarray(src.lats, dtype=np.float64)
    # xarray DataArray/Dataset with lat/lon coords
    if hasattr(src, "coords"):
        if "lon" in src.coords and "lat" in src.coords:
            return (
                np.asarray(src.coords["lon"].values, dtype=np.float64),
                np.asarray(src.coords["lat"].values, dtype=np.float64),
            )
        if "longitude" in src.coords and "latitude" in src.coords:
            return (
                np.asarray(src.coords["longitude"].values, dtype=np.float64),
                np.asarray(src.coords["latitude"].values, dtype=np.float64),
            )
    raise ValueError(
        "make_regridder: could not extract lon/lat from src_area; pass a pyresample "
        "AreaDefinition/SwathDefinition, a {'lons','lats'} dict, or an xarray object with "
        "lat/lon coordinates."
    )


def _area_to_gridspec(area: "Any") -> "GridSpec":
    """Best-effort conversion of a pyresample area definition to a :class:`GridSpec`."""
    from ...contracts import GridSpec  # lazy

    extent = getattr(area, "area_extent", None)
    width = getattr(area, "width", None)
    height = getattr(area, "height", None)
    if extent is None or width is None or height is None:
        raise ValueError("dst_area is not a GridSpec and lacks area_extent/width/height")
    west, south, east, north = extent
    return GridSpec(
        west=float(west), south=float(south), east=float(east), north=float(north),
        n_rows=int(height), n_cols=int(width),
    )

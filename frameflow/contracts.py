"""FrameFlow data contracts — the AUTHORITATIVE in-code interface every team builds against.

This module defines the typed, serializable contracts that flow between the six build
teams: the analysis-grid spec, the canonical Zarr cube schema, the training ``Sample``,
the inference NetCDF (.nc) output schema, and the web ``Manifest`` JSON schema (with its
nested ``FrameEntry`` / ``MetricRecord`` / ``CrossvalMethodResult`` records).

Design rules:
    * Pure standard library (``dataclasses`` + ``typing``) — NO third-party dependency at
      import time, so this module imports cleanly before any team lands code. Heavy deps
      (xarray) are lazily imported INSIDE helper functions only.
    * Every dataclass round-trips through ``to_dict()`` / ``from_dict()`` (JSON-friendly).
    * ``validate_manifest(d)`` returns a list of human-readable problems (empty == valid).

CODE-REVIEW CORRECTIONS BAKED IN:
    * P1 (metric data_range): the manifest carries a top-level ``metrics.data_range_k`` and
      a ``value_range_k`` that MUST equal the fixed physical range
      (:data:`frameflow.constants.BT_DATA_RANGE_K` / the [VMIN,VMAX] pair). Never per-image.
    * P2 (per-frame tile sources): a single raster PMTiles archive is addressed only by
      z/x/y and CANNOT select a timestamp. Therefore EACH :class:`FrameEntry` carries its
      OWN ``tiles_url_template`` AND ``pmtiles`` path; the timeline slider switches the
      active frame's tile source. ``validate_manifest`` enforces their presence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from . import constants as C

if TYPE_CHECKING:  # import only for type-checkers; never at runtime
    import numpy as np
    import xarray as xr


# Manifest schema version — bump when the JSON contract below changes incompatibly.
MANIFEST_SCHEMA_VERSION: str = "1.0"

# ISO-8601 UTC ("Zulu") timestamp regex, e.g. "2025-06-20T00:10:00Z".
_ISO8601_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
)


# ===========================================================================
# 1. GridSpec — the common analysis grid
# ===========================================================================
@dataclass(frozen=True)
class GridSpec:
    """A regular lat/lon analysis grid that all sensors are co-registered to.

    The bbox is given as west/south/east/north in CRS units (degrees for EPSG:4326). The
    grid has ``n_rows`` rows (y / latitude, north->south) and ``n_cols`` columns
    (x / longitude, west->east). ``resolution_deg`` is the (square) pixel size; it is
    advisory/derived and need not exactly equal ``(east-west)/n_cols`` if a producer
    rounds, but should be consistent.

    Attributes:
        west, south, east, north: bounding box edges (CRS units).
        n_rows: number of grid rows (latitude / y).
        n_cols: number of grid columns (longitude / x).
        crs: coordinate reference system (default EPSG:4326 lat/lon).
        resolution_deg: nominal pixel size in degrees.
    """

    west: float
    south: float
    east: float
    north: float
    n_rows: int
    n_cols: int
    crs: str = C.DEFAULT_GRID_CRS
    resolution_deg: float = C.DEFAULT_GRID_RESOLUTION_DEG

    @property
    def shape(self) -> tuple[int, int]:
        """Return the grid array shape ``(n_rows, n_cols)`` == ``(H, W)``."""
        return (self.n_rows, self.n_cols)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return the bounding box as ``(west, south, east, north)``."""
        return (self.west, self.south, self.east, self.north)

    def lat_coords(self) -> list[float]:
        """Return ``n_rows`` latitude pixel-centre coordinates, north -> south."""
        if self.n_rows <= 0:
            return []
        dy = (self.north - self.south) / self.n_rows
        return [self.north - (r + 0.5) * dy for r in range(self.n_rows)]

    def lon_coords(self) -> list[float]:
        """Return ``n_cols`` longitude pixel-centre coordinates, west -> east."""
        if self.n_cols <= 0:
            return []
        dx = (self.east - self.west) / self.n_cols
        return [self.west + (c + 0.5) * dx for c in range(self.n_cols)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridSpec:
        """Build a :class:`GridSpec` from a dict produced by :meth:`to_dict`."""
        return cls(
            west=float(d["west"]),
            south=float(d["south"]),
            east=float(d["east"]),
            north=float(d["north"]),
            n_rows=int(d["n_rows"]),
            n_cols=int(d["n_cols"]),
            crs=str(d.get("crs", C.DEFAULT_GRID_CRS)),
            resolution_deg=float(d.get("resolution_deg", C.DEFAULT_GRID_RESOLUTION_DEG)),
        )

    @classmethod
    def default(cls) -> GridSpec:
        """A small, fast default grid for the synthetic demo (128x128 over the Indian bbox).

        The full operational grid is larger (see :func:`default_operational_grid`); this
        small one keeps the synthetic generator and tests quick.
        """
        return cls(
            west=C.DEFAULT_GRID_BBOX[0],
            south=C.DEFAULT_GRID_BBOX[1],
            east=C.DEFAULT_GRID_BBOX[2],
            north=C.DEFAULT_GRID_BBOX[3],
            n_rows=128,
            n_cols=128,
            crs=C.DEFAULT_GRID_CRS,
            resolution_deg=(C.DEFAULT_GRID_BBOX[2] - C.DEFAULT_GRID_BBOX[0]) / 128.0,
        )


def default_operational_grid() -> GridSpec:
    """The full operational lat/lon grid over the default bbox at native resolution."""
    w, s, e, n = C.DEFAULT_GRID_BBOX
    res = C.DEFAULT_GRID_RESOLUTION_DEG
    n_cols = int(round((e - w) / res))
    n_rows = int(round((n - s) / res))
    return GridSpec(
        west=w, south=s, east=e, north=n,
        n_rows=n_rows, n_cols=n_cols,
        crs=C.DEFAULT_GRID_CRS, resolution_deg=res,
    )


# ===========================================================================
# 2. CubeSchema — the canonical Zarr analysis cube + empty_cube() helper
# ===========================================================================
class CubeSchema:
    """Documentation constants describing the canonical Zarr v3 analysis cube.

    The cube is the single source of truth for analysis-ready brightness temperature
    (R3 §1, §8.1). Layout:

        * dims:        ``("time", "y", "x")``
        * data var:    ``"bt"`` — float32, **Kelvin**, NaN-filled where off-disk/space
        * coords:      ``time`` (datetime64[ns]), ``lat`` (along ``y``), ``lon`` (along ``x``)
        * chunks:      ``(8, 512, 512)`` — one chunk serves a training window AND a 512 map tile
        * shards:      ``(32, 1024, 1024)`` — few large objects (Zarr v3 sharding codec)
        * compressor:  ``blosc-zstd-shuffle``
        * fill value:  ``NaN`` (space pixels)

    A read ``cube["bt"][t0:t1, y0:y1, x0:x1]`` maps by integer arithmetic to exactly the
    chunk(s) needed -> O(1) chunk address.
    """

    DIMS: tuple[str, str, str] = C.CUBE_DIMS
    DATA_VAR: str = C.CUBE_DATA_VAR
    DTYPE: str = C.CUBE_DTYPE
    COORDS: tuple[str, str, str] = ("time", "lat", "lon")
    CHUNKS: tuple[int, int, int] = C.CUBE_CHUNKS
    SHARDS: tuple[int, int, int] = C.CUBE_SHARDS
    COMPRESSOR: str = C.CUBE_COMPRESSOR
    UNITS: str = "K"
    LONG_NAME: str = "brightness_temperature"
    STANDARD_NAME: str = "toa_brightness_temperature"


def empty_cube(grid: GridSpec, times: list | Any) -> xr.Dataset:
    """Build a correctly-structured, empty (NaN-filled) :class:`xarray.Dataset` cube.

    The returned dataset matches :class:`CubeSchema`: dims ``(time, y, x)``, a float32
    ``bt`` data var filled with NaN, and ``time`` / ``lat`` / ``lon`` coordinates. xarray
    and numpy are imported lazily inside so this module stays import-light.

    Args:
        grid: the :class:`GridSpec` defining the spatial grid.
        times: a sequence of timestamps (anything ``np.array(..., dtype="datetime64[ns]")``
            accepts: ``datetime`` objects, ISO strings, or numpy datetimes).

    Returns:
        An :class:`xarray.Dataset` with one NaN-filled float32 ``bt`` variable and proper
        coordinates and attributes, ready for a producer to fill and write to Zarr.
    """
    import numpy as np  # lazy
    import xarray as xr  # lazy

    time_index = np.asarray(times, dtype="datetime64[ns]")
    n_t = int(time_index.shape[0])
    n_rows, n_cols = grid.shape

    bt = np.full((n_t, n_rows, n_cols), np.nan, dtype=np.float32)
    lat = np.asarray(grid.lat_coords(), dtype=np.float64)
    lon = np.asarray(grid.lon_coords(), dtype=np.float64)

    ds = xr.Dataset(
        data_vars={
            CubeSchema.DATA_VAR: (
                CubeSchema.DIMS,
                bt,
                {
                    "units": CubeSchema.UNITS,
                    "long_name": CubeSchema.LONG_NAME,
                    "standard_name": CubeSchema.STANDARD_NAME,
                },
            )
        },
        coords={
            "time": ("time", time_index),
            "lat": ("y", lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("x", lon, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "title": "FrameFlow brightness-temperature analysis cube",
            "crs": grid.crs,
            "bbox_west_south_east_north": list(grid.bbox),
            "resolution_deg": grid.resolution_deg,
            "data_var": CubeSchema.DATA_VAR,
            "compressor": CubeSchema.COMPRESSOR,
            "generated_by": C.GENERATED_BY,
            "institution": C.INSTITUTION,
        },
    )
    return ds


# ===========================================================================
# 3. Sample — one training triplet consumed by the dataset/model teams
# ===========================================================================
class Sample(TypedDict):
    """One frame-interpolation training/eval sample (a triplet + timestep).

    Tensors are single-channel brightness temperature, shape ``(1, H, W)`` (channel-first),
    either ``numpy.ndarray`` or a ``torch.Tensor``. Whether they are raw Kelvin or
    normalized is a pipeline convention recorded in ``meta`` (key ``"normalized"``); the
    contract is only the shape/dtype/keys.

    Keys:
        I0:  the earlier bracketing frame, shape ``(1, H, W)``.
        I1:  the later bracketing frame, shape ``(1, H, W)``.
        It:  the target intermediate frame at fraction ``t``, shape ``(1, H, W)``.
        t:   interpolation fraction in (0, 1); ``0.5`` == temporal midpoint.
        meta: free-form dict (e.g. satellite, timestamps, normalization flag, mask info).
    """

    I0: np.ndarray | Any
    I1: np.ndarray | Any
    It: np.ndarray | Any
    t: float
    meta: dict[str, Any]


# ===========================================================================
# 4. Inference NetCDF (.nc) output schema + helper
# ===========================================================================
class InferenceNetCDFSchema:
    """Documentation constants for the required interpolated-frame ``.nc`` output.

    PS-12 requires both input AND output to be ``.nc`` (idea.md l. 35). Each interpolated
    frame is written as a CF-style NetCDF4 file:

        * dims:   ``("time", "y", "x")`` (``time`` length 1 per interpolated instant)
        * var:    ``"bt"`` — float32, **Kelvin**, NaN where off-disk
        * coords: ``time`` (datetime64), ``lat`` (along ``y``), ``lon`` (along ``x``)
        * attrs (REQUIRED): ``source_frames``, ``t``, ``model``, ``model_version``,
          ``generated_by``, ``institution`` (+ ``crs``, ``interpolation_factor`` advisory).

    Producers SHOULD set internal NetCDF chunking ``(1, 512, 512)`` so downstream virtual
    (kerchunk/VirtualiZarr) access stays coarse-grained (R3 §1.4).
    """

    DIMS: tuple[str, str, str] = ("time", "y", "x")
    DATA_VAR: str = "bt"
    DTYPE: str = "float32"
    REQUIRED_ATTRS: tuple[str, ...] = (
        "source_frames",
        "t",
        "model",
        "model_version",
        "generated_by",
        "institution",
    )
    INTERNAL_CHUNKSIZES: tuple[int, int, int] = (1, 512, 512)


def netcdf_attrs(
    *,
    source_frames: list[str],
    t: float,
    model: str,
    model_version: str,
    kind: Literal["observed", "interpolated"] = "interpolated",
    interpolation_factor: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the REQUIRED global-attribute dict for an interpolated-frame ``.nc`` file.

    Args:
        source_frames: identifiers/paths of the bracketing frames used (e.g.
            ``["...00:00.nc", "...00:20.nc"]``).
        t: the interpolation fraction in (0, 1) for this output instant.
        model: model name (e.g. ``"RIFE"``).
        model_version: model/checkpoint version string (feeds the cache key, R6 §3.3).
        kind: ``"interpolated"`` (synthesized) or ``"observed"`` (passthrough real frame).
        interpolation_factor: optional integer up-sampling factor (2 == 30->15 min).
        extra: optional additional attributes to merge in.

    Returns:
        A dict containing every attribute in :data:`InferenceNetCDFSchema.REQUIRED_ATTRS`
        plus useful extras.
    """
    attrs: dict[str, Any] = {
        "source_frames": list(source_frames),
        "t": float(t),
        "kind": kind,
        "model": model,
        "model_version": model_version,
        "generated_by": C.GENERATED_BY,
        "institution": C.INSTITUTION,
        "units": "K",
        "data_var": InferenceNetCDFSchema.DATA_VAR,
        "Conventions": "CF-1.8",
    }
    if interpolation_factor is not None:
        attrs["interpolation_factor"] = int(interpolation_factor)
    if extra:
        attrs.update(extra)
    return attrs


# ===========================================================================
# 5. Manifest JSON schema (the WEB contract) + nested records
# ===========================================================================
FrameKind = Literal["observed", "interpolated"]


@dataclass
class FrameEntry:
    """One timeline frame in the web manifest (observed or interpolated).

    CODE-REVIEW CORRECTION P2: ``tiles_url_template`` AND ``pmtiles`` are PER-FRAME and
    REQUIRED. A single raster PMTiles archive is addressed only by z/x/y and cannot select
    a timestamp, so every frame needs its OWN tile source; the timeline slider switches the
    active frame's source. ``tiles_url_template`` must contain ``{z}``, ``{x}``, ``{y}``.

    Attributes:
        index: 0-based position on the timeline.
        time: ISO-8601 UTC timestamp ("...Z").
        kind: ``"observed"`` (real frame) or ``"interpolated"`` (synthesized).
        t: interpolation fraction in (0,1) for interpolated frames; ``None`` for observed.
        bracket: ``[i, j]`` indices of the two observed frames an interpolated frame sits
            between; ``None`` for observed frames.
        image: path to a full-frame rendered raster (PNG/WebP) for quick display.
        thumb: path to a small thumbnail raster.
        tiles_url_template: PER-FRAME XYZ tile template, e.g. ``"tiles/000/{z}/{x}/{y}.webp"``.
        pmtiles: PER-FRAME PMTiles archive path, e.g. ``"pmtiles/000.pmtiles"``.
        netcdf: path to this frame's ``.nc`` file (CF, Kelvin).
        flow_overlay: optional path to an optical-flow vector overlay asset (or ``None``).
    """

    index: int
    time: str
    kind: FrameKind
    image: str
    thumb: str
    tiles_url_template: str
    pmtiles: str
    netcdf: str
    t: float | None = None
    bracket: list[int] | None = None
    flow_overlay: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameEntry:
        bracket = d.get("bracket")
        return cls(
            index=int(d["index"]),
            time=str(d["time"]),
            kind=d["kind"],  # type: ignore[arg-type]
            image=str(d["image"]),
            thumb=str(d["thumb"]),
            tiles_url_template=str(d["tiles_url_template"]),
            pmtiles=str(d["pmtiles"]),
            netcdf=str(d["netcdf"]),
            t=(None if d.get("t") is None else float(d["t"])),
            bracket=(None if bracket is None else [int(b) for b in bracket]),
            flow_overlay=d.get("flow_overlay"),
        )


@dataclass
class MetricRecord:
    """Per-frame validation metrics for one interpolated frame vs. withheld ground truth.

    All full-reference metrics MUST have been computed with the fixed physical
    ``data_range`` (Kelvin) — see Manifest.metrics.data_range_k (P1). Fields are optional
    so a producer can emit whatever subset it computed; ``index`` ties the record to a
    :class:`FrameEntry`.
    """

    index: int
    time: str | None = None
    psnr: float | None = None
    ssim: float | None = None
    ms_ssim: float | None = None
    fsim: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mae: float | None = None
    bt_rmse_k: float | None = None
    bt_bias_k: float | None = None
    epe: float | None = None
    csi_235k: float | None = None
    fss: float | None = None
    lpips: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetricRecord:
        known = {
            "index", "time", "psnr", "ssim", "ms_ssim", "fsim", "mse", "rmse", "mae",
            "bt_rmse_k", "bt_bias_k", "epe", "csi_235k", "fss", "lpips", "extra",
        }
        extra = dict(d.get("extra", {}))
        return cls(
            index=int(d["index"]),
            time=d.get("time"),
            psnr=_opt_float(d.get("psnr")),
            ssim=_opt_float(d.get("ssim")),
            ms_ssim=_opt_float(d.get("ms_ssim")),
            fsim=_opt_float(d.get("fsim")),
            mse=_opt_float(d.get("mse")),
            rmse=_opt_float(d.get("rmse")),
            mae=_opt_float(d.get("mae")),
            bt_rmse_k=_opt_float(d.get("bt_rmse_k")),
            bt_bias_k=_opt_float(d.get("bt_bias_k")),
            epe=_opt_float(d.get("epe")),
            csi_235k=_opt_float(d.get("csi_235k")),
            fss=_opt_float(d.get("fss")),
            lpips=_opt_float(d.get("lpips")),
            extra={**{k: float(v) for k, v in d.items() if k not in known}, **extra},
        )


@dataclass
class CrossvalMethodResult:
    """Result of one cross-validation method from the 40-method framework (R5 §4).

    Attributes:
        method_id: short id, e.g. ``"M1"`` (leave-the-middle-out on GOES-19).
        name: human-readable method name.
        status: ``"run"``, ``"skipped"``, or ``"failed"``.
        summary: optional headline scalar(s) the method produced.
        details: optional free-form details (paths to plots, per-stratum tables, ...).
    """

    method_id: str
    name: str
    status: Literal["run", "skipped", "failed"] = "run"
    summary: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrossvalMethodResult:
        return cls(
            method_id=str(d["method_id"]),
            name=str(d.get("name", d["method_id"])),
            status=d.get("status", "run"),  # type: ignore[arg-type]
            summary={k: float(v) for k, v in d.get("summary", {}).items()},
            details=dict(d.get("details", {})),
        )


@dataclass
class Manifest:
    """The web dashboard manifest — the authoritative JSON contract the front-end reads.

    THE MANIFEST JSON SCHEMA (implemented exactly as the fields of this dataclass):

        version, scene_id, title, satellite, channel, wavelength_um,
        bbox[4], crs, colormap, value_range_k[2], tile_size, min_zoom, max_zoom,
        interpolation_factor, cadence_minutes_input, cadence_minutes_output,
        frames[]  (each FrameEntry: index, time, kind, t, bracket, image, thumb,
                   tiles_url_template, pmtiles, netcdf, flow_overlay),
        videos {observed, interpolated, side_by_side},
        metrics {data_range_k, value_range_k, per_frame[], summary{}, baselines{}},
        crossval {methods_run[]},
        generated,
        model {name, version, params_m}

    P1: ``metrics.data_range_k`` / ``value_range_k`` carry the FIXED physical Kelvin range
        used for every full-reference metric — never per-image min/max.
    P2: each frame in ``frames`` carries its OWN ``tiles_url_template`` and ``pmtiles`` so
        the slider can switch the active tile source per timestamp.
    """

    scene_id: str
    title: str
    satellite: str
    channel: str
    wavelength_um: float
    bbox: list[float]  # [west, south, east, north]
    colormap: str
    value_range_k: list[float]  # [vmin_k, vmax_k] — the fixed display/metric BT range
    tile_size: int
    min_zoom: int
    max_zoom: int
    interpolation_factor: int
    cadence_minutes_input: float
    cadence_minutes_output: float
    frames: list[FrameEntry]
    generated: str  # ISO-8601 UTC build timestamp
    model_name: str
    model_version: str
    model_params_m: float
    crs: str = C.DEFAULT_GRID_CRS
    version: str = MANIFEST_SCHEMA_VERSION
    videos: dict[str, str | None] = field(
        default_factory=lambda: {"observed": None, "interpolated": None, "side_by_side": None}
    )
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "data_range_k": C.BT_DATA_RANGE_K,
            "value_range_k": [C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K],
            "per_frame": [],
            "summary": {},
            "baselines": {},
        }
    )
    crossval: dict[str, Any] = field(default_factory=lambda: {"methods_run": []})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the exact manifest JSON structure (nested dicts/lists only)."""
        return {
            "version": self.version,
            "scene_id": self.scene_id,
            "title": self.title,
            "satellite": self.satellite,
            "channel": self.channel,
            "wavelength_um": self.wavelength_um,
            "bbox": list(self.bbox),
            "crs": self.crs,
            "colormap": self.colormap,
            "value_range_k": list(self.value_range_k),
            "tile_size": self.tile_size,
            "min_zoom": self.min_zoom,
            "max_zoom": self.max_zoom,
            "interpolation_factor": self.interpolation_factor,
            "cadence_minutes_input": self.cadence_minutes_input,
            "cadence_minutes_output": self.cadence_minutes_output,
            "frames": [f.to_dict() for f in self.frames],
            "videos": dict(self.videos),
            "metrics": _metrics_to_dict(self.metrics),
            "crossval": _crossval_to_dict(self.crossval),
            "generated": self.generated,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "params_m": self.model_params_m,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        """Build a :class:`Manifest` from a dict produced by :meth:`to_dict`."""
        model = d.get("model", {})
        metrics = dict(d.get("metrics", {}))
        # Re-hydrate nested records where present.
        if "per_frame" in metrics:
            metrics["per_frame"] = [
                MetricRecord.from_dict(m) if isinstance(m, dict) else m
                for m in metrics["per_frame"]
            ]
        crossval = dict(d.get("crossval", {"methods_run": []}))
        if "methods_run" in crossval:
            crossval["methods_run"] = [
                CrossvalMethodResult.from_dict(c) if isinstance(c, dict) else c
                for c in crossval["methods_run"]
            ]
        return cls(
            scene_id=str(d["scene_id"]),
            title=str(d["title"]),
            satellite=str(d["satellite"]),
            channel=str(d["channel"]),
            wavelength_um=float(d["wavelength_um"]),
            bbox=[float(x) for x in d["bbox"]],
            colormap=str(d["colormap"]),
            value_range_k=[float(x) for x in d["value_range_k"]],
            tile_size=int(d["tile_size"]),
            min_zoom=int(d["min_zoom"]),
            max_zoom=int(d["max_zoom"]),
            interpolation_factor=int(d["interpolation_factor"]),
            cadence_minutes_input=float(d["cadence_minutes_input"]),
            cadence_minutes_output=float(d["cadence_minutes_output"]),
            frames=[FrameEntry.from_dict(f) for f in d.get("frames", [])],
            generated=str(d.get("generated", "")),
            model_name=str(model.get("name", "")),
            model_version=str(model.get("version", "")),
            model_params_m=float(model.get("params_m", 0.0)),
            crs=str(d.get("crs", C.DEFAULT_GRID_CRS)),
            version=str(d.get("version", MANIFEST_SCHEMA_VERSION)),
            videos=dict(d.get("videos", {"observed": None, "interpolated": None, "side_by_side": None})),
            metrics=metrics,
            crossval=crossval,
        )


# ---------------------------------------------------------------------------
# Manifest validation (basic schema checks; returns list of problems)
# ---------------------------------------------------------------------------
def validate_manifest(d: dict[str, Any]) -> list[str]:
    """Validate a manifest dict against the schema; return a list of problems.

    An empty returned list means the manifest passed the basic checks. This intentionally
    accepts a plain ``dict`` (the JSON the web team consumes), not a :class:`Manifest`, so
    it can be run on serialized output. It enforces the two review corrections:

        * P1: ``metrics.data_range_k`` and ``metrics.value_range_k`` must be present and
          consistent with the fixed physical range; ``value_range_k`` (top-level) too.
        * P2: every frame must carry a ``tiles_url_template`` (with ``{z}/{x}/{y}``) AND a
          ``pmtiles`` path.

    Args:
        d: the manifest as a dict.

    Returns:
        List of human-readable problem strings (empty == valid).
    """
    problems: list[str] = []

    if not isinstance(d, dict):
        return ["manifest is not a dict"]

    # --- required top-level scalar fields -------------------------------------------------
    required_scalars = [
        "version", "scene_id", "title", "satellite", "channel", "wavelength_um",
        "bbox", "crs", "colormap", "value_range_k", "tile_size", "min_zoom", "max_zoom",
        "interpolation_factor", "cadence_minutes_input", "cadence_minutes_output",
        "frames", "videos", "metrics", "crossval", "generated", "model",
    ]
    for key in required_scalars:
        if key not in d:
            problems.append(f"missing required top-level field: '{key}'")

    # --- bbox -----------------------------------------------------------------------------
    bbox = d.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        problems.append("bbox must be a 4-element [west, south, east, north] list")
    elif bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        problems.append("bbox must satisfy west < east and south < north")

    # --- value_range_k (top-level) (P1) ---------------------------------------------------
    vr = d.get("value_range_k")
    if not (isinstance(vr, (list, tuple)) and len(vr) == 2):
        problems.append("value_range_k must be a 2-element [vmin_k, vmax_k] list")
    elif vr[0] >= vr[1]:
        problems.append("value_range_k must satisfy vmin_k < vmax_k")

    # --- zoom / tile_size -----------------------------------------------------------------
    if isinstance(d.get("min_zoom"), int) and isinstance(d.get("max_zoom"), int):
        if d["min_zoom"] > d["max_zoom"]:
            problems.append("min_zoom must be <= max_zoom")
    if "tile_size" in d and not isinstance(d["tile_size"], int):
        problems.append("tile_size must be an integer")

    # --- metrics block (P1) ---------------------------------------------------------------
    metrics = d.get("metrics")
    if isinstance(metrics, dict):
        if "data_range_k" not in metrics:
            problems.append(
                "metrics.data_range_k is REQUIRED (fixed physical Kelvin range; P1) — "
                "full-reference metrics must NOT use per-image min/max"
            )
        else:
            dr = metrics["data_range_k"]
            if not isinstance(dr, (int, float)) or dr <= 0:
                problems.append("metrics.data_range_k must be a positive number (Kelvin)")
            # Consistency with the top-level display range, if both present.
            if isinstance(vr, (list, tuple)) and len(vr) == 2:
                expected = float(vr[1]) - float(vr[0])
                if isinstance(dr, (int, float)) and abs(float(dr) - expected) > 1e-6:
                    problems.append(
                        f"metrics.data_range_k ({dr}) must equal value_range_k span "
                        f"({expected}) — fixed physical range (P1)"
                    )
        if "value_range_k" not in metrics:
            problems.append("metrics.value_range_k is REQUIRED (P1)")
        if "per_frame" not in metrics:
            problems.append("metrics.per_frame is REQUIRED (list, may be empty)")
    elif metrics is not None:
        problems.append("metrics must be a dict")

    # --- crossval block -------------------------------------------------------------------
    crossval = d.get("crossval")
    if isinstance(crossval, dict):
        if "methods_run" not in crossval:
            problems.append("crossval.methods_run is REQUIRED (list, may be empty)")
    elif crossval is not None:
        problems.append("crossval must be a dict")

    # --- videos block ---------------------------------------------------------------------
    videos = d.get("videos")
    if isinstance(videos, dict):
        for vk in ("observed", "interpolated", "side_by_side"):
            if vk not in videos:
                problems.append(f"videos.{vk} is REQUIRED (may be null)")
    elif videos is not None:
        problems.append("videos must be a dict")

    # --- model block ----------------------------------------------------------------------
    model = d.get("model")
    if isinstance(model, dict):
        for mk in ("name", "version", "params_m"):
            if mk not in model:
                problems.append(f"model.{mk} is REQUIRED")
    elif model is not None:
        problems.append("model must be a dict with name/version/params_m")

    # --- frames (P2: per-frame tile sources) ----------------------------------------------
    frames = d.get("frames")
    if not isinstance(frames, list):
        problems.append("frames must be a list")
    else:
        for i, fr in enumerate(frames):
            if not isinstance(fr, dict):
                problems.append(f"frames[{i}] must be a dict")
                continue
            for fk in ("index", "time", "kind", "image", "thumb",
                       "tiles_url_template", "pmtiles", "netcdf"):
                if fk not in fr:
                    problems.append(f"frames[{i}] missing required field '{fk}'")
            # P2: per-frame tile sources REQUIRED + well-formed XYZ template.
            tmpl = fr.get("tiles_url_template")
            if isinstance(tmpl, str):
                if not ("{z}" in tmpl and "{x}" in tmpl and "{y}" in tmpl):
                    problems.append(
                        f"frames[{i}].tiles_url_template must contain '{{z}}', '{{x}}', "
                        f"'{{y}}' (per-frame XYZ source; P2)"
                    )
            if not fr.get("pmtiles"):
                problems.append(
                    f"frames[{i}].pmtiles is REQUIRED — each frame needs its OWN PMTiles "
                    f"so the slider can switch sources per timestamp (P2)"
                )
            # kind / t / bracket coherence.
            kind = fr.get("kind")
            if kind not in ("observed", "interpolated"):
                problems.append(f"frames[{i}].kind must be 'observed' or 'interpolated'")
            elif kind == "interpolated":
                if fr.get("t") is None:
                    problems.append(f"frames[{i}] is interpolated but 't' is null")
                if fr.get("bracket") is None:
                    problems.append(f"frames[{i}] is interpolated but 'bracket' is null")
            # time format.
            tm = fr.get("time")
            if isinstance(tm, str) and not _ISO8601_Z_RE.match(tm):
                problems.append(f"frames[{i}].time '{tm}' is not ISO-8601 UTC (e.g. 2025-06-20T00:10:00Z)")

    return problems


# ---------------------------------------------------------------------------
# small internal helpers
# ---------------------------------------------------------------------------
def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _metrics_to_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    pf = out.get("per_frame", [])
    out["per_frame"] = [m.to_dict() if isinstance(m, MetricRecord) else m for m in pf]
    return out


def _crossval_to_dict(crossval: dict[str, Any]) -> dict[str, Any]:
    out = dict(crossval)
    mr = out.get("methods_run", [])
    out["methods_run"] = [c.to_dict() if isinstance(c, CrossvalMethodResult) else c for c in mr]
    return out


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "GridSpec",
    "default_operational_grid",
    "CubeSchema",
    "empty_cube",
    "Sample",
    "InferenceNetCDFSchema",
    "netcdf_attrs",
    "FrameKind",
    "FrameEntry",
    "MetricRecord",
    "CrossvalMethodResult",
    "Manifest",
    "validate_manifest",
]

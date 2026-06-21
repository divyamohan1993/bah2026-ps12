"""FrameFlow shared constants — REAL, fully-implemented values.

This is the single source of truth for physical and engineering constants used across
the whole monorepo: satellite band specs, the FIXED radiometric metric range (Kelvin),
normalization defaults, the canonical analysis grid, tile/zoom/colormap defaults, and S3
path templates. Everything here is grounded in the deep-research reports in ``research/``
(primarily ``02_satellite_data.md`` for band specs/access, ``03_data_engineering.md`` for
the cube/normalization recipe, ``04_web_viz.md`` for tiles/colormaps, ``05_validation.md``
for the metric data_range rule).

CODE-REVIEW CORRECTION P1 (BAKED IN HERE):
    All FULL-REFERENCE metrics (MSE/RMSE/MAE/PSNR/SSIM/MS-SSIM/FSIM/...) MUST be computed
    against a SHARED, FIXED physical brightness-temperature range in Kelvin
    (:data:`BT_METRIC_VMIN_K`, :data:`BT_METRIC_VMAX_K`, :data:`BT_DATA_RANGE_K`).
    NEVER use per-image min/max as the data_range: per-image scaling hides radiometric
    errors (warm/cold bias, contrast compression) because it re-normalizes each frame to
    its own extremes, so a frame that is uniformly biased or flattened can still score
    well. A fixed physical range makes PSNR/SSIM comparable across frames, satellites, and
    methods, and lets a real radiometric error actually move the metric.
"""

from __future__ import annotations

from typing import Final, TypedDict

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------
PROJECT_NAME: Final[str] = "FrameFlow"
PACKAGE_NAME: Final[str] = "frameflow"
INSTITUTION: Final[str] = "ISRO BAH 2026 PS-12 — FrameFlow Team"
GENERATED_BY: Final[str] = "frameflow"


# ---------------------------------------------------------------------------
# Satellite band catalogue (Thermal-IR ~10 µm "clean window")
# ---------------------------------------------------------------------------
class BandSpec(TypedDict):
    """Specification of one satellite's Thermal-IR clean-window channel.

    Keys:
        channel: human-readable channel/band identifier (e.g. ``"ABI C13"``).
        wavelength_um: central wavelength in micrometres.
        native_res_km: native nadir spatial resolution in km.
        cadence_min: native full-disk acquisition cadence in minutes.
        file_format: on-disk container the source data ships in (``"netcdf4"`` / ``"hdf5"`` / ``"hsd"``).
        access: short note on the fastest programmatic access path.
    """

    channel: str
    wavelength_um: float
    native_res_km: float
    cadence_min: int
    file_format: str
    access: str


# Values pulled from research/02_satellite_data.md (sections 1-7).
SATELLITE_BANDS: Final[dict[str, BandSpec]] = {
    # PRIMARY training source — dense (10-min) exact-match truth (R2 §1).
    "GOES-19": {
        "channel": "ABI C13",
        "wavelength_um": 10.3,
        "native_res_km": 2.0,
        "cadence_min": 10,
        "file_format": "netcdf4",
        "access": "anonymous S3 s3://noaa-goes19/ (ABI-L1b-RadF C13 or ABI-L2-CMIPF CMI, Kelvin-ready)",
    },
    "GOES-18": {
        "channel": "ABI C13",
        "wavelength_um": 10.3,
        "native_res_km": 2.0,
        "cadence_min": 10,
        "file_format": "netcdf4",
        "access": "anonymous S3 s3://noaa-goes18/ (GOES-West, second independent geometry)",
    },
    # SECONDARY training / Asia-Pacific; overlaps INSAT over the Indian Ocean (R2 §2).
    "Himawari-9": {
        "channel": "AHI B13",
        "wavelength_um": 10.4,
        "native_res_km": 2.0,
        "cadence_min": 10,
        "file_format": "netcdf4",
        "access": "anonymous S3 s3://noaa-himawari9/AHI-L1b-FLDK/ (HSD via satpy) or JAXA P-Tree gridded NetCDF",
    },
    # GEO cross-validation — Korea/Asia (R2 §6).
    "GK-2A": {
        "channel": "AMI IR105",
        "wavelength_um": 10.5,
        "native_res_km": 2.0,
        "cadence_min": 10,
        "file_format": "netcdf4",
        "access": "anonymous S3 s3://noaa-gk2a-pds/ (gk2a_ami_le1b_ir105_fd020ge_*.nc)",
    },
    # DEPLOYMENT TARGET (R2 §3). INSAT-3DS is the operational target of PS-12.
    "INSAT-3DS": {
        "channel": "Imager TIR1",
        "wavelength_um": 10.8,
        "native_res_km": 4.0,
        "cadence_min": 30,
        "file_format": "hdf5",
        "access": "MOSDAC mdapi (datasetId 3SIMG_L1B_STD / 3SIMG_L1C_SGP) or EUMETSAT eumdac EO:EUM:DAT:INSAT:INSAT3D-L1C",
    },
    "INSAT-3DR": {
        "channel": "Imager TIR1",
        "wavelength_um": 10.8,
        "native_res_km": 4.0,
        "cadence_min": 30,
        "file_format": "hdf5",
        "access": "MOSDAC mdapi (datasetId 3RIMG_L1B_STD / 3RIMG_L1C_SGP)",
    },
    # GEO cross-validation — Africa/Europe/Indian Ocean; IODC overlaps INSAT (R2 §4).
    "Meteosat": {
        "channel": "SEVIRI IR10.8 / FCI IR10.5",
        "wavelength_um": 10.8,
        "native_res_km": 3.0,
        "cadence_min": 15,
        "file_format": "netcdf4",
        "access": "EUMETSAT eumdac (EO:EUM:DAT:MSG:HRSEVIRI, -IODC at 45.5E; MTG FCI L1C)",
    },
    # GEO cross-validation — East/Central Asia; overlaps Himawari & INSAT (R2 §5).
    "FY-4B": {
        "channel": "AGRI ~10.8 µm window",
        "wavelength_um": 10.8,
        "native_res_km": 4.0,
        "cadence_min": 15,
        "file_format": "hdf5",
        "access": "NSMC FengYun Cloud data.nsmc.org.cn (AGRI L1, satpy agri_fy4b_l1)",
    },
    # GEO cross-validation — Indian Ocean (76E, co-located with INSAT-3DR) (R2 §7).
    "Electro-L": {
        "channel": "MSU-GS IR ~10.7-11.5 µm",
        "wavelength_um": 11.0,
        "native_res_km": 4.0,
        "cadence_min": 30,
        "file_format": "hrit",
        "access": "Roscosmos/Roshydromet NTs OMZ (ntsomz.gptl.ru) — HRIT/LRIT (verify portal/credentials)",
    },
}

#: The default satellite the demo / synthetic pipeline emulates (dense 10-min IR).
DEFAULT_SATELLITE: Final[str] = "GOES-19"


# ---------------------------------------------------------------------------
# Brightness-temperature metric constants (CODE-REVIEW CORRECTION P1)
# ---------------------------------------------------------------------------
#: Fixed lower bound (Kelvin) of the physical BT range used for ALL full-reference metrics.
#: ~180 K comfortably covers the coldest convective/overshooting cloud tops in TIR.
BT_METRIC_VMIN_K: Final[float] = 180.0

#: Fixed upper bound (Kelvin) of the physical BT range used for ALL full-reference metrics.
#: ~320 K comfortably covers the warmest clear-sky land/ocean surfaces in TIR.
BT_METRIC_VMAX_K: Final[float] = 320.0

#: The SHARED, FIXED ``data_range`` (Kelvin) every full-reference metric MUST pass.
#: This is ``BT_METRIC_VMAX_K - BT_METRIC_VMIN_K``. Do NOT use per-image min/max — that
#: hides radiometric (warm/cold-bias, contrast) errors. See module docstring (P1).
BT_DATA_RANGE_K: Final[float] = BT_METRIC_VMAX_K - BT_METRIC_VMIN_K  # 140.0 K


# ---------------------------------------------------------------------------
# Normalization defaults for standardizing model INPUT
# ---------------------------------------------------------------------------
# Two interchangeable normalization conventions are supported; both are deterministic and
# stored alongside the cube so inference matches training exactly (R3 §5.2, R1 §7.2):
#
#   1. Fixed physical min-max  ->  x' = (BT - BT_NORM_VMIN_K) / (BT_NORM_VMAX_K - BT_NORM_VMIN_K)
#      Reproducible across GOES/Himawari/INSAT; makes cross-sensor transfer cleaner.
#   2. Per-dataset z-score      ->  x' = (BT - mean) / std
#      The classic standardization used by Vandal & Nemani (R1 §1) and the Nature 3D-U-Net
#      IR nowcasting paper (R3 §5.2).
#
# The per-dataset mean/std below are DATA-DERIVED PLACEHOLDERS (physically plausible round
# numbers). The real statistics MUST be computed once over the training split and persisted
# to a stats JSON (see STATS_JSON_FILENAME); inference then reads that file, never these.

#: Fixed-range normalization bounds (Kelvin) for the [0,1] model-input convention.
BT_NORM_VMIN_K: Final[float] = 180.0
BT_NORM_VMAX_K: Final[float] = 330.0

#: Per-dataset z-score placeholders (Kelvin). REPLACE with values from a real stats JSON.
GOES_C13_MEAN_K: Final[float] = 270.0
GOES_C13_STD_K: Final[float] = 22.0
HIMAWARI_B13_MEAN_K: Final[float] = 270.0
HIMAWARI_B13_STD_K: Final[float] = 22.0
INSAT_TIR1_MEAN_K: Final[float] = 268.0
INSAT_TIR1_STD_K: Final[float] = 23.0

#: Filename (relative to a cube/run dir) where REAL per-dataset normalization statistics
#: live. Computed once from data; read at train/infer time. Schema: ``{"<dataset>": {"mean_k":
#: float, "std_k": float, "vmin_k": float, "vmax_k": float, "n_samples": int}}``.
STATS_JSON_FILENAME: Final[str] = "norm_stats.json"


# ---------------------------------------------------------------------------
# Default analysis grid (regional lat/lon over the Indian domain by default)
# ---------------------------------------------------------------------------
# A modest default that co-registers GOES/Himawari/INSAT into one lat/lon `AreaDefinition`
# (R3 §4). The Indian subcontinent bbox (~70-90E, 8-28N) matches MOSDAC examples (R2 §3).
# Synthetic/demo data uses a smaller, faster default (see GridSpec.default in contracts).
DEFAULT_GRID_BBOX: Final[tuple[float, float, float, float]] = (68.0, 6.0, 98.0, 38.0)  # W,S,E,N
DEFAULT_GRID_CRS: Final[str] = "EPSG:4326"
DEFAULT_GRID_RESOLUTION_DEG: Final[float] = 0.04  # ~4 km, matching INSAT TIR1 nadir res


# ---------------------------------------------------------------------------
# Canonical Zarr analysis-cube layout (R3 §1, §8.1)
# ---------------------------------------------------------------------------
CUBE_DIMS: Final[tuple[str, str, str]] = ("time", "y", "x")
CUBE_DATA_VAR: Final[str] = "bt"  # brightness temperature, float32 Kelvin, NaN-filled
CUBE_DTYPE: Final[str] = "float32"
CUBE_CHUNKS: Final[tuple[int, int, int]] = (8, 512, 512)
CUBE_SHARDS: Final[tuple[int, int, int]] = (32, 1024, 1024)
CUBE_COMPRESSOR: Final[str] = "blosc-zstd-shuffle"
CUBE_FILL_VALUE_NAN: Final[bool] = True


# ---------------------------------------------------------------------------
# Web / tiling / colormap defaults (R4)
# ---------------------------------------------------------------------------
DEFAULT_TILE_SIZE: Final[int] = 256
DEFAULT_MIN_ZOOM: Final[int] = 0
DEFAULT_MAX_ZOOM: Final[int] = 8
DEFAULT_TILE_FORMAT: Final[str] = "webp"

#: Perceptually-reasonable colormaps for TIR brightness temperature. ``ir_clouds`` is the
#: enhanced IR cloud ramp (cold tops bright/saturated), ``greyscale_ir`` the classic
#: inverted-grey IR, ``turbo`` a perceptually-uniform rainbow (R4 §7).
COLORMAP_NAMES: Final[tuple[str, ...]] = ("ir_clouds", "greyscale_ir", "turbo")
DEFAULT_COLORMAP: Final[str] = "ir_clouds"

#: Reference IR colormap definition: anchor stops as ``(bt_kelvin, "#rrggbb")`` spanning the
#: metric range. This is a compact, deterministic ramp the viz/precompute team can hand to
#: matplotlib's ``LinearSegmentedColormap`` (cold cloud tops -> bright; warm surface -> dark).
IR_CLOUDS_COLORMAP_STOPS: Final[tuple[tuple[float, str], ...]] = (
    (180.0, "#ffffff"),  # coldest overshooting tops — white
    (200.0, "#ff64ff"),  # deep convection — magenta
    (220.0, "#ff0000"),  # cold cloud — red
    (240.0, "#ffa500"),  # mid cloud — orange
    (260.0, "#ffff00"),  # low cloud — yellow
    (275.0, "#00c800"),  # cool surface — green
    (290.0, "#0a3d91"),  # warm surface — deep blue
    (320.0, "#000000"),  # warmest surface — black
)


# ---------------------------------------------------------------------------
# Temporal-resolution targets (PS-12: 30 -> 15 -> 7.5 min) (idea.md ll. 14, 45)
# ---------------------------------------------------------------------------
#: Default native input cadence (minutes) for the deployment target (INSAT).
DEFAULT_INPUT_CADENCE_MIN: Final[int] = 30
#: Default interpolation factors to produce (2x -> 15 min, 4x -> 7.5 min).
DEFAULT_INTERP_FACTORS: Final[tuple[int, ...]] = (2, 4)


# ---------------------------------------------------------------------------
# S3 / object-store path templates for the main public sources (R2 §1, §2, §6, §9)
# ---------------------------------------------------------------------------
# Format with e.g. ``GOES19_RADF_TEMPLATE.format(year=2025, doy=171, hour=14)``.
GOES19_RADF_TEMPLATE: Final[str] = "s3://noaa-goes19/ABI-L1b-RadF/{year}/{doy:03d}/{hour:02d}/"
GOES19_CMIPF_TEMPLATE: Final[str] = "s3://noaa-goes19/ABI-L2-CMIPF/{year}/{doy:03d}/{hour:02d}/"
GOES18_RADF_TEMPLATE: Final[str] = "s3://noaa-goes18/ABI-L1b-RadF/{year}/{doy:03d}/{hour:02d}/"
# Himawari AWS path is Y/M/D/HHMM (not day-of-year) (R2 §2).
HIMAWARI9_FLDK_TEMPLATE: Final[str] = (
    "s3://noaa-himawari9/AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/"
)
GK2A_FD_TEMPLATE: Final[str] = "s3://noaa-gk2a-pds/AMI/L1B/FD/{year}{month:02d}/{day:02d}/{hour:02d}/"
# NOAA Global Mosaic of Geostationary Satellite Imagery (longwave IR) — global cross-val (R2 §9).
GMGSI_LW_TEMPLATE: Final[str] = "s3://noaa-gmgsi-pds/GMGSI_LW/{year}/{month:02d}/{day:02d}/{hour:02d}/"
# INSAT has no public S3 bucket; access is via MOSDAC mdapi or EUMETSAT eumdac (R2 §3).
INSAT3DS_MOSDAC_DATASET_ID: Final[str] = "3SIMG_L1B_STD"
INSAT3DR_MOSDAC_DATASET_ID: Final[str] = "3RIMG_L1B_STD"
INSAT_EUMETSAT_COLLECTION: Final[str] = "EO:EUM:DAT:INSAT:INSAT3D-L1C"


__all__ = [
    "PROJECT_NAME",
    "PACKAGE_NAME",
    "INSTITUTION",
    "GENERATED_BY",
    "BandSpec",
    "SATELLITE_BANDS",
    "DEFAULT_SATELLITE",
    "BT_METRIC_VMIN_K",
    "BT_METRIC_VMAX_K",
    "BT_DATA_RANGE_K",
    "BT_NORM_VMIN_K",
    "BT_NORM_VMAX_K",
    "GOES_C13_MEAN_K",
    "GOES_C13_STD_K",
    "HIMAWARI_B13_MEAN_K",
    "HIMAWARI_B13_STD_K",
    "INSAT_TIR1_MEAN_K",
    "INSAT_TIR1_STD_K",
    "STATS_JSON_FILENAME",
    "DEFAULT_GRID_BBOX",
    "DEFAULT_GRID_CRS",
    "DEFAULT_GRID_RESOLUTION_DEG",
    "CUBE_DIMS",
    "CUBE_DATA_VAR",
    "CUBE_DTYPE",
    "CUBE_CHUNKS",
    "CUBE_SHARDS",
    "CUBE_COMPRESSOR",
    "CUBE_FILL_VALUE_NAN",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_MIN_ZOOM",
    "DEFAULT_MAX_ZOOM",
    "DEFAULT_TILE_FORMAT",
    "COLORMAP_NAMES",
    "DEFAULT_COLORMAP",
    "IR_CLOUDS_COLORMAP_STOPS",
    "DEFAULT_INPUT_CADENCE_MIN",
    "DEFAULT_INTERP_FACTORS",
    "GOES19_RADF_TEMPLATE",
    "GOES19_CMIPF_TEMPLATE",
    "GOES18_RADF_TEMPLATE",
    "HIMAWARI9_FLDK_TEMPLATE",
    "GK2A_FD_TEMPLATE",
    "GMGSI_LW_TEMPLATE",
    "INSAT3DS_MOSDAC_DATASET_ID",
    "INSAT3DR_MOSDAC_DATASET_ID",
    "INSAT_EUMETSAT_COLLECTION",
]

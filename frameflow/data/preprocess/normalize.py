"""Brightness-temperature normalization for MODEL INPUT (distinct from the metric range).

CRITICAL distinction (research/03 §5.2, constants module docstring):

* The **metric** ``data_range`` is the FIXED physical Kelvin span ``BT_DATA_RANGE_K`` (140 K
  over 180..320 K) used by *all full-reference metrics*. It is **never** touched here and is
  **never** per-image.
* This module implements **model-input standardization** only — mapping Kelvin to a small,
  zero-centred or [0,1] range so the network trains stably. Two interchangeable, fully
  deterministic conventions are supported, both persisted alongside the cube so inference
  matches training exactly:

  1. ``fixed_range`` — ``x = (BT - vmin) / (vmax - vmin)`` with the constants
     ``BT_NORM_VMIN_K`` / ``BT_NORM_VMAX_K`` (180..330 K). Reproducible across
     GOES/Himawari/INSAT; the cleanest choice for cross-sensor transfer.
  2. ``zscore`` — ``x = (BT - mean) / std`` with the *real* per-dataset statistics computed
     once over the training split (:func:`compute_stats`) and written to ``norm_stats.json``
     (:data:`frameflow.constants.STATS_JSON_FILENAME`). The placeholder means/stds in
     ``constants`` are only fallbacks.

All functions are NaN-safe: NaN (space) pixels stay NaN through normalize/denormalize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import constants as C

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import xarray as xr


__all__ = [
    "normalize_bt",
    "denormalize_bt",
    "compute_stats",
    "load_stats",
    "dataset_stats",
]


# Per-dataset z-score fallbacks (Kelvin) from constants, keyed by short dataset name.
_ZSCORE_FALLBACK: dict[str, tuple[float, float]] = {
    "goes": (C.GOES_C13_MEAN_K, C.GOES_C13_STD_K),
    "goes19": (C.GOES_C13_MEAN_K, C.GOES_C13_STD_K),
    "goes18": (C.GOES_C13_MEAN_K, C.GOES_C13_STD_K),
    "himawari": (C.HIMAWARI_B13_MEAN_K, C.HIMAWARI_B13_STD_K),
    "himawari9": (C.HIMAWARI_B13_MEAN_K, C.HIMAWARI_B13_STD_K),
    "gk2a": (C.HIMAWARI_B13_MEAN_K, C.HIMAWARI_B13_STD_K),
    "insat": (C.INSAT_TIR1_MEAN_K, C.INSAT_TIR1_STD_K),
    "insat3ds": (C.INSAT_TIR1_MEAN_K, C.INSAT_TIR1_STD_K),
    "insat3dr": (C.INSAT_TIR1_MEAN_K, C.INSAT_TIR1_STD_K),
    "synthetic": (C.GOES_C13_MEAN_K, C.GOES_C13_STD_K),
}


def dataset_stats(dataset: str, stats: dict[str, Any] | None = None) -> tuple[float, float]:
    """Resolve ``(mean_k, std_k)`` for a dataset from a stats dict, else constants fallback.

    Args:
        dataset: short dataset key (e.g. ``"goes"``, ``"insat3ds"``).
        stats: optional loaded stats dict (schema from :func:`compute_stats`).

    Returns:
        ``(mean_k, std_k)`` in Kelvin.
    """
    key = dataset.lower()
    if stats and key in stats:
        rec = stats[key]
        return float(rec["mean_k"]), float(rec["std_k"])
    if stats and dataset in stats:
        rec = stats[dataset]
        return float(rec["mean_k"]), float(rec["std_k"])
    return _ZSCORE_FALLBACK.get(key, (C.GOES_C13_MEAN_K, C.GOES_C13_STD_K))


def normalize_bt(
    bt: "np.ndarray | Any",
    dataset: str = "goes",
    *,
    mode: str = "fixed_range",
    vmin_k: float = C.BT_NORM_VMIN_K,
    vmax_k: float = C.BT_NORM_VMAX_K,
    stats: dict[str, Any] | None = None,
) -> "np.ndarray":
    """Normalize brightness temperature (Kelvin) to a model-input range.

    Args:
        bt: brightness-temperature array in Kelvin (NaNs preserved).
        dataset: dataset key used to pick z-score stats (``mode="zscore"`` only).
        mode: ``"fixed_range"`` (-> [0,1] via vmin/vmax) or ``"zscore"`` (-> standardized).
        vmin_k, vmax_k: fixed-range bounds in Kelvin (``mode="fixed_range"``).
        stats: optional loaded stats dict for the z-score mode.

    Returns:
        Normalized ``numpy.ndarray`` (float32), NaN where input was NaN.

    Raises:
        ValueError: on an unknown ``mode`` or a degenerate range/std.
    """
    import numpy as np  # lazy

    arr = np.asarray(bt, dtype=np.float32)
    if mode == "fixed_range":
        span = float(vmax_k) - float(vmin_k)
        if span <= 0:
            raise ValueError("normalize_bt: vmax_k must be > vmin_k")
        return ((arr - np.float32(vmin_k)) / np.float32(span)).astype(np.float32)
    if mode == "zscore":
        mean_k, std_k = dataset_stats(dataset, stats)
        if std_k <= 0:
            raise ValueError("normalize_bt: std must be > 0 for zscore mode")
        return ((arr - np.float32(mean_k)) / np.float32(std_k)).astype(np.float32)
    raise ValueError(f"normalize_bt: unknown mode {mode!r} (expected 'fixed_range' or 'zscore')")


def denormalize_bt(
    x: "np.ndarray | Any",
    dataset: str = "goes",
    *,
    mode: str = "fixed_range",
    vmin_k: float = C.BT_NORM_VMIN_K,
    vmax_k: float = C.BT_NORM_VMAX_K,
    stats: dict[str, Any] | None = None,
) -> "np.ndarray":
    """Invert :func:`normalize_bt`, returning brightness temperature in Kelvin.

    Args:
        x: normalized array (NaNs preserved).
        dataset: dataset key for z-score stats (``mode="zscore"`` only).
        mode: must match the mode used to normalize.
        vmin_k, vmax_k: fixed-range bounds in Kelvin.
        stats: optional loaded stats dict for the z-score mode.

    Returns:
        Brightness temperature in Kelvin (float32), NaN where input was NaN.

    Raises:
        ValueError: on an unknown ``mode``.
    """
    import numpy as np  # lazy

    arr = np.asarray(x, dtype=np.float32)
    if mode == "fixed_range":
        span = float(vmax_k) - float(vmin_k)
        return (arr * np.float32(span) + np.float32(vmin_k)).astype(np.float32)
    if mode == "zscore":
        mean_k, std_k = dataset_stats(dataset, stats)
        return (arr * np.float32(std_k) + np.float32(mean_k)).astype(np.float32)
    raise ValueError(f"denormalize_bt: unknown mode {mode!r} (expected 'fixed_range' or 'zscore')")


def compute_stats(
    cube: "xr.Dataset | xr.DataArray | np.ndarray | Any",
    *,
    dataset: str = "goes",
    out_path: str | Path | None = None,
    var: str = C.CUBE_DATA_VAR,
) -> dict[str, Any]:
    """Compute real per-dataset normalization statistics over a cube and (optionally) save.

    Computes NaN-aware ``mean_k``/``std_k`` plus the observed ``vmin_k``/``vmax_k`` and the
    valid-sample count, in the schema documented in :mod:`frameflow.constants`
    (``{"<dataset>": {"mean_k", "std_k", "vmin_k", "vmax_k", "n_samples"}}``). When
    ``out_path`` is given the dict is merged into any existing stats JSON and written back,
    so multiple datasets accumulate into one ``norm_stats.json``.

    Args:
        cube: a Zarr-backed :class:`xarray.Dataset`/``DataArray`` or a raw array of BT (K).
        dataset: short dataset key under which to store the stats.
        out_path: optional path to write/merge the stats JSON.
        var: data-variable name when ``cube`` is a Dataset.

    Returns:
        The stats dict ``{dataset: {...}}`` that was computed (and persisted if requested).
    """
    import numpy as np  # lazy

    values = _as_values(cube, var)
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    n = int(finite.sum())
    if n == 0:
        mean_k = std_k = vmin = vmax = float("nan")
    else:
        vals = arr[finite]
        mean_k = float(vals.mean())
        std_k = float(vals.std())
        vmin = float(vals.min())
        vmax = float(vals.max())

    record = {
        "mean_k": mean_k,
        "std_k": std_k,
        "vmin_k": vmin,
        "vmax_k": vmax,
        "n_samples": n,
    }
    stats = {dataset: record}

    if out_path is not None:
        path = Path(out_path)
        merged: dict[str, Any] = {}
        if path.exists():
            try:
                merged = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                merged = {}
        merged[dataset] = record
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=2, sort_keys=True))
        return merged
    return stats


def load_stats(path: str | Path) -> dict[str, Any]:
    """Load a normalization stats JSON (returns ``{}`` if the file is missing/unreadable).

    Args:
        path: path to a ``norm_stats.json`` file.

    Returns:
        The parsed stats dict (empty on any error).
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _as_values(cube: "Any", var: str) -> "Any":
    """Extract the numeric array of BT values from a Dataset/DataArray/ndarray input."""
    if hasattr(cube, "data_vars"):  # xarray.Dataset
        return cube[var].values
    if hasattr(cube, "values"):  # xarray.DataArray
        return cube.values
    return cube

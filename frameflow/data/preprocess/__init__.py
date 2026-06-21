"""FrameFlow data preprocessing — radiance->BT, regridding, normalization, patches.

This subpackage holds the offline, do-it-once preprocessing steps that turn raw sensor
arrays into analysis-ready brightness temperature on the common grid (research/03 §4-5):

* :mod:`.radiance` — Planck inverse (GOES/Himawari/GK-2A) and count->BT LUT (INSAT).
* :mod:`.regrid`   — cached pyresample KDTree regridder (+ scipy/xarray fallback).
* :mod:`.normalize`— model-input standardization (fixed-range / z-score) + real stats.
* :mod:`.patches`  — patch extraction / random crops for sample preparation.

It also re-exports a flat API matching ``CONTRACTS.md §3.3`` so callers can use either the
granular module functions or the contract's ``frameflow.data.preprocess.<fn>`` names:

    radiance_to_bt(da, **coeffs)           # DataArray-in / DataArray-out Planck inverse
    regrid_to_grid(da, grid, *, cache_dir) # DataArray -> common GridSpec
    normalize(bt, *, mode, vmin_k, vmax_k) # Kelvin -> [0,1] (or z-score)
    denormalize(x, *, mode, vmin_k, vmax_k)# inverse
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ... import constants as C
from .normalize import (
    compute_stats,
    denormalize_bt,
    load_stats,
    normalize_bt,
)
from .patches import extract_patches, random_crop, random_crop_stack
from .radiance import count_to_bt, radiance_to_bt, radiance_to_bt_dataarray
from .regrid import Regridder, make_regridder, regrid, regrid_to_grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import xarray as xr

    from ...contracts import GridSpec


def normalize(
    bt: "np.ndarray | Any",
    *,
    mode: str = "fixed_range",
    vmin_k: float = C.BT_NORM_VMIN_K,
    vmax_k: float = C.BT_NORM_VMAX_K,
    dataset: str = "goes",
) -> "np.ndarray":
    """Flat-contract alias of :func:`normalize_bt` (CONTRACTS.md §3.3).

    Maps brightness temperature (Kelvin) to the model-input range. See
    :func:`frameflow.data.preprocess.normalize.normalize_bt` for details.
    """
    return normalize_bt(bt, dataset=dataset, mode=mode, vmin_k=vmin_k, vmax_k=vmax_k)


def denormalize(
    x: "np.ndarray | Any",
    *,
    mode: str = "fixed_range",
    vmin_k: float = C.BT_NORM_VMIN_K,
    vmax_k: float = C.BT_NORM_VMAX_K,
    dataset: str = "goes",
) -> "np.ndarray":
    """Flat-contract alias of :func:`denormalize_bt` (CONTRACTS.md §3.3) -> Kelvin."""
    return denormalize_bt(x, dataset=dataset, mode=mode, vmin_k=vmin_k, vmax_k=vmax_k)


def radiance_to_bt_da(da: "xr.DataArray", **coeffs: float) -> "xr.DataArray":
    """Flat-contract alias: ``radiance_to_bt(da, **coeffs)`` on a DataArray (CONTRACTS §3.3).

    Note the granular array-level Planck inverse is exported as :func:`radiance_to_bt`; this
    DataArray-level wrapper is what the flat contract's ``preprocess.radiance_to_bt(da, ...)``
    signature describes. It is also exported under the name :func:`radiance_to_bt_dataarray`.
    """
    return radiance_to_bt_dataarray(da, **coeffs)


__all__ = [
    # radiance
    "radiance_to_bt",
    "count_to_bt",
    "radiance_to_bt_dataarray",
    "radiance_to_bt_da",
    # regrid
    "Regridder",
    "make_regridder",
    "regrid",
    "regrid_to_grid",
    # normalize
    "normalize_bt",
    "denormalize_bt",
    "normalize",
    "denormalize",
    "compute_stats",
    "load_stats",
    # patches
    "extract_patches",
    "random_crop",
    "random_crop_stack",
]

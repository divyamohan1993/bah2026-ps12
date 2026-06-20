"""FrameFlow Team DATA — raw ``.nc``/``.h5`` -> analysis-ready Zarr cube + training samples.

This package owns the whole data pipeline (CONTRACTS.md §3):

* :mod:`frameflow.data.access`     — per-satellite access drivers (GOES/Himawari/GK-2A/
  INSAT) that list, download, and open one source frame as a brightness-temperature
  :class:`xarray.DataArray` in Kelvin. Network access is always lazy with a clear
  :class:`~frameflow.data.access.base.OfflineError` when offline.
* :mod:`frameflow.data.preprocess` — radiance/count -> BT (Planck / LUT), cached pyresample
  regridding onto the common :class:`~frameflow.contracts.GridSpec`, model-input
  normalization (fixed-range / z-score), and patch extraction.
* :mod:`frameflow.data.ingest`     — assemble the canonical :class:`~frameflow.contracts.
  CubeSchema` Zarr v3 cube (:func:`~frameflow.data.ingest.cube.build_cube`) and build
  zero-copy kerchunk/VirtualiZarr virtual references over the originals.
* :mod:`frameflow.data.datasets`   — a torch-compatible
  :class:`~frameflow.data.datasets.triplet.TripletDataset` that yields
  :class:`~frameflow.contracts.Sample` triplets, plus sharded sample writers.

Nothing heavy is imported at package-import time: submodules import numpy/xarray/zarr/torch
lazily inside their functions, so ``import frameflow.data`` stays cheap and side-effect free.
The package is deliberately self-contained — it never imports the sibling ``models`` /
``train`` / ``infer`` / ``validate`` / ``serve`` / ``viz`` areas.
"""

from __future__ import annotations

__all__: list[str] = [
    "access",
    "preprocess",
    "ingest",
    "datasets",
]

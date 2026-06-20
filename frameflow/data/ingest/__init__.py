"""Ingest: build the canonical Zarr cube + zero-copy virtual references (CONTRACTS.md §3.2).

* :func:`build_cube` (from :mod:`.cube`) — read source ``.nc``/``.h5`` frames, convert to BT
  (Kelvin), regrid onto a common :class:`~frameflow.contracts.GridSpec` with a cached
  pyresample kernel, and write a schema-correct :class:`~frameflow.contracts.CubeSchema`
  Zarr v3 cube. Works on the synthetic ``.nc`` triplet from :mod:`frameflow.synthetic`.
* :func:`build_catalog` — write a Parquet (CSV fallback) catalog of frame timestamps and the
  ``(i-1, i, i+1)`` training triplets, for the dataset layer to index.
* virtual references (from :mod:`.virtual`) — :func:`virtual_reference` /
  :func:`virtual_references` build kerchunk/VirtualiZarr zero-copy refs over the originals;
  :func:`open_virtual` opens them lazily; :func:`write_references` persists them.

Everything is import-light: numpy/xarray/zarr/kerchunk are imported lazily inside functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import constants as C
from .cube import build_cube, read_bt_frame
from .virtual import (
    open_virtual,
    virtual_reference,
    virtual_references,
    write_references,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


__all__ = [
    "build_cube",
    "read_bt_frame",
    "build_catalog",
    "virtual_reference",
    "virtual_references",
    "write_references",
    "open_virtual",
]


def build_catalog(
    cube_path: str | Path,
    out_parquet: str | Path,
    *,
    cadence_min: int = C.DEFAULT_INPUT_CADENCE_MIN,
) -> Path:
    """Write a catalog of frame timestamps + ``(i-1, i, i+1)`` training triplets.

    Reads the cube's ``time`` coordinate and emits one row per interior frame describing the
    leave-the-middle-out triplet ``(prev=i-1, mid=i, next=i+1)`` with the bracket cadence —
    exactly the VFI training sampling "frames at 00:00 and 00:20 predict 00:10" (research/03
    §5.4). Written as Parquet when a Parquet engine (pyarrow/fastparquet) is available, else
    as CSV (same columns) so the call never hard-fails on a missing optional dependency.

    Args:
        cube_path: path to a Zarr cube (CubeSchema) to index.
        out_parquet: destination ``.parquet`` path (a ``.csv`` sibling is written if no
            Parquet engine is installed).
        cadence_min: nominal minutes between consecutive frames (recorded per row).

    Returns:
        The :class:`pathlib.Path` actually written (``.parquet`` or the ``.csv`` fallback).
    """
    import numpy as np  # lazy
    import pandas as pd  # lazy
    import xarray as xr  # lazy

    ds = xr.open_zarr(str(cube_path), consolidated=False)
    try:
        times = np.asarray(ds["time"].values)
    finally:
        ds.close()

    rows: list[dict[str, Any]] = []
    for i in range(1, len(times) - 1):
        rows.append(
            {
                "mid_index": i,
                "prev_index": i - 1,
                "next_index": i + 1,
                "prev_time": str(times[i - 1]),
                "mid_time": str(times[i]),
                "next_time": str(times[i + 1]),
                "t": 0.5,
                "cadence_min": int(cadence_min),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "mid_index", "prev_index", "next_index",
            "prev_time", "mid_time", "next_time", "t", "cadence_min",
        ],
    )

    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
        return out
    except Exception:
        # No Parquet engine installed — fall back to CSV with the same schema.
        csv_out = out.with_suffix(".csv")
        df.to_csv(csv_out, index=False)
        return csv_out

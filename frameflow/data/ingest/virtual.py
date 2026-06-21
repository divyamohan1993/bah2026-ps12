"""Zero-copy virtual Zarr references over the original ``.nc``/``.h5`` (kerchunk).

This is the single most important *ingest* technique for PS-12 (research/03 §0, §1.2): the
source archives (NOAA GOES on S3, MOSDAC INSAT) are already NetCDF4/HDF5, and every such
file is internally chunked. **kerchunk** scans a file once and records, for each internal
chunk, its byte ``(offset, length)`` and codec inside the *untouched* original. That metadata
is presented to Zarr/xarray as a *virtual store*, so reading a chunk later is a single ranged
GET into the original file — **no copy, no translation**, address precomputed -> O(1).

Public surface:

* :func:`virtual_reference`     — build a kerchunk reference set for ONE ``.nc``/``.h5``.
* :func:`virtual_references`    — build + combine references for a collection along ``time``.
* :func:`write_references`      — persist a reference set to JSON (or Parquet for big sets).
* :func:`open_virtual`          — open a reference set (dict or file) as an
  :class:`xarray.Dataset` (lazy, O(1) per chunk).

The modern, Zarr-native successor is **VirtualiZarr** (``open_virtual_mfdataset`` →
``.vz.to_kerchunk(...)`` / ``.vz.to_icechunk(...)``); when it is installed it is used for the
multi-file combine, otherwise we fall back to kerchunk's :class:`MultiZarrToZarr`. Either way
the on-disk artifact is a kerchunk reference set that :func:`open_virtual` can read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr


__all__ = [
    "virtual_reference",
    "virtual_references",
    "write_references",
    "open_virtual",
]


def _require_kerchunk() -> Any:
    """Import kerchunk's HDF5 translator with a clear error if it is missing.

    Returns:
        The :class:`kerchunk.hdf.SingleHdf5ToZarr` class.

    Raises:
        RuntimeError: if kerchunk is not importable.
    """
    try:
        from kerchunk.hdf import SingleHdf5ToZarr  # lazy
    except ImportError as exc:  # pragma: no cover - kerchunk is installed in this env
        raise RuntimeError(
            "Virtual (zero-copy) references need `kerchunk` (and `h5py`/`fsspec`). "
            "Install kerchunk, or use frameflow.data.ingest.cube.build_cube to materialize a "
            "Zarr cube instead."
        ) from exc
    return SingleHdf5ToZarr


def virtual_reference(
    path: str | Path,
    *,
    inline_threshold: int = 5000,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a kerchunk reference set for a single ``.nc``/``.h5`` file (zero copy).

    Args:
        path: a local path or ``s3://`` URI to one NetCDF4/HDF5 frame.
        inline_threshold: inline (base64) chunks smaller than this many bytes directly into
            the reference set so tiny coordinate arrays need no extra GET.
        storage_options: optional fsspec storage options for remote files (e.g.
            ``{"anon": True}`` for a public S3 bucket).

    Returns:
        The kerchunk reference dict (``{"version", "refs", ...}``), suitable for
        :func:`write_references` or :func:`open_virtual`.
    """
    SingleHdf5ToZarr = _require_kerchunk()
    p = str(path)
    so = dict(storage_options or {})
    if p.startswith("s3://"):
        import fsspec  # lazy

        so.setdefault("anon", True)
        with fsspec.open(p, **so) as f:
            return SingleHdf5ToZarr(f, url=p, inline_threshold=inline_threshold).translate()
    return SingleHdf5ToZarr(p, inline_threshold=inline_threshold).translate()


def virtual_references(
    files: Sequence[str | Path],
    *,
    concat_dim: str = "time",
    inline_threshold: int = 5000,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and combine virtual references for a file collection along ``concat_dim``.

    Each file is referenced once (no copy); the per-file reference sets are then stitched into
    one virtual dataset concatenated along ``concat_dim`` (default ``"time"``). Prefers
    VirtualiZarr when available, else uses kerchunk's :class:`MultiZarrToZarr`.

    Args:
        files: ordered source frame paths/URIs.
        concat_dim: dimension to concatenate the frames along.
        inline_threshold: small-chunk inlining threshold (bytes).
        storage_options: optional fsspec storage options for remote files.

    Returns:
        A combined kerchunk reference dict spanning all ``files``.

    Raises:
        ValueError: if ``files`` is empty.
    """
    file_list = [str(f) for f in files]
    if not file_list:
        raise ValueError("virtual_references: `files` is empty.")
    if len(file_list) == 1:
        return virtual_reference(
            file_list[0], inline_threshold=inline_threshold, storage_options=storage_options
        )

    # Preferred modern path: VirtualiZarr (Zarr-native, xarray-friendly).
    try:  # pragma: no cover - VirtualiZarr not installed in this env
        return _combine_virtualizarr(file_list, concat_dim, storage_options)
    except Exception:
        pass

    # Fallback: kerchunk MultiZarrToZarr over per-file single references.
    from kerchunk.combine import MultiZarrToZarr  # lazy

    singles = [
        virtual_reference(
            f, inline_threshold=inline_threshold, storage_options=storage_options
        )
        for f in file_list
    ]
    # For a CF time axis the cleanest concat reads the real ``time`` values from each file
    # (``cf:time``) rather than relying on a length-1 index coordinate; fall back to a plain
    # positional concat if the CF mapping is unavailable for the variable.
    if concat_dim == "time":
        try:
            mzz = MultiZarrToZarr(
                singles,
                concat_dims=[concat_dim],
                identical_dims=["lat", "lon"],
                coo_map={"time": "cf:time"},
            )
            return mzz.translate()
        except Exception:  # pragma: no cover - non-CF time falls through to positional
            pass
    mzz = MultiZarrToZarr(singles, concat_dims=[concat_dim], identical_dims=["lat", "lon"])
    return mzz.translate()


def write_references(
    refs: dict[str, Any],
    out_path: str | Path,
    *,
    fmt: str = "json",
) -> Path:
    """Persist a kerchunk reference set to disk.

    Args:
        refs: a reference dict from :func:`virtual_reference` / :func:`virtual_references`.
        out_path: destination file path.
        fmt: ``"json"`` (default) or ``"parquet"`` (use Parquet for large reference sets —
            it loads much faster than a giant JSON; research/03 §1.2).

    Returns:
        The :class:`pathlib.Path` written.

    Raises:
        ValueError: on an unknown ``fmt``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out.write_text(json.dumps(refs))
        return out
    if fmt == "parquet":  # pragma: no cover - exercised only when fastparquet is present
        from kerchunk.df import refs_to_dataframe  # lazy

        refs_to_dataframe(refs, str(out))
        return out
    raise ValueError(f"write_references: unknown fmt {fmt!r} (expected 'json' or 'parquet').")


def open_virtual(
    refs: "dict[str, Any] | str | Path",
    *,
    storage_options: dict[str, Any] | None = None,
) -> "xr.Dataset":
    """Open a kerchunk reference set as a lazy :class:`xarray.Dataset` (O(1) per chunk).

    Reading any chunk later resolves to a single ranged GET into the original ``.nc``/``.h5``
    via the fsspec ``reference`` filesystem — no data is copied.

    Args:
        refs: a reference dict, or a path to a reference JSON file written by
            :func:`write_references`.
        storage_options: optional fsspec ``target_options`` for the underlying (remote) files
            referenced by the set (e.g. ``{"anon": True}`` for a public S3 bucket).

    Returns:
        A lazily-backed :class:`xarray.Dataset` with the same variables/coords as the source.
    """
    import fsspec  # lazy
    import xarray as xr  # lazy

    if isinstance(refs, (str, Path)):
        fo: Any = str(refs)
    else:
        fo = refs

    fs = fsspec.filesystem(
        "reference", fo=fo, target_options=dict(storage_options or {})
    )
    mapper = fs.get_mapper("")
    return xr.open_dataset(
        mapper, engine="zarr", consolidated=False, backend_kwargs={"consolidated": False}
    )


def _combine_virtualizarr(
    files: list[str],
    concat_dim: str,
    storage_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine references using VirtualiZarr when it is installed (else raises to fall back)."""
    import virtualizarr  # type: ignore  # noqa: F401  (presence check)
    from virtualizarr import open_virtual_mfdataset  # type: ignore
    from virtualizarr.parsers import HDFParser  # type: ignore

    vds = open_virtual_mfdataset(
        files,
        parser=HDFParser(),
        combine="nested",
        concat_dim=concat_dim,
        combine_attrs="drop_conflicts",
    )
    return vds.vz.to_kerchunk(format="dict")

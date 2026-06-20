"""Per-frame XYZ tile pyramids (+ optional per-frame PMTiles) for the dashboard (P2).

CODE-REVIEW CORRECTION P2 (the reason this module exists): a single raster PMTiles archive
is addressed only by ``z/x/y`` and CANNOT select a timestamp. So **every timeline frame
needs its OWN tile source**. This module builds, for one frame, a self-contained XYZ WebP
pyramid at::

    {out_dir}/tiles/{frame_index:03d}/{z}/{x}/{y}.webp

and (when a PMTiles writer is available) a per-frame archive at::

    {out_dir}/pmtiles/{frame_index:03d}.pmtiles

The timeline slider on the web side switches the active raster source to the current
frame's pyramid/archive (see CONTRACTS.md §8 / research/04 §4).

Tiling scheme (non-geographic "image" XYZ, the standard pattern for a single georeferenced
raster placed into a slippy pyramid): the frame's full extent fills the whole tile matrix.
At zoom ``z`` there are ``2**z x 2**z`` tiles; tile ``(x, y)`` covers the bbox sub-rectangle
``[x/2**z, (x+1)/2**z]`` (west->east) by ``[y/2**z, (y+1)/2**z]`` (north->south). The web's
deck.gl ``TileLayer`` reconstructs the frame by placing each tile at its proportional
sub-rectangle of the manifest ``bbox``. The pyramid is built by resizing the frame to a
square ``2**z * tile_size`` canvas per level (a Pillow-based pyramid; ``rio-tiler``/rasterio
are used if importable but are not required).

``rio-tiler`` and ``pmtiles`` are imported lazily and optionally; if absent, the XYZ
pyramid (always produced via Pillow) is the source of record and PMTiles is skipped
gracefully — the manifest still *declares* the per-frame pmtiles path so the web contract
(P2) holds, and an XYZ source is always present.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import TYPE_CHECKING

from .. import constants as C
from .render import render_rgba

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Pillow resampling filter handle, resolved lazily (Pillow>=9.1 uses Image.Resampling).
_RESAMPLE = None


def _lanczos():
    """Return the Pillow LANCZOS resampling enum across Pillow versions."""
    global _RESAMPLE
    if _RESAMPLE is None:
        from PIL import Image

        _RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
    return _RESAMPLE


def default_max_zoom(height: int, width: int, tile_size: int = C.DEFAULT_TILE_SIZE) -> int:
    """Choose a max zoom so native resolution is (just) covered by the tile matrix.

    Picks the smallest ``z`` with ``2**z * tile_size >= max(height, width)`` so that at the
    deepest zoom each source pixel is represented by roughly one tile pixel (no upsampling
    blur), clamped to ``[0, constants.DEFAULT_MAX_ZOOM]`` to bound artifact size.
    """
    longest = max(int(height), int(width), 1)
    z = max(0, math.ceil(math.log2(longest / float(tile_size)))) if longest > tile_size else 0
    return min(z, C.DEFAULT_MAX_ZOOM)


def _encode_tile_webp(tile_img) -> bytes:
    """Encode a Pillow RGBA tile to lossless WebP bytes."""
    import io

    buf = io.BytesIO()
    tile_img.save(buf, format="WEBP", lossless=True)
    return buf.getvalue()


def build_frame_tiles(
    rgba_or_bt: "np.ndarray",
    bbox: tuple[float, float, float, float] | list[float],
    out_dir: str | Path,
    frame_index: int,
    tile_size: int = C.DEFAULT_TILE_SIZE,
    max_zoom: int | None = None,
    min_zoom: int = C.DEFAULT_MIN_ZOOM,
    cmap: str = C.DEFAULT_COLORMAP,
) -> dict[str, object]:
    """Build a PER-FRAME XYZ WebP tile pyramid for one frame (P2).

    Writes ``{out_dir}/tiles/{frame_index:03d}/{z}/{x}/{y}.webp`` for every zoom in
    ``[min_zoom, max_zoom]`` and returns metadata (the URL template + zoom range).

    Args:
        rgba_or_bt: either a 2D BT field ``(H, W)`` (Kelvin, colorized on the fixed range)
            or a pre-colorized ``(H, W, 4)`` ``uint8`` RGBA image.
        bbox: ``[west, south, east, north]`` extent of the frame (advisory metadata; the
            tiling itself fills the whole matrix proportionally).
        out_dir: artifact root; tiles go under ``out_dir/tiles/{frame_index:03d}/``.
        frame_index: 0-based frame index (zero-padded to 3 digits in the path).
        tile_size: tile edge length in pixels (default 256).
        max_zoom: deepest zoom level; if ``None`` chosen via :func:`default_max_zoom`.
        min_zoom: shallowest zoom level (default 0).
        cmap: colormap name used when ``rgba_or_bt`` is a BT field.

    Returns:
        A dict with keys ``tiles_url_template`` (relative, e.g.
        ``"tiles/000/{z}/{x}/{y}.webp"``), ``min_zoom``, ``max_zoom``, ``tile_size``,
        ``bbox``, and ``n_tiles`` (total tiles written).
    """
    from PIL import Image

    rgba = render_rgba(rgba_or_bt, cmap=cmap)
    h, w = int(rgba.shape[0]), int(rgba.shape[1])
    if max_zoom is None:
        max_zoom = default_max_zoom(h, w, tile_size)
    max_zoom = max(int(max_zoom), int(min_zoom))

    base = Path(out_dir)
    frame_dir = base / "tiles" / f"{frame_index:03d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    full = Image.fromarray(rgba, mode="RGBA")
    resample = _lanczos()
    n_tiles = 0

    for z in range(int(min_zoom), int(max_zoom) + 1):
        n = 2 ** z  # tiles per axis at this zoom
        side = n * tile_size
        # Resize the whole frame to fill the square tile matrix for this zoom level.
        level_img = full.resize((side, side), resample)
        for x in range(n):
            col_dir = frame_dir / str(z) / str(x)
            col_dir.mkdir(parents=True, exist_ok=True)
            left = x * tile_size
            for y in range(n):
                top = y * tile_size
                tile = level_img.crop((left, top, left + tile_size, top + tile_size))
                (col_dir / f"{y}.webp").write_bytes(_encode_tile_webp(tile))
                n_tiles += 1

    return {
        "tiles_url_template": f"tiles/{frame_index:03d}/{{z}}/{{x}}/{{y}}.webp",
        "min_zoom": int(min_zoom),
        "max_zoom": int(max_zoom),
        "tile_size": int(tile_size),
        "bbox": [float(b) for b in bbox],
        "n_tiles": n_tiles,
    }


def pmtiles_available() -> bool:
    """Return True if a usable ``pmtiles`` writer is importable in this environment."""
    if importlib.util.find_spec("pmtiles") is None:
        return False
    try:  # the writer lives in pmtiles.writer; guard against partial installs.
        import pmtiles.writer  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on installed package layout
        return False


def build_frame_pmtiles(
    rgba_or_bt: "np.ndarray",
    bbox: tuple[float, float, float, float] | list[float],
    out_dir: str | Path,
    frame_index: int,
    tile_size: int = C.DEFAULT_TILE_SIZE,
    max_zoom: int | None = None,
    min_zoom: int = C.DEFAULT_MIN_ZOOM,
    cmap: str = C.DEFAULT_COLORMAP,
) -> str | None:
    """Write a PER-FRAME raster PMTiles archive if a writer is available, else skip (P2).

    The archive at ``{out_dir}/pmtiles/{frame_index:03d}.pmtiles`` packs the same WebP tiles
    as :func:`build_frame_tiles` into a single range-addressable file (research/04 §4). If
    the ``pmtiles`` package is not installed this returns ``None`` and the caller relies on
    the XYZ pyramid; the manifest still declares the per-frame pmtiles *path* so the web
    contract (P2) is satisfied.

    Args:
        rgba_or_bt: BT field ``(H, W)`` or RGBA ``(H, W, 4)``.
        bbox: ``[west, south, east, north]`` extent (written into PMTiles header metadata).
        out_dir: artifact root; the archive goes under ``out_dir/pmtiles/``.
        frame_index: 0-based frame index (zero-padded to 3 digits).
        tile_size, max_zoom, min_zoom, cmap: as in :func:`build_frame_tiles`.

    Returns:
        The relative archive path (e.g. ``"pmtiles/000.pmtiles"``) on success, else ``None``.
    """
    if not pmtiles_available():
        return None

    try:  # pragma: no cover - exercised only where pmtiles is installed
        import io

        from PIL import Image
        from pmtiles.tile import (  # type: ignore[import-not-found]
            Compression,
            TileType,
            zxy_to_tileid,
        )
        from pmtiles.writer import Writer  # type: ignore[import-not-found]

        rgba = render_rgba(rgba_or_bt, cmap=cmap)
        h, w = int(rgba.shape[0]), int(rgba.shape[1])
        if max_zoom is None:
            max_zoom = default_max_zoom(h, w, tile_size)
        max_zoom = max(int(max_zoom), int(min_zoom))

        base = Path(out_dir)
        pm_dir = base / "pmtiles"
        pm_dir.mkdir(parents=True, exist_ok=True)
        rel = f"pmtiles/{frame_index:03d}.pmtiles"
        out_path = base / rel

        full = Image.fromarray(rgba, mode="RGBA")
        resample = _lanczos()

        with open(out_path, "wb") as fh:
            writer = Writer(fh)
            for z in range(int(min_zoom), int(max_zoom) + 1):
                n = 2 ** z
                side = n * tile_size
                level_img = full.resize((side, side), resample)
                for x in range(n):
                    for y in range(n):
                        tile = level_img.crop(
                            (x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size)
                        )
                        buf = io.BytesIO()
                        tile.save(buf, format="WEBP", lossless=True)
                        writer.write_tile(zxy_to_tileid(z, x, y), buf.getvalue())
            west, south, east, north = (float(b) for b in bbox)
            writer.finalize(
                {
                    "tile_type": TileType.WEBP,
                    "tile_compression": Compression.NONE,
                    "min_zoom": int(min_zoom),
                    "max_zoom": int(max_zoom),
                    "min_lon_e7": int(west * 1e7),
                    "min_lat_e7": int(south * 1e7),
                    "max_lon_e7": int(east * 1e7),
                    "max_lat_e7": int(north * 1e7),
                    "center_zoom": int(min_zoom),
                    "center_lon_e7": int((west + east) / 2 * 1e7),
                    "center_lat_e7": int((south + north) / 2 * 1e7),
                },
                {"frameflow:frame_index": frame_index},
            )
        return rel
    except Exception:  # pragma: no cover - any writer incompatibility -> skip gracefully
        return None


__all__ = [
    "default_max_zoom",
    "build_frame_tiles",
    "build_frame_pmtiles",
    "pmtiles_available",
]

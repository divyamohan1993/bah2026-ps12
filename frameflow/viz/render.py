"""Full-frame raster rendering for the web dashboard (SERVE+VIZ).

Turns a brightness-temperature field into an encoded full-frame image (WebP by default,
PNG also supported) using the fixed-range IR colormaps in :mod:`frameflow.viz.colormap`.

The encoded bytes feed the manifest's ``image``/``thumb`` quick-display rasters; the same
RGBA is also the source for the per-frame tile pyramid (:mod:`frameflow.viz.tiles`) and
the all-intra videos (:mod:`frameflow.viz.video`).

Functions:
    * :func:`render_rgba`  — BT (or pre-colorized RGBA) -> ``uint8`` ``(H, W, 4)`` RGBA.
    * :func:`render_frame` -> encoded image **bytes** (and optionally writes ``out_path``).
    * :func:`save_frame`   -> write an encoded image to disk, returning its path.
    * :func:`make_thumbnail` -> a downscaled WebP/PNG thumbnail (bytes).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import constants as C
from .colormap import bt_to_rgba

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Map a short format token to the PIL/Pillow encoder name.
_FMT_TO_PIL = {
    "webp": "WEBP",
    "png": "PNG",
    "jpeg": "JPEG",
    "jpg": "JPEG",
}


def _is_rgba(arr: "np.ndarray") -> bool:
    """Return True if ``arr`` already looks like an ``(H, W, 4)`` uint8 RGBA image."""
    import numpy as np

    a = np.asarray(arr)
    return a.ndim == 3 and a.shape[-1] == 4 and a.dtype == np.uint8


def render_rgba(
    bt_or_rgba: "np.ndarray",
    cmap: str = C.DEFAULT_COLORMAP,
    vmin: float = C.BT_METRIC_VMIN_K,
    vmax: float = C.BT_METRIC_VMAX_K,
) -> "np.ndarray":
    """Return an ``(H, W, 4)`` ``uint8`` RGBA image for a BT field or pass through RGBA.

    If ``bt_or_rgba`` is already an ``(H, W, 4)`` ``uint8`` array it is returned (copied)
    unchanged; otherwise it is treated as a 2D BT field and colorized via
    :func:`frameflow.viz.colormap.bt_to_rgba` using the FIXED range (P1).
    """
    import numpy as np

    if _is_rgba(bt_or_rgba):
        return np.array(bt_or_rgba, dtype=np.uint8, copy=True)
    return bt_to_rgba(bt_or_rgba, cmap=cmap, vmin=vmin, vmax=vmax)


def _encode_rgba(rgba: "np.ndarray", fmt: str) -> bytes:
    """Encode an ``(H, W, 4)`` uint8 RGBA array to image bytes via Pillow."""
    import io

    import numpy as np
    from PIL import Image

    pil_fmt = _FMT_TO_PIL.get(fmt.lower())
    if pil_fmt is None:
        raise ValueError(f"unsupported image format {fmt!r}; expected one of {sorted(_FMT_TO_PIL)}")

    arr = np.asarray(rgba, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    if pil_fmt == "JPEG":
        # JPEG has no alpha channel; composite onto black before encoding.
        img = img.convert("RGB")
        img.save(buf, format=pil_fmt, quality=90)
    elif pil_fmt == "WEBP":
        # Lossless WebP keeps the alpha channel crisp for the precision/scrub path.
        img.save(buf, format=pil_fmt, lossless=True)
    else:
        img.save(buf, format=pil_fmt)
    return buf.getvalue()


def render_frame(
    bt: "np.ndarray",
    cmap: str = C.DEFAULT_COLORMAP,
    out_path: str | Path | None = None,
    fmt: str = C.DEFAULT_TILE_FORMAT,
    vmin: float = C.BT_METRIC_VMIN_K,
    vmax: float = C.BT_METRIC_VMAX_K,
    colormap: str | None = None,
) -> bytes:
    """Render one BT field to a full-frame encoded image and return its bytes.

    The field is colorized on the FIXED physical range (P1) and encoded as ``fmt``
    (``"webp"`` by default; ``"png"``/``"jpeg"`` also supported). If ``out_path`` is given
    the bytes are also written there (parent dirs are created).

    Args:
        bt: brightness-temperature field ``(H, W)`` (Kelvin) OR a pre-colorized
            ``(H, W, 4)`` ``uint8`` RGBA image (passed through).
        cmap: colormap name (see :func:`frameflow.viz.colormap.get_colormap`).
        out_path: optional path to also write the encoded bytes to.
        fmt: image format token (``"webp"``/``"png"``/``"jpeg"``).
        vmin: fixed display range lower bound (Kelvin).
        vmax: fixed display range upper bound (Kelvin).
        colormap: alias for ``cmap`` (matches the CONTRACTS.md keyword); takes precedence
            over ``cmap`` when provided.

    Returns:
        The encoded image as ``bytes`` (re-openable by PIL).
    """
    name = colormap if colormap is not None else cmap
    rgba = render_rgba(bt, cmap=name, vmin=vmin, vmax=vmax)
    data = _encode_rgba(rgba, fmt)
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return data


def save_frame(
    bt: "np.ndarray",
    out_path: str | Path,
    cmap: str = C.DEFAULT_COLORMAP,
    fmt: str | None = None,
    vmin: float = C.BT_METRIC_VMIN_K,
    vmax: float = C.BT_METRIC_VMAX_K,
    colormap: str | None = None,
) -> Path:
    """Render a BT field and write it to ``out_path``; return the path.

    If ``fmt`` is omitted it is inferred from ``out_path``'s suffix (defaulting to WebP).
    """
    p = Path(out_path)
    chosen_fmt = fmt or (p.suffix.lstrip(".").lower() or C.DEFAULT_TILE_FORMAT)
    render_frame(
        bt, cmap=cmap, out_path=p, fmt=chosen_fmt, vmin=vmin, vmax=vmax, colormap=colormap
    )
    return p


def make_thumbnail(
    bt: "np.ndarray",
    max_size: int = 128,
    cmap: str = C.DEFAULT_COLORMAP,
    fmt: str = C.DEFAULT_TILE_FORMAT,
    out_path: str | Path | None = None,
    vmin: float = C.BT_METRIC_VMIN_K,
    vmax: float = C.BT_METRIC_VMAX_K,
) -> bytes:
    """Render a small thumbnail (longest side <= ``max_size``) and return its bytes.

    Args:
        bt: BT field ``(H, W)`` or RGBA ``(H, W, 4)``.
        max_size: maximum length (pixels) of the longer image side.
        cmap: colormap name.
        fmt: image format token.
        out_path: optional path to also write the thumbnail to.
        vmin/vmax: fixed display range (Kelvin).

    Returns:
        The encoded thumbnail as ``bytes``.
    """
    import io

    import numpy as np
    from PIL import Image

    rgba = render_rgba(bt, cmap=cmap, vmin=vmin, vmax=vmax)
    img = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    pil_fmt = _FMT_TO_PIL.get(fmt.lower(), "WEBP")
    buf = io.BytesIO()
    if pil_fmt == "WEBP":
        img.save(buf, format=pil_fmt, lossless=True)
    elif pil_fmt == "JPEG":
        img.convert("RGB").save(buf, format=pil_fmt, quality=85)
    else:
        img.save(buf, format=pil_fmt)
    data = buf.getvalue()
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return data


def _to_image_bytes(value: Any) -> bytes:
    """Coerce a render result to bytes (helper for callers that accept either)."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"expected image bytes, got {type(value)!r}")


__all__ = ["render_rgba", "render_frame", "save_frame", "make_thumbnail"]

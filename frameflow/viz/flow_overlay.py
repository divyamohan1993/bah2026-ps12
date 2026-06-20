"""Optical-flow -> deck.gl LineLayer overlay JSON (SERVE+VIZ, research/04 §7).

The dashboard renders the model's intermediate optical flow as motion vectors using a
deck.gl ``LineLayer`` — strong visual storytelling for an optical-flow VFI model. This
module converts a dense pixel-space flow field into a sparse set of geographic line
segments (``sourcePosition``/``targetPosition`` as ``[lon, lat]``) sampled on a grid.

The flow is expected in **pixels per frame**: ``u`` is the column (x / east-west)
displacement and ``v`` is the row (y / north-south, positive = downward = southward in
image space). Vectors are sub-sampled every ``step`` pixels and mapped to lon/lat using the
frame ``bbox`` (``[west, south, east, north]``). Off-disk samples (NaN flow) are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


def _as_uv(flow: np.ndarray) -> np.ndarray:
    """Normalize a flow array to shape ``(H, W, 2)`` with last axis ``(u, v)``."""
    import numpy as np

    arr = np.asarray(flow, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 2:  # (2, H, W) -> (H, W, 2)
        arr = np.transpose(arr, (1, 2, 0))
    if not (arr.ndim == 3 and arr.shape[-1] == 2):
        raise ValueError(
            f"flow must be (2, H, W) or (H, W, 2); got shape {np.asarray(flow).shape}"
        )
    return arr


def flow_to_overlay_json(
    flow: np.ndarray,
    step: int,
    bbox: tuple[float, float, float, float] | list[float],
    scale: float = 1.0,
) -> dict:
    """Convert a dense flow field to a deck.gl ``LineLayer``-ready overlay dict.

    Args:
        flow: dense flow ``(2, H, W)`` or ``(H, W, 2)`` in pixels/frame (``u`` = x/east,
            ``v`` = y/south in image space).
        step: spatial sub-sampling stride in pixels (e.g. 16 -> one vector per 16 px).
        bbox: ``[west, south, east, north]`` geographic extent of the frame.
        scale: multiplier applied to the displacement (visual exaggeration of motion).

    Returns:
        A JSON-serializable dict::

            {
              "type": "LineLayer",
              "bbox": [w, s, e, n],
              "step": step,
              "vectors": [
                {"sourcePosition": [lon, lat], "targetPosition": [lon, lat],
                 "u": <px>, "v": <px>, "mag": <px>},
                ...
              ]
            }

    Raises:
        ValueError: if ``step`` < 1 or ``flow`` has an unexpected shape.
    """
    import numpy as np

    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    uv = _as_uv(flow)
    h, w = uv.shape[0], uv.shape[1]
    west, south, east, north = (float(b) for b in bbox)

    # Pixel-centre -> lon/lat. Column 0 -> west edge, column w-1 -> east edge; row 0 -> north
    # edge, row h-1 -> south edge. Use (idx + 0.5)/n for pixel centres.
    def _lon(col: float) -> float:
        return west + (col + 0.5) / w * (east - west)

    def _lat(row: float) -> float:
        return north - (row + 0.5) / h * (north - south)

    # Per-pixel geographic size (degrees), for converting pixel displacement to lon/lat.
    deg_per_px_x = (east - west) / w
    deg_per_px_y = (north - south) / h

    vectors: list[dict] = []
    for row in range(0, h, step):
        for col in range(0, w, step):
            u = uv[row, col, 0]
            v = uv[row, col, 1]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            lon0 = _lon(col)
            lat0 = _lat(row)
            # u (east-west, +east) -> +lon; v (+south in image) -> -lat.
            lon1 = lon0 + u * scale * deg_per_px_x
            lat1 = lat0 - v * scale * deg_per_px_y
            mag = float(np.hypot(u, v))
            vectors.append(
                {
                    "sourcePosition": [round(lon0, 6), round(lat0, 6)],
                    "targetPosition": [round(lon1, 6), round(lat1, 6)],
                    "u": round(float(u), 4),
                    "v": round(float(v), 4),
                    "mag": round(mag, 4),
                }
            )

    return {
        "type": "LineLayer",
        "bbox": [west, south, east, north],
        "step": int(step),
        "scale": float(scale),
        "vectors": vectors,
    }


def save_flow_overlay(
    flow: np.ndarray,
    step: int,
    bbox: tuple[float, float, float, float] | list[float],
    out_dir: str | Path,
    frame_index: int,
    scale: float = 1.0,
) -> str:
    """Write a flow overlay to ``{out_dir}/flow/{frame_index:03d}.json``; return the rel path.

    Args:
        flow: dense flow field (see :func:`flow_to_overlay_json`).
        step: sub-sampling stride in pixels.
        bbox: ``[west, south, east, north]``.
        out_dir: artifact root; the JSON goes under ``out_dir/flow/``.
        frame_index: 0-based frame index (zero-padded to 3 digits).
        scale: displacement exaggeration factor.

    Returns:
        The relative path written, e.g. ``"flow/003.json"``.
    """
    overlay = flow_to_overlay_json(flow, step=step, bbox=bbox, scale=scale)
    base = Path(out_dir)
    flow_dir = base / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    rel = f"flow/{frame_index:03d}.json"
    (base / rel).write_text(json.dumps(overlay))
    return rel


__all__ = ["flow_to_overlay_json", "save_flow_overlay"]

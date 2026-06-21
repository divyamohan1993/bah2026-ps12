"""Patch extraction helpers for training-sample preparation.

VFI training reads small spatial patches over short temporal windows (research/03 §2.1,
§5.4). These helpers cut a 2-D frame (or a stack of frames) into:

* a regular grid of overlapping tiles (:func:`extract_patches`) — useful for tiling a whole
  frame for dense evaluation or for precomputing a sharded sample set; and
* random crops (:func:`random_crop`, :func:`random_crop_stack`) — the standard on-the-fly
  augmentation for training, with optional rejection of patches that are mostly NaN
  (off-disk) so the loss sees real cloud signal.

Everything is plain ``numpy``; the dataset code wraps these and converts to torch tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


__all__ = [
    "extract_patches",
    "random_crop",
    "random_crop_stack",
    "valid_fraction",
]


def extract_patches(
    frame: "np.ndarray | Any",
    size: int,
    stride: int | None = None,
    *,
    drop_incomplete: bool = True,
) -> "np.ndarray":
    """Cut a 2-D (or ``(C, H, W)``) frame into a regular grid of ``size x size`` patches.

    Args:
        frame: a 2-D ``(H, W)`` or channel-first ``(C, H, W)`` array.
        size: patch height/width in pixels.
        stride: step between patch origins (defaults to ``size`` == non-overlapping).
        drop_incomplete: if True, drop edge patches that would run past the frame; if False,
            the last row/column of patches is anchored flush to the bottom/right edge.

    Returns:
        ``numpy.ndarray`` of shape ``(n_patches, size, size)`` for 2-D input, or
        ``(n_patches, C, size, size)`` for channel-first input. Empty array if the frame is
        smaller than ``size`` and ``drop_incomplete`` is True.

    Raises:
        ValueError: if ``size`` <= 0 or the input is not 2-D/3-D.
    """
    import numpy as np  # lazy

    if size <= 0:
        raise ValueError("extract_patches: size must be positive")
    arr = np.asarray(frame)
    if arr.ndim == 2:
        chan = False
        h, w = arr.shape
    elif arr.ndim == 3:
        chan = True
        _c, h, w = arr.shape
    else:
        raise ValueError("extract_patches: frame must be 2-D (H,W) or 3-D (C,H,W)")

    step = size if stride is None else int(stride)
    if step <= 0:
        raise ValueError("extract_patches: stride must be positive")

    ys = _origins(h, size, step, drop_incomplete)
    xs = _origins(w, size, step, drop_incomplete)

    patches = []
    for y0 in ys:
        for x0 in xs:
            if chan:
                patches.append(arr[:, y0:y0 + size, x0:x0 + size])
            else:
                patches.append(arr[y0:y0 + size, x0:x0 + size])
    if not patches:
        shape = (0, size, size) if not chan else (0, arr.shape[0], size, size)
        return np.empty(shape, dtype=arr.dtype)
    return np.stack(patches, axis=0)


def _origins(extent: int, size: int, step: int, drop_incomplete: bool) -> list[int]:
    """Compute patch-origin coordinates along one axis."""
    if extent < size:
        return [] if drop_incomplete else [0]
    origins = list(range(0, extent - size + 1, step))
    if not drop_incomplete and origins and origins[-1] != extent - size:
        origins.append(extent - size)
    return origins


def valid_fraction(patch: "np.ndarray | Any") -> float:
    """Return the fraction of finite (non-NaN) pixels in a patch (in ``[0, 1]``)."""
    import numpy as np  # lazy

    arr = np.asarray(patch)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def random_crop(
    frame: "np.ndarray | Any",
    size: int,
    rng: "np.random.Generator | None" = None,
) -> "tuple[np.ndarray, tuple[int, int]]":
    """Take a single random ``size x size`` crop from a 2-D frame.

    Args:
        frame: a 2-D ``(H, W)`` array.
        size: crop side length.
        rng: optional seeded ``numpy`` generator (defaults to a fresh default_rng).

    Returns:
        ``(crop, (y0, x0))`` — the cropped array and its top-left origin.

    Raises:
        ValueError: if the frame is smaller than ``size``.
    """
    import numpy as np  # lazy

    arr = np.asarray(frame)
    h, w = arr.shape[-2:]
    if h < size or w < size:
        raise ValueError(f"random_crop: frame {h}x{w} smaller than crop size {size}")
    gen = rng if rng is not None else np.random.default_rng()
    y0 = int(gen.integers(0, h - size + 1))
    x0 = int(gen.integers(0, w - size + 1))
    return arr[..., y0:y0 + size, x0:x0 + size], (y0, x0)


def random_crop_stack(
    frames: "list[np.ndarray] | np.ndarray | Any",
    size: int,
    rng: "np.random.Generator | None" = None,
    *,
    min_valid: float = 0.0,
    max_tries: int = 8,
) -> "list[np.ndarray]":
    """Take the SAME random crop from each frame in a stack (spatially aligned).

    All frames are cropped at one shared origin so a (I0, It, I1) triplet stays registered.
    When ``min_valid > 0`` the crop is re-sampled up to ``max_tries`` times to find a window
    whose first frame has at least ``min_valid`` finite pixels (avoids all-space crops).

    Args:
        frames: a list of 2-D arrays (or a 3-D ``(N, H, W)`` array) sharing a grid.
        size: crop side length.
        rng: optional seeded generator.
        min_valid: minimum finite-pixel fraction required of the first frame's crop.
        max_tries: how many random origins to try before accepting the last one.

    Returns:
        A list of cropped 2-D arrays (one per input frame).

    Raises:
        ValueError: if no frames are given.
    """
    import numpy as np  # lazy

    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        stack = [frames[i] for i in range(frames.shape[0])]
    else:
        stack = list(frames)
    if not stack:
        raise ValueError("random_crop_stack: need at least one frame")

    gen = rng if rng is not None else np.random.default_rng()
    h, w = stack[0].shape[-2:]
    if h < size or w < size:
        raise ValueError(f"random_crop_stack: frame {h}x{w} smaller than crop size {size}")

    best: list[np.ndarray] | None = None
    for _ in range(max(1, max_tries)):
        y0 = int(gen.integers(0, h - size + 1))
        x0 = int(gen.integers(0, w - size + 1))
        crops = [f[..., y0:y0 + size, x0:x0 + size] for f in stack]
        best = crops
        if min_valid <= 0.0 or valid_fraction(crops[0]) >= min_valid:
            break
    assert best is not None  # loop runs at least once
    return best

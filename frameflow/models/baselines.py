"""Classical (non-AI) frame-interpolation baselines on single-channel numpy frames.

PS-12 motivates the project by claiming AI beats *traditional optical flow*
(``idea.md``; ``research/01_vfi_models.md`` §0, §8). These comparators make that story
concrete and quantitative:

    * :func:`linear_blend`        — the trivial ``(1-t)*I0 + t*I1`` cross-fade (no motion).
    * :func:`farneback_interpolate` — Farnebäck dense optical flow (``cv2.calcOpticalFlowFarneback``)
      → scale the flow by ``t`` → backward-warp both frames toward time ``t`` → blend.
    * :func:`tvl1_interpolate`    — the same recipe with the more accurate TV-L1 dense flow
      (``cv2.optflow.DualTVL1OpticalFlow`` / ``cv2.optflow.createOptFlow_DualTVL1``).
    * :func:`classical_interpolate` — a unified dispatcher over ``method`` in
      ``{"linear", "farneback", "tvl1"}``.

These operate on **numpy single-channel** brightness-temperature arrays (``(H, W)`` or
``(1, H, W)`` / ``(H, W, 1)``), in whatever units the caller passes (raw Kelvin or
normalized) — the math is unit-agnostic. ``cv2`` (OpenCV) is imported lazily inside the
functions so importing this module never hard-requires OpenCV.

Why these "fail" (the point of the baseline): classical dense flow assumes brightness
constancy and roughly *linear* motion between frames. Convective cloud tops grow, cool,
and deform non-linearly and violate brightness constancy (the cloud literally changes
temperature), so warping + blending produces ghosting/doubling on fast or evolving cloud
— exactly where the learned RIFE/IFNet model wins (``research/01`` §1, §7.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "linear_blend",
    "farneback_interpolate",
    "tvl1_interpolate",
    "classical_interpolate",
    "CLASSICAL_METHODS",
]

#: Names accepted by :func:`classical_interpolate`.
CLASSICAL_METHODS: tuple[str, ...] = ("linear", "farneback", "tvl1")


# ---------------------------------------------------------------------------
# small array helpers (numpy imported lazily by callers; typed loosely)
# ---------------------------------------------------------------------------
def _to_hw(img: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Squeeze a single-channel array to 2-D ``(H, W)`` float32, remembering the orig shape.

    Accepts ``(H, W)``, ``(1, H, W)`` (channel-first) or ``(H, W, 1)`` (channel-last).
    Returns ``(hw, original_shape)`` so the result can be restored to the input layout.
    """
    import numpy as np

    arr = np.asarray(img)
    orig = arr.shape
    if arr.ndim == 2:
        hw = arr
    elif arr.ndim == 3 and arr.shape[0] == 1:  # (1, H, W)
        hw = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:  # (H, W, 1)
        hw = arr[..., 0]
    else:
        raise ValueError(
            f"expected single-channel (H,W)/(1,H,W)/(H,W,1) array, got shape {orig}"
        )
    return hw.astype(np.float32, copy=False), orig


def _restore(hw: np.ndarray, orig_shape: tuple[int, ...]) -> np.ndarray:
    """Restore a 2-D ``(H, W)`` result to the caller's original channel layout."""
    import numpy as np

    if len(orig_shape) == 2:
        return hw
    if len(orig_shape) == 3 and orig_shape[0] == 1:
        return hw[np.newaxis, ...]
    if len(orig_shape) == 3 and orig_shape[-1] == 1:
        return hw[..., np.newaxis]
    return hw  # pragma: no cover - guarded in _to_hw


def _warp_by_flow(hw: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp a 2-D image by a ``(H, W, 2)`` pixel flow via ``cv2.remap``.

    ``flow[y, x] = (dx, dy)`` is the displacement to ADD to the output coordinate to find
    the source sample, matching :func:`frameflow.models.warp.backward_warp`'s convention.
    """
    import cv2
    import numpy as np

    h, w = hw.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        hw, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _normalize_for_flow(hw: np.ndarray) -> np.ndarray:
    """Scale a 2-D field to ``uint8`` [0,255] for OpenCV flow (its solvers expect 8-bit).

    Uses a robust min/max over the *pair-independent* single array; this is only an internal
    representation for the flow solver — the warp/blend still happens on the original float
    values, so radiometry is preserved in the output.
    """
    import numpy as np

    lo = float(np.nanmin(hw))
    hi = float(np.nanmax(hw))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(hw, dtype=np.uint8)
    scaled = (hw - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def linear_blend(I0: np.ndarray, I1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Linear temporal cross-fade ``(1 - t) * I0 + t * I1`` (no motion compensation).

    The simplest possible interpolation — and a required trivial baseline
    (``research/01`` §8). For ``t = 0.5`` this is exactly the midpoint average
    ``0.5 * (I0 + I1)``.

    Args:
        I0: earlier frame, single-channel numpy array.
        I1: later frame, same shape as ``I0``.
        t: interpolation fraction in ``[0, 1]``.

    Returns:
        The blended frame as a float32 array in the same layout as the inputs.
    """
    import numpy as np

    a = np.asarray(I0, dtype=np.float32)
    b = np.asarray(I1, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"I0 {a.shape} and I1 {b.shape} must have the same shape")
    return (1.0 - float(t)) * a + float(t) * b


def _flow_interpolate(
    I0: np.ndarray,
    I1: np.ndarray,
    t: float,
    flow01: np.ndarray,
    flow10: np.ndarray,
) -> np.ndarray:
    """Shared warp-and-blend given precomputed forward/backward dense flows.

    Given the flow ``flow01`` (I0->I1) and ``flow10`` (I1->I0), approximate the frame at
    time ``t`` by warping ``I0`` forward by ``t * flow01`` and ``I1`` backward by
    ``(1 - t) * flow10`` (both toward time ``t``), then linearly blend by ``t``. This is the
    standard classical OF interpolation used as the "traditional optical flow" comparator.
    """
    import numpy as np

    hw0, orig = _to_hw(I0)
    hw1, _ = _to_hw(I1)
    # Warp each frame to time t. To pull I0 toward t we sample I0 at -t*flow01 (backward
    # warp convention: add the displacement to the output coord to find the source).
    warp0 = _warp_by_flow(hw0, (-float(t)) * flow01)
    warp1 = _warp_by_flow(hw1, (-(1.0 - float(t))) * flow10)
    blended = (1.0 - float(t)) * warp0 + float(t) * warp1
    return _restore(blended.astype(np.float32), orig)


def farneback_interpolate(I0: np.ndarray, I1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Farnebäck dense-optical-flow interpolation (warp both frames to ``t``, blend).

    Computes dense flow with :func:`cv2.calcOpticalFlowFarneback` in both directions, scales
    by ``t`` / ``1 - t``, backward-warps each frame toward time ``t``, and blends. A classic,
    fully-classical baseline that exhibits ghosting on fast/evolving cloud (the failure mode
    the AI model fixes; ``research/01`` §1).

    Args:
        I0, I1: earlier/later single-channel numpy frames (same shape).
        t: interpolation fraction in ``[0, 1]``.

    Returns:
        The interpolated frame, float32, in the input layout.
    """
    import cv2

    hw0, _ = _to_hw(I0)
    hw1, _ = _to_hw(I1)
    g0 = _normalize_for_flow(hw0)
    g1 = _normalize_for_flow(hw1)
    fb_kwargs = dict(
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    flow01 = cv2.calcOpticalFlowFarneback(g0, g1, None, **fb_kwargs)  # I0 -> I1
    flow10 = cv2.calcOpticalFlowFarneback(g1, g0, None, **fb_kwargs)  # I1 -> I0
    return _flow_interpolate(I0, I1, t, flow01, flow10)


def _make_tvl1():
    """Construct a TV-L1 dense optical-flow solver across OpenCV API variants.

    The TV-L1 implementation lives in ``cv2.optflow`` (opencv-contrib) and its factory has
    moved across versions. Tries the known entry points and raises a clear error if none is
    available.
    """
    import cv2

    # Newer contrib API.
    if hasattr(cv2, "optflow"):
        opt = cv2.optflow
        for name in ("DualTVL1OpticalFlow_create", "createOptFlow_DualTVL1"):
            if hasattr(opt, name):
                return getattr(opt, name)()
    # Older top-level API.
    for name in ("DualTVL1OpticalFlow_create", "createOptFlow_DualTVL1"):
        if hasattr(cv2, name):
            return getattr(cv2, name)()
    raise RuntimeError(
        "TV-L1 optical flow is unavailable in this OpenCV build "
        "(needs cv2.optflow from opencv-contrib-python)."
    )


def tvl1_interpolate(I0: np.ndarray, I1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """TV-L1 dense-optical-flow interpolation (warp both frames to ``t``, blend).

    Same recipe as :func:`farneback_interpolate` but with the more accurate (and slower)
    TV-L1 variational dense flow. Falls back to Farnebäck only if the TV-L1 solver is not
    present in the OpenCV build (so the function still produces a result rather than crashing
    the pipeline).

    Args:
        I0, I1: earlier/later single-channel numpy frames (same shape).
        t: interpolation fraction in ``[0, 1]``.

    Returns:
        The interpolated frame, float32, in the input layout.
    """
    hw0, _ = _to_hw(I0)
    hw1, _ = _to_hw(I1)
    g0 = _normalize_for_flow(hw0)
    g1 = _normalize_for_flow(hw1)
    import cv2

    try:
        solver = _make_tvl1()
        flow01 = solver.calc(g0, g1, None)  # I0 -> I1
        flow10 = solver.calc(g1, g0, None)  # I1 -> I0
    except (RuntimeError, AttributeError, ModuleNotFoundError, cv2.error):
        # Graceful degradation: TV-L1 missing/broken in this OpenCV build (e.g. plain
        # opencv-python without the contrib `optflow` module) -> fall back to Farnebäck so
        # the comparator still runs end-to-end.
        return farneback_interpolate(I0, I1, t)
    return _flow_interpolate(I0, I1, t, flow01, flow10)


def classical_interpolate(
    I0: np.ndarray, I1: np.ndarray, t: float = 0.5, method: str = "tvl1"
) -> np.ndarray:
    """Unified dispatcher over the classical baselines.

    Args:
        I0, I1: earlier/later single-channel numpy frames (same shape).
        t: interpolation fraction in ``[0, 1]``.
        method: one of :data:`CLASSICAL_METHODS` — ``"linear"``, ``"farneback"``, ``"tvl1"``.

    Returns:
        The interpolated frame, float32, in the input layout.

    Raises:
        ValueError: if ``method`` is not recognized.
    """
    key = method.lower()
    if key == "linear":
        return linear_blend(I0, I1, t)
    if key == "farneback":
        return farneback_interpolate(I0, I1, t)
    if key == "tvl1":
        return tvl1_interpolate(I0, I1, t)
    raise ValueError(f"unknown classical method {method!r}; choose from {CLASSICAL_METHODS}")

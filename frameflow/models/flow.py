"""Optical-flow utilities: Middlebury color visualization, sparse vectors, RAFT wrapper.

These support the PS-12 dashboard's motion-vector overlay and the EPE/flow diagnostics
(``research/01_vfi_models.md`` §4, §10). Three pieces:

    * :func:`flow_to_image` — the standard **Middlebury** color wheel encoding (hue = flow
      direction, saturation/value = flow magnitude). The de-facto way to visualize a dense
      flow field; useful for the dashboard and for sanity-checking IFNet's estimated flow.
    * :func:`flow_to_vectors` — subsample a dense flow into a sparse list of arrows for an
      overlay (each entry ``{"x","y","dx","dy"}``), which the web/viz team can draw on top of
      a frame.
    * :class:`RAFTFlow` — an OPTIONAL lazy wrapper around torchvision's
      :func:`torchvision.models.optical_flow.raft_small` for computing high-quality dense
      flow on large cyclone-scale motion or for flow visualization. Single-channel inputs
      are adapted to the 3-channel RGB the pretrained RAFT expects. RIFE/IFNet estimates its
      own internal flow and needs none of this — RAFT is only for the comparator path / viz
      (``research/01`` §4).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = ["flow_to_image", "flow_to_vectors", "make_color_wheel", "RAFTFlow"]


# ---------------------------------------------------------------------------
# Middlebury color wheel
# ---------------------------------------------------------------------------
def make_color_wheel() -> np.ndarray:
    """Build the Middlebury optical-flow color wheel as an ``(N, 3)`` uint8 array.

    The wheel concatenates six hue ramps (RY, YG, GC, CB, BM, MR) so that flow *direction*
    maps to hue smoothly around the circle. This is the canonical wheel from the Middlebury
    flow benchmark (Baker et al.) reproduced in essentially every flow toolkit.
    """
    import numpy as np

    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    wheel = np.zeros((ncols, 3), dtype=np.float64)
    col = 0
    # Red -> Yellow
    wheel[0:RY, 0] = 255
    wheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY
    # Yellow -> Green
    wheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    wheel[col:col + YG, 1] = 255
    col += YG
    # Green -> Cyan
    wheel[col:col + GC, 1] = 255
    wheel[col:col + GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC
    # Cyan -> Blue
    wheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    wheel[col:col + CB, 2] = 255
    col += CB
    # Blue -> Magenta
    wheel[col:col + BM, 2] = 255
    wheel[col:col + BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM
    # Magenta -> Red
    wheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    wheel[col:col + MR, 0] = 255
    return wheel.astype(np.uint8)


def _flow_to_hw2(flow: Any) -> np.ndarray:
    """Coerce a flow to a contiguous ``(H, W, 2)`` float32 numpy array.

    Accepts numpy or torch, channel-first ``(2, H, W)`` / batched ``(1, 2, H, W)`` or
    channel-last ``(H, W, 2)``.
    """
    import numpy as np

    arr = flow
    # torch -> numpy
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:  # (1, 2, H, W)
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 2:  # (2, H, W) -> (H, W, 2)
        arr = np.transpose(arr, (1, 2, 0))
    if not (arr.ndim == 3 and arr.shape[-1] == 2):
        raise ValueError(
            f"flow must be (2,H,W)/(1,2,H,W)/(H,W,2); got shape {np.asarray(flow).shape}"
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def flow_to_image(flow: Any, *, max_magnitude: float | None = None) -> np.ndarray:
    """Convert a dense optical-flow field to a Middlebury-colored RGB image.

    Hue encodes flow direction (via :func:`make_color_wheel`); saturation/brightness encode
    normalized magnitude. NaNs are rendered black.

    Args:
        flow: dense flow, ``(2, H, W)`` / ``(1, 2, H, W)`` / ``(H, W, 2)`` (channel 0 = dx,
            channel 1 = dy), numpy or torch.
        max_magnitude: optional fixed magnitude to normalize by (for consistent coloring
            across frames). If ``None``, uses the per-field max (the usual Middlebury
            behavior).

    Returns:
        An ``(H, W, 3)`` uint8 RGB image.
    """
    import numpy as np

    f = _flow_to_hw2(flow)
    u = f[..., 0]
    v = f[..., 1]

    nan_mask = ~(np.isfinite(u) & np.isfinite(v))
    u = np.where(nan_mask, 0.0, u)
    v = np.where(nan_mask, 0.0, v)

    rad = np.sqrt(u ** 2 + v ** 2)
    if max_magnitude is not None and max_magnitude > 0:
        maxrad = float(max_magnitude)
    else:
        maxrad = float(rad.max()) if rad.size and rad.max() > 0 else 1.0

    eps = np.finfo(np.float32).eps
    u_n = u / (maxrad + eps)
    v_n = v / (maxrad + eps)
    rad_n = np.sqrt(u_n ** 2 + v_n ** 2)

    wheel = make_color_wheel().astype(np.float64)  # (ncols, 3)
    ncols = wheel.shape[0]

    angle = np.arctan2(-v_n, -u_n) / math.pi  # in [-1, 1]
    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int64)
    k1 = (k0 + 1) % ncols
    k0 = k0 % ncols
    frac = fk - np.floor(fk)

    h, w = u.shape
    img = np.zeros((h, w, 3), dtype=np.float64)
    for ch in range(3):
        c0 = wheel[k0, ch] / 255.0
        c1 = wheel[k1, ch] / 255.0
        col = (1.0 - frac) * c0 + frac * c1
        # Saturate toward white for small magnitude (Middlebury convention).
        idx = rad_n <= 1.0
        col_out = col.copy()
        col_out[idx] = 1.0 - rad_n[idx] * (1.0 - col[idx])
        col_out[~idx] = col[~idx] * 0.75  # out-of-range magnitude: darken
        img[..., ch] = col_out

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img[nan_mask] = 0
    return img


def flow_to_vectors(flow: Any, step: int = 16) -> list[dict[str, float]]:
    """Subsample a dense flow into a sparse list of vectors for an overlay.

    Samples the flow on a regular ``step``-pixel grid and returns one dict per sample,
    suitable for the dashboard's motion-vector overlay (drawn as arrows from ``(x, y)`` to
    ``(x + dx, y + dy)``; ``research/01`` §4, §10).

    Args:
        flow: dense flow ``(2,H,W)`` / ``(1,2,H,W)`` / ``(H,W,2)`` (numpy or torch).
        step: grid spacing in pixels between sampled vectors (>= 1).

    Returns:
        A list of ``{"x": float, "y": float, "dx": float, "dy": float}`` dicts (pixel
        coordinates and pixel displacements). Non-finite samples are skipped.
    """

    if step < 1:
        raise ValueError("step must be >= 1")
    f = _flow_to_hw2(flow)
    h, w, _ = f.shape
    # Centre the sampling grid within each step-cell.
    offset = step // 2
    out: list[dict[str, float]] = []
    for y in range(offset, h, step):
        for x in range(offset, w, step):
            dx = float(f[y, x, 0])
            dy = float(f[y, x, 1])
            if not (math.isfinite(dx) and math.isfinite(dy)):
                continue
            out.append({"x": float(x), "y": float(y), "dx": dx, "dy": dy})
    return out


# ---------------------------------------------------------------------------
# Optional RAFT wrapper (lazy torchvision)
# ---------------------------------------------------------------------------
class RAFTFlow:
    """Lazy wrapper around torchvision ``raft_small`` for dense optical flow (OPTIONAL).

    Used only for the comparator path / large-motion flow visualization
    (``research/01`` §4) — the primary RIFE/IFNet engine estimates flow internally and does
    NOT need this. The torchvision RAFT model and (optional) pretrained weights are imported
    and constructed lazily on first :meth:`__call__`, so importing this module is cheap and
    does not require network access.

    Single-channel adaptation: pretrained RAFT expects 3-channel RGB, so single-channel
    ``(B, 1, H, W)`` inputs are replicated to 3 channels before inference (the standard
    grayscale→RGB trick; ``research/01`` §7.1). RAFT also requires H and W to be multiples
    of 8, so inputs are reflect-padded to the next multiple of 8 and the flow cropped back.

    Note: torchvision's RAFT correlation pyramid needs the inputs to be **at least 128 px**
    in each spatial dim (they are downsampled by 8, and the feature maps must be ≥16). This
    is fine for the intended use (full cyclone-scale frames); it is not meant for the tiny
    64 px tensors used in unit tests — use :class:`~frameflow.models.ifnet.IFNet` there.

    Args:
        pretrained: if ``True`` (default), load torchvision's ``Raft_Small_Weights.DEFAULT``
            (downloads on first use; falls back to random init with a warning if unavailable).
        iters: number of RAFT refinement iterations at inference (more = slower/finer).
    """

    def __init__(self, *, pretrained: bool = True, iters: int = 12) -> None:
        self.pretrained = bool(pretrained)
        self.iters = int(iters)
        self._model: Any | None = None  # lazily built torchvision module

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        import warnings

        from torchvision.models.optical_flow import raft_small

        weights = None
        if self.pretrained:
            try:
                from torchvision.models.optical_flow import Raft_Small_Weights

                weights = Raft_Small_Weights.DEFAULT
            except Exception as exc:  # pragma: no cover - depends on torchvision version/network
                warnings.warn(
                    f"RAFTFlow: could not resolve pretrained weights ({exc!r}); "
                    "falling back to RANDOM init — flow will be meaningless until trained.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        try:
            model = raft_small(weights=weights, progress=False)
        except Exception as exc:  # pragma: no cover - e.g. no network for weights download
            warnings.warn(
                f"RAFTFlow: failed to load pretrained RAFT ({exc!r}); using random init.",
                RuntimeWarning,
                stacklevel=2,
            )
            model = raft_small(weights=None, progress=False)
        model = model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        return model

    @staticmethod
    def _to_rgb(x: torch.Tensor) -> torch.Tensor:
        """Replicate a single-channel ``(B,1,H,W)`` tensor to 3 channels; pass 3ch through."""
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 3:
            return x
        raise ValueError(f"RAFTFlow expects 1- or 3-channel input, got {x.shape[1]} channels")

    def __call__(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        """Estimate dense flow ``img0 -> img1``.

        Args:
            img0: ``(B, C, H, W)`` source frame (C in {1, 3}); values expected ~``[-1, 1]``
                or ``[0, 1]`` (RAFT is fairly robust to scale but trained on normalized RGB).
            img1: ``(B, C, H, W)`` target frame, same shape as ``img0``.

        Returns:
            The final-iteration flow ``(B, 2, H, W)`` (channel 0 = dx, channel 1 = dy),
            cropped back to the input ``H, W`` and detached (inference-only).
        """
        import torch
        import torch.nn.functional as F

        model = self._ensure_model()
        a = self._to_rgb(img0)
        b = self._to_rgb(img1)
        _bs, _c, h, w = a.shape
        # RAFT needs H,W divisible by 8: reflect-pad up, crop the flow back afterwards.
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            a = F.pad(a, (0, pw, 0, ph), mode="reflect")
            b = F.pad(b, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            flows = model(a, b, num_flow_updates=self.iters)
        flow = flows[-1] if isinstance(flows, (list, tuple)) else flows
        if ph or pw:
            flow = flow[..., :h, :w]
        return flow.detach()

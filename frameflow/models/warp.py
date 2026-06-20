"""Differentiable backward (inverse) warping via :func:`torch.nn.functional.grid_sample`.

Backward warping is the core operation of every flow-based VFI model
(``research/01_vfi_models.md`` §2): given a source image and a flow field that, for each
*output* pixel, says where to *sample from* in the source, it resamples the source onto
the output grid. RIFE/IFNet warps both input frames toward the (unknown) intermediate
frame using the estimated intermediate flows and then fuses them — so this function is
called many times per forward pass and must be cheap, batched, and fully differentiable
(gradients flow into both the image and the flow).

Conventions (IMPORTANT):
    * ``flow`` is in **pixel** units, layout ``(B, 2, H, W)`` with channel 0 = horizontal
      displacement ``dx`` (along width / x) and channel 1 = vertical displacement ``dy``
      (along height / y). ``out[y, x] = img[y + dy, x + dx]`` (we ADD the flow to the
      output coordinate to get the source coordinate — the standard backward-warp /
      ``grid_sample`` convention used by RIFE).
    * Works for any channel count, including the single-channel (C=1) brightness-temperature
      tensors FrameFlow uses.

``align_corners`` handling (documented per the task contract):
    We build the normalized sampling grid with ``align_corners=False`` semantics and pass
    ``align_corners=False`` to ``grid_sample``. With ``align_corners=False`` a pixel centre
    ``i`` (in ``0..N-1``) maps to normalized coordinate ``(2*i + 1) / N - 1``; i.e. the
    extreme pixel *centres* are NOT pushed to the corners ``±1`` (instead the pixel *edges*
    are at ``±1``). This is the modern, resolution-consistent convention and is what
    Practical-RIFE uses. A pixel displaced by ``d`` columns therefore moves by
    ``2*d / W`` in normalized x (and ``2*d / H`` in normalized y) — independent of the
    ``align_corners`` choice for the *step size*, only the origin differs. Keeping the grid
    construction and ``grid_sample`` call consistent (both ``align_corners=False``)
    guarantees that a **zero flow is an exact identity** (verified in ``tests/test_models``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["backward_warp", "make_base_grid"]


def make_base_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the normalized identity sampling grid for ``grid_sample`` (no batch dim).

    Returns a ``(H, W, 2)`` tensor whose ``[y, x] = (gx, gy)`` are the normalized
    coordinates (in ``[-1, 1]`` under ``align_corners=False``) that sample pixel
    ``(y, x)`` of the source — i.e. an identity map. Adding the (normalized) flow to this
    grid yields the backward-warp sampling grid.

    Args:
        height: grid height ``H`` in pixels.
        width: grid width ``W`` in pixels.
        device: torch device for the grid (defaults to CPU / caller's tensors).
        dtype: floating dtype for the grid.

    Returns:
        A ``(H, W, 2)`` identity grid in normalized coordinates (x then y), matching the
        layout ``grid_sample`` expects in its last dimension.
    """
    # Pixel-centre coordinates under align_corners=False: centre i -> (2i+1)/N - 1.
    ys = (torch.arange(height, device=device, dtype=dtype) * 2.0 + 1.0) / height - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) * 2.0 + 1.0) / width - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # each (H, W)
    # grid_sample wants the last dim ordered (x, y).
    return torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)


def backward_warp(
    img: torch.Tensor,
    flow: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """Backward-warp ``img`` by a pixel-unit ``flow`` field (differentiable).

    For every output pixel ``(y, x)`` the result samples the source at
    ``(y + flow_y, x + flow_x)`` using bilinear interpolation. Gradients propagate into
    both ``img`` and ``flow`` (``grid_sample`` is differentiable), so this is usable inside
    a trainable VFI network (RIFE/IFNet warps the inputs toward the intermediate frame and
    backprops through the warp; ``research/01`` §3.2).

    Args:
        img: source image, shape ``(B, C, H, W)``; any ``C`` (incl. single-channel C=1).
        flow: pixel-unit flow, shape ``(B, 2, H, W)``; channel 0 = ``dx`` (x / width),
            channel 1 = ``dy`` (y / height). ``out[...,y,x] = img[..., y+dy, x+dx]``.
        mode: ``grid_sample`` interpolation mode (``"bilinear"`` default; ``"nearest"`` /
            ``"bicubic"`` also valid).
        padding_mode: how out-of-bounds samples are handled (``"border"`` default — repeat
            edge values, sensible for off-disk/space borders; ``"zeros"`` / ``"reflection"``
            also valid).
        align_corners: kept ``False`` to match :func:`make_base_grid`; see module docstring.
            Both the grid and ``grid_sample`` use the same value so zero flow is an exact
            identity.

    Returns:
        The warped image, shape ``(B, C, H, W)`` (same dtype/device as ``img``).

    Raises:
        ValueError: if ``img`` / ``flow`` are not 4-D or ``flow`` lacks 2 channels.
    """
    if img.dim() != 4:
        raise ValueError(f"img must be (B, C, H, W); got shape {tuple(img.shape)}")
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be (B, 2, H, W); got shape {tuple(flow.shape)}")

    b, _c, h, w = img.shape
    if flow.shape[0] != b or flow.shape[2] != h or flow.shape[3] != w:
        raise ValueError(
            f"img {tuple(img.shape)} and flow {tuple(flow.shape)} must share B,H,W"
        )

    # Identity grid in normalized coords, broadcast to the batch.
    base = make_base_grid(h, w, device=img.device, dtype=flow.dtype)  # (H, W, 2)
    base = base.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)

    # Convert the pixel-unit flow to normalized-coordinate displacement. Under
    # align_corners=False the normalized span [-1, 1] covers N pixel widths, so a
    # displacement of d pixels equals 2*d / N in normalized units (x uses W, y uses H).
    dx = flow[:, 0, :, :] * (2.0 / w)  # (B, H, W)
    dy = flow[:, 1, :, :] * (2.0 / h)  # (B, H, W)
    norm_flow = torch.stack((dx, dy), dim=-1)  # (B, H, W, 2), (x, y) order

    sample_grid = base + norm_flow  # (B, H, W, 2)

    return F.grid_sample(
        img,
        sample_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

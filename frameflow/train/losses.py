"""FrameFlow training losses — single-channel (TIR) friendly reconstruction losses.

This module implements the loss recipe from ``research/06_inference_serving.md`` §5.3 for
fine-tuning a RIFE/IFRNet-class video-frame-interpolation (VFI) model on **single-channel
thermal-infrared (TIR) brightness-temperature** frames::

    L = 1.0 * Charbonnier      # robust L1 reconstruction (default fidelity term)
      + 0.5 * Census           # ternary/census transform — illumination-robust → clouds
      + 0.25 * (1 - MS-SSIM)   # structural similarity; aligns with the SSIM eval metric
      + 0.1  * Gradient        # Sobel edge preservation → crisp cloud boundaries
      (+ optional flow distillation when fine-tuning IFRNet/RIFE)

DELIBERATE DESIGN DECISION — NO VGG / LPIPS BY DEFAULT
-----------------------------------------------------
VGG-perceptual and LPIPS losses live in an **ImageNet-RGB** feature space. Our inputs are
**1-channel brightness temperature in Kelvin**, not 3-channel sRGB. Using VGG/LPIPS would
require replicating the single channel to three and accepting a domain mismatch (the
network never saw TIR radiometric statistics), which research/06 §5.1 flags as
"domain-mismatched … lower priority". The grading metrics are MSE/PSNR/SSIM/FSIM, all of
which Charbonnier + census + MS-SSIM + gradient optimize directly, so we explicitly omit
the RGB-perceptual trap. (One could later add ``0.05 * LPIPS`` via 1→3 replication and
ablate on SSIM/PSNR, but it is off by default.)

All losses are single-channel-friendly: they operate per-channel and so work for
``(B, 1, H, W)`` TIR tensors (and for multi-channel tensors too). Inputs are assumed to be
in the model's working range (normalized ``[0, 1]`` by convention); the MS-SSIM term takes
an explicit ``data_range`` because :func:`piq.multi_scale_ssim` needs a fixed peak value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .. import config as _config

if TYPE_CHECKING:  # for type-checkers only
    from torch import Tensor


__all__ = [
    "charbonnier_loss",
    "census_loss",
    "ms_ssim_loss",
    "gradient_loss",
    "flow_distillation_loss",
    "CombinedLoss",
]


# ===========================================================================
# Charbonnier — robust L1 (the default reconstruction term)
# ===========================================================================
def charbonnier_loss(pred: Tensor, gt: Tensor, eps: float = 1e-3) -> Tensor:
    r"""Charbonnier (smooth-L1 / pseudo-Huber) reconstruction loss.

    Computes ``mean( sqrt((pred - gt)^2 + eps^2) )``. The Charbonnier penalty
    :math:`\rho(x) = \sqrt{x^2 + \epsilon^2}` is a differentiable, outlier-robust
    approximation of the L1 norm and is the default VFI reconstruction term
    (research/06 §5.1; IFRNet's :math:`\rho` with :math:`\epsilon=10^{-3}`).

    Args:
        pred: predicted frame, shape ``(B, C, H, W)`` (C == 1 for TIR).
        gt: ground-truth frame, same shape as ``pred``.
        eps: small constant :math:`\epsilon` controlling the L2→L1 transition.

    Returns:
        A scalar tensor (mean over all elements) carrying gradient.
    """
    diff = pred - gt
    return torch.sqrt(diff * diff + eps * eps).mean()


# ===========================================================================
# Census / ternary transform — illumination-robust structural loss
# ===========================================================================
def _rgb_to_gray(x: Tensor) -> Tensor:
    """Collapse a tensor to a single channel for the census transform.

    Single-channel TIR is returned unchanged; multi-channel inputs are averaged across the
    channel dim (the census transform is defined on a scalar intensity field).
    """
    if x.shape[1] == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def _census_transform(img: Tensor, patch_size: int = 7, eps: float = 1e-2) -> Tensor:
    r"""Soft census (ternary) transform of a single-channel image.

    The census transform encodes, for every pixel, the *sign of the difference* between the
    centre pixel and each neighbour in a ``patch_size × patch_size`` window. It is invariant
    to additive/multiplicative illumination changes, which makes it robust to the
    brightness-temperature variations of clouds (research/06 §5.1: "excellent for clouds").

    We use a smooth/normalized variant (à la UnFlow / SelFlow): differences are divided by
    ``sqrt(0.81 + diff^2)`` so the transform is differentiable rather than a hard sign.

    Args:
        img: single-channel image, shape ``(B, 1, H, W)``.
        patch_size: odd window size for the local comparison.
        eps: kept for API symmetry / numerical safety (unused in the smooth form).

    Returns:
        A ``(B, patch_size*patch_size, H, W)`` tensor of per-neighbour soft comparisons.
    """
    assert patch_size % 2 == 1, "census patch_size must be odd"
    pad = patch_size // 2
    n = patch_size * patch_size
    # An identity stack of 1x1-in / n-out filters: each output channel picks one neighbour
    # offset, so convolving with reflect padding gathers the full local patch per pixel.
    weights = torch.eye(n, dtype=img.dtype, device=img.device).reshape(n, 1, patch_size, patch_size)
    padded = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    patches = F.conv2d(padded, weights)  # (B, n, H, W) — neighbourhood intensities
    # Difference of each neighbour from the centre pixel, then a smooth normalization.
    centre = img  # (B, 1, H, W) broadcasts against (B, n, H, W)
    diff = patches - centre
    transformed = diff / torch.sqrt(0.81 + diff * diff)
    return transformed


def census_loss(pred: Tensor, gt: Tensor, patch_size: int = 7) -> Tensor:
    r"""Census/ternary-transform structural loss (illumination-robust; good for clouds).

    Applies the soft :func:`_census_transform` to both ``pred`` and ``gt`` and returns the
    Charbonnier distance between the two transform fields. Because the census transform
    depends only on *local intensity ordering*, this term penalizes structural/edge
    mismatches while being insensitive to global brightness shifts — ideal for TIR cloud
    fields whose absolute brightness temperature drifts (research/06 §5.1, §5.3).

    Multi-channel inputs are first reduced to a single intensity channel.

    Args:
        pred: predicted frame, shape ``(B, C, H, W)``.
        gt: ground-truth frame, same shape.
        patch_size: odd census window size (default 7, as in IFRNet/UnFlow).

    Returns:
        A scalar tensor carrying gradient.
    """
    p = _census_transform(_rgb_to_gray(pred), patch_size=patch_size)
    g = _census_transform(_rgb_to_gray(gt), patch_size=patch_size)
    # Soft Hamming distance between the two census signatures (Charbonnier over channels).
    dist = torch.sqrt((p - g) ** 2 + 1e-6).mean()
    return dist


# ===========================================================================
# MS-SSIM loss (uses piq.multi_scale_ssim with a FIXED data_range)
# ===========================================================================
def _ms_ssim_scale_weights(min_hw: int, kernel_size: int = 7) -> Tensor | None:
    """Pick MS-SSIM scale weights that fit the smallest spatial dimension.

    :func:`piq.multi_scale_ssim` halves the image once per scale and requires the smallest
    scale to stay larger than the SSIM kernel. The default 5-scale weights need ≥161 px;
    small training patches (or tests) need fewer scales. We return a (renormalized) prefix
    of the canonical Wang et al. weights with as many scales as the input supports.

    Args:
        min_hw: the smaller of (H, W) of the input.
        kernel_size: SSIM Gaussian kernel size used by the MS-SSIM call.

    Returns:
        A 1-D tensor of scale weights (summing to 1), or ``None`` to use piq's default
        5-scale weights when the image is large enough.
    """
    canonical = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Wang et al. 2003
    # After (n-1) halvings the smallest side is min_hw / 2**(n-1); it must exceed kernel_size.
    max_scales = 1
    for n in range(1, len(canonical) + 1):
        smallest = min_hw // (2 ** (n - 1))
        if smallest > kernel_size:
            max_scales = n
        else:
            break
    if max_scales >= len(canonical):
        return None  # large enough → let piq use its default 5-scale weights
    w = torch.tensor(canonical[:max_scales], dtype=torch.float32)
    return w / w.sum()


def ms_ssim_loss(pred: Tensor, gt: Tensor, data_range: float = 1.0) -> Tensor:
    r"""Multi-scale SSIM loss, ``1 - MS-SSIM``, via :func:`piq.multi_scale_ssim`.

    MS-SSIM aligns the training objective with the SSIM-family evaluation metrics
    (research/06 §5.1). ``piq`` requires a **fixed** ``data_range`` (the peak signal value)
    rather than per-image min/max, and asserts the inputs lie within ``[0, data_range]`` —
    so we pass the model's working range and clamp first.

    Important: ``data_range`` here is the *model working range* (normalized ``1.0`` by
    convention), NOT the Kelvin metric range. Validation PSNR/SSIM in physical Kelvin uses
    the fixed :data:`frameflow.constants.BT_DATA_RANGE_K` separately (see the module/
    :class:`~frameflow.train.module.VFIModule` validation step). MS-SSIM is scale-invariant,
    so using the working range here does not violate the fixed-range metric rule (P1) for
    the reported metrics.

    For small inputs the number of pyramid scales is reduced automatically so the call does
    not raise on patches smaller than piq's 161-px default minimum.

    Args:
        pred: predicted frame, shape ``(B, C, H, W)``.
        gt: ground-truth frame, same shape.
        data_range: the fixed peak value of the working range (e.g. ``1.0`` for [0,1]).

    Returns:
        A scalar tensor ``1 - MS-SSIM`` in ``[0, ~1]`` carrying gradient.
    """
    import piq  # lazy (heavy import)

    dr = float(data_range)
    # piq strictly asserts inputs are within [0, data_range]; clamp to satisfy it while
    # keeping gradients on the in-range region.
    pred_c = pred.clamp(0.0, dr)
    gt_c = gt.clamp(0.0, dr)

    kernel_size = 7  # smaller kernel → works on smaller patches than the default 11
    min_hw = int(min(pred_c.shape[-2], pred_c.shape[-1]))
    weights = _ms_ssim_scale_weights(min_hw, kernel_size=kernel_size)
    if weights is not None:
        weights = weights.to(pred_c.device, dtype=pred_c.dtype)

    value = piq.multi_scale_ssim(
        pred_c,
        gt_c,
        kernel_size=kernel_size,
        data_range=dr,
        scale_weights=weights,
        reduction="mean",
    )
    return 1.0 - value


# ===========================================================================
# Gradient loss (Sobel edge preservation)
# ===========================================================================
def _sobel_kernels(dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return the (Gx, Gy) Sobel kernels as ``(1, 1, 3, 3)`` tensors."""
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).reshape(1, 1, 3, 3)
    gy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    ).reshape(1, 1, 3, 3)
    return gx, gy


def gradient_loss(pred: Tensor, gt: Tensor) -> Tensor:
    r"""Sobel-gradient (edge-preservation) loss.

    Compares the spatial image gradients (Sobel Gx/Gy) of ``pred`` and ``gt`` with a
    Charbonnier penalty. Matching gradients preserves sharp cloud boundaries / high-frequency
    structure that FSIM rewards (research/06 §5.1, §5.3). Applied per channel, so it works
    for single-channel TIR and multi-channel inputs alike.

    Args:
        pred: predicted frame, shape ``(B, C, H, W)``.
        gt: ground-truth frame, same shape.

    Returns:
        A scalar tensor carrying gradient.
    """
    c = pred.shape[1]
    gx, gy = _sobel_kernels(pred.dtype, pred.device)
    # Depthwise convolution: replicate the single-channel kernel across C groups.
    gx = gx.repeat(c, 1, 1, 1)
    gy = gy.repeat(c, 1, 1, 1)
    pred_p = F.pad(pred, (1, 1, 1, 1), mode="reflect")
    gt_p = F.pad(gt, (1, 1, 1, 1), mode="reflect")
    pred_gx = F.conv2d(pred_p, gx, groups=c)
    pred_gy = F.conv2d(pred_p, gy, groups=c)
    gt_gx = F.conv2d(gt_p, gx, groups=c)
    gt_gy = F.conv2d(gt_p, gy, groups=c)
    loss_x = torch.sqrt((pred_gx - gt_gx) ** 2 + 1e-6).mean()
    loss_y = torch.sqrt((pred_gy - gt_gy) ** 2 + 1e-6).mean()
    return loss_x + loss_y


# ===========================================================================
# Optional flow-distillation loss (IFRNet/RIFE privileged supervision)
# ===========================================================================
def flow_distillation_loss(
    student_flows: Tensor | list[Tensor] | None,
    teacher_flows: Tensor | list[Tensor] | None,
    eps: float = 1e-3,
) -> Tensor:
    r"""Optional task-oriented flow-distillation loss (IFRNet/RIFE; research/06 §5.2).

    When fine-tuning IFRNet/RIFE you can keep the privileged flow supervision term
    :math:`L_d = \sum_k \rho(U(F^k_{t\to l}) - F^p_{t\to l})` — a Charbonnier distance
    between the model's multi-scale intermediate flows (student) and a privileged/teacher
    flow estimate. This is OPTIONAL: it is only active when both ``student_flows`` and
    ``teacher_flows`` are provided; otherwise it returns ``0`` so the combined loss is
    unaffected for backbones that do not expose intermediate flow.

    Lists of multi-scale flows are upsampled to a common resolution and summed. The returned
    tensor is always a scalar; if no flows are given it is a non-grad zero.

    Args:
        student_flows: the model's predicted intermediate flow(s) ``(B, 2, H, W)`` or a list.
        teacher_flows: the privileged/teacher flow(s), same structure/shape.
        eps: Charbonnier epsilon.

    Returns:
        A scalar tensor (``0.0`` when flows are absent).
    """
    if student_flows is None or teacher_flows is None:
        return torch.zeros((), dtype=torch.float32)

    s_list = student_flows if isinstance(student_flows, (list, tuple)) else [student_flows]
    t_list = teacher_flows if isinstance(teacher_flows, (list, tuple)) else [teacher_flows]
    if len(s_list) == 0 or len(t_list) == 0:
        ref = (s_list or t_list)[0]
        return torch.zeros((), dtype=ref.dtype, device=ref.device)

    total = None
    for s, t in zip(s_list, t_list, strict=False):
        if s.shape[-2:] != t.shape[-2:]:
            # Upsample the coarser flow (and scale its magnitude) to match the finer one.
            target_hw = (max(s.shape[-2], t.shape[-2]), max(s.shape[-1], t.shape[-1]))
            if s.shape[-2:] != target_hw:
                scale = target_hw[0] / s.shape[-2]
                s = F.interpolate(s, size=target_hw, mode="bilinear", align_corners=False) * scale
            if t.shape[-2:] != target_hw:
                scale = target_hw[0] / t.shape[-2]
                t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False) * scale
        term = torch.sqrt((s - t) ** 2 + eps * eps).mean()
        total = term if total is None else total + term
    return total


# ===========================================================================
# CombinedLoss — the weighted sum used during training
# ===========================================================================
class CombinedLoss(torch.nn.Module):
    r"""Weighted sum of Charbonnier + census + MS-SSIM + gradient (research/06 §5.3).

    The default weights match :class:`frameflow.config.TrainConfig` and the fine-tune YAML::

        L = w_char * Charbonnier + w_cen * Census
          + w_ms * (1 - MS-SSIM) + w_grad * Gradient
          (+ w_flow * flow_distillation, only when flows are supplied)

    VGG/LPIPS are intentionally excluded (RGB/ImageNet mismatch for 1-channel TIR — see the
    module docstring). The ``__call__`` returns a ``(total, parts)`` tuple so the training
    loop can log each component.

    Args:
        weights: a mapping (or :class:`~frameflow.config.TrainConfig`) supplying the
            ``loss_charbonnier`` / ``loss_census`` / ``loss_ms_ssim`` / ``loss_gradient`` /
            ``loss_flow_distill`` / ``charbonnier_eps`` values. ``None`` uses the config
            defaults.
        data_range: the model's working data range for MS-SSIM (``1.0`` for normalized
            ``[0, 1]`` inputs); NOT the Kelvin metric range.
    """

    #: Default weights (mirrors ``TrainConfig`` / ``configs/train/finetune.yaml``).
    DEFAULTS: dict[str, float] = {
        "loss_charbonnier": 1.0,
        "loss_census": 0.5,
        "loss_ms_ssim": 0.25,
        "loss_gradient": 0.1,
        "loss_flow_distill": 0.01,
        "charbonnier_eps": 1.0e-3,
    }

    def __init__(self, weights: object | None = None, *, data_range: float = 1.0) -> None:
        super().__init__()
        self.data_range = float(data_range)
        self.weights = self._coerce_weights(weights)

    @staticmethod
    def _coerce_weights(weights: object | None) -> dict[str, float]:
        """Normalize ``weights`` (dict / TrainConfig / None) into a plain float dict."""
        out = dict(CombinedLoss.DEFAULTS)
        if weights is None:
            return out
        if isinstance(weights, _config.TrainConfig):
            src: dict[str, float] = {
                "loss_charbonnier": weights.loss_charbonnier,
                "loss_census": weights.loss_census,
                "loss_ms_ssim": weights.loss_ms_ssim,
                "loss_gradient": weights.loss_gradient,
                "loss_flow_distill": weights.loss_flow_distill,
                "charbonnier_eps": weights.charbonnier_eps,
            }
        elif isinstance(weights, dict):
            src = weights  # type: ignore[assignment]
        else:
            # Fall back to attribute access (e.g. an OmegaConf node or namespace).
            src = {}
            for k in CombinedLoss.DEFAULTS:
                if hasattr(weights, k):
                    src[k] = getattr(weights, k)
        for k, v in src.items():
            if k in out and v is not None:
                out[k] = float(v)
        return out

    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        *,
        student_flows: Tensor | list[Tensor] | None = None,
        teacher_flows: Tensor | list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the combined loss and a dict of its (unweighted) component values.

        Args:
            pred: predicted frame ``(B, C, H, W)``.
            gt: ground-truth frame, same shape.
            student_flows: optional model intermediate flow(s) for distillation.
            teacher_flows: optional privileged/teacher flow(s) for distillation.

        Returns:
            ``(total, parts)`` where ``total`` is the scalar weighted sum and ``parts`` maps
            each component name (``charbonnier``/``census``/``ms_ssim``/``gradient``/
            ``flow_distill``) to its raw (unweighted) scalar value (detached-friendly for
            logging, but kept attached so callers may inspect grads if desired).
        """
        eps = self.weights["charbonnier_eps"]
        parts: dict[str, Tensor] = {}

        char = charbonnier_loss(pred, gt, eps=eps)
        cen = census_loss(pred, gt)
        ms = ms_ssim_loss(pred, gt, data_range=self.data_range)
        grad = gradient_loss(pred, gt)
        parts["charbonnier"] = char
        parts["census"] = cen
        parts["ms_ssim"] = ms
        parts["gradient"] = grad

        total = (
            self.weights["loss_charbonnier"] * char
            + self.weights["loss_census"] * cen
            + self.weights["loss_ms_ssim"] * ms
            + self.weights["loss_gradient"] * grad
        )

        if student_flows is not None and teacher_flows is not None:
            flow = flow_distillation_loss(student_flows, teacher_flows, eps=eps)
            parts["flow_distill"] = flow
            total = total + self.weights["loss_flow_distill"] * flow

        return total, parts

r"""Full-reference + brightness-temperature metrics — the FIXED-range rule (P1).

CODE-REVIEW CORRECTION P1 (the central rule of this module):
    All FULL-REFERENCE image-quality metrics (MSE / RMSE / MAE / PSNR / SSIM / MS-SSIM /
    FSIM / GMSD / ...) are computed against a **shared, FIXED physical brightness-temperature
    range in Kelvin** — :data:`frameflow.constants.BT_METRIC_VMIN_K` (180 K) ..
    :data:`~frameflow.constants.BT_METRIC_VMAX_K` (320 K), a span of
    :data:`~frameflow.constants.BT_DATA_RANGE_K` (140 K).

    **Per-image min/max is FORBIDDEN.** WHY: re-normalizing each frame to its own extremes
    cancels exactly the radiometric errors we must catch. A uniformly warm/cold-biased frame,
    or a contrast-compressed (flattened) frame, gets *re-stretched* back onto [0,1] by
    per-image scaling and then scores almost perfectly on SSIM/PSNR — the error is hidden.
    Anchoring every frame to the same physical span makes PSNR/SSIM comparable across frames,
    satellites and methods, and lets a real bias actually move the metric. (See the unit
    test ``test_p1_fixed_range_penalizes_warm_bias`` which demonstrates exactly this.)

Implementation notes:
    * SSIM-family (SSIM / MS-SSIM / FSIM / GMSD) operate on the **fixed-[0,1] arrays**
      produced by :func:`to_unit_interval` (Kelvin -> [0,1] via the FIXED span) with
      ``data_range = 1.0`` — piq requires inputs within ``[0, data_range]``, so we cannot
      pass raw Kelvin with ``data_range = 140``; scaling by the fixed span is the equivalent,
      contract-correct transform.
    * PSNR / MSE / RMSE / MAE are reported BOTH on the fixed-[0,1] arrays (``psnr``, ``mse``,
      ``rmse``, ``mae`` — comparable to the VFI literature) AND in physical Kelvin
      (``bt_rmse_k``, ``bt_bias_k``) so radiometric error is legible in real units.
    * NaN / off-disk pixels are excluded via a mask (the union of the two inputs' NaNs, plus
      any caller mask). SSIM-family need a dense array, so masked pixels are filled with the
      fixed mid-range value before the windowed metrics (they are excluded from the pixel
      metrics exactly).
    * Metrics use ``piq`` (always present) on CPU tensors; the optional ``scikit-image``
      cross-check is used only if installed (else skipped), supporting the dual-implementation
      agreement check (R5 M40).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import constants as C

if TYPE_CHECKING:  # typing only
    import numpy as np
    import torch

    from ..contracts import MetricRecord


__all__ = [
    "to_unit_interval",
    "from_unit_interval",
    "compute_metrics",
    "per_frame_metrics",
    "METRIC_KEYS",
]


#: The metric keys :func:`compute_metrics` always returns.
METRIC_KEYS: tuple[str, ...] = (
    "mse",
    "rmse",
    "mae",
    "psnr",
    "ssim",
    "ms_ssim",
    "fsim",
    "gmsd",
    "bt_rmse_k",
    "bt_bias_k",
)


def to_unit_interval(
    bt: Any,
    *,
    vmin_k: float = C.BT_METRIC_VMIN_K,
    vmax_k: float = C.BT_METRIC_VMAX_K,
    clip: bool = True,
) -> np.ndarray:
    r"""Scale a Kelvin brightness-temperature array to ``[0, 1]`` with the FIXED span (P1).

    ``x' = (BT - vmin_k) / (vmax_k - vmin_k)`` using the **fixed, shared** physical bounds
    (defaults :data:`BT_METRIC_VMIN_K` .. :data:`BT_METRIC_VMAX_K`). This is the ONLY
    sanctioned scaling for the SSIM-family metrics — it is **never** per-image min/max,
    which would hide radiometric errors (see module docstring).

    NaNs are preserved (so callers can mask them); values are clipped to ``[0, 1]`` by
    default so the rare out-of-range pixel cannot violate piq's input-range assertion.

    Args:
        bt: brightness temperature in Kelvin (any shape; NaN allowed).
        vmin_k / vmax_k: the FIXED physical bounds (defaults are the project constants —
            override only with great care; the manifest publishes these).
        clip: clip the result to ``[0, 1]`` (recommended; keeps piq happy).

    Returns:
        A float32 array in ``[0, 1]`` (NaNs preserved), same shape as ``bt``.
    """
    import numpy as np  # lazy

    span = float(vmax_k) - float(vmin_k)
    if span <= 0:
        raise ValueError(f"vmax_k ({vmax_k}) must be > vmin_k ({vmin_k})")
    x = (np.asarray(bt, dtype=np.float32) - float(vmin_k)) / span
    if clip:
        # np.clip propagates NaN (NaN stays NaN), which is what we want.
        x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


def from_unit_interval(
    x: Any,
    *,
    vmin_k: float = C.BT_METRIC_VMIN_K,
    vmax_k: float = C.BT_METRIC_VMAX_K,
) -> np.ndarray:
    """Inverse of :func:`to_unit_interval`: map ``[0,1]`` back to Kelvin via the FIXED span."""
    import numpy as np  # lazy

    span = float(vmax_k) - float(vmin_k)
    return (np.asarray(x, dtype=np.float32) * span + float(vmin_k)).astype(np.float32)


def _as_nchw(x01: np.ndarray) -> torch.Tensor:
    """Wrap a 2D ``[0,1]`` array as a ``(1, 1, H, W)`` float32 CPU tensor for piq."""
    import numpy as np  # lazy
    import torch  # lazy

    a = np.asarray(x01, dtype=np.float32)
    return torch.from_numpy(a[np.newaxis, np.newaxis, :, :]).float()


def _ms_ssim_scales_for(h: int, w: int) -> tuple[int, torch.Tensor, int] | None:
    """Choose (kernel_size, scale_weights, n_scales) so MS-SSIM is valid for an ``H x W`` image.

    piq's MS-SSIM needs each successive 2x-downsampled level to remain larger than the
    Gaussian kernel. For small satellite test patches the default 5-scale / kernel-11 config
    fails, so we pick the largest feasible number of scales (>=1) with an odd kernel that fits
    the smallest dimension, and renormalize the standard MS-SSIM weights to that count.

    Returns ``None`` if the image is too small for even a single-scale windowed metric.
    """
    import torch  # lazy

    smaller = min(int(h), int(w))
    if smaller < 4:
        return None

    # Kernel must be odd and <= smaller dimension; cap at 11 (piq default).
    kernel = min(11, smaller if smaller % 2 == 1 else smaller - 1)
    if kernel < 3:
        kernel = 3
    if kernel % 2 == 0:
        kernel -= 1

    # Standard MS-SSIM level weights (Wang 2003). We keep a prefix and renormalize.
    full_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    # Number of scales such that the smallest level stays > kernel:
    #   after (n-1) halvings, size ~ smaller / 2**(n-1) must exceed `kernel`.
    n_scales = 1
    while n_scales < len(full_weights):
        nxt = n_scales + 1
        if smaller / (2 ** (nxt - 1)) > kernel:
            n_scales = nxt
        else:
            break

    w = full_weights[:n_scales]
    s = sum(w)
    weights = torch.tensor([v / s for v in w], dtype=torch.float32)
    return kernel, weights, n_scales


def _safe_ms_ssim(p01: np.ndarray, t01: np.ndarray) -> float:
    """MS-SSIM on fixed-[0,1] arrays, adapting the scale count to the image size.

    Falls back to single-scale SSIM if the image is too small for any multi-scale pyramid,
    and returns ``nan`` only if even that is impossible. Never raises.
    """
    import piq  # lazy

    h, w = p01.shape[-2:]
    cfg = _ms_ssim_scales_for(h, w)
    tp, tt = _as_nchw(p01), _as_nchw(t01)
    if cfg is None:
        return float("nan")
    kernel, weights, n_scales = cfg
    if n_scales == 1:
        # Degenerate "multi-scale" -> single-scale SSIM with a fitting kernel.
        try:
            return float(piq.ssim(tp, tt, data_range=1.0, kernel_size=kernel))
        except Exception:
            return float("nan")
    try:
        return float(
            piq.multi_scale_ssim(
                tp, tt, data_range=1.0, kernel_size=kernel, scale_weights=weights
            )
        )
    except Exception:
        try:
            return float(piq.ssim(tp, tt, data_range=1.0, kernel_size=kernel))
        except Exception:
            return float("nan")


def _safe_ssim(p01: np.ndarray, t01: np.ndarray) -> float:
    """Single-scale SSIM on fixed-[0,1] arrays with a kernel that fits the image."""
    import piq  # lazy

    h, w = p01.shape[-2:]
    smaller = min(int(h), int(w))
    kernel = min(11, smaller if smaller % 2 == 1 else smaller - 1)
    if kernel < 3:
        kernel = 3
    if kernel % 2 == 0:
        kernel -= 1
    try:
        return float(piq.ssim(_as_nchw(p01), _as_nchw(t01), data_range=1.0, kernel_size=kernel))
    except Exception:
        return float("nan")


def _safe_fsim(p01: np.ndarray, t01: np.ndarray) -> float:
    """FSIM on fixed-[0,1] arrays (replicate single channel to 3 — piq FSIM expects RGB)."""
    import piq  # lazy

    tp = _as_nchw(p01).repeat(1, 3, 1, 1)
    tt = _as_nchw(t01).repeat(1, 3, 1, 1)
    try:
        return float(piq.fsim(tp, tt, data_range=1.0, chromatic=False))
    except Exception:
        return float("nan")


def _safe_gmsd(p01: np.ndarray, t01: np.ndarray) -> float:
    """GMSD on fixed-[0,1] arrays (lower is better; 0 == identical)."""
    import piq  # lazy

    try:
        return float(piq.gmsd(_as_nchw(p01), _as_nchw(t01), data_range=1.0))
    except Exception:
        return float("nan")


def compute_metrics(
    pred_k: Any,
    gt_k: Any,
    *,
    mask: Any | None = None,
    vmin_k: float = C.BT_METRIC_VMIN_K,
    vmax_k: float = C.BT_METRIC_VMAX_K,
) -> dict[str, float]:
    r"""Compute the full-reference + BT-domain metric suite for ONE frame (FIXED range, P1).

    Returns a dict with keys :data:`METRIC_KEYS`:

        * ``mse`` / ``rmse`` / ``mae`` — pixel error on the **fixed-[0,1]** arrays
          (comparable across frames/methods; dimensionless).
        * ``psnr`` — peak-signal-to-noise (dB) on the fixed-[0,1] arrays
          (``data_range = 1.0``); a uniform bias degrades this (it is NOT hidden).
        * ``ssim`` / ``ms_ssim`` / ``fsim`` — structural / multi-scale / feature similarity
          (higher better), on the fixed-[0,1] arrays (``data_range = 1.0``).
        * ``gmsd`` — gradient-magnitude similarity deviation (lower better).
        * ``bt_rmse_k`` — RMSE in physical **Kelvin** (the headline radiometric error).
        * ``bt_bias_k`` — mean(pred - gt) in **Kelvin** (signed; catches warm/cold drift).

    ALL of these use the FIXED physical span (``vmin_k`` / ``vmax_k``); none use per-image
    min/max (P1). NaN / off-disk pixels (the union of both inputs' NaNs plus any ``mask``)
    are excluded from the exact pixel/Kelvin metrics and filled with the fixed mid-range
    value before the windowed SSIM-family metrics.

    Args:
        pred_k: predicted brightness temperature, Kelvin ``(H, W)`` (NaN allowed).
        gt_k: ground-truth brightness temperature, Kelvin ``(H, W)`` (same shape).
        mask: optional boolean array; ``True`` marks pixels to EXCLUDE (in addition to NaNs).
        vmin_k / vmax_k: the FIXED physical bounds (defaults are the project constants).

    Returns:
        ``dict[str, float]`` of the metric values (``nan`` for any metric that could not be
        computed on the given image size).

    Raises:
        ValueError: if ``pred_k`` and ``gt_k`` shapes differ.
    """
    import numpy as np  # lazy

    p = np.asarray(pred_k, dtype=np.float32)
    g = np.asarray(gt_k, dtype=np.float32)
    if p.shape != g.shape:
        raise ValueError(f"pred_k {p.shape} and gt_k {g.shape} must have the same shape")
    if p.ndim != 2:
        p = np.squeeze(p)
        g = np.squeeze(g)
    if p.ndim != 2:
        raise ValueError(f"compute_metrics expects 2D frames; got {p.shape!r}")

    # Exclusion mask: NaN in either input OR caller-supplied mask.
    exclude = np.isnan(p) | np.isnan(g)
    if mask is not None:
        exclude = exclude | np.asarray(mask, dtype=bool)
    valid = ~exclude

    out: dict[str, float] = {k: float("nan") for k in METRIC_KEYS}
    if not np.any(valid):
        return out

    # --- exact Kelvin-domain metrics (over valid pixels only) ----------------------------
    diff_k = (p - g)[valid]
    out["bt_bias_k"] = float(np.mean(diff_k))
    out["bt_rmse_k"] = float(np.sqrt(np.mean(diff_k ** 2)))

    # --- fixed-[0,1] scaling for everything else (P1) ------------------------------------
    p01 = to_unit_interval(p, vmin_k=vmin_k, vmax_k=vmax_k)
    t01 = to_unit_interval(g, vmin_k=vmin_k, vmax_k=vmax_k)

    # Exact pixel metrics on the fixed-[0,1] arrays (over valid pixels).
    dp = (p01 - t01)
    dp_valid = dp[valid]
    mse = float(np.mean(dp_valid ** 2))
    out["mse"] = mse
    out["rmse"] = float(np.sqrt(mse))
    out["mae"] = float(np.mean(np.abs(dp_valid)))
    # PSNR with the fixed data_range = 1.0 (the [0,1] peak). Uniform bias degrades this.
    if mse <= 0:
        out["psnr"] = float("inf")
    else:
        out["psnr"] = float(10.0 * np.log10(1.0 / mse))

    # --- windowed structure metrics need a dense array: fill excluded pixels -------------
    # Fill with the fixed mid-range (0.5) so masked regions are neutral and identical in both
    # arrays (they contribute structure-neutral, equal content and do not bias the score).
    fill = 0.5
    p01_f = np.where(valid, p01, fill).astype(np.float32)
    t01_f = np.where(valid, t01, fill).astype(np.float32)

    out["ssim"] = _safe_ssim(p01_f, t01_f)
    out["ms_ssim"] = _safe_ms_ssim(p01_f, t01_f)
    out["fsim"] = _safe_fsim(p01_f, t01_f)
    out["gmsd"] = _safe_gmsd(p01_f, t01_f)

    return out


def per_frame_metrics(
    pred_k: Any,
    true_k: Any,
    *,
    data_range_k: float = C.BT_DATA_RANGE_K,
    mask: Any | None = None,
    index: int = 0,
    time: str | None = None,
) -> MetricRecord:
    """Compute the full-reference + BT-domain metrics for ONE frame (CONTRACTS §6.3).

    Thin wrapper over :func:`compute_metrics` that returns a
    :class:`frameflow.contracts.MetricRecord` (the per-frame manifest record) instead of a
    bare dict. ALL full-reference metrics use the FIXED physical ``data_range_k`` (P1) — the
    fixed span is converted back into the ``vmin_k``/``vmax_k`` pair the metric core uses, so
    a caller passing the default 140 K gets exactly the project's [180, 320] K range. NaN /
    off-disk pixels are excluded via ``mask`` (plus the inputs' own NaNs).

    Args:
        pred_k: predicted brightness temperature, Kelvin ``(H, W)`` (NaN allowed).
        true_k: ground-truth brightness temperature, Kelvin ``(H, W)`` (same shape).
        data_range_k: the FIXED physical Kelvin span (default
            :data:`frameflow.constants.BT_DATA_RANGE_K` = 140 K). Anchored at
            :data:`frameflow.constants.BT_METRIC_VMIN_K`.
        mask: optional boolean array; ``True`` marks pixels to EXCLUDE.
        index: the frame index recorded on the returned :class:`MetricRecord`.
        time: optional ISO-8601 timestamp recorded on the record.

    Returns:
        A :class:`frameflow.contracts.MetricRecord` with the computed metrics; metrics not in
        the record's named fields (e.g. ``gmsd``) are stored under ``extra``.
    """
    from ..contracts import MetricRecord  # local import (contracts is import-light)

    vmin_k = float(C.BT_METRIC_VMIN_K)
    vmax_k = vmin_k + float(data_range_k)
    m = compute_metrics(pred_k, true_k, mask=mask, vmin_k=vmin_k, vmax_k=vmax_k)
    named = {"mse", "rmse", "mae", "psnr", "ssim", "ms_ssim", "fsim",
             "bt_rmse_k", "bt_bias_k"}
    extra = {k: v for k, v in m.items() if k not in named}
    return MetricRecord(
        index=int(index),
        time=time,
        psnr=m.get("psnr"),
        ssim=m.get("ssim"),
        ms_ssim=m.get("ms_ssim"),
        fsim=m.get("fsim"),
        mse=m.get("mse"),
        rmse=m.get("rmse"),
        mae=m.get("mae"),
        bt_rmse_k=m.get("bt_rmse_k"),
        bt_bias_k=m.get("bt_bias_k"),
        extra=extra,
    )

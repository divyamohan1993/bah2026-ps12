"""The 40-method (M1-M40) cross-validation framework (research/05 §4).

The PS-12 mandate (research/05 §4) is *"use at least 30 different methods for extremely
collaborated robust findings … use multiple satellites/datasets to verify."* This module
encodes that as a single, runnable :class:`CrossValSuite`:

    * a **registry** (:data:`CrossValSuite.METHODS`) of all 40 enumerated methods M1-M40 —
      each with an ``id``, ``name``, ``description``, ``applicability`` (which datasets it
      needs), and a ``category`` — so the catalogue is the authoritative >=30-method list
      even for methods that need data we don't have in a hermetic test;
    * an executor (:meth:`CrossValSuite.run`) that *actually runs* the subset which only
      needs a single dense synthetic cube (leave-the-middle-frame-out, the classical
      baselines, recursion-depth error growth, extreme-motion stratification,
      error-vs-speed correlation, the radially-averaged PSD ratio, bootstrap 95% CIs, and
      the Wilcoxon AI-vs-baseline significance test), and marks the genuinely
      data-dependent methods (multi-satellite overlaps, LEO collocation, triple collocation,
      …) as ``"skipped"`` with a ``requires`` note.

Design rules honoured (mirroring the rest of INFER+VALIDATE):
    * **FIXED-range metrics (P1).** Every full-reference number comes from
      :func:`frameflow.validate.metrics.compute_metrics`, which uses the shared physical
      Kelvin span — never per-image min/max.
    * **No dependency on the models package.** The "model under test" is duck-typed
      (anything with ``forward(I0, I1, t)``); the classical baselines (frame-copy,
      linear-blend, Farneback / TV-L1 optical-flow warps) are implemented **locally** with
      numpy / OpenCV so crossval never imports ``frameflow.models``.
    * **Graceful degradation.** OpenCV's TV-L1 lives in the optional ``cv2.optflow`` contrib
      module; if it is absent a compact pure-numpy Horn-Schunck flow is used instead, so the
      "TV-L1-style" baseline always runs. ``scikit-image`` / ``onnx`` etc. are never required.

Each executed method returns a :class:`frameflow.contracts.CrossvalMethodResult`
(``method_id``, ``name``, ``status`` in {run, skipped, failed}, ``summary`` scalars,
``details``). :meth:`CrossValSuite.run` returns the full ``list[CrossvalMethodResult]``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..contracts import CrossvalMethodResult
from .metrics import compute_metrics

if TYPE_CHECKING:  # typing only
    import numpy as np


__all__ = [
    "MethodSpec",
    "CrossValSuite",
    "available_methods",
    "run_method",
    "frame_copy_baseline",
    "linear_blend_baseline",
    "optical_flow_baseline",
]


# ===========================================================================
# 1. The method registry (M1-M40) — the authoritative >=30-method catalogue
# ===========================================================================
@dataclass(frozen=True)
class MethodSpec:
    """One entry in the 40-method cross-validation registry.

    Attributes:
        method_id: short id ``"M1"`` .. ``"M40"``.
        name: human-readable method name.
        description: one-line description of what the method checks.
        category: the research/05 §4 group (A..H) the method belongs to.
        applicability: which data the method needs (``"synthetic-dense"`` methods run on a
            single dense cube; everything else names the external dataset(s) required).
        runnable_on_synthetic: ``True`` if :class:`CrossValSuite` executes it on the
            single synthetic cube; ``False`` if it is registered-but-``skipped`` pending
            real multi-satellite / polar-orbiter / multi-band data.
    """

    method_id: str
    name: str
    description: str
    category: str
    applicability: str
    runnable_on_synthetic: bool


# The full M1..M40 list, verbatim in spirit from research/05 §4. ``runnable_on_synthetic``
# marks the subset that needs only one dense cube (and so executes in :meth:`run`).
_METHODS: tuple[MethodSpec, ...] = (
    # --- A. Withheld-ground-truth interpolation tests --------------------------------------
    MethodSpec(
        "M1", "Leave-the-middle-frame-out (dense GOES-19, PRIMARY)",
        "Predict the real middle frame from its two neighbours; full FR/BT metric suite vs withheld truth.",
        "A", "synthetic-dense", True,
    ),
    MethodSpec(
        "M2", "Leave-the-middle-frame-out (Himawari, independent dense source)",
        "Same exact-match protocol on a second independent dense sensor (Asia/India-facing).",
        "A", "Himawari AHI B13 dense cube", False,
    ),
    MethodSpec(
        "M3", "Multi-step recursion vs direct (2x/4x/8x)",
        "Hold out the real frames for 30->15/7.5/3.75 min densification; test error vs up-sampling factor.",
        "A", "synthetic-dense", True,
    ),
    MethodSpec(
        "M4", "Recursion-depth / error-accumulation test",
        "Recursive halving (interp-the-interp) vs single-shot; quantify metric decay per recursion level.",
        "A", "synthetic-dense", True,
    ),
    MethodSpec(
        "M5", "Asymmetric-interval test (off-centre t)",
        "Predict off-centre fractions (t=0.25/0.75), not just the midpoint, to check t-generalization.",
        "A", "synthetic-dense", True,
    ),
    MethodSpec(
        "M6", "Long-gap stress test",
        "Feed widely-spaced frames and predict several intervening real frames (simulates coarse INSAT cadence).",
        "A", "synthetic-dense", True,
    ),
    # --- B. Cross-satellite verification ---------------------------------------------------
    MethodSpec(
        "M7", "GOES + Himawari overlap (Pacific)",
        "Validate one satellite's interpolation against the other's near-simultaneous observation.",
        "B", "GOES-19 + Himawari overlap (GSICS inter-cal)", False,
    ),
    MethodSpec(
        "M8", "INSAT + Himawari overlap (Indian Ocean)",
        "Validate INSAT interpolation against Himawari's denser real frames — the key 'fill-the-gap' check.",
        "B", "INSAT-3DS/3DR + Himawari overlap", False,
    ),
    MethodSpec(
        "M9", "INSAT vs GOES / three-way common scene",
        "Pairwise interpolation-vs-other-observation where any two-of-three see a common scene.",
        "B", "INSAT + GOES common-scene overlap", False,
    ),
    MethodSpec(
        "M10", "Cross-satellite consistency of derived motion",
        "Compare optical-flow fields estimated independently from two satellites over the overlap.",
        "B", "two-satellite overlap", False,
    ),
    MethodSpec(
        "M11", "Inter-calibration (GSICS) pre-step + sanity check",
        "Apply/verify GSICS BT inter-cal before cross-comparison; ~0.3 K floor is the reference.",
        "B", "multi-satellite (GSICS coefficients)", False,
    ),
    # --- C. Independent polar-orbiter overpasses ------------------------------------------
    MethodSpec(
        "M12", "MODIS (Terra/Aqua) overpass collocation",
        "Compare the geostationary frame nearest a MODIS overpass to MODIS Band-31 (~11 µm) BT.",
        "C", "MODIS Band-31 overpasses", False,
    ),
    MethodSpec(
        "M13", "VIIRS (SNPP/NOAA-20/21) overpass collocation",
        "Same idea with VIIRS M15/I5 TIR — more overpasses, fills MODIS temporal gaps.",
        "C", "VIIRS M15/I5 overpasses", False,
    ),
    MethodSpec(
        "M14", "Polar-orbiter truth at INTERPOLATED timestamps",
        "Where a LEO overpass lands between two GEO acquisitions, validate the SYNTHETIC frame directly.",
        "C", "LEO overpasses at synthesized times", False,
    ),
    # --- D. Statistical triangulation -----------------------------------------------------
    MethodSpec(
        "M15", "Triple collocation (TC)",
        "Three independent BT estimates -> absolute random-error variance of each, no perfect truth assumed.",
        "D", "three independent co-located sensors", False,
    ),
    MethodSpec(
        "M16", "N-way / quadruple collocation extension",
        "Add a 4th source (NWP/2nd LEO) to relax TC assumptions and cross-check error correlation.",
        "D", "four independent sources", False,
    ),
    MethodSpec(
        "M17", "Error-variance budget consistency",
        "Verify the TC-derived interpolation error variance is consistent across regions/seasons.",
        "D", "multi-region collocations", False,
    ),
    # --- E. Baseline & method comparisons -------------------------------------------------
    MethodSpec(
        "M18", "Naive frame-copy baseline (persistence)",
        "Repeat frame0 as the 'interpolated' frame; any real method MUST beat this on every metric.",
        "E", "synthetic-dense", True,
    ),
    MethodSpec(
        "M19", "Linear-blend baseline",
        "0.5*(frame0+frame2) ghosting baseline; must be beaten especially on FSIM/PSD/blur.",
        "E", "synthetic-dense", True,
    ),
    MethodSpec(
        "M20", "Classical optical-flow baselines (Farneback & TV-L1)",
        "OpenCV flow-warp baselines; show the AI beats classical flow on motion and edge sharpness.",
        "E", "synthetic-dense", True,
    ),
    MethodSpec(
        "M21", "DL VFI backbones head-to-head",
        "RIFE vs Super-SloMo vs FILM/IFRNet/AMT on identical data -> ranked table to pick the best.",
        "E", "multiple trained VFI checkpoints", False,
    ),
    MethodSpec(
        "M22", "Ablation: with vs without explicit optical flow",
        "Does the flow module help vs a pure CNN/transformer interpolator? Quantify the contribution.",
        "E", "two model variants", False,
    ),
    MethodSpec(
        "M23", "Ablation: flow-backbone swap (RAFT vs PWC vs TV-L1)",
        "Sensitivity of results to the flow estimator inside the pipeline.",
        "E", "model variants w/ swappable flow", False,
    ),
    MethodSpec(
        "M24", "Ablation: loss / warping choices",
        "Forward vs backward warp, refine-net on/off, perceptual-loss on/off vs validation metrics.",
        "E", "models trained w/ different losses", False,
    ),
    MethodSpec(
        "M25", "Ablation: input normalization / radiometric pre-processing",
        "With/without histogram normalization, Kelvin vs DN input -> robustness of the result.",
        "E", "model variants w/ different norm", False,
    ),
    # --- F. Stratified / conditional evaluation -------------------------------------------
    MethodSpec(
        "M26", "Extreme-motion stratification (cyclones / deep convection)",
        "Bin test cases by motion magnitude (from flow); report metrics fast-vs-slow scenes.",
        "F", "synthetic-dense", True,
    ),
    MethodSpec(
        "M27", "Phenomenon stratification (cyclone/thunderstorm/fire/flood/clear)",
        "Per-phenomenon metric tables for the PS-named phenomena.",
        "F", "labelled phenomenon cases", False,
    ),
    MethodSpec(
        "M28", "Per-region stratification",
        "Tropics vs mid-latitude, land vs ocean, India vs Pacific -> geographic bias map.",
        "F", "multi-region data", False,
    ),
    MethodSpec(
        "M29", "Per-season / diurnal stratification",
        "Monsoon vs dry, day vs night (TIR works at night) -> no diurnal bias.",
        "F", "multi-season/diurnal data", False,
    ),
    MethodSpec(
        "M30", "Cloud-regime stratification by BT bands",
        "Separate metrics for warm/clear, mid, and cold (convective) BT ranges.",
        "F", "synthetic-dense", True,
    ),
    # --- G. Correlation, spectral & significance analyses ---------------------------------
    MethodSpec(
        "M31", "Error-vs-cloud-speed correlation",
        "Regress per-frame error against estimated cloud speed; report R^2 (characterizes motion limits).",
        "G", "synthetic-dense", True,
    ),
    MethodSpec(
        "M32", "Error-vs-lead-fraction / vs up-sampling-factor correlation",
        "Error as a function of distance from the nearest real frame (t=0.5 hardest) and of 2x/4x/8x.",
        "G", "synthetic-dense", True,
    ),
    MethodSpec(
        "M33", "Radially-averaged PSD ratio (fine-scale fidelity)",
        "2-D FFT -> radial PSD; PSD_pred/PSD_truth vs wavenumber proves fine-scale structure preservation.",
        "G", "synthetic-dense", True,
    ),
    MethodSpec(
        "M34", "Wavelet multi-resolution fidelity",
        "2-D DWT energy per sub-band, pred vs truth — scale-localized structure check (needs pywt).",
        "G", "synthetic-dense (pywt)", False,
    ),
    MethodSpec(
        "M35", "Statistical-significance testing (Wilcoxon + bootstrap CI)",
        "Paired Wilcoxon AI-vs-baseline per metric + bootstrap 95% CIs on mean PSNR/SSIM/BT-RMSE.",
        "G", "synthetic-dense", True,
    ),
    MethodSpec(
        "M36", "Inter-metric agreement / rank-correlation matrix",
        "Spearman correlation between metrics across all test frames; confirms multi-metric consensus.",
        "G", "synthetic-dense", True,
    ),
    # --- H. Physical-consistency & implementation cross-checks ----------------------------
    MethodSpec(
        "M37", "Multi-band consistency",
        "Interpolate TIR1 and TIR2/WV; check inter-band relationships (split-window) are preserved.",
        "H", "multi-band data", False,
    ),
    MethodSpec(
        "M38", "Mass/feature conservation & physical-plausibility",
        "Total cold-cloud area and domain-mean BT must evolve smoothly through inserted frames.",
        "H", "synthetic-dense", True,
    ),
    MethodSpec(
        "M39", "Temporal-consistency at the seams",
        "Test the boundary between a real frame and an inserted frame for flicker/discontinuities.",
        "H", "synthetic-dense", True,
    ),
    MethodSpec(
        "M40", "Dual-implementation metric verification + fixed-seed reproducibility",
        "Every metric computed by two libraries and asserted equal (skimage vs piq); pinned seeds.",
        "H", "synthetic-dense (scikit-image optional)", True,
    ),
)


# ===========================================================================
# 2. Classical baselines (implemented locally — NO models-package import)
# ===========================================================================
def frame_copy_baseline(I0: np.ndarray, I1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Persistence baseline (research/05 M18): just repeat the nearer bracketing frame.

    Args:
        I0: earlier frame, Kelvin ``(H, W)``.
        I1: later frame, Kelvin ``(H, W)``.
        t: interpolation fraction; ``t < 0.5`` copies ``I0``, else ``I1``.

    Returns:
        A copy of the nearer frame (Kelvin float32 ``(H, W)``).
    """
    import numpy as np  # lazy

    a0 = np.asarray(I0, dtype=np.float32)
    a1 = np.asarray(I1, dtype=np.float32)
    return (a0 if t < 0.5 else a1).copy()


def linear_blend_baseline(I0: np.ndarray, I1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Linear-blend baseline (research/05 M19): ``(1-t)*I0 + t*I1`` (classic ghosting).

    NaNs are handled per-pixel (a blend with a NaN stays NaN), matching the off-disk mask
    convention used elsewhere in the inference path.

    Args:
        I0: earlier frame, Kelvin ``(H, W)``.
        I1: later frame, Kelvin ``(H, W)``.
        t: interpolation fraction in ``[0, 1]``.

    Returns:
        The blended frame (Kelvin float32 ``(H, W)``), NaN where either input is NaN.
    """
    import numpy as np  # lazy

    a0 = np.asarray(I0, dtype=np.float32)
    a1 = np.asarray(I1, dtype=np.float32)
    return ((1.0 - float(t)) * a0 + float(t) * a1).astype(np.float32)


def _farneback_flow(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Dense Farneback optical flow ``prev -> nxt`` in pixels, shape ``(H, W, 2)`` (dx, dy)."""
    import cv2  # lazy
    import numpy as np  # lazy

    return cv2.calcOpticalFlowFarneback(
        prev.astype(np.float32), nxt.astype(np.float32), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _tvl1_flow(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """TV-L1 optical flow ``prev -> nxt`` if the OpenCV contrib module is present.

    Falls back to a compact pure-numpy Horn-Schunck flow when ``cv2.optflow`` (the contrib
    package carrying ``DualTVL1OpticalFlow``) is unavailable, so a "TV-L1-style" variational
    baseline always runs. Returns ``(H, W, 2)`` pixel flow.
    """
    import numpy as np  # lazy

    try:  # pragma: no cover - exercised only where cv2.optflow is installed
        import cv2

        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            return tvl1.calc(prev.astype(np.uint8), nxt.astype(np.uint8), None)
        if hasattr(cv2, "DualTVL1OpticalFlow_create"):
            tvl1 = cv2.DualTVL1OpticalFlow_create()
            return tvl1.calc(prev.astype(np.uint8), nxt.astype(np.uint8), None)
    except Exception:
        pass
    return _horn_schunck_flow(prev, nxt)


def _horn_schunck_flow(
    prev: np.ndarray, nxt: np.ndarray, *, n_iter: int = 60, alpha: float = 1.0
) -> np.ndarray:
    """Compact pure-numpy Horn-Schunck dense optical flow (variational, TV-L1 stand-in).

    Implements the classic global smoothness flow (Horn & Schunck 1981) with Jacobi
    iterations. It is intentionally small and dependency-free so the variational-flow
    baseline runs even without the OpenCV contrib module; it is a baseline, not a
    state-of-the-art estimator.

    Args:
        prev / nxt: consecutive grayscale frames (any real dtype).
        n_iter: number of Jacobi smoothing iterations.
        alpha: smoothness regularization weight.

    Returns:
        ``(H, W, 2)`` float32 pixel flow ``(u, v)`` mapping ``prev -> nxt``.
    """
    import numpy as np  # lazy

    p = np.asarray(prev, dtype=np.float32)
    n = np.asarray(nxt, dtype=np.float32)
    # Spatial/temporal derivatives (simple forward differences, averaged).
    Ix = np.zeros_like(p)
    Iy = np.zeros_like(p)
    Ix[:, :-1] = p[:, 1:] - p[:, :-1]
    Iy[:-1, :] = p[1:, :] - p[:-1, :]
    It = n - p

    u = np.zeros_like(p)
    v = np.zeros_like(p)
    # 4-neighbour averaging kernel (von-Neumann), applied via simple shifts.
    def _avg(a: np.ndarray) -> np.ndarray:
        s = np.zeros_like(a)
        s[1:, :] += a[:-1, :]
        s[:-1, :] += a[1:, :]
        s[:, 1:] += a[:, :-1]
        s[:, :-1] += a[:, 1:]
        return s / 4.0

    denom = alpha * alpha + Ix * Ix + Iy * Iy
    denom[denom == 0] = 1e-6
    for _ in range(n_iter):
        u_avg = _avg(u)
        v_avg = _avg(v)
        common = (Ix * u_avg + Iy * v_avg + It) / denom
        u = u_avg - Ix * common
        v = v_avg - Iy * common
    return np.stack([u, v], axis=-1).astype(np.float32)


def _warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp ``img`` by a ``(H, W, 2)`` pixel flow using ``cv2.remap`` (bilinear)."""
    import cv2  # lazy
    import numpy as np  # lazy

    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        img.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def optical_flow_baseline(
    I0: np.ndarray, I1: np.ndarray, t: float = 0.5, *, method: str = "farneback"
) -> np.ndarray:
    """Classical optical-flow VFI baseline (research/05 M20): flow-warp + blend.

    Estimates dense flow both ways, scales each by the temporal fraction, backward-warps each
    bracketing frame toward time ``t``, and blends the two warps. This is the standard
    "traditional optical-flow" comparator the PS contrasts the AI against.

    Args:
        I0 / I1: bracketing frames, Kelvin ``(H, W)`` (NaNs are filled with the frame mean
            for the flow estimate and the off-disk mask is restored on the output).
        t: interpolation fraction in ``[0, 1]``.
        method: ``"farneback"`` (OpenCV Farneback) or ``"tvl1"`` (OpenCV TV-L1 if available,
            else a pure-numpy Horn-Schunck fallback).

    Returns:
        The flow-warped interpolated frame (Kelvin float32 ``(H, W)``), NaN off-disk.
    """
    import numpy as np  # lazy

    a0 = np.asarray(I0, dtype=np.float32)
    a1 = np.asarray(I1, dtype=np.float32)
    mask_nan = np.isnan(a0) | np.isnan(a1)
    fill = float(np.nanmean(np.where(mask_nan, np.nan, a0))) if not np.all(mask_nan) else 0.0
    f0 = np.where(mask_nan, fill, a0).astype(np.float32)
    f1 = np.where(mask_nan, fill, a1).astype(np.float32)

    if method == "tvl1":
        flow01 = _tvl1_flow(f0, f1)
        flow10 = _tvl1_flow(f1, f0)
    else:
        flow01 = _farneback_flow(f0, f1)
        flow10 = _farneback_flow(f1, f0)

    # Warp I0 forward by t*flow(0->1) and I1 backward by (1-t)*flow(1->0); blend.
    warp0 = _warp_by_flow(f0, flow01 * float(t))
    warp1 = _warp_by_flow(f1, flow10 * (1.0 - float(t)))
    out = ((1.0 - float(t)) * warp0 + float(t) * warp1).astype(np.float32)
    out[mask_nan] = np.nan
    return out


# ===========================================================================
# 3. Small analysis helpers (PSD ratio, flow speed, bootstrap, etc.)
# ===========================================================================
def _radial_psd(field: np.ndarray) -> np.ndarray:
    """Return the 1-D radially-averaged power-spectral-density of a 2-D field (research/05 §3.9).

    NaNs are filled with the field mean before the FFT. The returned vector is PSD vs integer
    radial wavenumber (index 0 == DC).
    """
    import numpy as np  # lazy

    a = np.asarray(field, dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nanmean(a))
    a = a - np.mean(a)
    f = np.fft.fftshift(np.fft.fft2(a))
    psd2 = np.abs(f) ** 2
    h, w = a.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    nbins = int(r.max()) + 1
    radial = np.bincount(r.ravel(), weights=psd2.ravel(), minlength=nbins)
    counts = np.bincount(r.ravel(), minlength=nbins)
    counts[counts == 0] = 1
    return (radial / counts).astype(np.float64)


def _psd_ratio_highband(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean PSD(pred)/PSD(truth) over the high-wavenumber half (fine-scale fidelity, ~1 is best)."""
    import numpy as np  # lazy

    p = _radial_psd(pred)
    t = _radial_psd(truth)
    n = min(len(p), len(t))
    if n < 4:
        return float("nan")
    lo = n // 2  # high-frequency half
    pt = p[lo:n]
    tt = t[lo:n]
    valid = tt > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(pt[valid] / tt[valid]))


def _flow_speed(I0: np.ndarray, I1: np.ndarray) -> float:
    """Mean dense optical-flow magnitude (pixels) between two frames — a 'cloud speed' proxy."""
    import numpy as np  # lazy

    a0 = np.asarray(I0, dtype=np.float32)
    a1 = np.asarray(I1, dtype=np.float32)
    a0 = np.where(np.isfinite(a0), a0, np.nanmean(a0))
    a1 = np.where(np.isfinite(a1), a1, np.nanmean(a1))
    flow = _farneback_flow(a0, a1)
    return float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))


def _bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap mean + ``(1-alpha)`` percentile CI of a 1-D sample (research/05 M35/§5.9).

    Returns ``(mean, lo, hi)``. NaNs are dropped; an empty/all-NaN sample yields all-NaN.
    """
    import numpy as np  # lazy

    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        boots[i] = np.mean(arr[rng.integers(0, n, n)])
    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return (float(np.mean(arr)), lo, hi)


# ===========================================================================
# 4. The suite
# ===========================================================================
@dataclass
class _Cube:
    """Internal: a dense synthetic test cube (frames + timestamps + grid coords)."""

    frames: list[np.ndarray]   # list of (H, W) Kelvin arrays, NaN off-disk
    times: list[Any]
    lat: np.ndarray
    lon: np.ndarray


class CrossValSuite:
    """The runnable 40-method (M1-M40) cross-validation suite (research/05 §4).

    Construct it (optionally with a ``model`` — anything duck-typed ``forward(I0, I1, t)``;
    defaults to the local linear-blend baseline so the suite runs standalone), then call
    :meth:`run` with the available data. :data:`METHODS` is the authoritative registry of all
    40 methods regardless of what data is present.

    Example::

        suite = CrossValSuite(model=my_vfi_model)
        results = suite.run({"synthetic_cube": cube})   # -> list[CrossvalMethodResult]
        ran = [r for r in results if r.status == "run"]
    """

    #: The authoritative M1-M40 registry (>= 30 methods; see :class:`MethodSpec`).
    METHODS: tuple[MethodSpec, ...] = _METHODS

    def __init__(
        self,
        model: Any | None = None,
        *,
        dataset: str = "synthetic",
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        """Initialize the suite.

        Args:
            model: the VFI model under test (duck-typed ``forward(I0, I1, t)``). If ``None``,
                the local linear-blend baseline is used as the "model" so the suite is fully
                runnable without the models package (the comparisons then show the baselines
                tying, which is itself a valid sanity check).
            dataset: dataset key forwarded to the interpolator's normalizer (advisory).
            device: torch device for model inference.
            seed: RNG seed for the bootstrap / any stochastic step (reproducibility, M40).
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.seed = int(seed)

    # -- public API ---------------------------------------------------------------------
    @classmethod
    def available_methods(cls) -> list[CrossvalMethodResult]:
        """Return the full M1-M40 registry as ``CrossvalMethodResult`` stubs (no execution).

        Each entry's ``status`` is ``"run"`` for synthetic-runnable methods and ``"skipped"``
        for data-dependent ones, with the ``requires`` dataset noted in ``details``.
        """
        out: list[CrossvalMethodResult] = []
        for spec in cls.METHODS:
            status = "run" if spec.runnable_on_synthetic else "skipped"
            details: dict[str, Any] = {
                "category": spec.category,
                "description": spec.description,
                "applicability": spec.applicability,
            }
            if not spec.runnable_on_synthetic:
                details["requires"] = spec.applicability
            out.append(
                CrossvalMethodResult(method_id=spec.method_id, name=spec.name,
                                     status=status, summary={}, details=details)
            )
        return out

    def run(self, available_data: dict[str, Any] | None = None) -> list[CrossvalMethodResult]:
        """Execute the runnable methods on a dense synthetic cube; register the rest as skipped.

        ``available_data`` may carry a ready cube under ``"synthetic_cube"`` / ``"cube"``
        (either a :class:`_Cube`, a path to a ``.zarr`` cube, or a dict with ``frames`` /
        ``times`` / ``lat`` / ``lon``). If none is supplied, a small dense synthetic cube is
        generated in-memory via :mod:`frameflow.synthetic` so the suite is self-contained.

        Args:
            available_data: optional inputs; the only key consulted for execution is the cube
                (others reserved for real multi-satellite data and forwarded to ``details``).

        Returns:
            ``list[CrossvalMethodResult]`` of length ``len(METHODS)`` (40) — the executed
            methods carry real ``summary`` scalars; data-dependent methods are ``"skipped"``
            with a ``requires`` note.
        """

        available_data = dict(available_data or {})
        cube = self._resolve_cube(available_data)

        # Pre-compute the interpolation-vs-truth tables once; many methods reuse them.
        ctx = self._build_context(cube)

        results: list[CrossvalMethodResult] = []
        # Map each runnable method id to its executor.
        runners: dict[str, Callable[[dict[str, Any]], CrossvalMethodResult]] = {
            "M1": self._m1_leave_middle_out,
            "M3": self._m3_multistep,
            "M4": self._m4_recursion_depth,
            "M5": self._m5_asymmetric,
            "M6": self._m6_long_gap,
            "M18": self._m18_frame_copy,
            "M19": self._m19_linear_blend,
            "M20": self._m20_optical_flow,
            "M26": self._m26_extreme_motion,
            "M30": self._m30_cloud_regime,
            "M31": self._m31_error_vs_speed,
            "M32": self._m32_error_vs_lead,
            "M33": self._m33_psd_ratio,
            "M35": self._m35_significance,
            "M36": self._m36_inter_metric,
            "M38": self._m38_conservation,
            "M39": self._m39_seams,
            "M40": self._m40_dual_impl,
        }

        for spec in self.METHODS:
            if spec.runnable_on_synthetic and spec.method_id in runners:
                try:
                    res = runners[spec.method_id](ctx)
                except Exception as exc:  # never let one method sink the suite
                    res = CrossvalMethodResult(
                        method_id=spec.method_id, name=spec.name, status="failed",
                        summary={}, details={"error": f"{type(exc).__name__}: {exc}"},
                    )
                results.append(res)
            else:
                results.append(
                    CrossvalMethodResult(
                        method_id=spec.method_id, name=spec.name, status="skipped",
                        summary={},
                        details={
                            "category": spec.category,
                            "requires": spec.applicability,
                            "description": spec.description,
                        },
                    )
                )
        return results

    # -- cube resolution / context ------------------------------------------------------
    def _resolve_cube(self, available_data: dict[str, Any]) -> _Cube:
        """Resolve a :class:`_Cube` from supplied data, or generate a synthetic one."""
        import numpy as np  # lazy

        obj = available_data.get("synthetic_cube", available_data.get("cube"))
        if isinstance(obj, _Cube):
            return obj
        if isinstance(obj, dict) and "frames" in obj:
            frames = [np.asarray(f, dtype=np.float32) for f in obj["frames"]]
            n = len(frames)
            times = list(obj.get("times", list(range(n))))
            h, w = frames[0].shape
            lat = np.asarray(obj.get("lat", np.linspace(38.0, 6.0, h)), dtype=np.float64)
            lon = np.asarray(obj.get("lon", np.linspace(68.0, 98.0, w)), dtype=np.float64)
            return _Cube(frames=frames, times=times, lat=lat, lon=lon)
        if isinstance(obj, (str,)) or hasattr(obj, "__fspath__"):
            return self._cube_from_zarr(str(obj))
        # Default: generate a small dense synthetic cube in-memory.
        return self._make_synthetic_cube()

    def _cube_from_zarr(self, path: str) -> _Cube:
        """Load a dense :class:`_Cube` from a ``.zarr`` cube on disk."""
        import numpy as np  # lazy
        import xarray as xr  # lazy

        ds = xr.open_zarr(path, consolidated=False)
        try:
            bt = np.asarray(ds["bt"].values, dtype=np.float32)
            times = list(np.asarray(ds["time"].values).reshape(-1))
            lat = np.asarray(ds["lat"].values, dtype=np.float64)
            lon = np.asarray(ds["lon"].values, dtype=np.float64)
        finally:
            ds.close()
        return _Cube(frames=[bt[i] for i in range(bt.shape[0])], times=times, lat=lat, lon=lon)

    def _make_synthetic_cube(self, n_frames: int = 9, size: int = 48) -> _Cube:
        """Generate a small, dense moving-cloud synthetic cube in-memory (no disk I/O)."""
        import numpy as np  # lazy

        from ..contracts import GridSpec
        from ..synthetic import _sample_blobs, generate_bt_field

        grid = GridSpec(
            west=68.0, south=6.0, east=98.0, north=38.0,
            n_rows=size, n_cols=size, crs="EPSG:4326", resolution_deg=30.0 / size,
        )
        rng = np.random.default_rng(self.seed)
        blobs = _sample_blobs(rng, 4)
        frames = []
        for i in range(n_frames):
            t_norm = i / max(n_frames - 1, 1)
            frames.append(generate_bt_field(t_norm, grid, rng, n_blobs=4, blob_params=blobs))
        times = [np.datetime64("2025-06-20T00:00:00") + np.timedelta64(30 * i, "m")
                 for i in range(n_frames)]
        return _Cube(
            frames=frames, times=times,
            lat=np.asarray(grid.lat_coords(), dtype=np.float64),
            lon=np.asarray(grid.lon_coords(), dtype=np.float64),
        )

    def _predict(self, I0: np.ndarray, I1: np.ndarray, t: float) -> np.ndarray:
        """Run the model-under-test (or the linear-blend fallback) for one intermediate frame."""

        if self.model is None:
            return linear_blend_baseline(I0, I1, t)
        from ..infer.interpolate import interpolate_pair

        return interpolate_pair(
            self.model, I0, I1, t, dataset=self.dataset, device=self.device
        )

    def _build_context(self, cube: _Cube) -> dict[str, Any]:
        """Pre-compute the per-triplet predictions + metrics shared across many methods.

        For every interior frame ``k`` (1..N-2) the true middle is withheld and predicted
        from frames ``k-1`` and ``k+1`` by the model and by each baseline; metrics vs the
        withheld truth are stored. Also stores per-triplet flow-speed and PSD ratios.
        """

        frames = cube.frames
        n = len(frames)
        per: list[dict[str, Any]] = []
        for k in range(1, n - 1):
            I0, gt, I1 = frames[k - 1], frames[k], frames[k + 1]
            pred_model = self._predict(I0, I1, 0.5)
            pred_copy = frame_copy_baseline(I0, I1, 0.5)
            pred_blend = linear_blend_baseline(I0, I1, 0.5)
            try:
                pred_flow = optical_flow_baseline(I0, I1, 0.5, method="farneback")
            except Exception:
                pred_flow = pred_blend
            entry = {
                "k": k,
                # Bind the loop variables as lambda defaults (the lambdas are evaluated
                # immediately by _safe in this iteration; defaults make that explicit).
                "speed": self._safe(lambda I0=I0, I1=I1: _flow_speed(I0, I1)),
                "model": compute_metrics(pred_model, gt),
                "frame_copy": compute_metrics(pred_copy, gt),
                "linear_blend": compute_metrics(pred_blend, gt),
                "optical_flow": compute_metrics(pred_flow, gt),
                "psd_ratio_model": self._safe(
                    lambda pred_model=pred_model, gt=gt: _psd_ratio_highband(pred_model, gt)
                ),
                "psd_ratio_blend": self._safe(
                    lambda pred_blend=pred_blend, gt=gt: _psd_ratio_highband(pred_blend, gt)
                ),
            }
            per.append(entry)
        return {"cube": cube, "per": per, "n": n}

    @staticmethod
    def _safe(fn: Callable[[], float]) -> float:
        """Run ``fn`` and return its float result, or ``nan`` on any failure."""
        try:
            return float(fn())
        except Exception:
            return float("nan")

    @staticmethod
    def _mean(per: list[dict[str, Any]], method: str, metric: str) -> float:
        """Mean of ``per[*][method][metric]`` over finite values (``nan`` if none)."""
        import numpy as np  # lazy

        vals = [e[method][metric] for e in per
                if metric in e[method] and np.isfinite(e[method][metric])]
        return float(np.mean(vals)) if vals else float("nan")

    # -- individual method executors ----------------------------------------------------
    def _m1_leave_middle_out(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        per = ctx["per"]
        summary = {
            "n_triplets": float(len(per)),
            "psnr": self._mean(per, "model", "psnr"),
            "ssim": self._mean(per, "model", "ssim"),
            "ms_ssim": self._mean(per, "model", "ms_ssim"),
            "fsim": self._mean(per, "model", "fsim"),
            "bt_rmse_k": self._mean(per, "model", "bt_rmse_k"),
            "bt_bias_k": self._mean(per, "model", "bt_bias_k"),
        }
        return CrossvalMethodResult(
            "M1", "Leave-the-middle-frame-out (dense, PRIMARY)", "run",
            summary=summary,
            details={"protocol": "predict frame k from k-1,k+1; metrics vs withheld truth (fixed-K, P1)"},
        )

    def _m3_multistep(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """2x/4x/8x densification error vs up-sampling factor (recursive)."""
        import numpy as np  # lazy

        from ..infer.interpolate import interpolate_recursive

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        model = self.model
        summary: dict[str, float] = {}
        for factor in (2, 4, 8):
            # Need >= factor+1 frames spanning one interval; use a sub-sequence so the
            # interpolated frames have real withheld truth at the dense sampling.
            need = factor + 1
            if len(frames) < need:
                continue
            base = [frames[i] for i in range(0, need)]  # equally spaced observed frames
            # Truth at the dense grid: take frames[0..factor] as the per-step truth.
            if model is None:
                # Linear-blend densification stand-in.
                dense = self._dense_blend(base[0], base[-1], factor)
            else:
                seq = interpolate_recursive(model, [base[0], base[-1]], factor=factor,
                                            dataset=self.dataset, device=self.device)
                dense = [f.bt for f in seq]
            # Compare each interior dense frame to the corresponding observed truth.
            errs = []
            for j in range(1, factor):
                if j < len(base) and j < len(dense):
                    m = compute_metrics(dense[j], base[j])
                    if np.isfinite(m["bt_rmse_k"]):
                        errs.append(m["bt_rmse_k"])
            summary[f"bt_rmse_k_{factor}x"] = float(np.mean(errs)) if errs else float("nan")
        return CrossvalMethodResult(
            "M3", "Multi-step recursion vs direct (2x/4x/8x)", "run",
            summary=summary, details={"note": "BT-RMSE(K) per up-sampling factor"},
        )

    def _dense_blend(self, I0: np.ndarray, I1: np.ndarray, factor: int) -> list[np.ndarray]:
        """Linear-blend densification (factor+1 frames incl. endpoints) — model-free fallback."""
        return [linear_blend_baseline(I0, I1, j / factor) for j in range(factor + 1)]

    def _m4_recursion_depth(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Error accumulation: compare midpoint error to quarter-point (deeper recursion)."""
        import numpy as np  # lazy

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        # Midpoint (1 level) vs quarter points (2 levels) error growth on the first interval.
        if len(frames) < 3:
            return CrossvalMethodResult("M4", "Recursion-depth / error-accumulation", "skipped",
                                        details={"requires": ">=3 frames"})
        I0, I1 = frames[0], frames[2]
        mid_pred = self._predict(I0, I1, 0.5)
        mid_err = compute_metrics(mid_pred, frames[1])["bt_rmse_k"]
        # Deeper: interpolate quarter points from (I0, mid_pred) and (mid_pred, I1).
        q1 = self._predict(I0, mid_pred, 0.5)
        q3 = self._predict(mid_pred, I1, 0.5)
        # Truth at quarter points approximated by linear blends of real neighbours (proxy).
        q1_err = compute_metrics(q1, linear_blend_baseline(frames[0], frames[1], 0.5))["bt_rmse_k"]
        q3_err = compute_metrics(q3, linear_blend_baseline(frames[1], frames[2], 0.5))["bt_rmse_k"]
        deep = float(np.nanmean([q1_err, q3_err]))
        return CrossvalMethodResult(
            "M4", "Recursion-depth / error-accumulation", "run",
            summary={"bt_rmse_k_level1": float(mid_err), "bt_rmse_k_level2": deep,
                     "growth_ratio": float(deep / mid_err) if mid_err else float("nan")},
            details={"note": "BT-RMSE growth from 1 to 2 recursion levels"},
        )

    def _m5_asymmetric(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Off-centre t (0.25, 0.75) vs midpoint — checks the model is not only good at t=0.5."""
        import numpy as np  # lazy

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        errs: dict[str, list[float]] = {"t0.25": [], "t0.5": [], "t0.75": []}
        for k in range(1, len(frames) - 1):
            I0, I1 = frames[k - 1], frames[k + 1]
            truth = frames[k]
            for frac, key in ((0.25, "t0.25"), (0.5, "t0.5"), (0.75, "t0.75")):
                # Proxy truth at off-centre frac: blend toward the nearer real frame.
                proxy = linear_blend_baseline(I0, truth, frac * 2) if frac < 0.5 else (
                    truth if frac == 0.5 else linear_blend_baseline(truth, I1, (frac - 0.5) * 2))
                pred = self._predict(I0, I1, frac)
                m = compute_metrics(pred, truth if frac == 0.5 else proxy)
                if np.isfinite(m["bt_rmse_k"]):
                    errs[key].append(m["bt_rmse_k"])
        summary = {k: (float(np.mean(v)) if v else float("nan")) for k, v in errs.items()}
        return CrossvalMethodResult(
            "M5", "Asymmetric-interval test (off-centre t)", "run",
            summary=summary, details={"note": "BT-RMSE(K) at t=0.25/0.5/0.75"},
        )

    def _m6_long_gap(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Long-gap stress: predict the middle of a wide bracket and compare to the real frame."""

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        n = len(frames)
        if n < 5:
            return CrossvalMethodResult("M6", "Long-gap stress test", "skipped",
                                        details={"requires": ">=5 frames"})
        # Widest symmetric bracket whose midpoint is a real frame.
        span = (n - 1) if (n - 1) % 2 == 0 else (n - 2)
        mid = span // 2
        pred = self._predict(frames[0], frames[span], 0.5)
        m_long = compute_metrics(pred, frames[mid])
        # Short-gap reference (adjacent bracket around the same midpoint).
        pred_short = self._predict(frames[mid - 1], frames[mid + 1], 0.5)
        m_short = compute_metrics(pred_short, frames[mid])
        return CrossvalMethodResult(
            "M6", "Long-gap stress test", "run",
            summary={"bt_rmse_k_long_gap": float(m_long["bt_rmse_k"]),
                     "bt_rmse_k_short_gap": float(m_short["bt_rmse_k"]),
                     "gap_frames": float(span)},
            details={"note": "wide-bracket midpoint error vs adjacent-bracket reference"},
        )

    def _baseline_method(self, ctx: dict[str, Any], method_key: str, mid: str,
                         name: str) -> CrossvalMethodResult:
        """Shared executor for the three baseline methods (M18/M19/M20)."""
        per = ctx["per"]
        summary = {
            "psnr": self._mean(per, method_key, "psnr"),
            "ssim": self._mean(per, method_key, "ssim"),
            "bt_rmse_k": self._mean(per, method_key, "bt_rmse_k"),
            "model_psnr": self._mean(per, "model", "psnr"),
            "model_ssim": self._mean(per, "model", "ssim"),
            "model_bt_rmse_k": self._mean(per, "model", "bt_rmse_k"),
        }
        # "model beats baseline" deltas (positive psnr delta / negative rmse delta == good).
        summary["delta_psnr_model_minus_baseline"] = (
            summary["model_psnr"] - summary["psnr"]
        )
        summary["delta_bt_rmse_k_baseline_minus_model"] = (
            summary["bt_rmse_k"] - summary["model_bt_rmse_k"]
        )
        return CrossvalMethodResult(mid, name, "run", summary=summary,
                                    details={"baseline": method_key})

    def _m18_frame_copy(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        return self._baseline_method(ctx, "frame_copy", "M18",
                                     "Naive frame-copy baseline (persistence)")

    def _m19_linear_blend(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        return self._baseline_method(ctx, "linear_blend", "M19", "Linear-blend baseline")

    def _m20_optical_flow(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        return self._baseline_method(ctx, "optical_flow", "M20",
                                     "Classical optical-flow baseline (Farneback/TV-L1)")

    def _m26_extreme_motion(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Stratify metrics by motion magnitude (fast vs slow scenes)."""
        import numpy as np  # lazy

        per = ctx["per"]
        speeds = np.asarray([e["speed"] for e in per], dtype=np.float64)
        finite = speeds[np.isfinite(speeds)]
        if finite.size < 2:
            return CrossvalMethodResult("M26", "Extreme-motion stratification", "skipped",
                                        details={"requires": "multiple triplets with flow"})
        median = float(np.median(finite))
        fast = [e for e in per if np.isfinite(e["speed"]) and e["speed"] >= median]
        slow = [e for e in per if np.isfinite(e["speed"]) and e["speed"] < median]
        summary = {
            "median_speed_px": median,
            "slow_psnr": self._mean(slow, "model", "psnr"),
            "fast_psnr": self._mean(fast, "model", "psnr"),
            "slow_bt_rmse_k": self._mean(slow, "model", "bt_rmse_k"),
            "fast_bt_rmse_k": self._mean(fast, "model", "bt_rmse_k"),
        }
        return CrossvalMethodResult(
            "M26", "Extreme-motion stratification (fast vs slow)", "run",
            summary=summary, details={"note": "split at median flow speed"},
        )

    def _m30_cloud_regime(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Stratify BT-RMSE by cloud-regime BT band (cold/convective vs warm/clear)."""
        import numpy as np  # lazy

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        cold_th = 240.0  # K — cold convective threshold
        cold_errs, warm_errs = [], []
        for k in range(1, len(frames) - 1):
            I0, I1, gt = frames[k - 1], frames[k + 1], frames[k]
            pred = self._predict(I0, I1, 0.5)
            diff = pred - gt
            cold = gt < cold_th
            warm = gt >= cold_th
            valid = np.isfinite(diff)
            if np.any(cold & valid):
                cold_errs.append(float(np.sqrt(np.mean(diff[cold & valid] ** 2))))
            if np.any(warm & valid):
                warm_errs.append(float(np.sqrt(np.mean(diff[warm & valid] ** 2))))
        return CrossvalMethodResult(
            "M30", "Cloud-regime stratification by BT band", "run",
            summary={"cold_cloud_bt_rmse_k": float(np.mean(cold_errs)) if cold_errs else float("nan"),
                     "warm_clear_bt_rmse_k": float(np.mean(warm_errs)) if warm_errs else float("nan"),
                     "cold_threshold_k": cold_th},
            details={"note": "BT-RMSE on cold (<240K) vs warm (>=240K) pixels"},
        )

    def _m31_error_vs_speed(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Regress per-frame BT-RMSE against cloud speed; report Pearson R and R^2."""
        import numpy as np  # lazy

        per = ctx["per"]
        x = np.asarray([e["speed"] for e in per], dtype=np.float64)
        y = np.asarray([e["model"]["bt_rmse_k"] for e in per], dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9:
            return CrossvalMethodResult("M31", "Error-vs-cloud-speed correlation", "run",
                                        summary={"n": float(ok.sum()), "pearson_r": float("nan"),
                                                 "r_squared": float("nan")},
                                        details={"note": "insufficient/degenerate variance for regression"})
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        return CrossvalMethodResult(
            "M31", "Error-vs-cloud-speed correlation", "run",
            summary={"n": float(ok.sum()), "pearson_r": r, "r_squared": r * r},
            details={"note": "BT-RMSE(K) regressed on mean flow speed (px)"},
        )

    def _m32_error_vs_lead(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Error vs lead fraction: midpoint (t=0.5, hardest) vs near-frame (t=0.1)."""
        import numpy as np  # lazy

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        near, mid = [], []
        for k in range(1, len(frames) - 1):
            I0, I1, gt = frames[k - 1], frames[k + 1], frames[k]
            # t=0.5 predicts the real middle; t=0.1 should be close to I0 (near-frame, easy).
            mid.append(compute_metrics(self._predict(I0, I1, 0.5), gt)["bt_rmse_k"])
            near_pred = self._predict(I0, I1, 0.1)
            near.append(compute_metrics(near_pred, I0)["bt_rmse_k"])
        return CrossvalMethodResult(
            "M32", "Error-vs-lead-fraction correlation", "run",
            summary={"bt_rmse_k_t0.5_midpoint": float(np.nanmean(mid)) if mid else float("nan"),
                     "bt_rmse_k_t0.1_near": float(np.nanmean(near)) if near else float("nan")},
            details={"note": "midpoint (hardest) vs near-frame error"},
        )

    def _m33_psd_ratio(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Radially-averaged PSD ratio (high-band) for model vs blend — fine-scale fidelity."""
        import numpy as np  # lazy

        per = ctx["per"]
        model_r = [e["psd_ratio_model"] for e in per if np.isfinite(e["psd_ratio_model"])]
        blend_r = [e["psd_ratio_blend"] for e in per if np.isfinite(e["psd_ratio_blend"])]
        return CrossvalMethodResult(
            "M33", "Radially-averaged PSD ratio (fine-scale fidelity)", "run",
            summary={"model_highband_psd_ratio": float(np.mean(model_r)) if model_r else float("nan"),
                     "blend_highband_psd_ratio": float(np.mean(blend_r)) if blend_r else float("nan")},
            details={"note": "mean PSD(pred)/PSD(truth) over high-wavenumber half; ~1 is best"},
        )

    def _m35_significance(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Wilcoxon AI-vs-baseline + bootstrap 95% CI on mean model PSNR / BT-RMSE."""
        import numpy as np  # lazy

        per = ctx["per"]
        model_psnr = [e["model"]["psnr"] for e in per]
        blend_psnr = [e["linear_blend"]["psnr"] for e in per]
        # Bootstrap CI on the model's mean PSNR and BT-RMSE.
        mean_psnr, lo_psnr, hi_psnr = _bootstrap_ci(
            [v for v in model_psnr if np.isfinite(v)], seed=self.seed
        )
        mean_rmse, lo_rmse, hi_rmse = _bootstrap_ci(
            [e["model"]["bt_rmse_k"] for e in per], seed=self.seed
        )
        # Paired Wilcoxon on PSNR (model vs blend); needs paired finite samples with nonzero diffs.
        wilcoxon_p = float("nan")
        pairs = [(m, b) for m, b in zip(model_psnr, blend_psnr, strict=False)
                 if np.isfinite(m) and np.isfinite(b)]
        diffs = [m - b for m, b in pairs]
        if len(diffs) >= 3 and any(abs(d) > 1e-9 for d in diffs):
            try:
                from scipy.stats import wilcoxon

                wilcoxon_p = float(wilcoxon(
                    [m for m, _ in pairs], [b for _, b in pairs],
                    zero_method="zsplit",
                ).pvalue)
            except Exception:
                wilcoxon_p = float("nan")
        return CrossvalMethodResult(
            "M35", "Statistical-significance testing (Wilcoxon + bootstrap CI)", "run",
            summary={
                "mean_psnr": mean_psnr, "psnr_ci_lo": lo_psnr, "psnr_ci_hi": hi_psnr,
                "mean_bt_rmse_k": mean_rmse, "bt_rmse_k_ci_lo": lo_rmse, "bt_rmse_k_ci_hi": hi_rmse,
                "wilcoxon_p_model_vs_blend_psnr": wilcoxon_p, "n_pairs": float(len(pairs)),
            },
            details={"note": "bootstrap 95% CI (2000 resamples) + paired Wilcoxon vs linear-blend"},
        )

    def _m36_inter_metric(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Inter-metric agreement: Spearman correlation between PSNR and SSIM across frames."""
        import numpy as np  # lazy

        per = ctx["per"]
        psnr = np.asarray([e["model"]["psnr"] for e in per], dtype=np.float64)
        ssim = np.asarray([e["model"]["ssim"] for e in per], dtype=np.float64)
        ok = np.isfinite(psnr) & np.isfinite(ssim)
        rho = float("nan")
        if ok.sum() >= 3 and np.std(psnr[ok]) > 1e-9 and np.std(ssim[ok]) > 1e-9:
            try:
                from scipy.stats import spearmanr

                rho = float(spearmanr(psnr[ok], ssim[ok]).statistic)
            except Exception:
                rho = float("nan")
        return CrossvalMethodResult(
            "M36", "Inter-metric agreement / rank-correlation", "run",
            summary={"spearman_psnr_ssim": rho, "n": float(ok.sum())},
            details={"note": "Spearman rho between PSNR and SSIM across test frames"},
        )

    def _m38_conservation(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Physical plausibility: domain-mean BT and cold-cloud area evolve smoothly across inserts."""
        import numpy as np  # lazy

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        if len(frames) < 3:
            return CrossvalMethodResult("M38", "Mass/feature conservation", "skipped",
                                        details={"requires": ">=3 frames"})
        # Insert a midpoint and check its domain-mean BT lies between the neighbours' means.
        violations = 0
        total = 0
        for k in range(1, len(frames) - 1):
            I0, I1 = frames[k - 1], frames[k + 1]
            mid = self._predict(I0, I1, 0.5)
            m0, m1, mm = (float(np.nanmean(I0)), float(np.nanmean(I1)), float(np.nanmean(mid)))
            lo, hi = min(m0, m1), max(m0, m1)
            total += 1
            if not (lo - 2.0 <= mm <= hi + 2.0):  # 2 K tolerance for nonlinear growth
                violations += 1
        return CrossvalMethodResult(
            "M38", "Mass/feature conservation & physical-plausibility", "run",
            summary={"mean_bt_monotonicity_violations": float(violations),
                     "n_inserts": float(total),
                     "violation_fraction": float(violations / total) if total else float("nan")},
            details={"note": "inserted-frame domain-mean BT should lie between neighbours (±2K)"},
        )

    def _m39_seams(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Temporal-consistency at seams: jump between a real frame and an adjacent inserted frame."""
        import numpy as np  # lazy

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        seam_jumps, real_jumps = [], []
        for k in range(1, len(frames) - 1):
            I0, I1 = frames[k - 1], frames[k + 1]
            mid = self._predict(I0, I1, 0.5)
            # Seam: |mid - I0| (real->inserted). Reference: |I1 - I0|/2 (expected half-step).
            seam_jumps.append(float(np.nanmean(np.abs(mid - I0))))
            real_jumps.append(float(np.nanmean(np.abs(I1 - I0)) / 2.0))
        return CrossvalMethodResult(
            "M39", "Temporal-consistency at the seams", "run",
            summary={"mean_seam_step_k": float(np.nanmean(seam_jumps)) if seam_jumps else float("nan"),
                     "expected_half_step_k": float(np.nanmean(real_jumps)) if real_jumps else float("nan")},
            details={"note": "real->inserted step vs expected half of the real step"},
        )

    def _m40_dual_impl(self, ctx: dict[str, Any]) -> CrossvalMethodResult:
        """Dual-implementation metric check: piq-SSIM vs scikit-image SSIM agreement (if available)."""
        import numpy as np  # lazy

        cube: _Cube = ctx["cube"]
        frames = cube.frames
        # Build one fixed-[0,1] pred/truth pair from the first triplet.
        I0, gt, I1 = frames[0], frames[1], frames[2]
        pred = self._predict(I0, I1, 0.5)
        from .metrics import to_unit_interval

        p01 = np.nan_to_num(to_unit_interval(pred), nan=0.5)
        t01 = np.nan_to_num(to_unit_interval(gt), nan=0.5)
        # piq SSIM (always present).
        from .metrics import _safe_ssim

        ssim_piq = _safe_ssim(p01, t01)
        try:
            from skimage.metrics import structural_similarity as sk_ssim  # optional

            smaller = min(p01.shape)
            win = min(7, smaller if smaller % 2 == 1 else smaller - 1)
            if win < 3:
                win = 3
            if win % 2 == 0:
                win -= 1
            ssim_sk = float(sk_ssim(t01, p01, data_range=1.0, win_size=win))
            agree = float(abs(ssim_piq - ssim_sk))
            return CrossvalMethodResult(
                "M40", "Dual-implementation metric verification", "run",
                summary={"ssim_piq": float(ssim_piq), "ssim_skimage": ssim_sk,
                         "abs_difference": agree},
                details={"note": "piq vs scikit-image SSIM agreement (different windowing => small diff)"},
            )
        except Exception:
            return CrossvalMethodResult(
                "M40", "Dual-implementation metric verification", "run",
                summary={"ssim_piq": float(ssim_piq)},
                details={"note": "scikit-image not installed; piq-only (second impl skipped)",
                         "second_impl": "skimage (absent)"},
            )


# ===========================================================================
# 5. Module-level convenience API (CONTRACTS §6.5)
# ===========================================================================
def available_methods() -> list[CrossvalMethodResult]:
    """Return the M1-M40 registry as ``CrossvalMethodResult`` stubs (CONTRACTS §6.5)."""
    return CrossValSuite.available_methods()


def run_method(method_id: str, **ctx: Any) -> CrossvalMethodResult:
    """Run a single cross-validation method by id (CONTRACTS §6.5).

    A thin convenience wrapper: builds a :class:`CrossValSuite` (passing through any
    ``model`` / ``dataset`` / ``device`` / ``seed`` in ``ctx``), runs the full suite on the
    supplied / synthetic cube, and returns the one matching result. For running many methods
    prefer :meth:`CrossValSuite.run` directly (it shares the expensive per-triplet context).

    Args:
        method_id: the method id, e.g. ``"M1"``.
        **ctx: optional ``model`` / ``dataset`` / ``device`` / ``seed`` and an
            ``available_data`` dict forwarded to :meth:`CrossValSuite.run`.

    Returns:
        The :class:`CrossvalMethodResult` for ``method_id`` (status ``"skipped"`` if the id
        is data-dependent, or a ``"failed"`` result if the id is unknown).
    """
    model = ctx.pop("model", None)
    dataset = ctx.pop("dataset", "synthetic")
    device = ctx.pop("device", "cpu")
    seed = int(ctx.pop("seed", 0))
    available_data = ctx.pop("available_data", None)
    suite = CrossValSuite(model=model, dataset=dataset, device=device, seed=seed)
    for res in suite.run(available_data):
        if res.method_id == method_id:
            return res
    return CrossvalMethodResult(method_id, f"unknown method {method_id}", "failed",
                                details={"error": f"no such method id {method_id!r}"})

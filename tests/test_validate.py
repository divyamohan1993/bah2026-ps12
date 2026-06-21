"""Tests for the VALIDATE area — ``frameflow.validate.{metrics,crossval,report}``.

Covered:
    * ``compute_metrics`` on identical frames: ``mse ~= 0``, ``ssim ~= 1``, ``psnr`` huge.
    * **P1 demonstration** — a uniform ``+5 K`` warm bias clearly DEGRADES PSNR (and SSIM)
      AND surfaces as ``bt_bias_k ~= 5`` because the metrics use the FIXED physical Kelvin
      range. The test also shows the failure mode the fixed range PREVENTS: per-image
      min/max scaling would re-stretch the biased frame and hide the error (SSIM ~ 1).
    * ``CrossValSuite.METHODS`` has >= 30 entries; ``run()`` executes the runnable subset on
      a dense synthetic cube and returns ``CrossvalMethodResult``s; the local classical
      baselines (frame-copy / linear-blend / optical-flow) run.
    * ``make_report`` produces a manifest-consistent summary dict (fixed-range, P1).

NEVER imports the models package — the "model under test" is the in-test linear blend.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ===========================================================================
# tiny in-test model (NEVER import frameflow.models)
# ===========================================================================
class BlendModel(torch.nn.Module):
    """Linear-blend VFI stand-in (``t`` shaped (B, 1) per the contract)."""

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = t.view(-1, 1, 1, 1)
        return I0 * (1.0 - tt) + I1 * tt


def _structured_field(n: int = 64, seed: int = 0) -> np.ndarray:
    """A smooth structured Kelvin field within the metric range [180, 320]."""
    yy, xx = np.mgrid[0:n, 0:n]
    rng = np.random.default_rng(seed)
    field = 250.0 + 30.0 * np.sin(xx / 8.0) + 20.0 * np.cos(yy / 6.0)
    field += 3.0 * rng.standard_normal((n, n))
    return np.clip(field, 185.0, 315.0).astype(np.float32)


# ===========================================================================
# metrics: identity
# ===========================================================================
def test_compute_metrics_identity() -> None:
    from frameflow.validate.metrics import METRIC_KEYS, compute_metrics

    gt = _structured_field(64)
    m = compute_metrics(gt, gt)

    assert set(METRIC_KEYS).issubset(m.keys())
    assert m["mse"] == pytest.approx(0.0, abs=1e-9)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert m["mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["bt_rmse_k"] == pytest.approx(0.0, abs=1e-6)
    assert m["bt_bias_k"] == pytest.approx(0.0, abs=1e-6)
    assert m["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert m["ms_ssim"] == pytest.approx(1.0, abs=1e-3)
    assert m["fsim"] == pytest.approx(1.0, abs=1e-3)
    assert m["gmsd"] == pytest.approx(0.0, abs=1e-4)
    # PSNR of identical frames is +inf (mse == 0).
    assert m["psnr"] == float("inf") or m["psnr"] > 80.0


def test_compute_metrics_excludes_nan_pixels() -> None:
    """Off-disk NaN pixels (and any caller mask) must be excluded, never poison the metrics."""
    from frameflow.validate.metrics import compute_metrics

    gt = _structured_field(48)
    pred = gt.copy()
    # Put NaNs in both at the same corner; the metric must still be a clean identity.
    gt[0, 0] = np.nan
    pred[0, 0] = np.nan
    m = compute_metrics(pred, gt)
    assert np.isfinite(m["bt_rmse_k"]) and m["bt_rmse_k"] == pytest.approx(0.0, abs=1e-6)
    assert m["ssim"] == pytest.approx(1.0, abs=1e-3)


# ===========================================================================
# metrics: THE P1 demonstration (fixed range surfaces a warm bias; per-image hides it)
# ===========================================================================
def test_p1_fixed_range_penalizes_warm_bias() -> None:
    """A uniform +5 K bias must DEGRADE PSNR/SSIM and show as bt_bias_k ~= 5 (fixed range, P1)."""
    from frameflow.validate.metrics import compute_metrics

    gt = _structured_field(64)
    pred = gt + 5.0  # uniform warm bias

    m_ref = compute_metrics(gt, gt)
    m_bias = compute_metrics(pred, gt)

    # The radiometric error is surfaced, NOT hidden: signed bias is exactly ~ +5 K.
    assert m_bias["bt_bias_k"] == pytest.approx(5.0, abs=1e-3)
    assert m_bias["bt_rmse_k"] == pytest.approx(5.0, abs=1e-3)

    # PSNR clearly degrades from the (infinite) identity baseline to a finite, modest value.
    assert np.isfinite(m_bias["psnr"])
    assert m_bias["psnr"] < 40.0, f"a +5 K bias should drop PSNR well below 40 dB; got {m_bias['psnr']}"

    # SSIM also drops below the identity 1.0 (it is NOT re-stretched back to ~1).
    assert m_bias["ssim"] < m_ref["ssim"]
    assert m_bias["ssim"] < 0.999


def test_p1_per_image_minmax_would_hide_the_bias() -> None:
    """Contrast check: per-image min/max scaling (FORBIDDEN) re-stretches and HIDES the bias.

    This is the explicit justification for P1 — it demonstrates *why* the fixed range is
    mandatory by showing the (forbidden) per-image normalization gives a near-perfect SSIM
    for a frame that the fixed-range metric correctly penalizes.
    """
    import piq

    from frameflow.validate.metrics import compute_metrics

    gt = _structured_field(64)
    pred = gt + 5.0

    # FORBIDDEN per-image min/max scaling of each frame to its OWN extremes.
    def _per_image_01(a: np.ndarray) -> np.ndarray:
        return ((a - a.min()) / (np.ptp(a) + 1e-9)).astype(np.float32)

    p01 = torch.from_numpy(_per_image_01(pred)[None, None]).float()
    t01 = torch.from_numpy(_per_image_01(gt)[None, None]).float()
    ssim_per_image = float(piq.ssim(p01, t01, data_range=1.0, kernel_size=11))

    # A pure additive bias survives per-image scaling almost perfectly -> hidden.
    assert ssim_per_image == pytest.approx(1.0, abs=1e-3), (
        "per-image min/max should hide a uniform bias (that is the bug P1 fixes)"
    )

    # The FIXED-range metric, by contrast, does NOT hide it.
    m_fixed = compute_metrics(pred, gt)
    assert m_fixed["bt_bias_k"] == pytest.approx(5.0, abs=1e-3)
    assert m_fixed["ssim"] < ssim_per_image  # fixed range is strictly more honest here


def test_to_unit_interval_uses_fixed_constants() -> None:
    """to_unit_interval must scale by the FIXED physical span, not per-image extremes (P1)."""
    from frameflow import constants as C
    from frameflow.validate.metrics import from_unit_interval, to_unit_interval

    # vmin -> 0, vmax -> 1, midpoint -> 0.5 regardless of the array's own min/max.
    x = np.array([[C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K],
                  [(C.BT_METRIC_VMIN_K + C.BT_METRIC_VMAX_K) / 2.0, 250.0]], dtype=np.float32)
    u = to_unit_interval(x)
    assert u[0, 0] == pytest.approx(0.0)
    assert u[0, 1] == pytest.approx(1.0)
    assert u[1, 0] == pytest.approx(0.5)
    # Round-trips back to Kelvin via the same fixed span.
    back = from_unit_interval(u)
    assert np.allclose(back, x, atol=1e-3)


def test_per_frame_metrics_returns_record() -> None:
    from frameflow.contracts import MetricRecord
    from frameflow.validate.metrics import per_frame_metrics

    gt = _structured_field(48)
    rec = per_frame_metrics(gt + 5.0, gt, index=3, time="2025-06-20T00:10:00Z")
    assert isinstance(rec, MetricRecord)
    assert rec.index == 3
    assert rec.bt_bias_k == pytest.approx(5.0, abs=1e-3)
    assert "gmsd" in rec.extra  # metrics outside the named fields land in extra


# ===========================================================================
# crossval: registry size + run() executes runnable subset
# ===========================================================================
def test_crossval_registry_has_at_least_30_methods() -> None:
    from frameflow.validate.crossval import CrossValSuite

    assert len(CrossValSuite.METHODS) >= 30
    ids = {m.method_id for m in CrossValSuite.METHODS}
    # The headline primary methods from research/05 §4 are present.
    for mid in ("M1", "M8", "M15", "M18", "M19", "M20", "M35", "M40"):
        assert mid in ids
    # Some methods are runnable on a single dense cube; some are data-dependent.
    assert any(m.runnable_on_synthetic for m in CrossValSuite.METHODS)
    assert any(not m.runnable_on_synthetic for m in CrossValSuite.METHODS)


def test_available_methods_registry_stubs() -> None:
    from frameflow.validate.crossval import available_methods

    stubs = available_methods()
    assert len(stubs) == 40
    by_id = {s.method_id: s for s in stubs}
    # Data-dependent methods are catalogued as skipped with a 'requires' note.
    assert by_id["M8"].status == "skipped"
    assert "requires" in by_id["M8"].details
    # Synthetic-runnable methods are marked run.
    assert by_id["M1"].status == "run"


def test_crossval_run_executes_runnable_subset() -> None:
    from frameflow.contracts import CrossvalMethodResult
    from frameflow.validate.crossval import CrossValSuite

    suite = CrossValSuite(model=BlendModel(), seed=0)
    results = suite.run()  # auto-generates a small dense synthetic cube

    assert len(results) == len(CrossValSuite.METHODS) == 40
    assert all(isinstance(r, CrossvalMethodResult) for r in results)

    by_id = {r.method_id: r for r in results}
    ran = [r for r in results if r.status == "run"]
    skipped = [r for r in results if r.status == "skipped"]
    assert len(ran) >= 15, f"expected the runnable subset to execute; only {len(ran)} ran"
    assert len(skipped) >= 1

    # M1 (leave-the-middle-out) produced real headline numbers with the fixed range.
    m1 = by_id["M1"]
    assert m1.status == "run"
    assert np.isfinite(m1.summary["psnr"]) and np.isfinite(m1.summary["bt_rmse_k"])
    assert m1.summary["n_triplets"] >= 1

    # No method should have crashed the suite (failed status only on genuine internal error).
    failed = [r for r in results if r.status == "failed"]
    assert not failed, f"methods failed: {[(r.method_id, r.details) for r in failed]}"


def test_crossval_run_on_real_zarr_cube(synthetic_cube) -> None:
    """run() accepts a path to a real .zarr cube and validates against its withheld frames."""
    from frameflow.validate.crossval import CrossValSuite

    suite = CrossValSuite(model=BlendModel(), seed=0)
    results = suite.run({"synthetic_cube": str(synthetic_cube.cube_path)})
    by_id = {r.method_id: r for r in results}
    assert by_id["M1"].status == "run"
    assert by_id["M1"].summary["n_triplets"] == synthetic_cube.n_frames - 2


# ===========================================================================
# crossval: the local classical baselines actually run
# ===========================================================================
def test_local_baselines_run() -> None:
    from frameflow.validate.crossval import (
        frame_copy_baseline,
        linear_blend_baseline,
        optical_flow_baseline,
    )

    I0 = _structured_field(48, seed=1)
    I1 = _structured_field(48, seed=2)

    copy_out = frame_copy_baseline(I0, I1, 0.5)
    assert copy_out.shape == I0.shape
    assert np.array_equal(copy_out, I1)  # t >= 0.5 copies the later frame

    blend_out = linear_blend_baseline(I0, I1, 0.5)
    assert np.allclose(blend_out, 0.5 * (I0 + I1), atol=1e-4)

    # Farneback flow-warp baseline runs and returns a finite same-shape frame.
    flow_out = optical_flow_baseline(I0, I1, 0.5, method="farneback")
    assert flow_out.shape == I0.shape
    assert np.isfinite(flow_out).all()


def test_optical_flow_tvl1_fallback_runs() -> None:
    """The TV-L1 baseline runs even without cv2.optflow (pure-numpy Horn-Schunck fallback)."""
    from frameflow.validate.crossval import optical_flow_baseline

    I0 = _structured_field(32, seed=3)
    I1 = _structured_field(32, seed=4)
    out = optical_flow_baseline(I0, I1, 0.5, method="tvl1")
    assert out.shape == I0.shape
    assert np.isfinite(out).all()


def test_crossval_baseline_methods_present_in_results() -> None:
    """M18/M19/M20 baseline comparisons appear in run() output with model-vs-baseline deltas."""
    from frameflow.validate.crossval import CrossValSuite

    results = {r.method_id: r for r in CrossValSuite(model=BlendModel()).run()}
    for mid in ("M18", "M19", "M20"):
        r = results[mid]
        assert r.status == "run"
        assert "bt_rmse_k" in r.summary
        assert "model_bt_rmse_k" in r.summary


# ===========================================================================
# report: manifest-consistent summary dict (fixed range, P1)
# ===========================================================================
def test_make_report_summary_and_figures(tmp_path) -> None:
    from frameflow import constants as C
    from frameflow.validate.crossval import CrossValSuite
    from frameflow.validate.metrics import per_frame_metrics
    from frameflow.validate.report import make_report

    gt_frames = [_structured_field(48, seed=i) for i in range(5)]
    records = [
        per_frame_metrics(f + np.random.default_rng(i).normal(0, 1.0, f.shape).astype(np.float32),
                          f, index=i)
        for i, f in enumerate(gt_frames)
    ]
    crossval = CrossValSuite(model=BlendModel()).run()

    rep = make_report(records, out_dir=tmp_path, crossval_results=crossval,
                      baselines={"linear_blend": records})

    # P1: fixed physical range carried in the metrics block.
    assert rep["metrics"]["data_range_k"] == pytest.approx(C.BT_DATA_RANGE_K)
    assert rep["metrics"]["value_range_k"] == [C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K]
    assert len(rep["metrics"]["per_frame"]) == 5
    assert "psnr" in rep["metrics"]["summary"]
    assert rep["metrics"]["summary"]["psnr"]["n"] == 5
    # crossval methods carried through.
    assert len(rep["crossval"]["methods_run"]) == 40

    # Figures rendered headlessly (no display) without error.
    assert "error" not in rep["figures"]
    for key in ("scorecard", "psnr_vs_frame", "distribution"):
        assert key in rep["figures"]
        from pathlib import Path

        assert Path(rep["figures"][key]).exists()


def test_make_report_summary_only_no_figures(tmp_path) -> None:
    """make_report(make_figures=False) returns the numeric summary without touching matplotlib."""
    from frameflow.validate.metrics import per_frame_metrics
    from frameflow.validate.report import make_report

    gt = _structured_field(32)
    records = [per_frame_metrics(gt + 2.0, gt, index=0)]
    rep = make_report(records, out_dir=tmp_path, make_figures=False)
    assert rep["figures"] == {}
    assert rep["metrics"]["summary"]["bt_bias_k"]["mean"] == pytest.approx(2.0, abs=1e-2)

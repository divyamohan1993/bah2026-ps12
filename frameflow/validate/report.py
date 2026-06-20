"""Validation reporting — matplotlib figures + a manifest-ready summary dict (research/05 §5).

This module turns the raw per-frame :class:`~frameflow.contracts.MetricRecord` list and the
cross-validation results into the artefacts research/05 §5 prescribes:

    * **Figures** (saved as PNG): the headline scorecard bar chart (AI vs baselines), a
      metric-vs-frame line plot, and a per-frame metric distribution (box plot). All use a
      non-interactive ``Agg`` backend so they render headlessly in CI / batch jobs.
    * **A summary dict** shaped for the manifest's ``metrics`` / ``crossval`` blocks: the
      fixed physical ``data_range_k`` / ``value_range_k`` (P1), aggregate means + bootstrap
      95% CIs, the per-frame records, per-baseline summaries, and the list of
      cross-validation methods run. The dict is JSON-serializable and consistent with what
      ``frameflow.contracts.validate_manifest`` expects under ``metrics``.

The heavy plotting import (``matplotlib``) is done lazily inside :func:`make_report` and the
backend is forced to ``Agg`` so importing this module never opens a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import constants as C
from ..contracts import CrossvalMethodResult, MetricRecord
from .crossval import _bootstrap_ci

if TYPE_CHECKING:  # typing only
    pass


__all__ = ["make_report", "summarize_metrics"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _records_to_dicts(records: Sequence[Any]) -> list[dict[str, Any]]:
    """Normalize a mixed list of :class:`MetricRecord` / dicts to plain dicts."""
    out: list[dict[str, Any]] = []
    for r in records:
        out.append(r.to_dict() if isinstance(r, MetricRecord) else dict(r))
    return out


def summarize_metrics(
    records: Sequence[Any],
    *,
    metrics: Sequence[str] = ("psnr", "ssim", "ms_ssim", "fsim", "bt_rmse_k", "bt_bias_k"),
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate per-frame metric records into means + bootstrap 95% CIs (research/05 §5.9).

    Args:
        records: per-frame :class:`MetricRecord` instances or dicts.
        metrics: which metric keys to summarize.
        seed: RNG seed for the bootstrap (reproducibility).

    Returns:
        A dict ``{metric: {"mean": float, "ci_lo": float, "ci_hi": float, "n": int}}`` over
        the finite values of each metric (missing/NaN values are ignored).
    """
    import numpy as np  # lazy

    dicts = _records_to_dicts(records)
    summary: dict[str, Any] = {}
    for key in metrics:
        vals = [d[key] for d in dicts
                if d.get(key) is not None and np.isfinite(d.get(key, float("nan")))]
        mean, lo, hi = _bootstrap_ci(vals, seed=seed)
        summary[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(vals))}
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_scorecard(
    out_path: Path,
    model_summary: dict[str, Any],
    baseline_summaries: dict[str, dict[str, Any]],
    metrics: Sequence[str],
) -> Path:
    """Bar chart of headline metric means: AI model vs each baseline (research/05 §5.1)."""
    import matplotlib.pyplot as plt  # lazy
    import numpy as np  # lazy

    labels = list(metrics)
    series = {"model": model_summary, **baseline_summaries}
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 4))
    for i, (name, summ) in enumerate(series.items()):
        means = [float(summ.get(k, {}).get("mean", float("nan")))
                 if isinstance(summ.get(k), dict) else float(summ.get(k, float("nan")))
                 for k in labels]
        ax.bar(x + i * width, means, width, label=name)
    ax.set_xticks(x + width * (len(series) - 1) / 2.0)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("metric value")
    ax.set_title("Headline scorecard: AI vs baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


def _fig_metric_vs_frame(out_path: Path, records: list[dict[str, Any]], metric: str) -> Path:
    """Line plot of one metric over the frame index (research/05 §5.2)."""
    import matplotlib.pyplot as plt  # lazy

    idx = [d.get("index", i) for i, d in enumerate(records)]
    vals = [d.get(metric, float("nan")) for d in records]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(idx, vals, marker="o", ms=3)
    ax.set_xlabel("frame index")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs frame")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


def _fig_distribution(out_path: Path, records: list[dict[str, Any]],
                      metrics: Sequence[str]) -> Path:
    """Box plot of the per-frame distribution of each metric (research/05 §5.5)."""
    import matplotlib.pyplot as plt  # lazy
    import numpy as np  # lazy

    data, labels = [], []
    for k in metrics:
        vals = [d[k] for d in records
                if d.get(k) is not None and np.isfinite(d.get(k, float("nan")))]
        if vals:
            data.append(vals)
            labels.append(k)
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(labels)), 4))
    if data:
        # matplotlib >= 3.9 renamed boxplot's ``labels`` kwarg to ``tick_labels``; support both.
        try:
            ax.boxplot(data, tick_labels=labels)
        except TypeError:  # pragma: no cover - older matplotlib
            ax.boxplot(data, labels=labels)
    ax.set_title("Per-frame metric distribution")
    ax.set_ylabel("value")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def make_report(
    records: Sequence[Any],
    *,
    out_dir: str | Path = "out/validation",
    crossval_results: Sequence[CrossvalMethodResult] | None = None,
    baselines: dict[str, Sequence[Any]] | None = None,
    make_figures: bool = True,
    value_range_k: tuple[float, float] = (C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K),
    seed: int = 0,
) -> dict[str, Any]:
    """Render validation figures + a manifest-ready summary dict (research/05 §5).

    Builds (optionally) the headline scorecard, metric-vs-frame, and distribution figures,
    and returns a JSON-serializable summary structured exactly like the manifest's
    ``metrics`` block (plus a ``crossval`` block), carrying the FIXED physical Kelvin range
    (P1) and per-frame records.

    Args:
        records: per-frame :class:`MetricRecord` instances (or dicts) for the AI model.
        out_dir: directory to write figures into (created if needed).
        crossval_results: optional list of :class:`CrossvalMethodResult` from
            :meth:`frameflow.validate.crossval.CrossValSuite.run`.
        baselines: optional ``{baseline_name: [MetricRecord, ...]}`` for the scorecard +
            ``metrics.baselines`` summaries.
        make_figures: if ``False``, skip all matplotlib rendering (returns summary only).
        value_range_k: the FIXED physical ``(vmin_k, vmax_k)`` (P1); ``data_range_k`` is the
            span. Defaults to the project constants.
        seed: RNG seed for bootstrap CIs.

    Returns:
        A dict with keys ``metrics`` (``data_range_k``, ``value_range_k``, ``per_frame``,
        ``summary``, ``baselines``), ``crossval`` (``methods_run``), and ``figures`` (paths
        to any rendered PNGs).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    record_dicts = _records_to_dicts(records)
    headline = ("psnr", "ssim", "ms_ssim", "fsim", "bt_rmse_k", "bt_bias_k")
    model_summary = summarize_metrics(records, metrics=headline, seed=seed)

    baseline_summaries: dict[str, dict[str, Any]] = {}
    for name, recs in (baselines or {}).items():
        baseline_summaries[name] = summarize_metrics(recs, metrics=headline, seed=seed)

    figures: dict[str, str] = {}
    if make_figures:
        # Force a headless backend before importing pyplot anywhere.
        import matplotlib  # lazy

        matplotlib.use("Agg", force=True)
        try:
            figures["scorecard"] = str(
                _fig_scorecard(out / "scorecard.png", model_summary, baseline_summaries, headline)
            )
            figures["psnr_vs_frame"] = str(
                _fig_metric_vs_frame(out / "psnr_vs_frame.png", record_dicts, "psnr")
            )
            figures["distribution"] = str(
                _fig_distribution(out / "distribution.png", record_dicts,
                                  ("psnr", "ssim", "ms_ssim", "fsim"))
            )
        except Exception as exc:  # never let a plotting glitch sink the numeric report
            figures["error"] = f"{type(exc).__name__}: {exc}"

    vmin_k, vmax_k = float(value_range_k[0]), float(value_range_k[1])
    crossval_dicts = [
        (c.to_dict() if isinstance(c, CrossvalMethodResult) else dict(c))
        for c in (crossval_results or [])
    ]

    return {
        "metrics": {
            "data_range_k": vmax_k - vmin_k,            # FIXED physical span (P1)
            "value_range_k": [vmin_k, vmax_k],
            "per_frame": record_dicts,
            "summary": model_summary,
            "baselines": baseline_summaries,
        },
        "crossval": {"methods_run": crossval_dicts},
        "figures": figures,
    }

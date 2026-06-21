"""FrameFlow validation — rigorous, fixed-range metrics + a 40-method cross-val suite.

Team **INFER+VALIDATE** (see ``CONTRACTS.md`` §6 and ``research/05_validation.md``). This
subpackage proves the interpolated frames are correct, the scientific half of PS-12.

Public modules:
    * :mod:`frameflow.validate.metrics` — full-reference + BT-domain metrics computed with
      the **FIXED physical Kelvin range** (P1): :func:`~frameflow.validate.metrics.to_unit_interval`,
      :func:`~frameflow.validate.metrics.compute_metrics`, and
      :func:`~frameflow.validate.metrics.per_frame_metrics` (returns a
      :class:`frameflow.contracts.MetricRecord`). Per-image min/max is forbidden — it hides
      warm/cold bias and contrast errors.
    * :mod:`frameflow.validate.crossval` — the 40-method (M1-M40) cross-validation
      framework as a :class:`~frameflow.validate.crossval.CrossValSuite` registry; the
      runnable subset executes on a dense cube, data-dependent methods are catalogued as
      ``skipped``. Module-level :func:`~frameflow.validate.crossval.available_methods` /
      :func:`~frameflow.validate.crossval.run_method` mirror CONTRACTS §6.5.
    * :mod:`frameflow.validate.report` — :func:`~frameflow.validate.report.make_report`,
      which renders matplotlib figures + a JSON summary suitable for the manifest's
      ``metrics`` / ``crossval`` blocks.

Optional heavy/scientific deps (``onnx``, ``pywt``, ``pysteps``, ``xskillscore``,
``scikit-image``) are imported lazily and degrade gracefully; the core metric suite needs
only ``piq`` + ``torch`` + ``numpy`` (always present here).
"""

from __future__ import annotations

from .crossval import CrossValSuite, available_methods, run_method
from .metrics import compute_metrics, per_frame_metrics, to_unit_interval
from .report import make_report

__all__ = [
    "to_unit_interval",
    "compute_metrics",
    "per_frame_metrics",
    "CrossValSuite",
    "available_methods",
    "run_method",
    "make_report",
]

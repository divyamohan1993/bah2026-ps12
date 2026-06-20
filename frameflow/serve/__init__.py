"""FrameFlow serving package (SERVE+VIZ ownership).

The dashboard's primary delivery path is the **precomputed, static** artifact set produced
by :mod:`frameflow.precompute` (per-frame tiles + PMTiles + video + manifest), served O(1)
from a CDN. This package provides:

    * :mod:`frameflow.serve.cache` — a content-addressed, disk-backed (optional Redis)
      cache keyed by ``sha256(I0, I1, t, model_version)`` for the on-demand fallback.
    * :mod:`frameflow.serve.api`   — a FastAPI app (``create_app``) exposing ``/health``,
      ``/manifest/{scene}`` (serve a precomputed manifest), and ``/interpolate`` (the
      optional on-demand model fallback, research/06 §3.4).
    * :mod:`frameflow.serve.app`   — a thin ``run(...)`` launcher used by ``frameflow.cli``.

Model loading and heavy deps (fastapi, redis, torch, the model/infer code) are imported
lazily so importing this package is cheap and never forces optional dependencies.

The public serve entry points named in ``CONTRACTS.md`` are re-exported lazily:
``create_app`` (the FastAPI factory) and ``InterpolationCache`` (the content-addressed
cache). They are resolved on first access via :func:`__getattr__` so that importing this
package never eagerly imports FastAPI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .api import create_app
    from .cache import InterpolationCache

__all__ = ["cache", "api", "app", "create_app", "InterpolationCache"]


def __getattr__(name: str) -> Any:
    """Lazily resolve the public serve API without forcing optional deps at import time."""
    if name == "create_app":
        from .api import create_app

        return create_app
    if name == "InterpolationCache":
        from .cache import InterpolationCache

        return InterpolationCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

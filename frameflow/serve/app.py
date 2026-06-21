"""Uvicorn launcher for the FrameFlow serving app (used by ``frameflow.cli serve``).

This is a thin wrapper so the CLI can do ``from .serve import app as serve_app;
serve_app.run(artifacts=..., host=..., port=...)``. The real application is built by
:func:`frameflow.serve.api.create_app`; static artifacts are also mounted at ``/artifacts``
so the precomputed tiles/PMTiles/video/manifest can be served directly during local demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .api import create_app


def build_app(artifacts: str | Path = "artifacts", **kwargs: Any) -> Any:
    """Build the FastAPI app and mount ``artifacts`` as static files at ``/artifacts``.

    Args:
        artifacts: directory of precomputed scene artifacts.
        **kwargs: forwarded to :func:`frameflow.serve.api.create_app` (e.g. ``model``,
            ``redis_url``).

    Returns:
        The configured FastAPI app.
    """
    app = create_app(artifacts_dir=artifacts, **kwargs)
    art = Path(artifacts)
    if art.exists():
        try:
            from fastapi.staticfiles import StaticFiles

            app.mount("/artifacts", StaticFiles(directory=str(art)), name="artifacts")
        except Exception:  # pragma: no cover - static mount is best-effort
            pass
    return app


def run(
    artifacts: str | Path = "artifacts",
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs: Any,
) -> None:
    """Launch the serving app with uvicorn (blocking).

    Args:
        artifacts: directory of precomputed scene artifacts to serve.
        host: bind host.
        port: bind port.
        **kwargs: forwarded to :func:`build_app` / ``create_app``.
    """
    import uvicorn

    app = build_app(artifacts=artifacts, **kwargs)
    uvicorn.run(app, host=host, port=int(port))


__all__ = ["build_app", "run", "create_app"]

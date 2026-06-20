"""FastAPI app for FrameFlow serving (SERVE, research/06 §3.4).

The precomputed artifact set (per-frame tiles + PMTiles + video + manifest) is the primary
O(1) delivery path; this app exposes:

    * ``GET /health``            -> ``{"status": "ok"}`` (liveness).
    * ``GET /manifest/{scene}``  -> the precomputed ``manifest.json`` for a scene.
    * ``POST /interpolate``      -> the OPTIONAL on-demand model fallback: given two frames
      (inline arrays or ``.nc`` paths) + ``t``, run the model through the content-addressed
      cache and return the synthesized frame as PNG or NetCDF bytes.

Model loading is **lazy** and **injectable**: tests (and deployments) pass a model or a
``model_runner`` callable into :func:`create_app`, so the serving layer never hard-depends on
Team MODELS / Team INFER. If neither is provided, ``/interpolate`` lazily tries
``frameflow.infer`` and otherwise returns HTTP 503.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from .. import constants as C
from .cache import InterpolationCache

# Type of an injected model runner: (I0_2d, I1_2d, t) -> It_2d  (all numpy float arrays, K).
ModelRunner = Callable[[Any, Any, float], Any]


def _coerce_to_2d(data: Any) -> "Any":
    """Coerce a JSON array / nested list to a 2D float32 numpy array."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:  # (1, H, W) -> (H, W)
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D frame, got shape {arr.shape}")
    return arr


def _read_nc_frame(path: str) -> "Any":
    """Read a single-frame ``.nc`` file's ``bt`` field as a 2D float32 array (lazy xarray)."""
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        var = "bt" if "bt" in ds else list(ds.data_vars)[0]
        arr = np.asarray(ds[var].values, dtype=np.float32)
    finally:
        ds.close()
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a single 2D frame, got shape {arr.shape}")
    return arr


def _runner_from_model(model: Any) -> ModelRunner:
    """Wrap a torch-style ``VFIModel`` (forward(I0,I1,t) with (B,1,H,W)/t=(B,1)) as a runner.

    Honours the contract that ``t`` has a leading batch dim ``(B, 1)`` (P2-ONNX). Falls back
    to treating ``model`` as a plain ``callable(I0_2d, I1_2d, t)`` if it has no ``forward``.
    """

    def _run(i0: Any, i1: Any, t: float) -> Any:
        import numpy as np

        forward = getattr(model, "forward", None)
        if forward is None and callable(model):
            return np.asarray(model(i0, i1, t), dtype=np.float32)
        import torch

        with torch.no_grad():
            x0 = torch.as_tensor(np.asarray(i0, dtype=np.float32))[None, None]
            x1 = torch.as_tensor(np.asarray(i1, dtype=np.float32))[None, None]
            tt = torch.as_tensor([[float(t)]], dtype=torch.float32)  # (B, 1) leading batch
            out = forward(x0, x1, tt)
            out = out.detach().cpu().numpy()
        return np.squeeze(out).astype(np.float32)

    return _run


def _infer_runner() -> Optional[ModelRunner]:
    """Lazily build a runner from ``frameflow.infer`` if that team's module is available."""
    try:  # pragma: no cover - depends on Team INFER landing
        from ..infer import interpolate as _inf  # noqa: F401
    except Exception:
        return None

    # Team INFER's public API is file-based (interpolate_pair); a generic array runner is not
    # guaranteed. We return None here so the endpoint reports 503 rather than guessing an API.
    return None


def create_app(
    artifacts_dir: str | Path = "artifacts",
    *,
    model: Any = None,
    model_runner: ModelRunner | None = None,
    model_version: str = "v0.1.0",
    cache: InterpolationCache | None = None,
    cache_dir: str | Path = ".cache/frameflow/interp",
    redis_url: str | None = None,
) -> "Any":
    """Create and return the FrameFlow FastAPI application.

    Args:
        artifacts_dir: root dir holding precomputed scenes (``artifacts/<scene>/manifest.json``).
        model: optional injected model (torch-style ``forward`` or a plain callable). Used by
            ``/interpolate`` if ``model_runner`` is not given.
        model_runner: optional injected callable ``(I0_2d, I1_2d, t) -> It_2d`` (takes
            precedence over ``model``). Lets tests inject a tiny identity model with no
            dependency on Team MODELS.
        model_version: version string folded into the cache key.
        cache: an explicit :class:`InterpolationCache` (else one is built from
            ``cache_dir``/``redis_url``).
        cache_dir: disk cache dir (used when ``cache`` is not supplied).
        redis_url: optional Redis URL for the cache front.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(
        title="FrameFlow Serving API",
        version="1.0",
        description=(
            "Precomputed-artifact serving + optional on-demand interpolation fallback "
            "for ISRO BAH 2026 PS-12 (FrameFlow)."
        ),
    )

    art_root = Path(artifacts_dir)
    interp_cache = cache or InterpolationCache(cache_dir=cache_dir, redis_url=redis_url)

    # Resolve the model runner once (lazily building from `model` if needed).
    resolved_runner: ModelRunner | None = model_runner
    if resolved_runner is None and model is not None:
        resolved_runner = _runner_from_model(model)

    # Stash on app.state so tests/handlers can introspect.
    app.state.artifacts_dir = art_root
    app.state.cache = interp_cache
    app.state.model_version = model_version
    app.state.model_runner = resolved_runner

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/manifest/{scene}")
    def get_manifest(scene: str) -> "Any":
        """Serve a precomputed scene manifest (``artifacts/<scene>/manifest.json``)."""
        # Guard against path traversal in the scene id.
        if "/" in scene or ".." in scene or "\\" in scene:
            raise HTTPException(status_code=400, detail="invalid scene id")
        manifest_path = art_root / scene / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"manifest for scene '{scene}' not found")
        import json

        try:
            data = json.loads(manifest_path.read_text())
        except Exception as exc:  # pragma: no cover - corrupt file
            raise HTTPException(status_code=500, detail=f"failed to read manifest: {exc}") from exc
        return JSONResponse(content=data)

    @app.post("/interpolate")
    def interpolate(payload: dict) -> "Any":
        """Run on-demand interpolation between two frames and return PNG or NetCDF bytes.

        Request JSON (one of ``i0``/``i1`` inline arrays OR ``i0_nc``/``i1_nc`` paths)::

            {
              "i0": [[...]], "i1": [[...]],         # inline 2D (or (1,H,W)) Kelvin arrays
              # or: "i0_nc": "path/a.nc", "i1_nc": "path/b.nc",
              "t": 0.5,                              # interpolation fraction (0,1)
              "format": "png" | "nc"                # response format (default "png")
            }

        The result is cached under ``sha256(i0, i1, t, model_version)``; a repeat request is
        an O(1) cache hit. Returns 503 if no model/runner is configured.
        """
        import numpy as np

        runner = app.state.model_runner or _infer_runner()
        if runner is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "no interpolation model configured; the precomputed manifest/tiles are "
                    "the primary O(1) path. Inject `model`/`model_runner` into create_app() "
                    "or land frameflow.infer for the on-demand fallback."
                ),
            )

        t = float(payload.get("t", 0.5))
        out_fmt = str(payload.get("format", "png")).lower()

        # Load the two source frames (inline arrays preferred, else .nc paths).
        try:
            if payload.get("i0") is not None and payload.get("i1") is not None:
                i0 = _coerce_to_2d(payload["i0"])
                i1 = _coerce_to_2d(payload["i1"])
            elif payload.get("i0_nc") and payload.get("i1_nc"):
                i0 = _read_nc_frame(str(payload["i0_nc"]))
                i1 = _read_nc_frame(str(payload["i1_nc"]))
            else:
                raise HTTPException(
                    status_code=422,
                    detail="provide either inline 'i0'/'i1' arrays or 'i0_nc'/'i1_nc' paths",
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"could not read inputs: {exc}") from exc

        if i0.shape != i1.shape:
            raise HTTPException(
                status_code=422, detail=f"i0 {i0.shape} and i1 {i1.shape} must match"
            )

        cache_key = interp_cache.make_key(i0, i1, t, app.state.model_version)
        cache_hit = interp_cache.get(cache_key)

        if out_fmt == "nc":
            media = "application/x-netcdf"
            if cache_hit is not None:
                return Response(
                    content=cache_hit,
                    media_type=media,
                    headers={"X-FrameFlow-Cache": "hit", "X-FrameFlow-Key": cache_key},
                )
            it = np.asarray(runner(i0, i1, t), dtype=np.float32)
            body = _encode_nc_bytes(it, t=t, model_version=app.state.model_version)
            interp_cache.set(cache_key, body)
            return Response(
                content=body,
                media_type=media,
                headers={"X-FrameFlow-Cache": "miss", "X-FrameFlow-Key": cache_key},
            )

        # default: PNG (colorized on the fixed display range, P1).
        media = "image/png"
        if cache_hit is not None:
            return Response(
                content=cache_hit,
                media_type=media,
                headers={"X-FrameFlow-Cache": "hit", "X-FrameFlow-Key": cache_key},
            )
        it = np.asarray(runner(i0, i1, t), dtype=np.float32)
        from ..viz.render import render_frame

        body = render_frame(it, cmap=C.DEFAULT_COLORMAP, fmt="png")
        interp_cache.set(cache_key, body)
        return Response(
            content=body,
            media_type=media,
            headers={"X-FrameFlow-Cache": "miss", "X-FrameFlow-Key": cache_key},
        )

    return app


def _encode_nc_bytes(frame_2d: "Any", *, t: float, model_version: str) -> bytes:
    """Encode a single 2D BT frame as NetCDF bytes (CF-ish, schema-aligned attrs)."""
    import tempfile

    import numpy as np
    import xarray as xr

    from ..contracts import InferenceNetCDFSchema, netcdf_attrs

    arr = np.asarray(frame_2d, dtype=np.float32)
    h, w = arr.shape
    data = arr[np.newaxis, :, :]
    ds = xr.Dataset(
        data_vars={
            InferenceNetCDFSchema.DATA_VAR: (
                InferenceNetCDFSchema.DIMS,
                data,
                {"units": "K", "long_name": "brightness_temperature"},
            )
        },
        coords={
            "time": ("time", np.array([np.datetime64("2025-06-20T00:00:00")])),
            "lat": ("y", np.arange(h, dtype=np.float64)),
            "lon": ("x", np.arange(w, dtype=np.float64)),
        },
        attrs=netcdf_attrs(
            source_frames=["i0", "i1"],
            t=float(t),
            model="on-demand",
            model_version=model_version,
            kind="interpolated",
        ),
    )
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp:
        for engine in ("netcdf4", "h5netcdf", "scipy"):
            try:
                ds.to_netcdf(tmp.name, engine=engine)
                return Path(tmp.name).read_bytes()
            except Exception:  # pragma: no cover - try next engine
                continue
    raise RuntimeError("no usable NetCDF engine available to encode response")


__all__ = ["create_app", "InterpolationCache"]

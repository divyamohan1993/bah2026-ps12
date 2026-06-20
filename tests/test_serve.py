"""Tests for the SERVE layer: the FastAPI app + content-addressed cache.

Covers (CONTRACTS.md §7.3, research/06 §3.3-§3.4):
    * ``GET /health`` -> ``200 {"status": "ok"}``.
    * ``GET /manifest/{scene}`` -> serves a precomputed ``manifest.json`` (404 when absent,
      400 on a traversal-y scene id).
    * ``POST /interpolate`` with an INJECTED tiny dummy model -> ``200`` image (PNG) and
      NetCDF bytes; the result is content-addressed so a 2nd identical call is a cache HIT.
    * The cache key is deterministic in ``(i0, i1, t, model_version)``.

Self-contained: a tiny NaN-aware linear-blend model is injected via ``create_app``'s
``model``/``model_runner`` hooks, so the serving layer is exercised WITHOUT importing the
models area. FastAPI/httpx are required and skipped if unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient (Starlette TestClient)

from fastapi.testclient import TestClient  # noqa: E402  (after importorskip)

from frameflow.serve.api import create_app  # noqa: E402
from frameflow.serve.cache import InterpolationCache, key  # noqa: E402


# ===========================================================================
# tiny injectable model (torch-free) — never imports the models area
# ===========================================================================
def _dummy_runner(i0: np.ndarray, i1: np.ndarray, t: float) -> np.ndarray:
    """A VFI stand-in: NaN-tolerant linear blend ``(I0_2d, I1_2d, t) -> It_2d`` (Kelvin)."""
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i1, dtype=np.float32)
    return ((1.0 - t) * a + t * b).astype(np.float32)


class _DummyModel:
    """A torch-free object exposing the ``VFIModel``-style callable surface.

    ``create_app(model=...)`` falls back to treating a non-``forward`` object as a plain
    ``callable(I0_2d, I1_2d, t)``, so this exercises the ``model=`` injection path without
    pulling in torch.
    """

    def __call__(self, i0: np.ndarray, i1: np.ndarray, t: float) -> np.ndarray:
        return _dummy_runner(i0, i1, t)

    @property
    def params_m(self) -> float:  # mirrors the contract's VFIModel.params_m
        return 0.001


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """A TestClient over an app with an injected dummy runner + a temp disk cache."""
    app = create_app(
        artifacts_dir=tmp_path / "artifacts",
        model_runner=_dummy_runner,
        model_version="vtest",
        cache_dir=tmp_path / "cache",
    )
    return TestClient(app)


def _i0_i1(h: int = 8, w: int = 8) -> tuple[list, list]:
    i0 = np.full((h, w), 200.0, dtype=np.float32)
    i1 = np.full((h, w), 280.0, dtype=np.float32)
    return i0.tolist(), i1.tolist()


# ===========================================================================
# /health
# ===========================================================================
def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ===========================================================================
# /interpolate  (injected dummy model)
# ===========================================================================
def test_interpolate_returns_png_bytes(client: TestClient) -> None:
    import io

    from PIL import Image

    i0, i1 = _i0_i1()
    resp = client.post("/interpolate", json={"i0": i0, "i1": i1, "t": 0.5, "format": "png"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["X-FrameFlow-Cache"] == "miss"

    img = Image.open(io.BytesIO(resp.content))
    img.load()  # raises if not a valid image
    assert img.size == (8, 8)
    assert img.format == "PNG"


def test_interpolate_returns_netcdf_bytes(client: TestClient) -> None:
    i0, i1 = _i0_i1()
    resp = client.post("/interpolate", json={"i0": i0, "i1": i1, "t": 0.5, "format": "nc"})
    assert resp.status_code == 200, resp.text
    assert "netcdf" in resp.headers["content-type"]
    assert len(resp.content) > 0


def test_interpolate_second_identical_call_hits_cache(client: TestClient) -> None:
    """A repeat request for the same (i0, i1, t, model_version) is an O(1) cache HIT."""
    i0, i1 = _i0_i1()
    body = {"i0": i0, "i1": i1, "t": 0.5, "format": "png"}

    first = client.post("/interpolate", json=body)
    assert first.status_code == 200
    assert first.headers["X-FrameFlow-Cache"] == "miss"

    second = client.post("/interpolate", json=body)
    assert second.status_code == 200
    assert second.headers["X-FrameFlow-Cache"] == "hit"
    # Same content + same content-addressed key on both calls.
    assert second.content == first.content
    assert second.headers["X-FrameFlow-Key"] == first.headers["X-FrameFlow-Key"]


def test_interpolate_model_object_injection(tmp_path: Path) -> None:
    """The ``model=`` hook accepts a plain callable object (no torch needed)."""
    import io

    from PIL import Image

    app = create_app(
        artifacts_dir=tmp_path / "art",
        model=_DummyModel(),
        model_version="vobj",
        cache_dir=tmp_path / "cache",
    )
    c = TestClient(app)
    i0, i1 = _i0_i1()
    resp = c.post("/interpolate", json={"i0": i0, "i1": i1, "t": 0.25, "format": "png"})
    assert resp.status_code == 200, resp.text
    img = Image.open(io.BytesIO(resp.content))
    img.load()
    assert img.size == (8, 8)


def test_interpolate_without_model_returns_503(tmp_path: Path) -> None:
    """With no model/runner configured the on-demand endpoint reports 503 (precompute is primary)."""
    app = create_app(artifacts_dir=tmp_path / "art", cache_dir=tmp_path / "cache")
    c = TestClient(app)
    i0, i1 = _i0_i1()
    resp = c.post("/interpolate", json={"i0": i0, "i1": i1, "t": 0.5})
    assert resp.status_code == 503


def test_interpolate_shape_mismatch_is_422(client: TestClient) -> None:
    i0 = np.zeros((8, 8), dtype=np.float32).tolist()
    i1 = np.zeros((8, 4), dtype=np.float32).tolist()
    resp = client.post("/interpolate", json={"i0": i0, "i1": i1, "t": 0.5})
    assert resp.status_code == 422


def test_interpolate_missing_inputs_is_422(client: TestClient) -> None:
    resp = client.post("/interpolate", json={"t": 0.5})
    assert resp.status_code == 422


# ===========================================================================
# /manifest/{scene}
# ===========================================================================
def test_get_manifest_served_from_artifacts(tmp_path: Path) -> None:
    """A precomputed manifest under ``artifacts/<scene>/manifest.json`` is served as JSON."""
    from frameflow.viz.manifest import build_manifest

    art = tmp_path / "artifacts"
    scene = "demo-scene"
    frames_meta = [
        {
            "index": i,
            "time": f"2025-06-20T00:{i * 10:02d}:00Z",
            "kind": "observed" if i != 1 else "interpolated",
            "t": None if i != 1 else 0.5,
            "bracket": None if i != 1 else [0, 2],
            "image": f"img/{i:03d}.webp",
            "thumb": f"thumb/{i:03d}.webp",
            "tiles_url_template": f"tiles/{i:03d}/{{z}}/{{x}}/{{y}}.webp",
            "pmtiles": f"pmtiles/{i:03d}.pmtiles",
            "netcdf": f"nc/{i:03d}.nc",
        }
        for i in range(3)
    ]
    manifest = build_manifest(
        scene_id=scene,
        bbox=[68.0, 6.0, 98.0, 38.0],
        frames_meta=frames_meta,
        model_info={"name": "RIFE", "version": "v0.1.0", "params_m": 9.8},
    )
    scene_dir = art / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "manifest.json").write_text(json.dumps(manifest.to_dict()))

    app = create_app(artifacts_dir=art, cache_dir=tmp_path / "cache")
    c = TestClient(app)

    resp = c.get(f"/manifest/{scene}")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["scene_id"] == scene
    assert len(data["frames"]) == 3


def test_get_manifest_missing_is_404(client: TestClient) -> None:
    resp = client.get("/manifest/does-not-exist")
    assert resp.status_code == 404


def test_get_manifest_rejects_path_traversal(client: TestClient) -> None:
    resp = client.get("/manifest/..%2f..%2fetc")
    # Either the route rejects the id (400) or it simply isn't found (404); never a 200.
    assert resp.status_code in (400, 404)


# ===========================================================================
# cache: deterministic content-addressed key
# ===========================================================================
def test_cache_key_is_deterministic() -> None:
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    b = a + 1.0

    k1 = key(a, b, 0.5, "v0.1.0")
    k2 = key(a, b, 0.5, "v0.1.0")
    assert k1 == k2, "same inputs must yield the same key"

    # Any input change perturbs the key.
    assert key(a, b, 0.6, "v0.1.0") != k1            # different t
    assert key(a, b, 0.5, "v0.2.0") != k1            # different model version
    assert key(b, a, 0.5, "v0.1.0") != k1            # swapped frames
    assert isinstance(k1, str) and len(k1) == 64     # sha256 hex digest


def test_cache_disk_roundtrip(tmp_path: Path) -> None:
    cache = InterpolationCache(cache_dir=tmp_path / "c", redis_url=None)
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    k = cache.make_key(a, b, 0.5, "v0")

    assert cache.get(k) is None and not cache.has(k)
    cache.set(k, b"payload-bytes")
    assert cache.has(k)
    assert cache.get(k) == b"payload-bytes"

    cache.clear()
    assert cache.get(k) is None

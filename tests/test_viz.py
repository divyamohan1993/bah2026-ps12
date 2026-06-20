"""Tests for the SERVE+VIZ visualization + precompute layer.

Covers (CONTRACTS.md §7, §8):
    * :func:`frameflow.viz.colormap.bt_to_rgba` — uint8 RGBA shape, NaN -> alpha 0, and the
      FIXED physical value range (P1: never per-image min/max).
    * :func:`frameflow.viz.render.render_frame` — encoded bytes PIL can reopen.
    * :func:`frameflow.viz.tiles.build_frame_tiles` — a PER-FRAME ``{z}/{x}/{y}.webp`` XYZ
      pyramid is written and each tile is a valid WebP (P2).
    * :func:`frameflow.viz.video.encode_video` — ~4 frames encode to a non-empty file
      (skipped if ``imageio_ffmpeg`` is unavailable).
    * :func:`frameflow.viz.flow_overlay.flow_to_overlay_json` — deck.gl LineLayer vectors.
    * :func:`frameflow.viz.manifest.build_manifest` — passes ``validate_manifest`` INCLUDING
      per-frame ``tiles_url_template`` (with ``{z}/{x}/{y}``) + ``pmtiles`` (P2).
    * :func:`frameflow.precompute.precompute_scene` — end-to-end, degrades gracefully with no
      model and still writes per-frame artifacts + a valid manifest to ``artifacts/`` (not web/).

Tests are self-contained: they build tiny inputs via :mod:`frameflow.synthetic` and a NaN-aware
linear-blend dummy model, and never import the models area.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from frameflow import constants as C


# ===========================================================================
# helpers
# ===========================================================================
def _bt_field(h: int = 24, w: int = 24, *, fill: float = 250.0) -> np.ndarray:
    """A small BT field with a NaN corner plus the two range extremes embedded."""
    bt = np.full((h, w), float(fill), dtype=np.float32)
    bt[0, 0] = np.nan                      # off-disk / space pixel
    bt[1, 1] = C.BT_METRIC_VMIN_K          # coldest in the fixed range
    bt[2, 2] = C.BT_METRIC_VMAX_K          # warmest in the fixed range
    return bt


def _dummy_runner(i0: np.ndarray, i1: np.ndarray, t: float) -> np.ndarray:
    """A torch-free VFI stand-in: NaN-tolerant linear blend (I0, I1, t) -> It."""
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i1, dtype=np.float32)
    return ((1.0 - t) * a + t * b).astype(np.float32)


# ===========================================================================
# colormap.bt_to_rgba
# ===========================================================================
@pytest.mark.parametrize("cmap", list(C.COLORMAP_NAMES))
def test_bt_to_rgba_shape_dtype_and_nan_alpha(cmap: str) -> None:
    from frameflow.viz.colormap import bt_to_rgba

    bt = _bt_field()
    rgba = bt_to_rgba(bt, cmap)

    # uint8 RGBA of the right shape.
    assert rgba.shape == (bt.shape[0], bt.shape[1], 4)
    assert rgba.dtype == np.uint8

    # NaN pixel -> fully transparent; finite pixels -> fully opaque.
    assert rgba[0, 0, 3] == 0, "NaN pixel must map to alpha 0"
    assert rgba[1, 1, 3] == 255
    assert rgba[3, 3, 3] == 255


def test_bt_to_rgba_uses_fixed_range_not_per_image() -> None:
    """P1: colorization uses the FIXED Kelvin range, never per-image min/max.

    Two fields containing the SAME values but DIFFERENT surrounding extremes must colorize
    identically. A per-image stretch would map them differently; a fixed range maps the same
    Kelvin value to the same colour every time.
    """
    from frameflow.viz.colormap import bt_to_rgba

    base = np.full((8, 8), 250.0, dtype=np.float32)
    cold = base.copy()
    cold[0, 0] = 190.0  # introduces a colder extreme (would shift a per-image stretch)
    warm = base.copy()
    warm[0, 0] = 310.0  # introduces a warmer extreme

    rgba_cold = bt_to_rgba(cold, "ir_clouds")
    rgba_warm = bt_to_rgba(warm, "ir_clouds")

    # The shared 250 K pixels (everything except [0,0]) must be identical across both fields.
    assert np.array_equal(rgba_cold[1:, 1:], rgba_warm[1:, 1:]), (
        "fixed-range colorization should map identical Kelvin values identically (P1)"
    )

    # And the explicit range endpoints land at the colormap extremes (position 0 and 1).
    endpoints = np.array([[C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K]], dtype=np.float32)
    rgba_ep = bt_to_rgba(endpoints, "greyscale_ir")
    # greyscale_ir: cold (vmin) -> white, warm (vmax) -> black.
    assert tuple(rgba_ep[0, 0, :3]) == (255, 255, 255)
    assert tuple(rgba_ep[0, 1, :3]) == (0, 0, 0)


def test_bt_to_rgba_rejects_bad_range() -> None:
    from frameflow.viz.colormap import bt_to_rgba

    with pytest.raises(ValueError):
        bt_to_rgba(_bt_field(), "ir_clouds", vmin=300.0, vmax=200.0)


# ===========================================================================
# render.render_frame
# ===========================================================================
@pytest.mark.parametrize("fmt", ["webp", "png"])
def test_render_frame_bytes_reopenable_by_pil(fmt: str) -> None:
    from PIL import Image

    from frameflow.viz.render import render_frame

    bt = _bt_field(20, 28)
    data = render_frame(bt, fmt=fmt)
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0

    img = Image.open(io.BytesIO(bytes(data)))
    img.load()  # force decode — raises if the bytes are not a valid image
    assert img.size == (bt.shape[1], bt.shape[0])  # PIL size is (W, H)
    assert img.format == fmt.upper()


def test_render_frame_colormap_kwarg_and_writes_file(tmp_path: Path) -> None:
    from PIL import Image

    from frameflow.viz.render import render_frame

    out = tmp_path / "frame.webp"
    # `colormap=` is the CONTRACTS.md keyword alias and must take precedence.
    data = render_frame(_bt_field(), colormap="turbo", out_path=out, fmt="webp")
    assert out.exists() and out.read_bytes() == data
    img = Image.open(out)
    img.load()
    assert img.mode in ("RGBA", "RGB")


# ===========================================================================
# tiles.build_frame_tiles  (P2: per-frame XYZ source)
# ===========================================================================
def test_build_frame_tiles_writes_valid_xyz_webp(tmp_path: Path) -> None:
    from PIL import Image

    from frameflow.viz.tiles import build_frame_tiles

    bt = _bt_field(16, 16)
    bbox = [68.0, 6.0, 98.0, 38.0]
    info = build_frame_tiles(
        bt, bbox, tmp_path, frame_index=3, tile_size=8, min_zoom=0, max_zoom=1
    )

    # The returned URL template is per-frame and well-formed for XYZ (P2).
    tmpl = info["tiles_url_template"]
    assert tmpl == "tiles/003/{z}/{x}/{y}.webp"
    assert "{z}" in tmpl and "{x}" in tmpl and "{y}" in tmpl
    assert info["min_zoom"] == 0 and info["max_zoom"] == 1
    assert info["tile_size"] == 8

    # Every tile in the matrix exists at the expected path and is a valid 8x8 WebP.
    frame_root = tmp_path / "tiles" / "003"
    assert frame_root.is_dir()
    n_seen = 0
    for z in range(info["min_zoom"], info["max_zoom"] + 1):
        n = 2 ** z
        for x in range(n):
            for y in range(n):
                p = frame_root / str(z) / str(x) / f"{y}.webp"
                assert p.exists(), f"missing tile {p}"
                tile = Image.open(p)
                tile.load()
                assert tile.size == (8, 8)
                assert tile.format == "WEBP"
                n_seen += 1
    assert n_seen == info["n_tiles"]


def test_build_frame_pmtiles_graceful_skip(tmp_path: Path) -> None:
    """Per-frame PMTiles is optional: returns None (skips) when no writer is installed."""
    from frameflow.viz.tiles import build_frame_pmtiles, pmtiles_available

    rel = build_frame_pmtiles(_bt_field(8, 8), [68.0, 6.0, 98.0, 38.0], tmp_path, 0, tile_size=8, max_zoom=0)
    if pmtiles_available():
        assert rel == "pmtiles/000.pmtiles" and (tmp_path / rel).exists()
    else:
        assert rel is None  # graceful skip; XYZ pyramid is the source of record


# ===========================================================================
# video.encode_video  (all-intra; needs imageio-ffmpeg)
# ===========================================================================
def test_encode_video_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    from frameflow.viz.video import encode_video

    frames = [np.full((16, 16), 200.0 + 25.0 * i, dtype=np.float32) for i in range(4)]
    out = encode_video(frames, tmp_path / "clip.mp4", fps=4, all_intra=True)
    assert out.exists() and out.stat().st_size > 0


def test_encode_side_by_side_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    from frameflow.viz.video import encode_side_by_side

    left = [np.full((16, 16), 210.0 + 10.0 * i, dtype=np.float32) for i in range(4)]
    right = [np.full((16, 16), 280.0 - 10.0 * i, dtype=np.float32) for i in range(4)]
    out = encode_side_by_side(left, right, tmp_path / "sbs.mp4", fps=4)
    assert out.exists() and out.stat().st_size > 0


def test_encode_video_empty_raises() -> None:
    from frameflow.viz.video import encode_video

    with pytest.raises(ValueError):
        encode_video([], "unused.mp4")


# ===========================================================================
# flow_overlay.flow_to_overlay_json  (deck.gl LineLayer)
# ===========================================================================
def test_flow_to_overlay_json_vectors() -> None:
    from frameflow.viz.flow_overlay import flow_to_overlay_json

    h = w = 16
    flow = np.zeros((2, h, w), dtype=np.float32)
    flow[0] = 2.0   # u: eastward
    flow[1] = -1.0  # v: northward (negative row displacement)
    flow[0, 0, 0] = np.nan  # off-disk sample must be skipped
    bbox = [68.0, 6.0, 98.0, 38.0]

    overlay = flow_to_overlay_json(flow, step=4, bbox=bbox, scale=1.0)
    assert overlay["type"] == "LineLayer"
    assert overlay["bbox"] == bbox
    assert overlay["step"] == 4
    assert len(overlay["vectors"]) > 0

    v0 = overlay["vectors"][0]
    for key in ("sourcePosition", "targetPosition", "u", "v", "mag"):
        assert key in v0
    assert len(v0["sourcePosition"]) == 2 and len(v0["targetPosition"]) == 2
    # Eastward + northward flow -> target lon increases, target lat increases.
    assert v0["targetPosition"][0] > v0["sourcePosition"][0]
    assert v0["targetPosition"][1] > v0["sourcePosition"][1]


def test_flow_to_overlay_json_bad_inputs() -> None:
    from frameflow.viz.flow_overlay import flow_to_overlay_json

    with pytest.raises(ValueError):
        flow_to_overlay_json(np.zeros((2, 8, 8)), step=0, bbox=[0, 0, 1, 1])
    with pytest.raises(ValueError):
        flow_to_overlay_json(np.zeros((8, 8, 3)), step=2, bbox=[0, 0, 1, 1])


# ===========================================================================
# manifest.build_manifest  (passes validate_manifest, P1 + P2)
# ===========================================================================
def _frames_meta() -> list[dict]:
    """Three-frame timeline (observed / interpolated / observed) with per-frame sources."""
    specs = [("observed", None, None), ("interpolated", 0.5, [0, 2]), ("observed", None, None)]
    out: list[dict] = []
    for i, (kind, t, bracket) in enumerate(specs):
        out.append(
            {
                "index": i,
                "time": f"2025-06-20T00:{i * 10:02d}:00Z",
                "kind": kind,
                "t": t,
                "bracket": bracket,
                "image": f"img/{i:03d}.webp",
                "thumb": f"thumb/{i:03d}.webp",
                # P2: each frame carries its OWN XYZ template + PMTiles archive path.
                "tiles_url_template": f"tiles/{i:03d}/{{z}}/{{x}}/{{y}}.webp",
                "pmtiles": f"pmtiles/{i:03d}.pmtiles",
                "netcdf": f"nc/{i:03d}.nc",
            }
        )
    return out


def test_build_manifest_passes_validation_with_per_frame_sources() -> None:
    from frameflow.contracts import validate_manifest
    from frameflow.viz.manifest import build_manifest

    manifest = build_manifest(
        scene_id="demo-0001",
        bbox=[68.0, 6.0, 98.0, 38.0],
        frames_meta=_frames_meta(),
        metrics={"per_frame": [{"index": 1, "psnr": 44.0, "ssim": 0.95, "bt_rmse_k": 0.9}]},
        crossval=[{"method_id": "M1", "name": "leave-the-middle-out"}],
        videos={"observed": "videos/observed.mp4", "interpolated": "videos/interpolated.mp4"},
        model_info={"name": "RIFE", "version": "v0.1.0", "params_m": 9.8},
    )
    d = manifest.to_dict()
    assert validate_manifest(d) == [], "build_manifest output must pass validate_manifest"

    # P1: the fixed Kelvin range is forced and consistent.
    assert d["metrics"]["data_range_k"] == pytest.approx(C.BT_DATA_RANGE_K)
    assert d["value_range_k"] == [C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K]
    assert d["metrics"]["data_range_k"] == pytest.approx(
        d["value_range_k"][1] - d["value_range_k"][0]
    )

    # P2: every frame carries a valid per-frame XYZ template + a non-empty pmtiles path.
    for fr in d["frames"]:
        tmpl = fr["tiles_url_template"]
        assert "{z}" in tmpl and "{x}" in tmpl and "{y}" in tmpl
        assert fr["pmtiles"]
    # The interpolated frame keeps its t + bracket.
    assert d["frames"][1]["t"] == pytest.approx(0.5)
    assert d["frames"][1]["bracket"] == [0, 2]
    # The three video keys exist.
    for vk in ("observed", "interpolated", "side_by_side"):
        assert vk in d["videos"]


def test_build_manifest_rejects_missing_pmtiles() -> None:
    """A frame without a per-frame pmtiles path (P2) must fail manifest assembly."""
    from frameflow.viz.manifest import build_manifest

    bad = _frames_meta()
    bad[1]["pmtiles"] = ""  # violate P2
    with pytest.raises(ValueError):
        build_manifest(
            scene_id="demo",
            bbox=[68.0, 6.0, 98.0, 38.0],
            frames_meta=bad,
            model_info={"name": "RIFE", "version": "v0", "params_m": 1.0},
        )


def test_write_manifest_roundtrips(tmp_path: Path) -> None:
    from frameflow.contracts import validate_manifest
    from frameflow.viz.manifest import build_manifest, write_manifest

    manifest = build_manifest(
        scene_id="demo",
        bbox=[68.0, 6.0, 98.0, 38.0],
        frames_meta=_frames_meta(),
        model_info={"name": "RIFE", "version": "v0", "params_m": 1.0},
    )
    path = write_manifest(manifest, tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert validate_manifest(loaded) == []


# ===========================================================================
# precompute.precompute_scene  (end-to-end; graceful degrade; writes to artifacts/)
# ===========================================================================
def _make_cube(tmp_path: Path, *, n_frames: int = 3, n: int = 24) -> Path:
    from frameflow.contracts import GridSpec
    from frameflow.synthetic import generate_cube

    grid = GridSpec(
        west=68.0, south=6.0, east=98.0, north=38.0,
        n_rows=n, n_cols=n, crs="EPSG:4326", resolution_deg=30.0 / n,
    )
    return Path(
        generate_cube(
            n_frames=n_frames,
            grid=grid,
            out_path=str(tmp_path / "cube.zarr"),
            cadence_min=30,
            seed=0,
            n_blobs=3,
        )
    )


def test_precompute_scene_degrades_without_model(tmp_path: Path) -> None:
    """No model injected -> still renders observed + interpolated frames + a valid manifest."""
    from frameflow.contracts import validate_manifest
    from frameflow.precompute import precompute_scene

    cube = _make_cube(tmp_path)
    out_dir = tmp_path / "artifacts" / "scene"
    result = precompute_scene(
        cube, model=None, out_dir=out_dir, factor=2, scene_id="scene", make_video=False
    )

    # Densified to 5 frames (3 observed + 2 interpolated midpoints).
    assert result.n_observed == 3
    assert result.n_interpolated == 2
    assert result.n_frames == 5

    manifest = json.loads(result.manifest_path.read_text())
    assert validate_manifest(manifest) == []
    assert len(manifest["frames"]) == 5

    # Per-frame artifacts exist on disk (img/thumb/nc + an XYZ tile).
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "img" / "000.webp").exists()
    assert (out_dir / "thumb" / "000.webp").exists()
    assert (out_dir / "nc" / "000.nc").exists()
    assert (out_dir / "tiles" / "000" / "0" / "0" / "0.webp").exists()


def test_precompute_scene_default_out_is_artifacts_not_web(tmp_path: Path, monkeypatch) -> None:
    """The default output root is ``artifacts/<scene>/`` — precompute must NEVER target web/."""
    from frameflow.precompute import precompute_scene

    cube = _make_cube(tmp_path, n_frames=2)
    monkeypatch.chdir(tmp_path)  # so the relative "artifacts/" default lands in the temp dir
    result = precompute_scene(cube, model=None, scene_id="abc", factor=2, make_video=False)

    assert result.out_dir == Path("artifacts") / "abc"
    assert "web" not in result.out_dir.parts
    assert (tmp_path / "artifacts" / "abc" / "manifest.json").exists()


def test_precompute_scene_with_injected_model(tmp_path: Path) -> None:
    """An injected (torch-free) model runner drives densification and per-frame metrics."""
    from frameflow.contracts import validate_manifest
    from frameflow.precompute import precompute_scene

    cube = _make_cube(tmp_path, n_frames=3, n=20)
    out_dir = tmp_path / "artifacts" / "modelled"
    result = precompute_scene(
        cube, model=_dummy_runner, out_dir=out_dir, factor=2, scene_id="modelled", make_video=False
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert validate_manifest(manifest) == []
    # Metrics were computed for the interpolated frames against the fixed-K range (P1).
    assert manifest["metrics"]["data_range_k"] == pytest.approx(C.BT_DATA_RANGE_K)
    assert len(manifest["metrics"]["per_frame"]) == result.n_interpolated

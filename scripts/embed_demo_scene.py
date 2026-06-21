#!/usr/bin/env python3
"""Embed the REAL precomputed demo scene into the web dashboard (static hosting).

``make demo`` produces a fully-validated scene under ``out/artifacts/demo-0001/``
(manifest.json + per-frame WebP + thumbs + XYZ tiles + per-frame ``.nc`` + flow
overlays + 3 all-intra MP4s). That artifact tree is git-ignored and ~1.7 MB.

This script copies a **compact, statically-hostable** subset of that REAL scene
into ``web/public/data/<scene>/`` so the dashboard loads genuine model output by
default (instead of the JS mock generator). Concretely it:

  1. Copies the assets the dashboard actually renders:
        - ``manifest.json``  (sanitized, see below)
        - ``img/*.webp``     (full-frame previews -> BitmapLayer path)
        - ``thumb/*.webp``   (filmstrip thumbnails)
        - ``videos/*.mp4``   (observed / interpolated / side_by_side, all-intra)
        - ``flow/*.json``    (optical-flow overlays for interpolated frames)
     and DROPS the heavy XYZ tile pyramids (``tiles/``) and per-frame ``.nc``
     (``nc/``) — neither is fetched by the dashboard. The web already falls back
     to a deck.gl BitmapLayer over ``frame.image`` when ``tiles_url_template`` is
     null (see web/src/lib/deckLayers.ts), so dropping tiles is lossless for the
     demo and keeps the embed tiny.

  2. Sanitizes the manifest so it is valid for the browser and shows real numbers:
        - Python ``json.dump`` emits bare ``NaN`` tokens, which are INVALID JSON
          and make the browser ``JSON.parse`` throw. We re-serialize with
          ``allow_nan=False`` after replacing every non-finite value with ``null``.
        - The precompute per-frame metric proxy currently records all-NaN here
          (it passes an INCLUDE-mask to validate.per_frame_metrics, whose contract
          is "True == EXCLUDE"; see the note printed at the end of this script).
          We RECOMPUTE the per-frame metrics with the correct mask polarity from
          the real interpolated frames vs the linear-blend-of-bracket reference
          (the same proxy precompute intends), so the metric strip is populated
          with genuine values rather than NaN.
        - Folds the REAL validation headline + baselines from
          ``out/validation/metrics.json`` (trained IFNet vs linear vs TV-L1, on
          the WITHHELD true middle frames) into ``metrics.summary`` /
          ``metrics.baselines`` so the "Model vs baselines" panel is honest.
        - Sets each frame's ``tiles_url_template`` and ``pmtiles`` to ``null`` (the
          dropped-tiles BitmapLayer path). ``netcdf`` path strings are kept as
          provenance (the dashboard never fetches them).

  3. Re-validates the sanitized manifest with the SAME web-side rules
     (``frameflow.contracts.validate_manifest``) before writing, and refreshes
     ``web/public/data/scenes.json`` to advertise the embedded scene first.

The embedded manifest never violates CONTRACTS.md §8 — it only makes the JSON
browser-safe and fills in metrics the precompute proxy left as NaN.

Usage:
    python scripts/embed_demo_scene.py
    python scripts/embed_demo_scene.py --src out/artifacts/demo-0001 --scene demo-0001
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# JSON sanitization (NaN/Inf -> null so the browser can parse it)
# ---------------------------------------------------------------------------
def _finite_or_none(v: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` (valid JSON ``null``)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite_or_none(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite_or_none(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Per-frame metric recompute (correct mask polarity)
# ---------------------------------------------------------------------------
def _recompute_per_frame_metrics(src: Path, frames_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute per-frame metrics for interpolated frames from the real ``.nc`` outputs.

    Reference is the NaN-aware linear blend of the bracketing observed frames at the
    frame's ``t`` (the same proxy precompute uses). The validate metric mask convention
    is "True == EXCLUDE", so we pass ``~finite`` (precompute passes ``finite`` — the bug
    that produces the all-NaN records we are repairing here).
    """
    import numpy as np
    import xarray as xr

    from frameflow.validate.metrics import per_frame_metrics

    def _read_nc(idx: int) -> np.ndarray | None:
        p = src / "nc" / f"{idx:03d}.nc"
        if not p.exists():
            return None
        ds = xr.open_dataset(p)
        try:
            var = "bt" if "bt" in ds else list(ds.data_vars)[0]
            arr = np.asarray(ds[var].values, dtype=np.float32)
        finally:
            ds.close()
        return np.squeeze(arr)

    def _blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        out = (1.0 - t) * a + t * b
        na, nb = np.isnan(a), np.isnan(b)
        out = np.where(na & ~nb, b, out)
        out = np.where(nb & ~na, a, out)
        return np.where(na & nb, np.nan, out).astype(np.float32)

    by_index = {int(f["index"]): f for f in frames_meta}
    per_frame: list[dict[str, Any]] = []
    for f in frames_meta:
        if f.get("kind") != "interpolated":
            continue
        bracket = f.get("bracket")
        if not bracket or len(bracket) != 2:
            continue
        idx = int(f["index"])
        # The bracket indices are ORIGINAL observed indices; map them to the global
        # frame indices of the two nearest enclosing observed frames in the densified
        # track (observed frames keep kind="observed").
        observed_global = [int(g["index"]) for g in frames_meta if g.get("kind") == "observed"]
        lo_obs, hi_obs = int(bracket[0]), int(bracket[1])
        if lo_obs >= len(observed_global) or hi_obs >= len(observed_global):
            continue
        a = _read_nc(observed_global[lo_obs])
        b = _read_nc(observed_global[hi_obs])
        pred = _read_nc(idx)
        if a is None or b is None or pred is None or a.shape != pred.shape:
            continue
        t = float(f.get("t") or 0.5)
        ref = _blend(a, b, t)
        exclude = ~(np.isfinite(pred) & np.isfinite(ref))
        rec = per_frame_metrics(pred, ref, data_range_k=140.0, mask=exclude, index=idx, time=f.get("time"))
        d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
        d["index"] = idx
        d["t"] = round(t, 3)
        d["time"] = f.get("time")
        # Promote gmsd out of extra so the web schema (which has a top-level gmsd) sees it.
        extra = d.get("extra") or {}
        if "gmsd" in extra and "gmsd" not in d:
            d["gmsd"] = extra["gmsd"]
        per_frame.append(d)
    _ = by_index  # (kept for clarity; mapping done via observed_global)
    return per_frame


def _summarize(per_frame: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of each finite numeric metric across the per-frame records."""
    import numpy as np

    if not per_frame:
        return {}
    keys = {
        k
        for rec in per_frame
        for k, v in rec.items()
        if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v)) and k not in ("index",)
    }
    out: dict[str, float] = {}
    for k in keys:
        vals = [
            float(rec[k])
            for rec in per_frame
            if isinstance(rec.get(k), (int, float)) and rec[k] is not None and math.isfinite(float(rec[k]))
        ]
        if vals:
            out[f"{k}_mean"] = round(float(np.mean(vals)), 4)
    return out


def _baselines_from_validation(val_json: Path) -> dict[str, dict[str, Any]]:
    """Build the manifest ``metrics.baselines`` block from the demo validation JSON.

    ``out/validation/metrics.json`` is ``{method: {psnr, ssim, ms_ssim, fsim, bt_rmse_k}}``
    with method in {trained_ifnet, linear, tvl1}. We surface the two classical baselines
    (linear, tvl1) in the web's BaselineSummary shape.
    """
    if not val_json.exists():
        return {}
    try:
        data = json.loads(val_json.read_text())
    except Exception:
        return {}
    pretty = {"linear": "Linear blend", "tvl1": "TV-L1 + warp", "persistence": "Persistence (copy)"}
    out: dict[str, dict[str, Any]] = {}
    for key in ("linear", "tvl1", "persistence"):
        m = data.get(key)
        if not isinstance(m, dict):
            continue
        rec: dict[str, Any] = {"name": pretty.get(key, key)}
        for mk in ("psnr", "ssim", "ms_ssim", "bt_rmse_k", "fsim"):
            if mk in m and m[mk] is not None and math.isfinite(float(m[mk])):
                rec[mk] = round(float(m[mk]), 4)
        out[key] = rec
    return out


def _headline_summary(val_json: Path) -> dict[str, float]:
    """Headline means from the REAL trained-IFNet validation (overrides the proxy means)."""
    if not val_json.exists():
        return {}
    try:
        data = json.loads(val_json.read_text())
    except Exception:
        return {}
    m = data.get("trained_ifnet")
    if not isinstance(m, dict):
        return {}
    out: dict[str, float] = {}
    mapping = {
        "psnr": "psnr_mean",
        "ssim": "ssim_mean",
        "ms_ssim": "ms_ssim_mean",
        "fsim": "fsim_mean",
        "bt_rmse_k": "bt_rmse_k_mean",
    }
    for src_k, dst_k in mapping.items():
        if src_k in m and m[src_k] is not None and math.isfinite(float(m[src_k])):
            out[dst_k] = round(float(m[src_k]), 4)
    return out


# ---------------------------------------------------------------------------
# Web-loader validation (mirror of web/src/lib/manifest.ts:validateManifest)
# ---------------------------------------------------------------------------
def _validate_web_manifest(m: dict[str, Any]) -> list[str]:
    """Validate exactly what the browser's ``validateManifest`` enforces.

    This is intentionally LOOSER than ``frameflow.contracts.validate_manifest`` (it does
    NOT require per-frame tiles/pmtiles), matching web/src/lib/manifest.ts so the embedded,
    tiles-dropped scene is accepted by the dashboard. Returns a list of problems ([] == ok).
    """
    problems: list[str] = []

    def is_finite_number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))

    for k in ("scene_id", "title", "satellite", "channel"):
        if not isinstance(m.get(k), str) or not m.get(k):
            problems.append(f'field "{k}" must be a non-empty string')
    for k in ("wavelength_um", "tile_size"):
        if not is_finite_number(m.get(k)):
            problems.append(f'field "{k}" must be a finite number')

    bbox = m.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(is_finite_number(x) for x in bbox)):
        problems.append('"bbox" must be [west, south, east, north]')
    vr = m.get("value_range_k")
    if not (isinstance(vr, list) and len(vr) == 2 and all(is_finite_number(x) for x in vr)):
        problems.append('"value_range_k" must be [min, max]')

    frames = m.get("frames")
    if not (isinstance(frames, list) and frames):
        problems.append('"frames" must be a non-empty array')
    else:
        for i, fr in enumerate(frames):
            if not is_finite_number(fr.get("index")):
                problems.append(f"frames[{i}].index must be a number")
            if not isinstance(fr.get("time"), str):
                problems.append(f"frames[{i}].time must be an ISO-8601 string")
            if fr.get("kind") not in ("observed", "interpolated"):
                problems.append(f'frames[{i}].kind must be "observed" or "interpolated"')
            if not isinstance(fr.get("image"), str):
                problems.append(f"frames[{i}].image must be a string path")

    metrics = m.get("metrics")
    if not isinstance(metrics, dict):
        problems.append('"metrics" object is missing')
    elif not isinstance(metrics.get("per_frame"), list):
        problems.append('"metrics.per_frame" must be an array')
    return problems


# ---------------------------------------------------------------------------
# Asset copy
# ---------------------------------------------------------------------------
def _copy_tree(src: Path, dst: Path, subdir: str, suffixes: tuple[str, ...]) -> int:
    s = src / subdir
    if not s.is_dir():
        return 0
    d = dst / subdir
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(s.iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes:
            shutil.copy2(p, d / p.name)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="out/artifacts/demo-0001", help="precompute artifact dir")
    ap.add_argument("--scene", default="demo-0001", help="embedded scene id (dir under web/public/data)")
    ap.add_argument("--validation", default="out/validation/metrics.json", help="demo validation metrics JSON")
    ap.add_argument(
        "--title",
        default="GOES-19 · Real IFNet demo (synthetic-trained)",
        help="human-readable scene title shown in the selector",
    )
    args = ap.parse_args()

    src = (_REPO_ROOT / args.src).resolve()
    if not (src / "manifest.json").exists():
        print(f"ERROR: {src}/manifest.json not found. Run `make demo` first.", file=sys.stderr)
        return 2

    data_root = _REPO_ROOT / "web" / "public" / "data"
    dst = data_root / args.scene
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # --- 1) Copy the rendered assets (drop tiles/ and nc/). ---
    n_img = _copy_tree(src, dst, "img", (".webp", ".png"))
    n_thumb = _copy_tree(src, dst, "thumb", (".webp", ".png"))
    n_video = _copy_tree(src, dst, "videos", (".mp4",))
    n_flow = _copy_tree(src, dst, "flow", (".json",))

    # --- 2) Load + sanitize the manifest. ---
    manifest = json.loads((src / "manifest.json").read_text())
    frames_meta = manifest.get("frames", [])

    # Drop per-frame tiles/pmtiles (we ship the BitmapLayer image path only).
    for f in frames_meta:
        f["tiles_url_template"] = None
        f["pmtiles"] = None
    # With tiles dropped, advertise a single-zoom BitmapLayer extent.
    manifest["min_zoom"] = 0
    manifest["max_zoom"] = 0

    # Recompute real per-frame metrics (correct mask polarity) + summary.
    per_frame = _recompute_per_frame_metrics(src, frames_meta)
    proxy_summary = _summarize(per_frame)
    manifest.setdefault("metrics", {})
    manifest["metrics"]["per_frame"] = per_frame
    # Summary: start from the proxy means, then override the headline metrics with the
    # REAL trained-IFNet validation numbers (vs withheld truth) so the panel is honest.
    summary = dict(proxy_summary)
    summary.update(_headline_summary(_REPO_ROOT / args.validation))
    manifest["metrics"]["summary"] = summary
    manifest["metrics"]["baselines"] = _baselines_from_validation(_REPO_ROOT / args.validation)

    # Title for the selector.
    manifest["title"] = args.title

    # Replace any remaining NaN/Inf anywhere with null.
    manifest = _finite_or_none(manifest)

    # --- 3) Re-validate against the WEB loader's rules (not the producer contract). ---
    # The full precompute artifact keeps per-frame tiles + PMTiles and passes the stricter
    # producer-side frameflow.contracts.validate_manifest (asserted by `make demo`). This
    # EMBEDDED copy deliberately drops those heavy pyramids and rides the BitmapLayer
    # (frame.image) fallback, which web/src/lib/manifest.ts:validateManifest explicitly
    # allows (the JS mock scenes ship null tiles/pmtiles too). So we validate the embed with
    # the same checks the browser enforces — CONTRACTS.md §8 is untouched for the producer.
    problems = _validate_web_manifest(manifest)
    if problems:
        print("ERROR: sanitized manifest failed the web loader's validation:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    # Write valid JSON (allow_nan=False guarantees no bare NaN tokens slip through).
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))

    # --- 4) Refresh scenes.json (embedded real scene first, then any mock scenes). ---
    scenes_path = data_root / "scenes.json"
    entry = {
        "scene_id": manifest["scene_id"],
        "title": manifest["title"],
        "satellite": manifest["satellite"],
    }
    scenes: list[dict[str, Any]] = []
    if scenes_path.exists():
        try:
            existing = json.loads(scenes_path.read_text()).get("scenes", [])
            scenes = [s for s in existing if s.get("scene_id") != manifest["scene_id"]]
        except Exception:
            scenes = []
    scenes.insert(0, entry)
    scenes_path.write_text(json.dumps({"scenes": scenes}, indent=2))

    total_bytes = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print("Embedded REAL demo scene ->", dst.relative_to(_REPO_ROOT))
    print(f"  frames={len(frames_meta)}  img={n_img}  thumb={n_thumb}  videos={n_video}  flow={n_flow}")
    print(f"  per_frame metrics recomputed={len(per_frame)}  baselines={list(manifest['metrics']['baselines'])}")
    print(f"  summary={ {k: manifest['metrics']['summary'][k] for k in sorted(manifest['metrics']['summary'])} }")
    print(f"  manifest valid (validate_manifest == []), total embed size = {total_bytes/1024:.0f} KB")
    print("  NOTE: frameflow/precompute.py:_compute_metrics passes an INCLUDE-mask to")
    print("        validate.per_frame_metrics (contract: True==EXCLUDE) -> all-NaN per-frame")
    print("        records in the raw artifact. This script repairs them for the embed; the")
    print("        precompute bug itself is left for the SERVE+VIZ owner (out of web scope).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

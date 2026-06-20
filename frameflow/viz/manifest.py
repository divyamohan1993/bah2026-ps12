"""Assemble + validate the web dashboard manifest (SERVE+VIZ, CONTRACTS.md §8).

This module builds a :class:`frameflow.contracts.Manifest` from per-frame metadata and the
artifact paths produced by the rest of :mod:`frameflow.viz` / :mod:`frameflow.precompute`,
runs :func:`frameflow.contracts.validate_manifest` on it, and writes ``manifest.json``.

Both code-review corrections are enforced here:
    * P1 — ``metrics.data_range_k`` / ``value_range_k`` carry the FIXED physical Kelvin range
      (:data:`frameflow.constants.BT_DATA_RANGE_K` and the ``[VMIN, VMAX]`` pair), never
      per-image min/max.
    * P2 — every :class:`frameflow.contracts.FrameEntry` carries its OWN
      ``tiles_url_template`` (with ``{z}/{x}/{y}``) and a non-empty ``pmtiles`` path, so the
      slider can switch the active tile source per timestamp.

``build_manifest`` raises :class:`ValueError` if the assembled manifest fails validation, so
a malformed manifest can never be written.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import constants as C
from ..contracts import (
    CrossvalMethodResult,
    FrameEntry,
    Manifest,
    MetricRecord,
    validate_manifest,
)


def _now_iso_z() -> str:
    """Current UTC time as an ISO-8601 'Z' timestamp (e.g. ``2026-06-20T00:00:00Z``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frame_entry_from_meta(meta: Mapping[str, Any], frame_index: int) -> FrameEntry:
    """Build a :class:`FrameEntry` from a loose per-frame metadata mapping.

    Required keys: ``time`` (ISO-8601 Z), ``kind`` ("observed"/"interpolated"),
    ``image``, ``thumb``, ``tiles_url_template``, ``pmtiles``, ``netcdf``. Optional:
    ``index`` (defaults to position), ``t``, ``bracket``, ``flow_overlay``.
    """
    idx = int(meta.get("index", frame_index))
    return FrameEntry(
        index=idx,
        time=str(meta["time"]),
        kind=meta["kind"],  # validated by validate_manifest
        image=str(meta["image"]),
        thumb=str(meta["thumb"]),
        tiles_url_template=str(meta["tiles_url_template"]),
        pmtiles=str(meta["pmtiles"]),
        netcdf=str(meta["netcdf"]),
        t=(None if meta.get("t") is None else float(meta["t"])),
        bracket=(None if meta.get("bracket") is None else [int(b) for b in meta["bracket"]]),
        flow_overlay=meta.get("flow_overlay"),
    )


def _coerce_metric_records(per_frame: Sequence[Any] | None) -> list[MetricRecord]:
    """Coerce a per-frame metrics list (dicts or MetricRecords) to MetricRecords."""
    out: list[MetricRecord] = []
    for i, m in enumerate(per_frame or []):
        if isinstance(m, MetricRecord):
            out.append(m)
        elif isinstance(m, Mapping):
            d = dict(m)
            d.setdefault("index", i)
            out.append(MetricRecord.from_dict(d))
        else:  # pragma: no cover - defensive
            raise TypeError(f"per_frame metric must be dict or MetricRecord, got {type(m)!r}")
    return out


def _coerce_crossval(methods: Sequence[Any] | None) -> list[CrossvalMethodResult]:
    """Coerce a crossval methods list (dicts or CrossvalMethodResult) to records."""
    out: list[CrossvalMethodResult] = []
    for m in methods or []:
        if isinstance(m, CrossvalMethodResult):
            out.append(m)
        elif isinstance(m, Mapping):
            out.append(CrossvalMethodResult.from_dict(dict(m)))
        else:  # pragma: no cover - defensive
            raise TypeError(f"crossval method must be dict or CrossvalMethodResult, got {type(m)!r}")
    return out


def build_manifest(
    scene_id: str,
    bbox: Sequence[float],
    frames_meta: Sequence[Mapping[str, Any]],
    *,
    title: str = "FrameFlow scene",
    satellite: str = C.DEFAULT_SATELLITE,
    channel: str = "C13",
    wavelength_um: float | None = None,
    colormap: str = C.DEFAULT_COLORMAP,
    value_range_k: Sequence[float] | None = None,
    tile_size: int = C.DEFAULT_TILE_SIZE,
    min_zoom: int = C.DEFAULT_MIN_ZOOM,
    max_zoom: int = C.DEFAULT_MAX_ZOOM,
    interpolation_factor: int = 2,
    cadence_minutes_input: float = float(C.DEFAULT_INPUT_CADENCE_MIN),
    cadence_minutes_output: float | None = None,
    crs: str = C.DEFAULT_GRID_CRS,
    metrics: Mapping[str, Any] | None = None,
    crossval: Sequence[Any] | Mapping[str, Any] | None = None,
    videos: Mapping[str, str | None] | None = None,
    model_info: Mapping[str, Any] | None = None,
    generated: str | None = None,
) -> Manifest:
    """Assemble a validated :class:`frameflow.contracts.Manifest`.

    Args:
        scene_id: unique scene identifier (e.g. ``"demo-0001"``).
        bbox: ``[west, south, east, north]`` geographic extent.
        frames_meta: ordered per-frame metadata mappings (see
            :func:`_frame_entry_from_meta` for required/optional keys). Each MUST carry its
            own ``tiles_url_template`` and ``pmtiles`` (P2).
        title, satellite, channel, wavelength_um: scene/source descriptors. ``wavelength_um``
            defaults to the satellite's clean-window band from
            :data:`frameflow.constants.SATELLITE_BANDS`.
        colormap: display colormap name.
        value_range_k: the FIXED ``[vmin_k, vmax_k]`` display/metric range; defaults to the
            shared metric range (P1).
        tile_size, min_zoom, max_zoom: tile pyramid parameters.
        interpolation_factor: temporal up-sampling factor (2 -> 30->15 min).
        cadence_minutes_input: native input cadence (minutes).
        cadence_minutes_output: output cadence; defaults to input / ``interpolation_factor``.
        crs: coordinate reference system.
        metrics: optional metrics block; ``data_range_k``/``value_range_k`` are forced to the
            fixed range, ``per_frame`` records are coerced to :class:`MetricRecord` (P1).
        crossval: a list of cross-val method results OR a full ``{"methods_run": [...]}``
            mapping.
        videos: ``{observed, interpolated, side_by_side}`` paths (missing keys -> ``None``).
        model_info: ``{name, version, params_m}`` describing the model.
        generated: build timestamp (ISO-8601 Z); defaults to now.

    Returns:
        A validated :class:`frameflow.contracts.Manifest`.

    Raises:
        ValueError: if the assembled manifest fails ``validate_manifest``.
    """
    if wavelength_um is None:
        band = C.SATELLITE_BANDS.get(satellite)
        wavelength_um = float(band["wavelength_um"]) if band else 10.8

    if value_range_k is None:
        value_range_k = [C.BT_METRIC_VMIN_K, C.BT_METRIC_VMAX_K]
    value_range_k = [float(value_range_k[0]), float(value_range_k[1])]

    if cadence_minutes_output is None:
        factor = interpolation_factor if interpolation_factor else 1
        cadence_minutes_output = float(cadence_minutes_input) / float(factor)

    frames = [_frame_entry_from_meta(m, i) for i, m in enumerate(frames_meta)]

    # --- metrics block (P1): force the fixed Kelvin range, coerce per-frame records. ------
    data_range_k = float(value_range_k[1] - value_range_k[0])
    metrics_block: dict[str, Any] = {
        "data_range_k": data_range_k,
        "value_range_k": list(value_range_k),
        "per_frame": [],
        "summary": {},
        "baselines": {},
    }
    if metrics:
        if "summary" in metrics:
            metrics_block["summary"] = dict(metrics["summary"])
        if "baselines" in metrics:
            metrics_block["baselines"] = dict(metrics["baselines"])
        metrics_block["per_frame"] = _coerce_metric_records(metrics.get("per_frame"))
        # Allow callers to pass extra scalar keys through (but never override the fixed range).
        for k, v in metrics.items():
            if k not in ("data_range_k", "value_range_k", "per_frame", "summary", "baselines"):
                metrics_block[k] = v

    # --- crossval block -------------------------------------------------------------------
    if isinstance(crossval, Mapping):
        methods = crossval.get("methods_run")
    else:
        methods = crossval
    crossval_block = {"methods_run": _coerce_crossval(methods)}

    # --- videos block (always carry the three keys) ---------------------------------------
    videos_block: dict[str, str | None] = {"observed": None, "interpolated": None, "side_by_side": None}
    if videos:
        for k in videos_block:
            if k in videos:
                videos_block[k] = videos[k]

    model_info = dict(model_info or {})

    manifest = Manifest(
        scene_id=str(scene_id),
        title=str(title),
        satellite=str(satellite),
        channel=str(channel),
        wavelength_um=float(wavelength_um),
        bbox=[float(b) for b in bbox],
        colormap=str(colormap),
        value_range_k=list(value_range_k),
        tile_size=int(tile_size),
        min_zoom=int(min_zoom),
        max_zoom=int(max_zoom),
        interpolation_factor=int(interpolation_factor),
        cadence_minutes_input=float(cadence_minutes_input),
        cadence_minutes_output=float(cadence_minutes_output),
        frames=frames,
        generated=generated or _now_iso_z(),
        model_name=str(model_info.get("name", "")),
        model_version=str(model_info.get("version", "")),
        model_params_m=float(model_info.get("params_m", 0.0)),
        crs=str(crs),
        videos=videos_block,
        metrics=metrics_block,
        crossval=crossval_block,
    )

    problems = validate_manifest(manifest.to_dict())
    if problems:
        raise ValueError(
            "assembled manifest failed validation:\n  - " + "\n  - ".join(problems)
        )
    return manifest


def write_manifest(
    manifest: Manifest,
    out_dir: str | Path,
    filename: str = "manifest.json",
    indent: int = 2,
) -> Path:
    """Serialize a validated manifest to ``{out_dir}/{filename}`` and return its path.

    Re-validates before writing as a final guard (so a hand-mutated manifest cannot be
    persisted in a broken state).

    Raises:
        ValueError: if the manifest fails ``validate_manifest`` at write time.
    """
    d = manifest.to_dict()
    problems = validate_manifest(d)
    if problems:
        raise ValueError(
            "refusing to write invalid manifest:\n  - " + "\n  - ".join(problems)
        )
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / filename
    out.write_text(json.dumps(d, indent=indent))
    return out


__all__ = ["build_manifest", "write_manifest"]

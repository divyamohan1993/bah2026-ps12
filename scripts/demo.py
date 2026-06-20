#!/usr/bin/env python3
"""FrameFlow end-to-end demo — the ``make demo`` target (the real integration proof).

Now that every module area (data, models, train, infer, validate, serve/viz, precompute)
is implemented, this script runs the WHOLE pipeline on synthetic data, end to end:

    synthetic Zarr cube (frameflow.synthetic)
        -> TripletDataset  (frameflow.data.datasets.TripletDataset)
        -> briefly TRAIN the real IFNet via the Lightning VFIModule
           (frameflow.models.get_model('ifnet') + frameflow.train.run_training) -> checkpoint
        -> recursive interpolation (frameflow.infer.interpolate_recursive, factor=2)
           writing per-frame .nc outputs
        -> validation metrics (frameflow.validate.compute_metrics) comparing the trained
           model vs the linear-blend and TV-L1 baselines on the WITHHELD true middle frames
           (PSNR / SSIM / MS-SSIM / FSIM / bt_rmse_k, fixed-K data_range, P1)
        -> a subset of the 40-method CrossValSuite (frameflow.validate.crossval)
        -> web artifacts (frameflow.precompute.precompute_scene) -> artifacts/<scene>/
           (manifest.json + per-frame webp + tiles + all-intra videos + flow overlay + .nc),
           asserting the manifest passes contracts.validate_manifest.

Everything runs on CPU in a few minutes (small grid, 64-px patches, a few hundred steps).
The point is that the integration RUNS and produces real numbers — not that a model trained
for two minutes on tiny synthetic data beats a linear blend.

Usage:
    python scripts/demo.py
    python scripts/demo.py --steps 200 --grid 96 --frames 9 --patch 64
    python scripts/demo.py --quick           # fewer steps / smaller, for a fast smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Make the repo root importable when run as `python scripts/demo.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Lightning / torch emit a lot of benign warnings on a tiny CPU run; keep the log readable.
warnings.filterwarnings("ignore")

_T0 = time.time()


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] ({time.time() - _T0:5.1f}s) {title}")


# ---------------------------------------------------------------------------
# Stage 1 — synthetic Zarr cube
# ---------------------------------------------------------------------------
def stage_synth(out_root: Path, grid_px: int, n_frames: int, cadence_min: int) -> Path:
    """Generate a schema-correct synthetic moving-cloud Zarr cube."""
    from frameflow import synthetic
    from frameflow.contracts import GridSpec

    grid = GridSpec(
        west=68.0, south=6.0, east=98.0, north=38.0,
        n_rows=grid_px, n_cols=grid_px, resolution_deg=(98.0 - 68.0) / grid_px,
    )
    cube = synthetic.generate_cube(
        n_frames=n_frames,
        grid=grid,
        out_path=str(out_root / "cubes" / "synthetic.zarr"),
        cadence_min=cadence_min,
        seed=7,
    )
    print(f"      cube  -> {cube}  (grid {grid_px}x{grid_px}, {n_frames} frames @ {cadence_min} min)")
    return cube


# ---------------------------------------------------------------------------
# Stage 2 — TripletDataset
# ---------------------------------------------------------------------------
def stage_dataset(cube: Path, patch: int) -> Any:
    """Build a TripletDataset over the cube and sanity-check one Sample."""
    import numpy as np

    from frameflow.data.datasets.triplet import TripletDataset

    ds = TripletDataset(str(cube), split="all", patch_size=patch, normalized=True)
    sample = ds[0]
    i0 = np.asarray(sample["I0"])
    print(f"      TripletDataset: {len(ds)} triplets; Sample I0 {i0.shape} t={sample['t']} "
          f"keys={sorted(sample.keys())}")
    return ds


# ---------------------------------------------------------------------------
# Stage 3 — brief real training of IFNet via the Lightning VFIModule
# ---------------------------------------------------------------------------
def stage_train(cube: Path, out_root: Path, patch: int, steps: int) -> tuple[Any, Path]:
    """Briefly train the real IFNet (small CPU run) and save a checkpoint.

    Returns ``(trained_model, ckpt_path)``. The model is returned in-memory so inference and
    precompute use the *exact* trained weights without a checkpoint round-trip.
    """
    import torch

    import frameflow.config as cfgmod
    from frameflow.models.registry import get_model
    from frameflow.train.cli import run_training
    from frameflow.train.datamodule import VFIDataModule

    # A small IFNet (2 scales, narrow width) trains fast on CPU yet exercises the real
    # flow-estimation / warping / fusion path (not the tiny blend fallback).
    model = get_model("ifnet", in_channels=1, hidden=24, scales=(2, 1))
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    dm = VFIDataModule(
        cube_path=str(cube),
        patch_size=patch,
        num_workers=0,
        batch_size=4,
        train_frac=0.7,
        val_frac=0.2,
    )
    cfg = cfgmod.FrameFlowConfig()
    # Let max_steps (not max_epochs) bound the run: on a tiny cube an epoch is only a couple
    # of batches, so a high epoch cap ensures we actually take `steps` optimizer steps.
    # (build_trainer reads max_epochs from cfg.train.epochs.) Disable the experiment tracker.
    cfg.train.epochs = 10_000
    cfg.train.tracker = "none"

    print(f"      training IFNet ({n_params_m:.3f}M params) for {steps} steps "
          f"on {patch}px patches (CPU) ...")
    ckpt = run_training(
        cfg,
        model=model,
        datamodule=dm,
        accelerator="cpu",
        devices=1,
        trainer_kwargs=dict(
            max_steps=steps,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=2,
        ),
    )
    print(f"      training returned -> {ckpt}")

    # Persist a small standalone checkpoint (state-dict) for reproducibility / inference.
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "demo_ifnet.pt"
    torch.save(
        {"model": model.state_dict(),
         "arch": {"name": "ifnet", "in_channels": 1, "hidden": 24, "scales": [2, 1]},
         "params_m": n_params_m},
        ckpt_path,
    )
    size_kb = ckpt_path.stat().st_size / 1024.0
    print(f"      saved checkpoint -> {ckpt_path}  ({size_kb:.0f} KB)")
    model.eval()
    return model, ckpt_path


# ---------------------------------------------------------------------------
# Stage 4 — recursive interpolation (writes .nc) on a held-out split
# ---------------------------------------------------------------------------
def stage_interpolate(cube: Path, model: Any, out_root: Path) -> dict[str, Any]:
    """Leave-middle-out densification: observed = even frames, withheld truth = odd frames.

    Runs ``interpolate_recursive(factor=2)`` to synthesize the midpoints, writes each
    interpolated frame to a schema-correct ``.nc`` (via the INFER netcdf writer), and returns
    the arrays needed for validation (predicted middles + the withheld true middles).
    """
    import numpy as np
    import xarray as xr

    from frameflow.contracts import netcdf_attrs
    from frameflow.infer.interpolate import interpolate_recursive
    from frameflow.infer.netcdf_io import write_netcdf

    ds = xr.open_zarr(str(cube), consolidated=False)
    bt = np.asarray(ds["bt"].values, dtype=np.float32)  # (T, H, W) Kelvin
    times = list(ds["time"].values)
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    bbox = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))

    n = bt.shape[0]
    obs_idx = list(range(0, n, 2))
    truth_idx = list(range(1, n, 2))
    observed = [bt[i] for i in obs_idx]
    obs_times = [times[i] for i in obs_idx]
    truth_mid = [bt[i] for i in truth_idx]  # withheld ground-truth middles

    frames = interpolate_recursive(model, observed, factor=2, times=obs_times, device="cpu")
    interp = [f for f in frames if f.kind == "interpolated"]

    # interp[k] is the midpoint of observed[k]..observed[k+1] == original frame truth_idx[k].
    n_pairs = min(len(interp), len(truth_mid))
    pred_mid = [interp[k].bt for k in range(n_pairs)]
    true_mid = [truth_mid[k] for k in range(n_pairs)]

    # Write per-frame .nc for the whole densified sequence (observed + interpolated).
    nc_dir = out_root / "interp_nc"
    nc_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, fr in enumerate(frames):
        when = fr.time if fr.time is not None else times[min(idx, len(times) - 1)]
        attrs = netcdf_attrs(
            source_frames=[f"obs_{b}" for b in (fr.bracket or [])] or ["synthetic"],
            t=(fr.t_local if fr.t_local is not None else 0.0),
            model="ifnet",
            model_version="demo-v0.1.0",
            kind="observed" if fr.kind == "observed" else "interpolated",
            interpolation_factor=2,
            extra={"crs": "EPSG:4326", "bbox_west_south_east_north": list(bbox)},
        )
        write_netcdf(
            bt=np.asarray(fr.bt, dtype=np.float32),
            lat=lat,
            lon=lon,
            time=when,
            attrs=attrs,
            path=nc_dir / f"{idx:03d}.nc",
        )
        written += 1
    print(f"      densified {len(observed)} observed -> {len(frames)} frames "
          f"({len(interp)} interpolated); wrote {written} .nc -> {nc_dir}")
    print(f"      held-out middles for validation: {n_pairs} "
          f"(observed idx {obs_idx} | withheld truth idx {truth_idx[:n_pairs]})")

    return {
        "pred_mid": pred_mid,
        "true_mid": true_mid,
        "observed": observed,
        "bbox": bbox,
        "nc_dir": nc_dir,
    }


# ---------------------------------------------------------------------------
# Stage 5 — validation: trained IFNet vs linear vs TV-L1 (fixed-K data_range, P1)
# ---------------------------------------------------------------------------
def stage_validate(model: Any, interp: dict[str, Any], out_root: Path) -> dict[str, Any]:
    """Compute the headline metric set for the trained model and the two baselines."""
    import numpy as np

    from frameflow.models.registry import get_model
    from frameflow.validate.metrics import compute_metrics

    pred_mid = interp["pred_mid"]
    true_mid = interp["true_mid"]
    observed = interp["observed"]

    methods: dict[str, list[np.ndarray]] = {}
    # Trained IFNet predictions were already produced in stage 4.
    methods["trained_ifnet"] = pred_mid

    # Baselines synthesize the SAME middles directly from the bracketing observed pair.
    for name in ("linear", "tvl1"):
        baseline = get_model(name)
        import torch
        preds: list[np.ndarray] = []
        for k in range(len(true_mid)):
            i0 = torch.from_numpy(np.asarray(observed[k], dtype=np.float32))[None, None]
            i1 = torch.from_numpy(np.asarray(observed[k + 1], dtype=np.float32))[None, None]
            tt = torch.full((1, 1), 0.5, dtype=torch.float32)
            out = baseline(i0, i1, tt)["pred"]
            preds.append(np.asarray(out.squeeze().numpy(), dtype=np.float32))
        methods[name] = preds

    # Per-frame metrics (fixed-K data_range, P1) -> per-method means.
    keys = ("psnr", "ssim", "ms_ssim", "fsim", "bt_rmse_k")
    table: dict[str, dict[str, float]] = {}
    for name, preds in methods.items():
        per = [compute_metrics(p, g) for p, g in zip(preds, true_mid, strict=False)]
        table[name] = {
            k: float(np.nanmean([m[k] for m in per])) if per else float("nan") for k in keys
        }

    out_dir = out_root / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(table, indent=2))
    print(f"      wrote per-method metrics -> {out_dir / 'metrics.json'}")
    _print_metrics_table(table, keys)
    return {"table": table, "keys": keys}


def _print_metrics_table(table: dict[str, dict[str, float]], keys: tuple[str, ...]) -> None:
    """Pretty-print the trained-vs-baselines metric table to stdout."""
    name_w = max(len("method"), *(len(n) for n in table))
    header = "method".ljust(name_w) + "".join(f"  {k:>10}" for k in keys)
    print("\n      " + header)
    print("      " + "-" * len(header))
    for name, row in table.items():
        line = name.ljust(name_w) + "".join(f"  {row[k]:10.4f}" for k in keys)
        print("      " + line)


# ---------------------------------------------------------------------------
# Stage 6 — a subset of the 40-method CrossValSuite
# ---------------------------------------------------------------------------
def stage_crossval(cube: Path, model: Any) -> dict[str, Any]:
    """Run a subset of the cross-validation suite against the trained model."""
    from frameflow.validate.crossval import CrossValSuite

    suite = CrossValSuite(model=model, dataset="synthetic", seed=0)
    results = suite.run({"synthetic_cube": str(cube)})
    ran = [r for r in results if r.status == "run"]
    print(f"      CrossValSuite: {len(results)} methods registered, {len(ran)} executed.")
    for r in ran[:6]:
        summ = ", ".join(f"{k}={v:.3f}" for k, v in list(r.summary.items())[:3]
                         if isinstance(v, (int, float)))
        print(f"        {r.method_id:>4}  {r.name[:38]:38}  {summ}")
    return {"n_methods": len(results), "n_run": len(ran),
            "ran_ids": [r.method_id for r in ran]}


# ---------------------------------------------------------------------------
# Stage 7 — web artifacts (precompute) + manifest validation
# ---------------------------------------------------------------------------
def stage_precompute(cube: Path, model: Any, out_root: Path) -> dict[str, Any]:
    """Produce artifacts/<scene>/ and assert the manifest passes contracts.validate_manifest."""
    from frameflow.contracts import validate_manifest
    from frameflow.precompute import precompute_scene

    artifacts_root = out_root / "artifacts"
    res = precompute_scene(
        str(cube),
        model=model,
        out_dir=str(artifacts_root / "demo-0001"),
        factor=2,
        scene_id="demo-0001",
        model_name="RIFE-IFNet",
        model_version="demo-v0.1.0",
        make_video=True,
        make_pmtiles=True,
        title="FrameFlow synthetic demo scene",
    )
    manifest = json.loads(Path(res.manifest_path).read_text())
    problems = validate_manifest(manifest)
    if problems:
        raise AssertionError(f"validate_manifest reported problems: {problems}")

    n_flow = sum(1 for f in manifest["frames"] if f.get("flow_overlay"))
    videos = {k: v for k, v in manifest["videos"].items() if v}
    print(f"      artifacts -> {res.out_dir}")
    print(f"      manifest  -> {res.manifest_path}  (validate_manifest == [] OK)")
    print(f"      frames={res.n_frames} (observed={res.n_observed}, "
          f"interpolated={res.n_interpolated}); flow_overlays={n_flow}; videos={list(videos)}")
    return {
        "out_dir": str(res.out_dir),
        "manifest_path": str(res.manifest_path),
        "n_frames": res.n_frames,
        "n_observed": res.n_observed,
        "n_interpolated": res.n_interpolated,
        "n_flow": n_flow,
        "videos": list(videos),
        "manifest_valid": True,
    }


# ---------------------------------------------------------------------------
# Results summary (RESULTS.md)
# ---------------------------------------------------------------------------
def write_results_md(
    val: dict[str, Any],
    crossval: dict[str, Any],
    precompute: dict[str, Any],
    n_held_out: int,
) -> Path:
    """Write a small RESULTS.md with the metric table + how to reproduce."""
    keys = val["keys"]
    table = val["table"]
    lines: list[str] = []
    lines.append("# FrameFlow — end-to-end demo results\n")
    lines.append("Auto-generated by `make demo` (`scripts/demo.py`). Reproduce with:\n")
    lines.append("```bash\nmake demo\n```\n")
    lines.append("The demo runs the full pipeline on synthetic data: synthetic Zarr cube → "
                 "`TripletDataset` → brief real **IFNet** training (Lightning `VFIModule`, CPU) "
                 "→ recursive interpolation (`factor=2`, writes `.nc`) → validation "
                 "(`compute_metrics`, fixed-K `data_range`, P1) vs the linear-blend and TV-L1 "
                 "baselines on **withheld** true middle frames → a subset of the 40-method "
                 "`CrossValSuite` → `precompute_scene` web artifacts (manifest + per-frame "
                 "webp + tiles + videos + flow overlay + `.nc`), asserting "
                 "`contracts.validate_manifest == []`.\n")

    lines.append(f"## Metrics — trained IFNet vs baselines ({n_held_out} held-out middle frames)\n")
    lines.append("Means over the withheld true intermediate frames. All full-reference "
                 "metrics use the FIXED Kelvin `data_range` (P1); `bt_rmse_k` is the headline "
                 "radiometric error in Kelvin. PSNR/SSIM/MS-SSIM/FSIM: higher is better; "
                 "`bt_rmse_k`: lower is better.\n")
    header = "| method | " + " | ".join(keys) + " |"
    sep = "|" + "---|" * (len(keys) + 1)
    lines.append(header)
    lines.append(sep)
    for name, row in table.items():
        lines.append("| " + name + " | " + " | ".join(f"{row[k]:.4f}" for k in keys) + " |")
    lines.append("")
    lines.append("> Note: the model is trained only briefly on tiny synthetic data (a few "
                 "hundred CPU steps); the primary purpose of `make demo` is to prove the "
                 "modules wire together and produce real numbers end to end. Exact values "
                 "vary run to run with the random init / step budget — read the table emitted "
                 "by your own run. On real multi-day satellite data with full training the "
                 "learned model is the intended quality win.\n")

    lines.append("## Cross-validation\n")
    lines.append(f"- `CrossValSuite`: **{crossval['n_run']}** of {crossval['n_methods']} "
                 f"methods executed on the synthetic cube (the rest require real "
                 f"multi-satellite data and are registered as skipped).\n")

    lines.append("## Web artifacts\n")
    lines.append(f"- `precompute_scene` wrote `{precompute['out_dir']}/` with "
                 f"**{precompute['n_frames']}** frames "
                 f"({precompute['n_observed']} observed + {precompute['n_interpolated']} "
                 f"interpolated), per-frame WebP + XYZ tiles + `.nc`, "
                 f"**{precompute['n_flow']}** flow overlays, and videos "
                 f"{precompute['videos']}.")
    lines.append(f"- Manifest `{precompute['manifest_path']}` passes "
                 f"`contracts.validate_manifest` (== `[]`).\n")
    lines.append("_(Generated artifacts under `out/` and `artifacts/` are git-ignored.)_\n")

    path = _REPO_ROOT / "RESULTS.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FrameFlow end-to-end demo.")
    parser.add_argument("--out", default="out", help="Root output directory (git-ignored).")
    parser.add_argument("--grid", type=int, default=96, help="Synthetic grid size (pixels).")
    parser.add_argument("--frames", type=int, default=9, help="Number of cube frames.")
    parser.add_argument("--cadence", type=int, default=30, help="Minutes between frames.")
    parser.add_argument("--patch", type=int, default=64, help="Training patch size (pixels).")
    parser.add_argument("--steps", type=int, default=300, help="Training steps (CPU).")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke run (fewer steps, smaller grid).")
    args = parser.parse_args(argv)

    if args.quick:
        args.steps = min(args.steps, 60)
        args.grid = min(args.grid, 64)
        args.frames = min(args.frames, 7)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    _hr("FrameFlow end-to-end demo (data → models → train → infer → validate → viz)")
    total = 7

    _step(1, total, "Generate synthetic Zarr cube ...")
    cube = stage_synth(out_root, args.grid, args.frames, args.cadence)

    _step(2, total, "Build TripletDataset ...")
    stage_dataset(cube, args.patch)

    _step(3, total, "Train the real IFNet (brief, CPU) ...")
    model, _ckpt = stage_train(cube, out_root, args.patch, args.steps)

    _step(4, total, "Recursive interpolation (factor=2) + write .nc ...")
    interp = stage_interpolate(cube, model, out_root)

    _step(5, total, "Validate: trained IFNet vs linear vs TV-L1 (fixed-K, P1) ...")
    val = stage_validate(model, interp, out_root)

    _step(6, total, "Cross-validation suite (subset) ...")
    crossval = stage_crossval(cube, model)

    _step(7, total, "Precompute web artifacts + validate manifest ...")
    precompute = stage_precompute(cube, model, out_root)

    results_md = write_results_md(val, crossval, precompute, n_held_out=len(interp["true_mid"]))

    _hr(f"Demo complete in {time.time() - _T0:.1f}s — all stages ran end to end.")
    print(f"  metrics table + reproduce steps -> {results_md}")
    print(f"  web artifacts (git-ignored)     -> {precompute['out_dir']}")
    print(f"  manifest (validate_manifest==[]) -> {precompute['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

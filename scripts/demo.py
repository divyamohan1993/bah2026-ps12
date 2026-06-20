#!/usr/bin/env python3
"""FrameFlow end-to-end demo script — the ``make demo`` target.

Runs the full chain with clear stdout logging and graceful skips when a stage's
dependencies or (not-yet-implemented) team modules are missing:

    synthetic data  ->  (tiny) train  ->  interpolate  ->  validate  ->  precompute web

All imports are LAZY (inside functions) so this script runs even before the six build
teams land their code. Only the synthetic-data stage is guaranteed end-to-end today; every
other stage prints a helpful "skipped" note if its module/deps are absent.

Usage:
    python scripts/demo.py
    python scripts/demo.py --no-skip-train      # also attempt the tiny training stage
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Make the repo root importable when run as `python scripts/demo.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}")


def _skip(module: str, team: str, exc: Exception) -> None:
    print(f"      SKIPPED: '{module}' not available yet (Team {team}).")
    print(f"               reason: {exc}")
    print("               see CONTRACTS.md for the interface it must implement.")


def stage_synth() -> tuple[Path, list[Path]]:
    """Generate the synthetic cube + demo .nc triplet (always works)."""
    from frameflow import synthetic  # lazy

    cube = synthetic.generate_cube()
    nc_paths = synthetic.generate_pair_nc()
    print(f"      cube  -> {cube}")
    for p in nc_paths:
        print(f"      .nc   -> {p}")
    return cube, nc_paths


def stage_train(skip: bool) -> None:
    """Tiny training run (optional)."""
    if skip:
        print("      skipped (--skip-train).")
        return
    try:
        from frameflow.train import trainer  # lazy; Team TRAIN
    except ImportError as exc:
        _skip("frameflow.train.trainer", "TRAIN", exc)
        return
    trainer.run(config="configs/config.yaml", overrides=["train.epochs=1"])  # type: ignore[attr-defined]


def stage_interpolate(nc_paths: list[Path], out_root: Path) -> Path | None:
    """Interpolate the middle frame of the demo triplet."""
    out_nc = out_root / "interp_nc" / "mid.nc"
    try:
        from frameflow.infer import interpolate as inf  # lazy; Team INFER+VALIDATE
    except ImportError as exc:
        _skip("frameflow.infer.interpolate", "INFER+VALIDATE", exc)
        return None
    observed = [p for p in nc_paths if "observed" in str(p)]
    inf.interpolate_pair(str(observed[0]), str(observed[1]), t=0.5, out_nc=str(out_nc))  # type: ignore[attr-defined]
    print(f"      wrote -> {out_nc}")
    return out_nc


def stage_validate(out_root: Path) -> None:
    """Validate interpolated frames vs withheld ground truth (fixed-K data_range; P1)."""
    try:
        from frameflow.validate import runner  # lazy; Team INFER+VALIDATE
    except ImportError as exc:
        _skip("frameflow.validate.runner", "INFER+VALIDATE", exc)
        return
    runner.run(  # type: ignore[attr-defined]
        pred_dir=str(out_root / "interp_nc"),
        truth_dir="data/demo_nc",
        out_dir=str(out_root / "validation"),
    )


def stage_precompute(cube: Path, out_root: Path) -> None:
    """Precompute per-frame tiles + PMTiles + video + manifest (P2)."""
    try:
        from frameflow import precompute as pc  # lazy; Team SERVE+VIZ
    except ImportError as exc:
        _skip("frameflow.precompute", "SERVE+VIZ", exc)
        return
    pc.build_web_artifacts(cube_path=str(cube), out_dir=str(out_root / "web"))  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FrameFlow end-to-end demo.")
    parser.add_argument("--out", default="out", help="Root output directory.")
    parser.add_argument(
        "--skip-train",
        dest="skip_train",
        action="store_true",
        default=True,
        help="Skip the tiny training stage (default: skip).",
    )
    parser.add_argument(
        "--no-skip-train",
        dest="skip_train",
        action="store_false",
        help="Attempt the tiny training stage.",
    )
    args = parser.parse_args(argv)
    out_root = Path(args.out)

    _hr("FrameFlow end-to-end demo")
    total = 5
    cube: Path | None = None
    nc_paths: list[Path] = []

    _step(1, total, "Generating synthetic data ...")
    try:
        cube, nc_paths = stage_synth()
    except Exception:  # the one stage that must work — surface failures loudly
        traceback.print_exc()
        print("\nFATAL: synthetic data generation failed (this stage must work). "
              "Ensure numpy/xarray/zarr are installed.")
        return 1

    _step(2, total, "Train (tiny) ...")
    stage_train(args.skip_train)

    _step(3, total, "Interpolate ...")
    if nc_paths:
        stage_interpolate(nc_paths, out_root)

    _step(4, total, "Validate ...")
    stage_validate(out_root)

    _step(5, total, "Precompute web artifacts ...")
    if cube is not None:
        stage_precompute(cube, out_root)

    _hr("Demo chain complete (stages with missing modules were skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

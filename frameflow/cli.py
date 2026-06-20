"""FrameFlow command-line interface (Typer).

Subcommands: ``synth``, ``ingest``, ``train``, ``interpolate``, ``validate``,
``precompute``, ``serve``, ``demo``. Each command uses LAZY imports inside its body so the
CLI loads even before the six build teams land their module code; commands that call into a
not-yet-existing subpackage wrap the import in ``try/except ImportError`` and print a
helpful message instead of crashing.

The only command guaranteed to work end-to-end today is ``synth`` (it calls
:mod:`frameflow.synthetic`, which is fully implemented). ``demo`` runs the full chain,
gracefully skipping stages whose dependencies/modules are not yet present.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="FrameFlow — satellite frame interpolation (ISRO BAH 2026 PS-12).",
)


def _missing(module: str, team: str, exc: Exception) -> None:
    """Print a consistent, helpful message when a team's module isn't implemented yet."""
    typer.secho(
        f"[frameflow] '{module}' is not available yet (owned by Team {team}).\n"
        f"            Reason: {exc}\n"
        f"            See CONTRACTS.md for the interface this module must implement.",
        fg=typer.colors.YELLOW,
    )


@app.command()
def synth(
    n_frames: int = typer.Option(24, help="Number of frames in the synthetic cube."),
    out: str = typer.Option("data/cubes/synthetic.zarr", help="Output Zarr cube path."),
    cadence_min: int = typer.Option(30, help="Minutes between frames."),
    seed: int = typer.Option(0, help="RNG seed."),
    nc_dir: str = typer.Option("data/demo_nc", help="Output dir for the demo .nc triplet."),
    pair_only: bool = typer.Option(False, help="Only write the .nc triplet, skip the cube."),
) -> None:
    """Generate the synthetic moving-cloud BT cube and demo .nc triplet."""
    from . import synthetic  # lazy (fully implemented)

    if not pair_only:
        cube = synthetic.generate_cube(n_frames=n_frames, out_path=out, cadence_min=cadence_min, seed=seed)
        typer.secho(f"[frameflow] wrote Zarr cube -> {cube}", fg=typer.colors.GREEN)
    paths = synthetic.generate_pair_nc(out_dir=nc_dir, seed=seed)
    for p in paths:
        typer.secho(f"[frameflow] wrote NetCDF -> {p}", fg=typer.colors.GREEN)


@app.command()
def ingest(
    source: str = typer.Option("goes19", help="Source: goes19 | himawari9 | gk2a | insat3ds."),
    out: str = typer.Option("data/cubes/cube.zarr", help="Output Zarr cube path."),
    config: str = typer.Option("", help="Optional Hydra/YAML config path."),
) -> None:
    """Ingest real satellite data into a schema-correct Zarr cube (Team DATA)."""
    try:
        from .data import ingest as data_ingest  # lazy; owned by Team DATA
    except ImportError as exc:
        _missing("frameflow.data.ingest", "DATA", exc)
        raise typer.Exit(code=0) from None
    data_ingest.build_cube(source=source, out_path=out, config=config)  # type: ignore[attr-defined]


@app.command()
def train(
    config: str = typer.Option("configs/config.yaml", help="Hydra config path."),
    overrides: list[str] = typer.Argument(None, help="Hydra-style overrides, e.g. model=ifrnet."),
) -> None:
    """Train / fine-tune the VFI model (Team TRAIN)."""
    try:
        from .train import trainer  # lazy; owned by Team TRAIN
    except ImportError as exc:
        _missing("frameflow.train.trainer", "TRAIN", exc)
        raise typer.Exit(code=0) from None
    trainer.run(config=config, overrides=list(overrides or []))  # type: ignore[attr-defined]


@app.command()
def interpolate(
    input_nc: list[str] = typer.Argument(..., help="Two consecutive input .nc frames."),
    t: float = typer.Option(0.5, help="Interpolation fraction in (0,1); 0.5 == midpoint."),
    ckpt: str = typer.Option("runs/ckpt/best_ssim.ckpt", help="Model checkpoint."),
    out_nc: str = typer.Option("out/interp_nc/mid.nc", help="Output .nc path."),
) -> None:
    """Interpolate the intermediate frame between two .nc inputs (Team INFER+VALIDATE)."""
    try:
        from .infer import interpolate as inf  # lazy; owned by Team INFER+VALIDATE
    except ImportError as exc:
        _missing("frameflow.infer.interpolate", "INFER+VALIDATE", exc)
        raise typer.Exit(code=0) from None
    if len(input_nc) != 2:
        typer.secho("[frameflow] interpolate expects exactly two input .nc frames.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    inf.interpolate_pair(input_nc[0], input_nc[1], t=t, ckpt=ckpt, out_nc=out_nc)  # type: ignore[attr-defined]


@app.command()
def validate(
    pred_dir: str = typer.Option("out/interp_nc", help="Dir of interpolated .nc frames."),
    truth_dir: str = typer.Option("data/demo_nc", help="Dir of withheld ground-truth .nc frames."),
    out: str = typer.Option("out/validation", help="Output dir for metrics/plots."),
) -> None:
    """Validate interpolated frames vs ground truth (Team INFER+VALIDATE).

    All full-reference metrics use the FIXED physical Kelvin data_range (P1).
    """
    try:
        from .validate import runner  # lazy; owned by Team INFER+VALIDATE
    except ImportError as exc:
        _missing("frameflow.validate.runner", "INFER+VALIDATE", exc)
        raise typer.Exit(code=0) from None
    runner.run(pred_dir=pred_dir, truth_dir=truth_dir, out_dir=out)  # type: ignore[attr-defined]


@app.command()
def precompute(
    cube: str = typer.Option("data/cubes/synthetic.zarr", help="Input cube/frames."),
    out: str = typer.Option("out/web", help="Output dir for tiles/video/manifest."),
) -> None:
    """Precompute web artifacts: per-frame tiles + PMTiles + video + manifest (Team SERVE+VIZ).

    Each frame gets its OWN tile source so the slider can switch per timestamp (P2).
    """
    try:
        from . import precompute as pc  # lazy; owned by Team SERVE+VIZ (top-level module)
    except ImportError as exc:
        _missing("frameflow.precompute", "SERVE+VIZ", exc)
        raise typer.Exit(code=0) from None
    pc.build_web_artifacts(cube_path=cube, out_dir=out)  # type: ignore[attr-defined]


@app.command()
def serve(
    artifacts: str = typer.Option("out/web", help="Dir of precomputed artifacts to serve."),
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Serve precomputed artifacts (+ optional on-demand endpoint) (Team SERVE+VIZ)."""
    try:
        from .serve import app as serve_app  # lazy; owned by Team SERVE+VIZ
    except ImportError as exc:
        _missing("frameflow.serve", "SERVE+VIZ", exc)
        raise typer.Exit(code=0) from None
    serve_app.run(artifacts=artifacts, host=host, port=port)  # type: ignore[attr-defined]


@app.command()
def demo(
    out: str = typer.Option("out", help="Root output dir for the demo chain."),
    skip_train: bool = typer.Option(True, help="Skip the (tiny) training stage."),
) -> None:
    """Run the full end-to-end demo chain (synth -> train -> interpolate -> validate -> precompute).

    Delegates to scripts/demo.py logic via :func:`frameflow.cli._run_demo`, which lazily
    imports each stage and gracefully skips any whose deps/modules are missing.
    """
    _run_demo(out=out, skip_train=skip_train)


def _run_demo(out: str = "out", skip_train: bool = True) -> None:
    """Programmatic full-chain demo with graceful skips (shared by `demo` and scripts/demo.py)."""
    import os

    typer.secho("=" * 70, fg=typer.colors.CYAN)
    typer.secho("FrameFlow end-to-end demo", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 70, fg=typer.colors.CYAN)

    # 1) Synthetic data (always works).
    from . import synthetic
    typer.secho("\n[1/5] Generating synthetic data ...", fg=typer.colors.CYAN)
    cube = synthetic.generate_cube()
    nc_paths = synthetic.generate_pair_nc()
    typer.secho(f"      cube={cube}; demo .nc frames={len(nc_paths)}", fg=typer.colors.GREEN)

    # 2) Tiny train (optional / skippable).
    typer.secho("\n[2/5] Train (tiny) ...", fg=typer.colors.CYAN)
    if skip_train:
        typer.secho("      skipped (skip_train=True).", fg=typer.colors.YELLOW)
    else:
        try:
            from .train import trainer
            trainer.run(config="configs/config.yaml", overrides=["train.epochs=1"])  # type: ignore[attr-defined]
        except ImportError as exc:
            _missing("frameflow.train.trainer", "TRAIN", exc)

    # 3) Interpolate the middle frame of the demo triplet.
    typer.secho("\n[3/5] Interpolate ...", fg=typer.colors.CYAN)
    out_nc = os.path.join(out, "interp_nc", "mid.nc")
    try:
        from .infer import interpolate as inf
        observed = [p for p in nc_paths if "observed" in str(p)]
        inf.interpolate_pair(str(observed[0]), str(observed[1]), t=0.5, out_nc=out_nc)  # type: ignore[attr-defined]
        typer.secho(f"      wrote {out_nc}", fg=typer.colors.GREEN)
    except ImportError as exc:
        _missing("frameflow.infer.interpolate", "INFER+VALIDATE", exc)

    # 4) Validate against the withheld true middle frame.
    typer.secho("\n[4/5] Validate ...", fg=typer.colors.CYAN)
    try:
        from .validate import runner
        runner.run(pred_dir=os.path.join(out, "interp_nc"), truth_dir="data/demo_nc",  # type: ignore[attr-defined]
                   out_dir=os.path.join(out, "validation"))
    except ImportError as exc:
        _missing("frameflow.validate.runner", "INFER+VALIDATE", exc)

    # 5) Precompute web artifacts.
    typer.secho("\n[5/5] Precompute web artifacts ...", fg=typer.colors.CYAN)
    try:
        from . import precompute as pc
        pc.build_web_artifacts(cube_path=str(cube), out_dir=os.path.join(out, "web"))  # type: ignore[attr-defined]
    except ImportError as exc:
        _missing("frameflow.precompute", "SERVE+VIZ", exc)

    typer.secho("\nDemo chain complete (stages with missing modules were skipped).", fg=typer.colors.CYAN, bold=True)


if __name__ == "__main__":
    app()

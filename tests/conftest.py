"""Shared pytest fixtures for the FrameFlow test suite.

Provides ``synthetic_cube``: a tiny synthetic Zarr cube built in a temp dir via
:mod:`frameflow.synthetic`, plus the demo ``.nc`` triplet. Fixtures are deliberately small
(few frames, small grid) so the whole suite runs in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class SyntheticCubeFixture:
    """Bundle of paths/handles for a generated synthetic cube and demo triplet."""

    cube_path: Path
    nc_paths: list[Path]
    n_frames: int
    grid_rows: int
    grid_cols: int


@pytest.fixture(scope="session")
def synthetic_cube(tmp_path_factory: pytest.TempPathFactory) -> SyntheticCubeFixture:
    """Build a tiny synthetic Zarr cube + demo .nc triplet in a temp dir.

    Session-scoped so the (already fast) generation runs once for the whole suite.
    """
    from frameflow.contracts import GridSpec
    from frameflow.synthetic import generate_cube, generate_pair_nc

    base = tmp_path_factory.mktemp("frameflow_synth")
    grid = GridSpec(
        west=68.0, south=6.0, east=98.0, north=38.0,
        n_rows=48, n_cols=48, crs="EPSG:4326", resolution_deg=30.0 / 48.0,
    )
    n_frames = 6
    cube_path = generate_cube(
        n_frames=n_frames,
        grid=grid,
        out_path=str(base / "cube.zarr"),
        cadence_min=30,
        seed=0,
        n_blobs=4,
    )
    nc_paths = generate_pair_nc(out_dir=str(base / "demo_nc"), grid=grid, seed=0, n_blobs=4)
    return SyntheticCubeFixture(
        cube_path=Path(cube_path),
        nc_paths=[Path(p) for p in nc_paths],
        n_frames=n_frames,
        grid_rows=grid.n_rows,
        grid_cols=grid.n_cols,
    )

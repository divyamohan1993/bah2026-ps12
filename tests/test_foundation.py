"""Foundation smoke tests — these MUST pass before the six build teams start.

Covers: constants import & invariants; contracts dataclass round-trips via
``to_dict``/``from_dict``; ``validate_manifest`` accepts a good manifest and catches a bad
one (including the two review corrections P1/P2); and that
``frameflow.synthetic.generate_cube`` writes a readable Zarr cube with the canonical dims
and ``bt`` variable.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# constants
# ===========================================================================
def test_constants_import_and_invariants() -> None:
    from frameflow import constants as C

    # Fixed BT metric range (P1).
    assert C.BT_METRIC_VMIN_K == 180.0
    assert C.BT_METRIC_VMAX_K == 320.0
    assert C.BT_DATA_RANGE_K == pytest.approx(C.BT_METRIC_VMAX_K - C.BT_METRIC_VMIN_K)
    assert C.BT_DATA_RANGE_K == pytest.approx(140.0)

    # Satellite catalogue has the key sensors with sane TIR ~10 µm specs.
    for sat in ("GOES-19", "Himawari-9", "GK-2A", "INSAT-3DS"):
        spec = C.SATELLITE_BANDS[sat]
        assert 9.5 <= spec["wavelength_um"] <= 12.5
        assert spec["cadence_min"] > 0
        assert spec["native_res_km"] > 0
        assert spec["file_format"] in {"netcdf4", "hdf5", "hsd", "hrit"}

    # Cube layout constants.
    assert C.CUBE_DIMS == ("time", "y", "x")
    assert C.CUBE_DATA_VAR == "bt"
    assert C.CUBE_CHUNKS == (8, 512, 512)

    # S3 templates format cleanly.
    url = C.GOES19_RADF_TEMPLATE.format(year=2025, doy=171, hour=14)
    assert url.startswith("s3://noaa-goes19/") and "/171/14/" in url


# ===========================================================================
# contracts: GridSpec
# ===========================================================================
def test_gridspec_shape_and_roundtrip() -> None:
    from frameflow.contracts import GridSpec

    g = GridSpec(west=68.0, south=6.0, east=98.0, north=38.0, n_rows=100, n_cols=200)
    assert g.shape == (100, 200)
    assert g.bbox == (68.0, 6.0, 98.0, 38.0)
    assert len(g.lat_coords()) == 100
    assert len(g.lon_coords()) == 200
    # latitudes go north -> south (descending)
    lats = g.lat_coords()
    assert lats[0] > lats[-1]

    d = g.to_dict()
    g2 = GridSpec.from_dict(d)
    assert g2 == g


# ===========================================================================
# contracts: empty_cube + netcdf_attrs
# ===========================================================================
def test_empty_cube_structure() -> None:
    from frameflow.contracts import CubeSchema, GridSpec, empty_cube

    g = GridSpec.default()
    times = np.array(
        ["2025-06-20T00:00:00", "2025-06-20T00:30:00"], dtype="datetime64[ns]"
    )
    ds = empty_cube(g, times)
    assert ds[CubeSchema.DATA_VAR].dims == ("time", "y", "x")
    assert ds[CubeSchema.DATA_VAR].shape == (2, g.n_rows, g.n_cols)
    assert str(ds[CubeSchema.DATA_VAR].dtype) == "float32"
    assert np.isnan(ds[CubeSchema.DATA_VAR].values).all()  # NaN-filled
    assert "lat" in ds.coords and "lon" in ds.coords and "time" in ds.coords


def test_netcdf_attrs_required_keys() -> None:
    from frameflow.contracts import InferenceNetCDFSchema, netcdf_attrs

    attrs = netcdf_attrs(
        source_frames=["a.nc", "b.nc"],
        t=0.5,
        model="RIFE",
        model_version="v0.1.0",
        interpolation_factor=2,
    )
    for key in InferenceNetCDFSchema.REQUIRED_ATTRS:
        assert key in attrs, f"missing required .nc attr: {key}"
    assert attrs["t"] == 0.5
    assert attrs["interpolation_factor"] == 2


# ===========================================================================
# contracts: nested record round-trips
# ===========================================================================
def test_frame_entry_roundtrip() -> None:
    from frameflow.contracts import FrameEntry

    fe = FrameEntry(
        index=1,
        time="2025-06-20T00:10:00Z",
        kind="interpolated",
        image="img/001.webp",
        thumb="thumb/001.webp",
        tiles_url_template="tiles/001/{z}/{x}/{y}.webp",
        pmtiles="pmtiles/001.pmtiles",
        netcdf="nc/001.nc",
        t=0.5,
        bracket=[0, 2],
        flow_overlay="flow/001.webp",
    )
    fe2 = FrameEntry.from_dict(fe.to_dict())
    assert fe2 == fe


def test_metric_record_roundtrip_with_extra() -> None:
    from frameflow.contracts import MetricRecord

    mr = MetricRecord(index=2, psnr=42.0, ssim=0.95, bt_rmse_k=0.9, extra={"gmsd": 0.01})
    d = mr.to_dict()
    mr2 = MetricRecord.from_dict(d)
    assert mr2.index == 2
    assert mr2.psnr == pytest.approx(42.0)
    assert mr2.extra["gmsd"] == pytest.approx(0.01)


def test_crossval_method_result_roundtrip() -> None:
    from frameflow.contracts import CrossvalMethodResult

    c = CrossvalMethodResult(
        method_id="M1",
        name="leave-the-middle-out (GOES-19)",
        status="run",
        summary={"psnr": 45.4, "bt_rmse_k": 0.99},
    )
    c2 = CrossvalMethodResult.from_dict(c.to_dict())
    assert c2 == c


# ===========================================================================
# contracts: Manifest round-trip + validation (P1 + P2)
# ===========================================================================
def _good_manifest_dict() -> dict:
    from frameflow.contracts import (
        CrossvalMethodResult,
        FrameEntry,
        Manifest,
        MetricRecord,
    )

    frames = [
        FrameEntry(
            index=0, time="2025-06-20T00:00:00Z", kind="observed",
            image="img/000.webp", thumb="thumb/000.webp",
            tiles_url_template="tiles/000/{z}/{x}/{y}.webp",
            pmtiles="pmtiles/000.pmtiles", netcdf="nc/000.nc",
        ),
        FrameEntry(
            index=1, time="2025-06-20T00:10:00Z", kind="interpolated",
            image="img/001.webp", thumb="thumb/001.webp",
            tiles_url_template="tiles/001/{z}/{x}/{y}.webp",
            pmtiles="pmtiles/001.pmtiles", netcdf="nc/001.nc",
            t=0.5, bracket=[0, 2],
        ),
        FrameEntry(
            index=2, time="2025-06-20T00:20:00Z", kind="observed",
            image="img/002.webp", thumb="thumb/002.webp",
            tiles_url_template="tiles/002/{z}/{x}/{y}.webp",
            pmtiles="pmtiles/002.pmtiles", netcdf="nc/002.nc",
        ),
    ]
    m = Manifest(
        scene_id="demo-0001",
        title="FrameFlow demo scene",
        satellite="GOES-19",
        channel="C13",
        wavelength_um=10.3,
        bbox=[68.0, 6.0, 98.0, 38.0],
        colormap="ir_clouds",
        value_range_k=[180.0, 320.0],
        tile_size=256,
        min_zoom=0,
        max_zoom=8,
        interpolation_factor=2,
        cadence_minutes_input=20.0,
        cadence_minutes_output=10.0,
        frames=frames,
        generated="2026-06-20T00:00:00Z",
        model_name="RIFE",
        model_version="v0.1.0",
        model_params_m=9.8,
    )
    m.metrics["per_frame"] = [MetricRecord(index=1, psnr=44.0, ssim=0.95).to_dict()]
    m.crossval["methods_run"] = [
        CrossvalMethodResult(method_id="M1", name="leave-the-middle-out").to_dict()
    ]
    return m.to_dict()


def test_manifest_roundtrip() -> None:
    from frameflow.contracts import Manifest

    d = _good_manifest_dict()
    m = Manifest.from_dict(d)
    d2 = m.to_dict()
    # structural equality on the JSON-able dicts
    assert d2["scene_id"] == d["scene_id"]
    assert len(d2["frames"]) == 3
    assert d2["metrics"]["data_range_k"] == pytest.approx(140.0)
    assert d2["frames"][1]["t"] == pytest.approx(0.5)


def test_validate_manifest_accepts_good() -> None:
    from frameflow.contracts import validate_manifest

    problems = validate_manifest(_good_manifest_dict())
    assert problems == [], f"good manifest reported problems: {problems}"


def test_validate_manifest_catches_bad() -> None:
    from frameflow.contracts import validate_manifest

    bad = _good_manifest_dict()
    # Remove a required top-level field.
    del bad["title"]
    # Break P1: data_range inconsistent with value_range span.
    bad["metrics"]["data_range_k"] = 999.0
    # Break P2: drop the per-frame pmtiles and corrupt the XYZ template on a frame.
    bad["frames"][1].pop("pmtiles", None)
    bad["frames"][1]["tiles_url_template"] = "tiles/001/static.webp"  # no {z}/{x}/{y}
    # Make an interpolated frame inconsistent (null t).
    bad["frames"][1]["t"] = None
    # Bad bbox ordering.
    bad["bbox"] = [98.0, 6.0, 68.0, 38.0]

    problems = validate_manifest(bad)
    joined = " | ".join(problems)
    assert problems, "bad manifest incorrectly passed validation"
    assert "title" in joined
    assert "data_range_k" in joined  # P1 caught
    assert "pmtiles" in joined       # P2 caught
    assert "{z}" in joined or "tiles_url_template" in joined  # P2 caught
    assert "bbox" in joined


def test_validate_manifest_rejects_non_dict() -> None:
    from frameflow.contracts import validate_manifest

    assert validate_manifest([])  # type: ignore[arg-type]  # returns a non-empty problem list


# ===========================================================================
# synthetic: generate_cube writes a readable Zarr with correct dims/var
# ===========================================================================
def test_generate_cube_writes_readable_zarr(synthetic_cube) -> None:
    import xarray as xr

    from frameflow.contracts import CubeSchema

    ds = xr.open_zarr(synthetic_cube.cube_path, consolidated=False)
    try:
        assert tuple(ds[CubeSchema.DATA_VAR].dims) == ("time", "y", "x")
        assert ds.sizes["time"] == synthetic_cube.n_frames
        assert ds.sizes["y"] == synthetic_cube.grid_rows
        assert ds.sizes["x"] == synthetic_cube.grid_cols
        assert str(ds[CubeSchema.DATA_VAR].dtype) == "float32"
        assert "lat" in ds.coords and "lon" in ds.coords

        arr = ds[CubeSchema.DATA_VAR].values
        # Physically plausible Kelvin values within the metric range, NaN space pixels present.
        assert np.isnan(arr).any(), "expected NaN 'space' corner pixels"
        finite = arr[np.isfinite(arr)]
        assert finite.min() >= 180.0 and finite.max() <= 320.0
    finally:
        ds.close()


def test_generate_pair_nc_triplet(synthetic_cube) -> None:
    import xarray as xr

    from frameflow.contracts import InferenceNetCDFSchema

    assert len(synthetic_cube.nc_paths) == 3
    # The middle file is the withheld ground truth (kind=interpolated marker in the name).
    names = [p.name for p in synthetic_cube.nc_paths]
    assert any("interpolated" in n for n in names)
    assert sum("observed" in n for n in names) == 2

    for p in synthetic_cube.nc_paths:
        ds = xr.open_dataset(p)
        try:
            assert InferenceNetCDFSchema.DATA_VAR in ds.data_vars
            assert tuple(ds[InferenceNetCDFSchema.DATA_VAR].dims) == ("time", "y", "x")
            # required global attrs present
            for key in InferenceNetCDFSchema.REQUIRED_ATTRS:
                assert key in ds.attrs, f"{p.name} missing attr {key}"
        finally:
            ds.close()


def test_generate_bt_field_nonlinear(synthetic_cube) -> None:
    """The true middle frame must NOT equal the linear blend of its neighbours."""
    import xarray as xr

    from frameflow.contracts import CubeSchema

    ds = xr.open_zarr(synthetic_cube.cube_path, consolidated=False)
    try:
        arr = ds[CubeSchema.DATA_VAR].values
        f0, f1, f2 = arr[0], arr[1], arr[2]
        blend = 0.5 * (f0 + f2)
        mean_abs_dev = float(np.nanmean(np.abs(f1 - blend)))
        # Non-trivial deviation proves motion is non-linear (else interpolation is trivial).
        assert mean_abs_dev > 0.1, f"motion looks linear (dev={mean_abs_dev} K)"
    finally:
        ds.close()

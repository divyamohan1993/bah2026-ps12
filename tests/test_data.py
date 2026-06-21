"""Tests for Team DATA — frameflow/data/** (CONTRACTS.md §3).

Self-contained: every test builds its own tiny inputs from :mod:`frameflow.synthetic` into a
pytest ``tmp_path`` (no network, no fixtures beyond the stdlib/pytest). Coverage:

* ``radiance_to_bt`` on known inputs (inverse Planck) + NaN/non-positive masking.
* ``normalize`` / ``denormalize`` round-trip (< 1e-4) for both modes, NaN-safe.
* ``build_cube`` from the synthetic ``.nc`` triplet -> a schema-correct Zarr cube readable by
  xarray (dims / data var / dtype / chunks / compressor / NaN corners), value-preserving.
* ``TripletDataset[0]`` returns ``(1, H, W)`` tensors with ``t`` in ``(0, 1)``, NaN-safe,
  with fixed and random ``t`` and leakage-free time splits.
* regrid-to-the-same-grid ≈ identity.
* virtual (kerchunk) zero-copy reference round-trip; WebDataset shard round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from frameflow.contracts import CubeSchema, GridSpec


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _small_grid() -> GridSpec:
    """A tiny 40x40 lat/lon grid over the Indian bbox (fast for tests)."""
    return GridSpec(
        west=68.0, south=6.0, east=98.0, north=38.0,
        n_rows=40, n_cols=40, crs="EPSG:4326", resolution_deg=30.0 / 40.0,
    )


def _make_nc_triplet(tmp_path, grid: GridSpec | None = None):
    """Materialize the synthetic .nc triplet (00:00, 00:10, 00:20) into ``tmp_path``."""
    from frameflow.synthetic import generate_pair_nc

    grid = grid or _small_grid()
    paths = generate_pair_nc(out_dir=str(tmp_path / "nc"), grid=grid, seed=0, n_blobs=4)
    return [str(p) for p in paths], grid


def _make_cube(tmp_path, n_frames: int = 6, grid: GridSpec | None = None):
    """Materialize a synthetic Zarr cube into ``tmp_path``."""
    from frameflow.synthetic import generate_cube

    grid = grid or _small_grid()
    cube = generate_cube(
        n_frames=n_frames, grid=grid, out_path=str(tmp_path / "synthetic.zarr"),
        cadence_min=30, seed=0, n_blobs=4,
    )
    return str(cube), grid


# ===========================================================================
# 1. radiance -> brightness temperature (inverse Planck)
# ===========================================================================
def test_radiance_to_bt_known_inputs():
    """``radiance_to_bt`` matches the closed-form inverse Planck on known coefficients."""
    from frameflow.data.preprocess.radiance import radiance_to_bt

    # Representative GOES ABI C13 (10.3 µm) Planck coefficients.
    fk1, fk2, bc1, bc2 = 1.08033e4, 1.39274e3, 0.07396, 0.99961
    rad = np.array([50.0, 80.0, 100.0], dtype=np.float64)

    bt = radiance_to_bt(rad, fk1, fk2, bc1, bc2)
    expected = (fk2 / np.log(fk1 / rad + 1.0) - bc1) / bc2

    assert np.allclose(bt, expected, atol=1e-6)
    # Physically sane window-IR brightness temperatures.
    assert np.all(bt > 180.0) and np.all(bt < 320.0)
    # Monotonic: more radiance -> warmer BT.
    assert bt[0] < bt[1] < bt[2]


def test_radiance_to_bt_masks_nan_and_nonpositive():
    """NaN and non-positive radiance become NaN (no log-of-nonpositive warnings/inf)."""
    from frameflow.data.preprocess.radiance import radiance_to_bt

    fk1, fk2, bc1, bc2 = 1.08033e4, 1.39274e3, 0.07396, 0.99961
    rad = np.array([np.nan, -1.0, 0.0, 100.0], dtype=np.float64)

    bt = radiance_to_bt(rad, fk1, fk2, bc1, bc2)
    assert np.isnan(bt[0]) and np.isnan(bt[1]) and np.isnan(bt[2])
    assert np.isfinite(bt[3])


def test_radiance_to_bt_rejects_degenerate_coeffs():
    """Zero ``fk1`` or ``bc2`` is a degenerate coefficient set and must raise."""
    from frameflow.data.preprocess.radiance import radiance_to_bt

    with pytest.raises(ValueError):
        radiance_to_bt(np.array([100.0]), 0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        radiance_to_bt(np.array([100.0]), 1.0, 1.0, 0.0, 0.0)


def test_count_to_bt_lut_gather():
    """The INSAT count->BT path gathers straight from the temperature LUT."""
    from frameflow.data.preprocess.radiance import count_to_bt

    lut = np.linspace(150.0, 350.0, 1024)
    counts = np.array([0, 511, 1023])
    bt = count_to_bt(counts, lut=lut)
    assert np.allclose(bt, lut[counts])
    # Out-of-range index -> NaN, not a crash.
    bt_oob = count_to_bt(np.array([5000]), lut=lut)
    assert np.isnan(bt_oob[0])


# ===========================================================================
# 2. normalize / denormalize round-trip
# ===========================================================================
def test_normalize_denormalize_roundtrip_fixed_range():
    """fixed-range normalize -> denormalize recovers Kelvin to < 1e-4."""
    from frameflow.data.preprocess import denormalize, normalize

    bt = np.array([180.0, 210.5, 250.0, 299.9, 330.0], dtype=np.float32)
    x = normalize(bt, mode="fixed_range")
    back = denormalize(x, mode="fixed_range")
    assert np.max(np.abs(back - bt)) < 1e-4
    # Endpoints of the [vmin, vmax] span map to [0, 1].
    assert abs(float(x[0]) - 0.0) < 1e-6
    assert abs(float(x[-1]) - 1.0) < 1e-6


def test_normalize_denormalize_roundtrip_zscore():
    """z-score normalize -> denormalize recovers Kelvin to < 1e-4."""
    from frameflow.data.preprocess import denormalize, normalize

    bt = np.array([200.0, 240.0, 270.0, 300.0], dtype=np.float32)
    x = normalize(bt, mode="zscore", dataset="goes")
    back = denormalize(x, mode="zscore", dataset="goes")
    assert np.max(np.abs(back - bt)) < 1e-4


def test_normalize_is_nan_safe():
    """NaN (space) pixels stay NaN through normalize and denormalize."""
    from frameflow.data.preprocess import denormalize, normalize

    bt = np.array([np.nan, 250.0, np.nan], dtype=np.float32)
    x = normalize(bt, mode="fixed_range")
    assert np.isnan(x[0]) and np.isnan(x[2]) and np.isfinite(x[1])
    back = denormalize(x, mode="fixed_range")
    assert np.isnan(back[0]) and np.isnan(back[2])


# ===========================================================================
# 3. regrid-to-the-same-grid ≈ identity
# ===========================================================================
def test_regrid_to_same_grid_is_identity(tmp_path):
    """Regridding a frame onto its OWN grid returns (essentially) the same field."""
    import xarray as xr

    from frameflow.data.preprocess import regrid_to_grid

    nc_paths, grid = _make_nc_triplet(tmp_path)
    ds = xr.open_dataset(nc_paths[0])
    try:
        da = ds["bt"]  # (time=1, y, x) with lat/lon coords
        out = regrid_to_grid(da, grid, cache_dir=str(tmp_path / "kernel"))
        a = da.isel(time=0).values
        b = out.isel(time=0).values if "time" in out.dims else out.values

        assert out.shape[-2:] == grid.shape
        finite = np.isfinite(a) & np.isfinite(b)
        # Nearest-neighbour onto the identical grid picks the same source pixel everywhere.
        assert finite.sum() > 0
        assert np.max(np.abs(a[finite] - b[finite])) < 1e-3
    finally:
        ds.close()


# ===========================================================================
# 4. build_cube from the synthetic .nc triplet
# ===========================================================================
def test_build_cube_schema_correct(tmp_path):
    """build_cube turns the synthetic .nc triplet into a schema-correct, readable Zarr cube."""
    import xarray as xr

    from frameflow.data.ingest import build_cube

    nc_paths, grid = _make_nc_triplet(tmp_path)
    out_path = tmp_path / "cube.zarr"

    out = build_cube(
        nc_paths,
        grid=grid,
        out_path=str(out_path),
        source_meta={"satellite": "SYNTHETIC", "channel": "TIR", "cadence_min": 10},
    )
    assert out.exists()

    ds = xr.open_zarr(str(out), consolidated=False)
    try:
        # dims / data var / coords
        assert tuple(ds[CubeSchema.DATA_VAR].dims) == CubeSchema.DIMS
        assert ds.sizes["time"] == len(nc_paths)
        assert ds.sizes["y"] == grid.n_rows
        assert ds.sizes["x"] == grid.n_cols
        assert CubeSchema.DATA_VAR in ds.data_vars
        for coord in ("time", "lat", "lon"):
            assert coord in ds.coords

        # dtype
        assert ds[CubeSchema.DATA_VAR].dtype == np.dtype(CubeSchema.DTYPE)

        # chunks: canonical (8,512,512) clamped to the (small) array dims
        enc_chunks = ds[CubeSchema.DATA_VAR].encoding.get("chunks")
        expected_chunks = (
            min(CubeSchema.CHUNKS[0], ds.sizes["time"]),
            min(CubeSchema.CHUNKS[1], ds.sizes["y"]),
            min(CubeSchema.CHUNKS[2], ds.sizes["x"]),
        )
        assert tuple(enc_chunks) == expected_chunks

        # compressor is blosc-zstd-shuffle (the CubeSchema codec)
        compressors = ds[CubeSchema.DATA_VAR].encoding.get("compressors")
        assert compressors, "expected a compressor in the cube encoding"
        comp_repr = repr(compressors).lower()
        assert "blosc" in comp_repr and "zstd" in comp_repr

        # time axis is datetime64
        assert np.issubdtype(ds["time"].dtype, np.datetime64)

        # values are physical Kelvin and NaN corners (space) survived ingest
        vals = ds[CubeSchema.DATA_VAR].values
        assert np.isnan(vals).any(), "expected NaN 'space' corners to be preserved"
        finite = vals[np.isfinite(vals)]
        assert finite.min() >= 180.0 and finite.max() <= 320.0
    finally:
        ds.close()


def test_build_cube_preserves_values(tmp_path):
    """Regrid-to-same-grid is value-preserving: cube[i] == source[i] within tolerance."""
    import xarray as xr

    from frameflow.data.ingest import build_cube

    nc_paths, grid = _make_nc_triplet(tmp_path)
    out = build_cube(nc_paths, grid=grid, out_path=str(tmp_path / "cube.zarr"))

    cube = xr.open_zarr(str(out), consolidated=False)
    try:
        for i, p in enumerate(nc_paths):
            src = xr.open_dataset(p)
            try:
                a = cube[CubeSchema.DATA_VAR].isel(time=i).values
                b = src["bt"].isel(time=0).values
                finite = np.isfinite(a) & np.isfinite(b)
                assert np.max(np.abs(a[finite] - b[finite])) < 1e-3
            finally:
                src.close()
    finally:
        cube.close()


def test_build_cube_empty_files_raises(tmp_path):
    """build_cube with no files is a usage error."""
    from frameflow.data.ingest import build_cube

    with pytest.raises(ValueError):
        build_cube([], grid=_small_grid(), out_path=str(tmp_path / "x.zarr"))


# ===========================================================================
# 5. TripletDataset
# ===========================================================================
def _as_numpy(t):
    """Return a numpy view of a tensor-or-array sample field."""
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def test_triplet_dataset_item_shapes_and_t(tmp_path):
    """TripletDataset[0] yields (1,H,W) tensors, valid t in (0,1), and is NaN-safe."""
    from frameflow.data.datasets import TripletDataset

    cube_path, grid = _make_cube(tmp_path, n_frames=6)
    ds = TripletDataset(cube_path, split="all", patch_size=None, normalized=True, t_mode="fixed")
    try:
        assert len(ds) == 4  # interior frames 1..4 of a 6-frame cube
        s = ds[0]
        assert set(s.keys()) == {"I0", "I1", "It", "t", "meta"}

        for key in ("I0", "I1", "It"):
            arr = _as_numpy(s[key])
            assert arr.shape == (1, grid.n_rows, grid.n_cols)
            assert arr.dtype == np.float32
            # NaN-safe: the model never sees NaN after fill.
            assert np.isfinite(arr).all()

        # t strictly inside the open interval (0, 1); fixed mode == midpoint.
        assert 0.0 < s["t"] < 1.0
        assert abs(s["t"] - 0.5) < 1e-9

        # meta carries the leave-the-middle-out bracket + a validity mask.
        assert s["meta"]["bracket"] == [s["meta"]["mid_index"] - 1, s["meta"]["mid_index"] + 1]
        assert s["meta"]["normalized"] is True
        mask = _as_numpy(s["meta"]["mask"])
        assert mask.shape == (1, grid.n_rows, grid.n_cols)
        assert 0.0 < float(mask.mean()) <= 1.0  # some valid, some NaN corners
    finally:
        ds.close()


def test_triplet_dataset_random_t_in_open_interval(tmp_path):
    """random-t mode draws t strictly inside (0,1) and varies across items."""
    from frameflow.data.datasets import TripletDataset

    cube_path, _ = _make_cube(tmp_path, n_frames=6)
    ds = TripletDataset(cube_path, split="all", patch_size=None, t_mode="random", seed=3)
    try:
        ts = [ds[i]["t"] for i in range(len(ds))]
        assert all(0.0 < t < 1.0 for t in ts)
        assert len(set(round(t, 4) for t in ts)) > 1  # not all identical
    finally:
        ds.close()


def test_triplet_dataset_normalized_values_in_unit_range(tmp_path):
    """Normalized fixed-range samples fall in [0,1] (a normalized BT field, NaNs filled)."""
    from frameflow.data.datasets import TripletDataset

    cube_path, _ = _make_cube(tmp_path, n_frames=6)
    ds = TripletDataset(cube_path, split="all", patch_size=None, normalized=True)
    try:
        arr = _as_numpy(ds[0]["I0"])
        assert arr.min() >= 0.0 - 1e-6
        assert arr.max() <= 1.0 + 1e-6
    finally:
        ds.close()


def test_triplet_dataset_patch_crop(tmp_path):
    """A patch_size smaller than the grid yields aligned (1, p, p) crops."""
    from frameflow.data.datasets import TripletDataset

    cube_path, _ = _make_cube(tmp_path, n_frames=6)
    ds = TripletDataset(cube_path, split="all", patch_size=16, normalized=True, seed=1)
    try:
        s = ds[0]
        for key in ("I0", "I1", "It"):
            assert _as_numpy(s[key]).shape == (1, 16, 16)
        # All three frames cropped at the SAME origin (registered triplet).
        assert len(s["meta"]["crop_origin"]) == 2
    finally:
        ds.close()


def test_triplet_dataset_time_split_no_leakage(tmp_path):
    """train/val/test are contiguous, disjoint time slices (no temporal leakage)."""
    from frameflow.data.datasets import TripletDataset

    cube_path, _ = _make_cube(tmp_path, n_frames=10)
    tr = TripletDataset(cube_path, split="train")
    va = TripletDataset(cube_path, split="val")
    te = TripletDataset(cube_path, split="test")
    try:
        # len() triggers the lazy open that populates ``_mid_indices``.
        assert len(tr) >= 1 and len(va) >= 1 and len(te) >= 1
        mids_tr = set(tr._mid_indices)
        mids_va = set(va._mid_indices)
        mids_te = set(te._mid_indices)
        # disjoint
        assert mids_tr.isdisjoint(mids_va)
        assert mids_tr.isdisjoint(mids_te)
        assert mids_va.isdisjoint(mids_te)
        # contiguous and ordered (train before val before test)
        assert max(mids_tr) < min(mids_va) <= max(mids_va) < min(mids_te)
        # cover all interior frames
        assert mids_tr | mids_va | mids_te == set(range(1, 10 - 1))
    finally:
        tr.close()
        va.close()
        te.close()


def test_triplet_dataset_index_out_of_range(tmp_path):
    """Out-of-range indexing raises IndexError (so DataLoader iteration terminates)."""
    from frameflow.data.datasets import TripletDataset

    cube_path, _ = _make_cube(tmp_path, n_frames=6)
    ds = TripletDataset(cube_path, split="all", patch_size=None)
    try:
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
    finally:
        ds.close()


# ===========================================================================
# 6. virtual (kerchunk) zero-copy references
# ===========================================================================
def test_virtual_reference_roundtrip(tmp_path):
    """A kerchunk virtual reference opens to the same data as the original .nc (zero-copy)."""
    import xarray as xr

    from frameflow.data.ingest import open_virtual, virtual_reference

    nc_paths, _ = _make_nc_triplet(tmp_path)
    refs = virtual_reference(nc_paths[0])
    vds = open_virtual(refs)
    truth = xr.open_dataset(nc_paths[0])
    try:
        assert "bt" in vds.data_vars
        assert tuple(vds["bt"].shape) == tuple(truth["bt"].shape)
        a = vds["bt"].values
        b = truth["bt"].values
        finite = np.isfinite(a) & np.isfinite(b)
        assert np.array_equal(a[finite], b[finite])
    finally:
        vds.close()
        truth.close()


def test_write_references_json(tmp_path):
    """Reference sets persist to JSON and reopen identically."""
    import json

    from frameflow.data.ingest import open_virtual, virtual_reference, write_references

    nc_paths, _ = _make_nc_triplet(tmp_path)
    refs = virtual_reference(nc_paths[0])
    ref_path = write_references(refs, tmp_path / "refs.json", fmt="json")
    assert ref_path.exists()
    # File is valid JSON and reopenable via the opener.
    json.loads(ref_path.read_text())
    vds = open_virtual(str(ref_path))
    try:
        assert "bt" in vds.data_vars
    finally:
        vds.close()


# ===========================================================================
# 7. WebDataset / npz shards
# ===========================================================================
def test_write_shards_tar_roundtrip(tmp_path):
    """Triplets bake into WebDataset tar shards and read back byte-identical."""
    from frameflow.data.datasets import TripletDataset, read_shard, write_shards

    cube_path, _ = _make_cube(tmp_path, n_frames=6)
    ds = TripletDataset(cube_path, split="all", patch_size=None, normalized=True)
    try:
        samples = [ds[i] for i in range(len(ds))]
    finally:
        ds.close()

    shards = write_shards(samples, tmp_path / "shards", fmt="tar", maxcount=2)
    assert len(shards) == 2  # 4 samples / 2 per shard
    assert all(p.suffix == ".tar" for p in shards)

    recs = []
    for p in shards:
        recs += read_shard(p)
    assert len(recs) == len(samples)
    assert recs[0]["i0"].shape == (1, _small_grid().n_rows, _small_grid().n_cols)
    assert 0.0 < recs[0]["t"] < 1.0
    assert np.array_equal(recs[0]["i0"], _as_numpy(samples[0]["I0"]))


def test_write_shards_npz_roundtrip(tmp_path):
    """The .npz shard fallback round-trips arrays + t."""
    from frameflow.data.datasets import TripletDataset, read_shard, write_shards

    cube_path, _ = _make_cube(tmp_path, n_frames=6)
    ds = TripletDataset(cube_path, split="all", patch_size=None, normalized=True)
    try:
        samples = [ds[i] for i in range(len(ds))]
    finally:
        ds.close()

    shards = write_shards(samples, tmp_path / "npz", fmt="npz", maxcount=3)
    assert all(p.suffix == ".npz" for p in shards)
    recs = []
    for p in shards:
        recs += read_shard(p)
    assert len(recs) == len(samples)
    assert np.array_equal(recs[1]["it"], _as_numpy(samples[1]["It"]))

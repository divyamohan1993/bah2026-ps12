"""Tests for the INFER area — ``frameflow.infer.{netcdf_io,interpolate,export,batch}``.

These exercise the inference scientific core WITHOUT importing the models package: a tiny
in-test :class:`BlendModel` (linear blend of ``I0``/``I1`` by ``t``, with ``t`` shaped
``(B, 1)`` per the P2-ONNX contract) and an identity model stand in for a real VFI model.

Covered:
    * ``.nc`` write -> read round-trip preserves the Kelvin float32 ``bt`` field (incl. NaN
      off-disk pixels) to < 1e-5 and carries every REQUIRED attribute.
    * ``interpolate_pair`` on identical frames is (numerically) the identity.
    * ``interpolate_pair`` builds a batched ``t`` of shape ``(1, 1)`` (asserted via a probe
      model that records the tensor it receives).
    * ``interpolate_recursive(factor=2)`` returns the right frame count and observed/
      interpolated tags; arbitrary ``factor`` in {2,4,8} and arbitrary ``t`` work.
    * ``to_onnx`` exports a file and the ONNX graph's ``t`` input carries a dynamic BATCH
      dim (loaded via ``onnx`` if importable; otherwise the whole ONNX test is skipped).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ===========================================================================
# tiny in-test models (NEVER import frameflow.models)
# ===========================================================================
class BlendModel(torch.nn.Module):
    """Linear-blend VFI stand-in: ``It = (1-t)*I0 + t*I1`` with ``t`` of shape ``(B, 1)``."""

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Honour the P2-ONNX contract: t arrives as (B, 1); broadcast to (B, 1, 1, 1).
        tt = t.view(-1, 1, 1, 1)
        return I0 * (1.0 - tt) + I1 * tt


class IdentityModel(torch.nn.Module):
    """Returns ``I0`` unchanged (a degenerate VFI model, useful for identity checks)."""

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return I0


class TShapeProbe(torch.nn.Module):
    """Records the shape of the ``t`` tensor it is called with (to assert the (B,1) contract)."""

    def __init__(self) -> None:
        super().__init__()
        self.t_shape: tuple[int, ...] | None = None

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.t_shape = tuple(t.shape)
        tt = t.view(-1, 1, 1, 1)
        return I0 * (1.0 - tt) + I1 * tt


# ===========================================================================
# helpers
# ===========================================================================
def _grid(n: int = 32):
    from frameflow.contracts import GridSpec

    return GridSpec(
        west=68.0, south=6.0, east=98.0, north=38.0,
        n_rows=n, n_cols=n, crs="EPSG:4326", resolution_deg=30.0 / n,
    )


def _bt_field(n: int = 32, seed: int = 0) -> np.ndarray:
    """A smooth physical-ish Kelvin field with a couple of NaN 'space' corners."""
    yy, xx = np.mgrid[0:n, 0:n]
    field = (250.0 + 30.0 * np.sin(xx / 6.0) + 20.0 * np.cos(yy / 5.0)).astype(np.float32)
    field[0, 0] = np.nan
    field[-1, -1] = np.nan
    return field


# ===========================================================================
# netcdf_io: write -> read round-trip + required attrs
# ===========================================================================
def test_netcdf_roundtrip_preserves_bt_and_attrs(tmp_path) -> None:
    from frameflow.contracts import InferenceNetCDFSchema, netcdf_attrs
    from frameflow.infer.netcdf_io import read_netcdf, write_netcdf

    g = _grid(32)
    bt = _bt_field(32)
    attrs = netcdf_attrs(
        source_frames=["a.nc", "b.nc"], t=0.5,
        model="RIFE", model_version="v0.1.0", interpolation_factor=2,
    )
    out = write_netcdf(
        bt, g.lat_coords(), g.lon_coords(), "2025-06-20T00:10:00", attrs,
        tmp_path / "mid.nc",
    )
    assert out.exists()

    ds = read_netcdf(out)
    try:
        # Schema: dims, var, dtype.
        assert tuple(ds[InferenceNetCDFSchema.DATA_VAR].dims) == ("time", "y", "x")
        bt2 = ds[InferenceNetCDFSchema.DATA_VAR].values[0]
        assert str(ds[InferenceNetCDFSchema.DATA_VAR].dtype) == "float32"

        # Finite values round-trip to < 1e-5; NaN off-disk pixels preserved.
        finite = np.isfinite(bt)
        assert np.max(np.abs(bt2[finite] - bt[finite])) < 1e-5
        assert np.isnan(bt2[0, 0]) and np.isnan(bt2[-1, -1])

        # Every REQUIRED global attribute present and faithful.
        for key in InferenceNetCDFSchema.REQUIRED_ATTRS:
            assert key in ds.attrs, f"missing required .nc attr: {key}"
        assert float(ds.attrs["t"]) == pytest.approx(0.5)
        assert str(ds.attrs["model"]) == "RIFE"
        assert str(ds.attrs["model_version"]) == "v0.1.0"
    finally:
        ds.close()


def test_write_netcdf_rejects_missing_required_attr(tmp_path) -> None:
    """A malformed attrs dict (missing a REQUIRED key) must be refused, never written."""
    from frameflow.infer.netcdf_io import write_netcdf

    g = _grid(16)
    bt = _bt_field(16)
    bad_attrs = {"t": 0.5, "model": "RIFE"}  # missing source_frames/model_version/...
    with pytest.raises(ValueError):
        write_netcdf(bt, g.lat_coords(), g.lon_coords(), "2025-06-20T00:00:00",
                     bad_attrs, tmp_path / "bad.nc")


# ===========================================================================
# interpolate_pair: identity, NaN mask, batched-t contract
# ===========================================================================
def test_interpolate_pair_identity_on_identical_frames() -> None:
    from frameflow.infer.interpolate import interpolate_pair

    I0 = _bt_field(32)
    I1 = I0.copy()
    out = interpolate_pair(BlendModel(), I0, I1, t=0.5)

    assert out.shape == I0.shape
    finite = np.isfinite(I0)
    # Blend of identical frames == the frame itself (within float32 normalize round-trip).
    assert np.max(np.abs(out[finite] - I0[finite])) < 1e-3
    # Off-disk NaN mask restored on the output.
    assert np.isnan(out[0, 0]) and np.isnan(out[-1, -1])


def test_interpolate_pair_blends_toward_t() -> None:
    """A blend model should move the output toward I1 as t increases (arbitrary-t support)."""
    from frameflow.infer.interpolate import interpolate_pair

    I0 = np.full((24, 24), 220.0, dtype=np.float32)
    I1 = np.full((24, 24), 300.0, dtype=np.float32)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = interpolate_pair(BlendModel(), I0, I1, t=t)
        expected = (1.0 - t) * 220.0 + t * 300.0
        assert float(np.mean(out)) == pytest.approx(expected, abs=0.05)


def test_interpolate_pair_uses_batched_t_shape_B1() -> None:
    """P2-ONNX: interpolate_pair MUST hand the model a (1, 1) t tensor, never a scalar."""
    from frameflow.infer.interpolate import interpolate_pair

    probe = TShapeProbe()
    I0 = _bt_field(16)
    interpolate_pair(probe, I0, I0.copy(), t=0.5)
    assert probe.t_shape == (1, 1), f"t must be shape (B, 1)=(1,1); got {probe.t_shape}"


# ===========================================================================
# interpolate_recursive: counts + observed/interpolated tags
# ===========================================================================
def test_interpolate_recursive_factor2_counts_and_tags() -> None:
    from frameflow.infer.interpolate import interpolate_recursive

    frames = [_bt_field(24, seed=i) for i in range(3)]
    seq = interpolate_recursive(BlendModel(), frames, factor=2)

    # (n_obs - 1) * factor + 1 == (3-1)*2 + 1 == 5
    assert len(seq) == (len(frames) - 1) * 2 + 1
    kinds = [f.kind for f in seq]
    assert kinds == ["observed", "interpolated", "observed", "interpolated", "observed"]

    # Observed frames pass through unchanged; interpolated carry t_local + bracket.
    for f in seq:
        if f.kind == "observed":
            assert f.t_local is None and f.bracket is None
        else:
            assert f.t_local == pytest.approx(0.5)
            assert f.bracket is not None and len(f.bracket) == 2


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_interpolate_recursive_factor_counts(factor: int) -> None:
    from frameflow.infer.interpolate import interpolate_recursive

    frames = [_bt_field(16, seed=i) for i in range(3)]
    seq = interpolate_recursive(BlendModel(), frames, factor=factor)
    assert len(seq) == (len(frames) - 1) * factor + 1
    n_obs = sum(1 for f in seq if f.kind == "observed")
    assert n_obs == len(frames)


def test_interpolate_recursive_rejects_bad_factor() -> None:
    from frameflow.infer.interpolate import interpolate_recursive

    with pytest.raises(ValueError):
        interpolate_recursive(BlendModel(), [_bt_field(8), _bt_field(8)], factor=3)


def test_interpolate_recursive_propagates_times() -> None:
    from frameflow.infer.interpolate import interpolate_recursive

    frames = [_bt_field(16, seed=i) for i in range(3)]
    times = np.array(
        ["2025-06-20T00:00:00", "2025-06-20T00:30:00", "2025-06-20T01:00:00"],
        dtype="datetime64[ns]",
    ).tolist()
    seq = interpolate_recursive(BlendModel(), frames, factor=2, times=times)
    # The first inserted (interpolated) frame sits at 00:15 (midpoint of 00:00..00:30).
    interp = [f for f in seq if f.kind == "interpolated"]
    assert np.datetime64(interp[0].time) == np.datetime64("2025-06-20T00:15:00")


# ===========================================================================
# interpolate_pair_nc + batch.interpolate_cube (the .nc / cube paths)
# ===========================================================================
def test_interpolate_pair_nc_roundtrip(tmp_path) -> None:
    """Read two .nc frames, synthesize the middle, write a schema-correct .nc."""
    from frameflow.contracts import InferenceNetCDFSchema, netcdf_attrs
    from frameflow.infer.interpolate import interpolate_pair_nc
    from frameflow.infer.netcdf_io import read_netcdf, write_netcdf

    g = _grid(24)
    f0 = np.full((24, 24), 230.0, dtype=np.float32)
    f1 = np.full((24, 24), 290.0, dtype=np.float32)
    attrs0 = netcdf_attrs(source_frames=["x"], t=0.0, model="obs", model_version="v0")
    attrs1 = netcdf_attrs(source_frames=["x"], t=1.0, model="obs", model_version="v0")
    p0 = write_netcdf(f0, g.lat_coords(), g.lon_coords(), "2025-06-20T00:00:00", attrs0, tmp_path / "a.nc")
    p1 = write_netcdf(f1, g.lat_coords(), g.lon_coords(), "2025-06-20T00:20:00", attrs1, tmp_path / "b.nc")

    out = interpolate_pair_nc(p0, p1, t=0.5, model=BlendModel(), out_nc=tmp_path / "mid.nc")
    assert out.exists()
    ds = read_netcdf(out)
    try:
        mid = ds[InferenceNetCDFSchema.DATA_VAR].values[0]
        assert float(np.mean(mid)) == pytest.approx(260.0, abs=0.1)  # blend of 230 & 290
        for key in InferenceNetCDFSchema.REQUIRED_ATTRS:
            assert key in ds.attrs
        # The synthesized instant's time is the midpoint 00:10.
        assert np.datetime64(ds["time"].values.reshape(-1)[0]) == np.datetime64("2025-06-20T00:10:00")
    finally:
        ds.close()


def test_interpolate_cube_writes_nc_per_frame(synthetic_cube) -> None:
    """batch.interpolate_cube densifies a real Zarr cube and writes one .nc per output frame."""
    from frameflow.infer.batch import interpolate_cube
    from frameflow.infer.netcdf_io import read_netcdf

    res = interpolate_cube(
        synthetic_cube.cube_path, BlendModel(),
        out_dir=str(synthetic_cube.cube_path.parent / "interp_nc"), factor=2,
    )
    n_obs = synthetic_cube.n_frames
    assert len(res.nc_paths) == (n_obs - 1) * 2 + 1
    assert all(p.exists() for p in res.nc_paths)
    # Observed + interpolated tags present in order.
    kinds = [f.kind for f in res.frames]
    assert kinds[0] == "observed" and kinds[1] == "interpolated"
    # A written .nc reads back with the right dims.
    ds = read_netcdf(res.nc_paths[1])
    try:
        assert tuple(ds["bt"].dims) == ("time", "y", "x")
    finally:
        ds.close()


# ===========================================================================
# export.to_onnx: file exported + dynamic BATCH dim on `t` (P2-ONNX)
# ===========================================================================
def test_to_onnx_exports_with_dynamic_batch_t(tmp_path) -> None:
    """to_onnx writes a graph whose `t` input has a dynamic (symbolic) batch dim (P2-ONNX).

    Requires the optional ``onnx`` package both to perform the export (torch needs it to
    serialize) and to inspect the graph; the whole test is skipped if it is absent.
    """
    onnx = pytest.importorskip("onnx")  # absent -> skip (env note: onnx may be missing)

    from frameflow.infer.export import to_onnx

    out = to_onnx(BlendModel(), tmp_path / "vfi.onnx", sample_hw=(16, 16), opset=17,
                  channels=1, batch=1)
    assert out.exists() and out.stat().st_size > 0

    graph = onnx.load(str(out))
    onnx.checker.check_model(graph)

    # Locate the `t` input and assert its first (batch) dim is dynamic (a dim_param, not a
    # fixed dim_value). The whole point of P2-ONNX is that Triton can batch along t's axis 0.
    inputs = {i.name: i for i in graph.graph.input}
    assert "t" in inputs, f"expected a 't' input; got {list(inputs)}"
    t_dims = inputs["t"].type.tensor_type.shape.dim
    assert len(t_dims) == 2, "t must be rank-2 (B, 1) — a leading BATCH dim, not a scalar"
    batch_dim = t_dims[0]
    assert batch_dim.dim_param != "", "t's batch dim must be dynamic (dim_param), not fixed"

    # The image inputs and the output must also have a dynamic batch dim.
    for name in ("img0", "img1", "mid"):
        node = (inputs.get(name)
                or next((o for o in graph.graph.output if o.name == name), None))
        assert node is not None, f"missing graph io: {name}"
        assert node.type.tensor_type.shape.dim[0].dim_param != "", f"{name} batch dim not dynamic"


def test_to_onnx_rejects_low_opset(tmp_path) -> None:
    """opset < 16 would break grid_sample export (RIFE/IFRNet backward warp) -> ValueError."""
    from frameflow.infer.export import to_onnx

    with pytest.raises(ValueError):
        to_onnx(BlendModel(), tmp_path / "vfi.onnx", opset=15)


def test_export_helpers_document_no_int8() -> None:
    """The TensorRT/Triton helpers must emit FP16 and never offer INT8 (research/06 §1.4)."""
    from frameflow.infer.export import to_tensorrt, triton_config_pbtxt

    cmd = to_tensorrt("vfi.onnx", "vfi.plan", fp16=True)
    assert "--fp16" in cmd
    assert "int8" not in cmd.lower()
    # t carries its batch axis in the trtexec shapes (e.g. "t:1x1").
    assert "t:" in cmd

    cfg = triton_config_pbtxt(max_batch_size=16)
    assert "dynamic_batching" in cfg
    assert "int8" not in cfg.lower()

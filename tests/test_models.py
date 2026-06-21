"""Tests for Team MODELS — ``frameflow/models/**`` (CONTRACTS.md §4).

Self-contained and CPU-only: this suite imports ONLY ``frameflow.models`` (and the shared
foundation it transitively needs), never another team's area, and builds all inputs from
small random tensors / numpy arrays so the whole file runs in well under a few seconds on
CPU. It verifies the load-bearing model contracts:

    * :func:`frameflow.models.warp.backward_warp` — zero-flow is an EXACT identity, a known
      pixel shift warps correctly, gradients flow into both image and flow, single-channel OK.
    * :class:`frameflow.models.ifnet.IFNet` — the PRIMARY engine: ``forward(I0, I1, t)`` with
      ``I0, I1`` shaped ``(B, 1, H, W)`` and **``t`` a ``(B, 1)`` tensor** (P2-ONNX) returns
      ``{"pred":(B,1,H,W), "flow":(B,4,H,W), "mask":(B,1,H,W)}``; a python ``float`` ``t`` also
      works; ``pred.mean().backward()`` populates grads on EVERY parameter (trainable); the
      untrained forward is ~the average blend; runs at 64–128 px on CPU.
    * :mod:`frameflow.models.baselines` — ``linear_blend(I0,I1,0.5) == 0.5*(I0+I1)`` exactly;
      Farnebäck/TV-L1/classical dispatchers run on small numpy single-channel frames.
    * :mod:`frameflow.models.flow` — Middlebury color, sparse vectors, color wheel.
    * :mod:`frameflow.models.pretrained` — graceful load with no weights; single-channel adapt.
    * :func:`frameflow.models.registry.get_model` — ``'ifnet'`` and ``'linear'`` return objects
      whose ``forward`` / ``__call__`` give a dict containing ``'pred'``; they satisfy the
      :class:`~frameflow.models.registry.VFIModel` protocol.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from frameflow.models import baselines as B
from frameflow.models import flow as FL
from frameflow.models.ifnet import IFBlock, IFNet
from frameflow.models.pretrained import (
    PRETRAINED_SOURCES,
    adapt_to_single_channel,
    load_pretrained,
)
from frameflow.models.registry import (
    MODEL_NAMES,
    ClassicalBaseline,
    VFIModel,
    get_model,
)
from frameflow.models.warp import backward_warp, make_base_grid

# Keep everything tiny + deterministic.
torch.manual_seed(0)
np.random.seed(0)


# ===========================================================================
# warp.backward_warp
# ===========================================================================
class TestBackwardWarp:
    def test_zero_flow_is_exact_identity(self) -> None:
        """A zero flow must reproduce the input EXACTLY (the core warp invariant)."""
        img = torch.randn(2, 1, 32, 32)
        flow = torch.zeros(2, 2, 32, 32)
        out = backward_warp(img, flow)
        assert out.shape == img.shape
        assert torch.allclose(out, img, atol=1e-6)

    def test_zero_flow_identity_multichannel(self) -> None:
        """Zero-flow identity also holds for C != 1 (e.g. 3-channel adaptations)."""
        img = torch.randn(1, 3, 16, 24)
        out = backward_warp(img, torch.zeros(1, 2, 16, 24))
        assert torch.allclose(out, img, atol=1e-6)

    def test_known_pixel_shift(self) -> None:
        """``out[y, x] = img[y, x + dx]``: a constant dx=+2 shifts a bright pixel left by 2."""
        img = torch.zeros(1, 1, 8, 8)
        img[0, 0, 4, 4] = 1.0
        flow = torch.zeros(1, 2, 8, 8)
        flow[:, 0, :, :] = 2.0  # dx = +2 (sample from x+2)
        out = backward_warp(img, flow, padding_mode="zeros")
        ys, xs = torch.where(out[0, 0] > 0.5)
        assert ys.tolist() == [4]
        assert xs.tolist() == [2]  # bright pixel moved from x=4 to x=2

    def test_gradients_flow_into_image_and_flow(self) -> None:
        """grid_sample is differentiable: grads must reach BOTH the image and the flow."""
        img = torch.randn(1, 1, 12, 12, requires_grad=True)
        # Build flow as a LEAF tensor that requires grad (a trailing op would make it
        # non-leaf and its .grad would not be populated).
        flow = (torch.randn(1, 2, 12, 12) * 0.5).requires_grad_(True)
        out = backward_warp(img, flow)
        out.mean().backward()
        assert img.grad is not None and img.grad.abs().sum() > 0
        assert flow.grad is not None and flow.grad.abs().sum() > 0

    def test_make_base_grid_shape_and_range(self) -> None:
        grid = make_base_grid(10, 14)
        assert grid.shape == (10, 14, 2)
        # align_corners=False pixel centres stay strictly inside (-1, 1).
        assert grid.min() > -1.0 and grid.max() < 1.0

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            backward_warp(torch.randn(1, 1, 8), torch.zeros(1, 2, 8, 8))  # img not 4-D
        with pytest.raises(ValueError):
            backward_warp(torch.randn(1, 1, 8, 8), torch.zeros(1, 3, 8, 8))  # flow !=2 ch
        with pytest.raises(ValueError):
            backward_warp(torch.randn(1, 1, 8, 8), torch.zeros(1, 2, 4, 4))  # mismatched H,W


# ===========================================================================
# ifnet.IFNet — the primary engine
# ===========================================================================
class TestIFNet:
    def test_forward_batched_t_shapes(self) -> None:
        """forward(I0, I1, t) with t shaped (B,1) returns the documented dict + shapes."""
        m = IFNet(in_channels=1)
        I0 = torch.randn(2, 1, 64, 64)
        I1 = torch.randn(2, 1, 64, 64)
        t = torch.tensor([[0.3], [0.7]])  # (B, 1) — the P2-ONNX batched form
        out = m(I0, I1, t)
        assert set(out) >= {"pred", "flow", "mask"}
        assert out["pred"].shape == (2, 1, 64, 64)
        assert out["flow"].shape == (2, 4, 64, 64)
        assert out["mask"].shape == (2, 1, 64, 64)
        # mask is a (0,1) fusion weight (post-sigmoid).
        mask = out["mask"].detach()
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
        # outputs are finite (no NaN/Inf from the warps/blend).
        assert torch.isfinite(out["pred"]).all()
        assert torch.isfinite(out["flow"]).all()

    def test_scalar_float_t_also_works(self) -> None:
        """A python float t is accepted and broadcast across the batch (convenience form)."""
        m = IFNet(in_channels=1)
        I0 = torch.randn(2, 1, 64, 64)
        I1 = torch.randn(2, 1, 64, 64)
        out = m(I0, I1, 0.5)
        assert out["pred"].shape == (2, 1, 64, 64)
        assert torch.isfinite(out["pred"]).all()

    def test_t_as_1d_and_scalar_tensor(self) -> None:
        """A (B,) vector and a 0-d scalar tensor for t are also handled (broadcast to (B,1))."""
        m = IFNet(in_channels=1)
        I0 = torch.randn(3, 1, 32, 32)
        I1 = torch.randn(3, 1, 32, 32)
        out_vec = m(I0, I1, torch.tensor([0.2, 0.5, 0.8]))  # (B,)
        out_scalar = m(I0, I1, torch.tensor(0.5))  # 0-d
        assert out_vec["pred"].shape == (3, 1, 32, 32)
        assert out_scalar["pred"].shape == (3, 1, 32, 32)

    def test_backward_populates_grads_on_every_param(self) -> None:
        """`pred.mean().backward()` must populate a NONZERO grad on EVERY parameter.

        This is the trainability contract: the model fine-tunes end-to-end. (The earlier
        zero-flow-init on the final block silently killed the gradient into that block's
        feature trunk; this guards the fix.)
        """
        m = IFNet(in_channels=1)
        I0 = torch.randn(2, 1, 64, 64)
        I1 = torch.randn(2, 1, 64, 64)
        t = torch.tensor([[0.3], [0.7]])
        out = m(I0, I1, t)
        out["pred"].mean().backward()
        dead = [
            name
            for name, p in m.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0
        ]
        assert dead == [], f"parameters with zero/None grad (untrainable): {dead}"

    def test_grads_flow_into_inputs(self) -> None:
        """Gradients also reach the input frames (needed for distillation / perceptual losses)."""
        m = IFNet(in_channels=1)
        I0 = torch.randn(1, 1, 32, 32, requires_grad=True)
        I1 = torch.randn(1, 1, 32, 32, requires_grad=True)
        out = m(I0, I1, torch.tensor([[0.5]]))
        out["pred"].sum().backward()
        assert I0.grad is not None and I0.grad.abs().sum() > 0
        assert I1.grad is not None and I1.grad.abs().sum() > 0

    def test_untrained_forward_is_approximately_average_blend(self) -> None:
        """Sensible init: an untrained forward should be ~the simple average of the inputs.

        The flow heads init tiny (≈0 applied flow) and the mask logits ~0 (sigmoid ≈ 0.5),
        so pred ≈ 0.5*(I0+I1) before any training.
        """
        m = IFNet(in_channels=1)
        I0 = torch.randn(2, 1, 64, 64)
        I1 = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            out = m(I0, I1, torch.tensor([[0.3], [0.7]]))
        avg = 0.5 * (I0 + I1)
        assert torch.allclose(out["pred"], avg, atol=5e-2)
        # the untrained applied flow is negligible.
        assert float(out["flow"].abs().max()) < 0.5

    def test_runs_at_128px_on_cpu(self) -> None:
        """The net must be runnable at 128 px on CPU (with the default 3-scale stack)."""
        m = IFNet(in_channels=1)
        I0 = torch.randn(1, 1, 128, 128)
        I1 = torch.randn(1, 1, 128, 128)
        out = m(I0, I1, torch.tensor([[0.5]]))
        assert out["pred"].shape == (1, 1, 128, 128)
        assert torch.isfinite(out["pred"]).all()

    def test_three_channel_variant(self) -> None:
        """in_channels=3 (for RGB / pretrained adaptation) produces 3-channel preds."""
        m = IFNet(in_channels=3)
        I0 = torch.randn(1, 3, 32, 32)
        I1 = torch.randn(1, 3, 32, 32)
        out = m(I0, I1, torch.tensor([[0.5]]))
        assert out["pred"].shape == (1, 3, 32, 32)
        assert out["flow"].shape == (1, 4, 32, 32)  # flow is always bidirectional (4ch)

    def test_forward_pred_convenience(self) -> None:
        m = IFNet(in_channels=1)
        I0 = torch.randn(1, 1, 32, 32)
        I1 = torch.randn(1, 1, 32, 32)
        pred = m.forward_pred(I0, I1, 0.5)
        assert pred.shape == (1, 1, 32, 32)

    def test_params_count_and_repr(self) -> None:
        m = IFNet(in_channels=1)
        assert m.num_parameters() == sum(p.numel() for p in m.parameters())
        assert m.params_m == pytest.approx(m.num_parameters() / 1e6)
        assert m.params_m > 0
        assert "in_channels=1" in repr(m)

    def test_mismatched_input_shapes_raise(self) -> None:
        m = IFNet(in_channels=1)
        with pytest.raises(ValueError):
            m(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 16, 16), 0.5)

    def test_empty_scales_rejected(self) -> None:
        with pytest.raises(ValueError):
            IFNet(in_channels=1, scales=())

    def test_two_scale_stack_runs(self) -> None:
        """A smaller 2-scale stack also runs (used for fast CPU smoke paths)."""
        m = IFNet(in_channels=1, hidden=16, scales=(2, 1))
        out = m(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32), 0.5)
        assert out["pred"].shape == (1, 1, 32, 32)

    def test_ifblock_direct(self) -> None:
        """An IFBlock returns (flow_update (B,4,H,W), mask (B,1,H,W)) at the input resolution."""
        blk = IFBlock(in_planes=2 * 1 + 1, hidden=16, scale=2)
        x = torch.randn(1, 3, 32, 32)
        flow_out, mask_out = blk(x, flow=None)
        assert flow_out.shape == (1, 4, 32, 32)
        assert mask_out.shape == (1, 1, 32, 32)


# ===========================================================================
# baselines
# ===========================================================================
class TestBaselines:
    def test_linear_blend_half_is_exact_average(self) -> None:
        """linear_blend(I0, I1, 0.5) MUST equal 0.5*(I0+I1) exactly (the trivial baseline)."""
        I0 = np.random.rand(1, 32, 32).astype(np.float32)
        I1 = np.random.rand(1, 32, 32).astype(np.float32)
        out = B.linear_blend(I0, I1, 0.5)
        assert np.allclose(out, 0.5 * (I0 + I1))

    def test_linear_blend_endpoints_and_fraction(self) -> None:
        I0 = np.random.rand(16, 16).astype(np.float32)
        I1 = np.random.rand(16, 16).astype(np.float32)
        assert np.allclose(B.linear_blend(I0, I1, 0.0), I0)
        assert np.allclose(B.linear_blend(I0, I1, 1.0), I1)
        assert np.allclose(B.linear_blend(I0, I1, 0.25), 0.75 * I0 + 0.25 * I1)

    def test_linear_blend_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            B.linear_blend(np.zeros((8, 8)), np.zeros((8, 4)), 0.5)

    def test_farneback_runs_on_small_frames(self) -> None:
        I0 = np.random.rand(1, 32, 32).astype(np.float32)
        I1 = np.random.rand(1, 32, 32).astype(np.float32)
        out = B.farneback_interpolate(I0, I1, 0.5)
        assert out.shape == I0.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_classical_tvl1_runs_on_small_frames(self) -> None:
        """classical_interpolate(method='tvl1') must RUN on small numpy frames.

        OpenCV's TV-L1 (``cv2.optflow``) is only in opencv-contrib; on a plain/headless
        build :func:`tvl1_interpolate` gracefully falls back to Farnebäck, so this still
        produces a finite, correctly-shaped frame either way.
        """
        I0 = np.random.rand(1, 24, 24).astype(np.float32)
        I1 = np.random.rand(1, 24, 24).astype(np.float32)
        out = B.classical_interpolate(I0, I1, 0.5, method="tvl1")
        assert out.shape == I0.shape
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("method", ["linear", "farneback", "tvl1"])
    def test_classical_dispatch_all_methods(self, method: str) -> None:
        I0 = np.random.rand(1, 24, 24).astype(np.float32)
        I1 = np.random.rand(1, 24, 24).astype(np.float32)
        out = B.classical_interpolate(I0, I1, 0.3, method=method)
        assert out.shape == I0.shape
        assert np.isfinite(out).all()

    def test_classical_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            B.classical_interpolate(np.zeros((8, 8)), np.zeros((8, 8)), 0.5, method="nope")

    def test_baselines_accept_hw_and_channel_layouts(self) -> None:
        """The numpy helpers accept (H,W), (1,H,W) and (H,W,1) and preserve the layout."""
        hw = np.random.rand(16, 16).astype(np.float32)
        chw = hw[np.newaxis, ...]
        hwc = hw[..., np.newaxis]
        assert B.linear_blend(hw, hw, 0.5).shape == (16, 16)
        assert B.farneback_interpolate(chw, chw, 0.5).shape == (1, 16, 16)
        assert B.farneback_interpolate(hwc, hwc, 0.5).shape == (16, 16, 1)


# ===========================================================================
# flow utilities
# ===========================================================================
class TestFlowUtils:
    def test_flow_to_image_numpy(self) -> None:
        f = (np.random.randn(2, 24, 24) * 3).astype(np.float32)  # (2,H,W)
        img = FL.flow_to_image(f)
        assert img.shape == (24, 24, 3)
        assert img.dtype == np.uint8

    def test_flow_to_image_torch_batched(self) -> None:
        img = FL.flow_to_image(torch.randn(1, 2, 20, 20))  # (1,2,H,W)
        assert img.shape == (20, 20, 3)

    def test_flow_to_image_zero_flow(self) -> None:
        """A zero flow must not divide-by-zero; it renders as ~white (zero magnitude)."""
        img = FL.flow_to_image(np.zeros((2, 8, 8), dtype=np.float32))
        assert img.shape == (8, 8, 3)
        assert np.isfinite(img).all()

    def test_flow_to_vectors(self) -> None:
        f = (np.random.randn(2, 32, 32) * 2).astype(np.float32)
        vecs = FL.flow_to_vectors(f, step=8)
        assert isinstance(vecs, list) and len(vecs) > 0
        assert set(vecs[0]) == {"x", "y", "dx", "dy"}

    def test_flow_to_vectors_bad_step(self) -> None:
        with pytest.raises(ValueError):
            FL.flow_to_vectors(np.zeros((2, 8, 8), dtype=np.float32), step=0)

    def test_color_wheel(self) -> None:
        w = FL.make_color_wheel()
        assert w.ndim == 2 and w.shape[1] == 3
        assert w.dtype == np.uint8


# ===========================================================================
# pretrained loading + single-channel adaptation
# ===========================================================================
class TestPretrained:
    def test_load_pretrained_no_ckpt_is_graceful(self) -> None:
        """No checkpoint -> a usable randomly-initialized IFNet (with a warning), no crash."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = load_pretrained("ifnet", None, in_channels=1)
        assert isinstance(m, IFNet)
        out = m(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32), 0.5)
        assert out["pred"].shape == (1, 1, 32, 32)

    def test_load_pretrained_missing_ckpt_is_graceful(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = load_pretrained("rife", "/no/such/checkpoint.pkl", in_channels=1)
        assert isinstance(m, IFNet)

    def test_load_pretrained_missing_ckpt_strict_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pretrained("ifnet", "/no/such/checkpoint.pkl", strict=True)

    def test_load_pretrained_roundtrip_state_dict(self, tmp_path) -> None:
        """A checkpoint saved FROM an IFNet loads back fully (name+shape match)."""
        ref = IFNet(in_channels=1, hidden=16, scales=(2, 1))
        ckpt = tmp_path / "ifnet.pth"
        torch.save(ref.state_dict(), str(ckpt))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = load_pretrained("ifnet", str(ckpt), in_channels=1, hidden=16, scales=(2, 1))
        for (kr, vr), (km, vm) in zip(
            ref.state_dict().items(), m.state_dict().items(), strict=False
        ):
            assert kr == km
            assert torch.allclose(vr, vm)

    def test_adapt_to_single_channel_average(self) -> None:
        """A 6-ch (two stacked RGB frames) stem collapses to a 2-ch single-channel stem."""
        import torch.nn as nn

        class RGBNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Conv2d(6, 8, 3, padding=1)
                self.head = nn.Conv2d(8, 1, 3, padding=1)

        net = RGBNet()
        adapted = adapt_to_single_channel(net, mode="average")
        assert adapted.stem.in_channels == 2  # 6 // 3
        y = adapted.head(adapted.stem(torch.randn(1, 2, 16, 16)))
        assert y.shape == (1, 1, 16, 16)

    def test_adapt_to_single_channel_no_rgb_stem_raises(self) -> None:
        import torch.nn as nn

        class GrayNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Conv2d(1, 4, 3, padding=1)

        with pytest.raises(ValueError):
            adapt_to_single_channel(GrayNet())

    def test_pretrained_sources_documented(self) -> None:
        assert "rife" in PRETRAINED_SOURCES and PRETRAINED_SOURCES["rife"]


# ===========================================================================
# registry.get_model + VFIModel protocol
# ===========================================================================
class TestRegistry:
    def test_get_model_ifnet_returns_dict_with_pred(self) -> None:
        m = get_model("ifnet")
        assert isinstance(m, IFNet)
        out = m.forward(
            torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32), torch.tensor([[0.3], [0.7]])
        )
        assert isinstance(out, dict) and "pred" in out
        assert out["pred"].shape == (2, 1, 32, 32)

    def test_get_model_linear_returns_dict_with_pred(self) -> None:
        m = get_model("linear")
        assert isinstance(m, ClassicalBaseline)
        out = m(
            torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32), torch.tensor([[0.3], [0.7]])
        )
        assert isinstance(out, dict) and "pred" in out
        assert out["pred"].shape == (2, 1, 32, 32)

    def test_get_model_satisfies_vfimodel_protocol(self) -> None:
        """Both the learned model and a baseline structurally satisfy VFIModel."""
        assert isinstance(get_model("ifnet"), VFIModel)
        assert isinstance(get_model("linear"), VFIModel)

    def test_rife_is_alias_for_ifnet(self) -> None:
        assert isinstance(get_model("rife", hidden=16, scales=(2, 1)), IFNet)

    @pytest.mark.parametrize("name", ["farneback", "tvl1"])
    def test_get_model_classical_baselines(self, name: str) -> None:
        m = get_model(name)
        out = m(torch.randn(1, 1, 24, 24), torch.randn(1, 1, 24, 24), 0.5)
        assert "pred" in out and out["pred"].shape == (1, 1, 24, 24)

    def test_get_model_case_insensitive(self) -> None:
        assert isinstance(get_model("IFNet"), IFNet)
        assert isinstance(get_model("LINEAR"), ClassicalBaseline)

    def test_get_model_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            get_model("not-a-model")

    def test_model_names_listed(self) -> None:
        assert "ifnet" in MODEL_NAMES and "linear" in MODEL_NAMES

    def test_classical_baseline_per_element_t(self) -> None:
        """The baseline wrapper reads a per-sample scalar t from a (B,1) tensor."""
        m = ClassicalBaseline("linear")
        I0 = torch.randn(2, 1, 16, 16)
        I1 = torch.randn(2, 1, 16, 16)
        out = m(I0, I1, torch.tensor([[0.0], [1.0]]))
        # element 0 uses t=0 -> I0; element 1 uses t=1 -> I1.
        assert torch.allclose(out["pred"][0], I0[0], atol=1e-5)
        assert torch.allclose(out["pred"][1], I1[1], atol=1e-5)


# ===========================================================================
# package exports (CONTRACTS.md §4)
# ===========================================================================
def test_package_exports() -> None:
    import frameflow.models as M

    for name in ("IFNet", "VFIModel", "get_model", "backward_warp"):
        assert name in M.__all__
        assert hasattr(M, name)

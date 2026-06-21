"""Tests for Team TRAIN — :mod:`frameflow.train` (losses, datamodule, module, runner).

These tests are deliberately SELF-CONTAINED and CPU-FAST (whole file < 30 s):

    * They never import the DATA or MODELS areas. Instead they use a tiny in-test dummy
      ``nn.Module`` (a trainable conv that refines a ``t``-blend of ``I0``/``I1``) and small
      in-test triplets — either random tensors or fields from the foundation-owned
      :mod:`frameflow.synthetic` generator (NOT ``frameflow.data``).
    * They exercise the real code paths: every loss returns a finite scalar carrying a
      gradient; :class:`~frameflow.train.losses.CombinedLoss` respects its weights; and a
      :class:`~frameflow.train.module.VFIModule` wrapping the dummy model trains for four
      steps via a real :class:`pytorch_lightning.Trainer` and overfits a single batch (the
      last training-step loss is below the first).
    * Validation metrics are asserted to use the FIXED Kelvin ``data_range`` (code-review
      correction P1) — never per-image min/max.

PyTorch Lightning is required for the module/runner tests; they are skipped (not failed) if
it is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from frameflow import constants as C
from frameflow.train import losses as L

pl = pytest.importorskip("pytorch_lightning")

# Keep the CPU thread count modest so the (many small) tensor ops in the loss/training tests
# don't thrash on busy CI machines; this keeps the whole file comfortably fast.
torch.set_num_threads(min(4, torch.get_num_threads()))


# ===========================================================================
# In-test dummies (NO dependency on frameflow.data / frameflow.models)
# ===========================================================================
class _DummyVFI(nn.Module):
    """A minimal, genuinely-trainable VFI backbone for the tests.

    Implements the VFI contract ``forward(I0, I1, t) -> It`` with ``t`` shaped ``(B, 1)``
    (the P2-ONNX leading-batch-dim convention). The output is the linear ``t``-blend of the
    two inputs plus a learnable convolutional refinement, so the loss can be reduced by
    gradient descent (it is initialised to the identity blend, i.e. zero refinement):

        ``out = blend + conv([I0, I1, blend])``,  ``blend = (1-t)*I0 + t*I1``.
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.refine = nn.Conv2d(3 * channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = t.view(t.shape[0], 1, 1, 1).to(I0.dtype)
        blend = (1.0 - tt) * I0 + tt * I1
        return blend + self.refine(torch.cat([I0, I1, blend], dim=1))


class _TripletDataset(Dataset):
    """A tiny in-memory dataset of :class:`frameflow.contracts.Sample`-style dicts.

    Each item is ``{"I0", "I1", "It", "t", "meta"}`` with single-channel ``(1, H, W)``
    float32 tensors in ``[0, 1]`` and a ``meta["time"]`` index (so the datamodule's
    time-based split has a timestamp to order by).
    """

    def __init__(self, triplets: list[dict[str, object]]) -> None:
        self._items = triplets

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self._items[idx]


def _make_random_triplets(
    n: int = 16, hw: int = 32, *, seed: int = 0
) -> list[dict[str, object]]:
    """Build ``n`` random ``(I0, I1, It, t)`` triplets with a learnable target residual.

    The target ``It`` is the ``0.5`` blend plus a structured (but fixed) spatial residual,
    so it is genuinely NOT the linear blend of the inputs — there is real signal for the
    dummy model to fit, which lets the overfit-one-batch sanity check reduce the loss. All
    tensors are float32 in ``[0, 1]`` with shape ``(1, hw, hw)``.
    """
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0.0, 1.0, hw).view(1, hw, 1).expand(1, hw, hw)
    items: list[dict[str, object]] = []
    for i in range(n):
        I0 = torch.rand(1, hw, hw, generator=g)
        I1 = torch.rand(1, hw, hw, generator=g)
        # Non-trivial, learnable target: blend + a structured residual (≠ linear blend).
        It = (0.5 * I0 + 0.5 * I1 + 0.3 * torch.sin(5.0 * yy)).clamp(0.0, 1.0)
        items.append({"I0": I0, "I1": I1, "It": It, "t": 0.5, "meta": {"time": i}})
    return items


def _make_synthetic_triplets(n_frames: int = 18, hw: int = 32) -> list[dict[str, object]]:
    """Build triplets from the foundation :mod:`frameflow.synthetic` generator.

    Uses the moving-cloud TIR field generator (NOT the DATA area) to produce ``n_frames``
    consecutive Kelvin frames, normalises them to ``[0, 1]`` with the fixed input range, and
    forms ``(i-1, i, i+1)`` triplets with ``t = 0.5`` (so ``It`` is the true middle frame —
    a non-linear interpolation target). Returns ``n_frames - 2`` triplets.
    """
    from frameflow.contracts import GridSpec
    from frameflow.synthetic import _sample_blobs, generate_bt_field

    grid = GridSpec(
        west=68.0, south=6.0, east=98.0, north=38.0,
        n_rows=hw, n_cols=hw, resolution_deg=30.0 / hw,
    )
    rng = np.random.default_rng(0)
    blobs = _sample_blobs(rng, 3)
    span = C.BT_NORM_VMAX_K - C.BT_NORM_VMIN_K
    fields = []
    for i in range(n_frames):
        t_norm = i / max(n_frames - 1, 1)
        f = generate_bt_field(t_norm, grid, rng, n_blobs=3, blob_params=blobs)
        f = np.nan_to_num(np.asarray(f, dtype=np.float32), nan=float(C.BT_NORM_VMIN_K))
        fields.append(np.clip((f - C.BT_NORM_VMIN_K) / span, 0.0, 1.0).astype(np.float32))
    items: list[dict[str, object]] = []
    for i in range(1, n_frames - 1):
        items.append(
            {
                "I0": torch.from_numpy(fields[i - 1])[None],
                "I1": torch.from_numpy(fields[i + 1])[None],
                "It": torch.from_numpy(fields[i])[None],
                "t": 0.5,
                "meta": {"time": i},
            }
        )
    return items


# ===========================================================================
# 1. Losses — finite scalar with grad
# ===========================================================================
@pytest.mark.parametrize("hw", [16, 32])
def test_individual_losses_finite_and_differentiable(hw: int) -> None:
    """Each loss returns a finite scalar tensor that carries a usable gradient."""
    torch.manual_seed(0)
    gt = torch.rand(2, 1, hw, hw)

    for fn in (L.charbonnier_loss, L.census_loss, L.gradient_loss):
        pred = torch.rand(2, 1, hw, hw, requires_grad=True)
        value = fn(pred, gt)
        assert value.shape == (), f"{fn.__name__} must return a scalar"
        assert torch.isfinite(value).item(), f"{fn.__name__} produced a non-finite value"
        value.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0, f"{fn.__name__} produced a zero gradient"

    # MS-SSIM takes an explicit fixed data_range (piq requirement).
    pred = torch.rand(2, 1, hw, hw, requires_grad=True)
    value = L.ms_ssim_loss(pred, gt, data_range=1.0)
    assert value.shape == ()
    assert torch.isfinite(value).item()
    value.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_ms_ssim_loss_uses_fixed_data_range() -> None:
    """``ms_ssim_loss`` is perfect (loss ≈ 0) for identical inputs and ≥0 otherwise."""
    torch.manual_seed(1)
    x = torch.rand(2, 1, 32, 32)
    same = L.ms_ssim_loss(x, x, data_range=1.0)
    diff = L.ms_ssim_loss(x, torch.rand(2, 1, 32, 32), data_range=1.0)
    assert float(same) == pytest.approx(0.0, abs=1e-4)
    assert float(diff) > float(same)


def test_charbonnier_zero_for_identical_inputs() -> None:
    """Charbonnier of identical inputs equals ``eps`` (its floor), and is positive."""
    x = torch.rand(2, 1, 16, 16)
    eps = 1e-3
    value = L.charbonnier_loss(x, x, eps=eps)
    assert float(value) == pytest.approx(eps, abs=1e-5)


def test_flow_distillation_optional_and_finite() -> None:
    """Flow-distillation returns zero when flows are absent, and a finite grad value else."""
    zero = L.flow_distillation_loss(None, None)
    assert float(zero) == 0.0

    student = torch.rand(2, 2, 16, 16, requires_grad=True)
    teacher = torch.rand(2, 2, 16, 16)
    value = L.flow_distillation_loss(student, teacher)
    assert value.shape == () and torch.isfinite(value).item()
    value.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_losses_on_synthetic_triplets_are_finite() -> None:
    """All losses are finite on real synthetic moving-cloud TIR triplets (domain check)."""
    triplets = _make_synthetic_triplets(n_frames=6, hw=32)
    I0 = torch.stack([t["I0"] for t in triplets])  # type: ignore[misc]
    It = torch.stack([t["It"] for t in triplets])  # type: ignore[misc]
    for fn in (L.charbonnier_loss, L.census_loss, L.gradient_loss):
        assert torch.isfinite(fn(I0, It)).item()
    assert torch.isfinite(L.ms_ssim_loss(I0, It, data_range=1.0)).item()


# ===========================================================================
# 2. CombinedLoss — respects weights
# ===========================================================================
def test_combined_loss_returns_total_and_parts() -> None:
    """``CombinedLoss`` returns a finite scalar total and a dict of its components."""
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    gt = torch.rand(2, 1, 32, 32)
    total, parts = L.CombinedLoss()(pred, gt)
    assert total.shape == () and torch.isfinite(total).item()
    assert set(parts) == {"charbonnier", "census", "ms_ssim", "gradient"}
    for name, value in parts.items():
        assert torch.isfinite(value).item(), f"part {name} is non-finite"
    total.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_combined_loss_respects_weights() -> None:
    """Zero weights ⇒ zero total; a single unit weight ⇒ exactly that component's value."""
    torch.manual_seed(2)
    pred = torch.rand(2, 1, 32, 32)
    gt = torch.rand(2, 1, 32, 32)

    zero_w = {
        "loss_charbonnier": 0.0, "loss_census": 0.0,
        "loss_ms_ssim": 0.0, "loss_gradient": 0.0,
    }
    total_zero, _ = L.CombinedLoss(zero_w)(pred, gt)
    assert float(total_zero) == pytest.approx(0.0, abs=1e-7)

    # Only Charbonnier active ⇒ total must equal the standalone Charbonnier term.
    char_w = dict(zero_w, loss_charbonnier=1.0, charbonnier_eps=1e-3)
    total_char, parts = L.CombinedLoss(char_w)(pred, gt)
    expected = L.charbonnier_loss(pred, gt, eps=1e-3)
    assert float(total_char) == pytest.approx(float(expected), rel=1e-5)
    assert float(parts["charbonnier"]) == pytest.approx(float(expected), rel=1e-5)

    # Doubling a single weight doubles the total (linearity in the weights).
    double_char = dict(char_w, loss_charbonnier=2.0)
    total_double, _ = L.CombinedLoss(double_char)(pred, gt)
    assert float(total_double) == pytest.approx(2.0 * float(total_char), rel=1e-5)


def test_combined_loss_accepts_trainconfig_weights() -> None:
    """``CombinedLoss`` can be configured from a :class:`frameflow.config.TrainConfig`."""
    from frameflow.config import TrainConfig

    cfg = TrainConfig()
    crit = L.CombinedLoss(cfg)
    assert crit.weights["loss_charbonnier"] == pytest.approx(cfg.loss_charbonnier)
    assert crit.weights["loss_census"] == pytest.approx(cfg.loss_census)
    assert crit.weights["loss_ms_ssim"] == pytest.approx(cfg.loss_ms_ssim)
    assert crit.weights["loss_gradient"] == pytest.approx(cfg.loss_gradient)


# ===========================================================================
# 3. DataModule — strict time-based split (no leakage)
# ===========================================================================
def test_time_based_split_is_contiguous_and_disjoint() -> None:
    """The split helper yields contiguous, time-ordered, non-overlapping index blocks."""
    from frameflow.train.datamodule import time_based_split_indices

    timestamps = list(range(10))
    train, val, test = time_based_split_indices(timestamps, train_frac=0.6, val_frac=0.2)
    # Disjoint and complete.
    assert set(train) & set(val) == set()
    assert set(val) & set(test) == set()
    assert sorted(train + val + test) == timestamps
    # Time-ordered: every train timestamp precedes every val/test timestamp (no leakage).
    assert max(train) < min(val)
    assert max(val) < min(test)


def test_datamodule_injected_split_no_overlap() -> None:
    """Injected dataset is split by time into non-overlapping train/val subsets."""
    from frameflow.train.datamodule import VFIDataModule

    ds = _TripletDataset(_make_random_triplets(n=16, hw=16))
    dm = VFIDataModule(dataset=ds, batch_size=4, num_workers=0, train_frac=0.75, val_frac=0.25)
    dm.setup()
    train_times = {ds[i]["meta"]["time"] for i in dm.train_dataset.indices}  # type: ignore[union-attr]
    val_times = {ds[i]["meta"]["time"] for i in dm.val_dataset.indices}  # type: ignore[union-attr]
    assert len(train_times) > 0 and len(val_times) > 0
    assert train_times & val_times == set(), "train/val splits must not share timestamps"
    # Time-based: the latest train timestamp precedes the earliest val timestamp.
    assert max(train_times) < min(val_times)


def test_datamodule_dataloaders_yield_batches() -> None:
    """The datamodule produces working train/val dataloaders over Sample dicts."""
    from frameflow.train.datamodule import VFIDataModule

    ds = _TripletDataset(_make_random_triplets(n=16, hw=16))
    dm = VFIDataModule(dataset=ds, batch_size=4, num_workers=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert set(["I0", "I1", "It", "t"]).issubset(batch.keys())
    assert batch["I0"].shape[1:] == (1, 16, 16)
    assert next(iter(dm.val_dataloader())) is not None


# ===========================================================================
# 4. VFIModule — validation uses FIXED-range metrics (P1)
# ===========================================================================
def test_module_uses_default_model_when_none() -> None:
    """``VFIModule(model=None)`` falls back to a runnable built-in backbone."""
    from frameflow.train.module import VFIModule

    mod = VFIModule(model=None, in_channels=1)
    assert isinstance(mod.model, nn.Module)
    I0 = torch.rand(2, 1, 16, 16)
    I1 = torch.rand(2, 1, 16, 16)
    t = torch.full((2, 1), 0.5)
    out = mod(I0, I1, t)
    assert out.shape == (2, 1, 16, 16)


def test_validation_psnr_uses_fixed_kelvin_range_not_per_image() -> None:
    """Validation PSNR uses the FIXED Kelvin ``data_range`` (P1), never per-image min/max.

    We compare the module's fixed-range PSNR against a hand-computed PSNR using the SHARED
    physical peak ``BT_DATA_RANGE_K``. If the implementation (wrongly) used per-image
    min/max, the numbers would diverge for a low-contrast frame whose own range is far
    smaller than the fixed physical span.
    """
    from frameflow.train.module import VFIModule

    mod = VFIModule(model=_DummyVFI(), normalized_io=False)  # inputs already in Kelvin

    torch.manual_seed(0)
    # A LOW-CONTRAST Kelvin field: its own min/max span (~few K) ≪ the fixed 140 K span, so
    # per-image vs fixed-range PSNR would disagree markedly.
    gt_k = 250.0 + 2.0 * torch.rand(1, 1, 32, 32)
    pred_k = gt_k + 0.5 * torch.randn(1, 1, 32, 32)

    psnr, ssim = mod._fixed_range_psnr_ssim(pred_k, gt_k)

    mse = torch.mean((pred_k - gt_k) ** 2)
    expected_psnr = 10.0 * torch.log10(
        torch.tensor(C.BT_DATA_RANGE_K ** 2) / mse
    )
    assert float(psnr) == pytest.approx(float(expected_psnr), rel=1e-4)
    # The module must be configured with the fixed physical range from constants (P1).
    assert mod.bt_data_range_k == pytest.approx(C.BT_DATA_RANGE_K)
    assert mod.bt_vmin_k == pytest.approx(C.BT_METRIC_VMIN_K)
    assert mod.bt_vmax_k == pytest.approx(C.BT_METRIC_VMAX_K)
    assert 0.0 <= float(ssim) <= 1.0 + 1e-4


def test_fixed_range_psnr_differs_from_per_image_psnr() -> None:
    """A guard that fixed-range PSNR is genuinely NOT the per-image-min/max PSNR (P1).

    For a low-contrast frame the per-image data_range is tiny, so a per-image PSNR would be
    much smaller than the fixed-physical-range PSNR. They must differ — proving the module
    does not silently use per-image extremes.
    """
    from frameflow.train.module import VFIModule

    mod = VFIModule(model=_DummyVFI(), normalized_io=False)
    torch.manual_seed(3)
    gt_k = 260.0 + 1.0 * torch.rand(1, 1, 32, 32)
    pred_k = gt_k + 0.3 * torch.randn(1, 1, 32, 32)

    fixed_psnr, _ = mod._fixed_range_psnr_ssim(pred_k, gt_k)

    mse = torch.mean((pred_k - gt_k) ** 2)
    per_image_range = float((gt_k.max() - gt_k.min()).item())
    per_image_psnr = 10.0 * torch.log10(torch.tensor(per_image_range ** 2) / mse)
    # Fixed physical range (140 K) ≫ per-image range (~1 K) ⇒ fixed PSNR is much higher.
    assert float(fixed_psnr) > float(per_image_psnr) + 5.0


def test_configure_optimizers_adamw_cosine() -> None:
    """``configure_optimizers`` returns AdamW + a cosine schedule, even when unattached."""
    from frameflow.train.module import VFIModule

    mod = VFIModule(model=_DummyVFI(), lr=1e-3, lr_min=1e-5, max_epochs=10)
    cfg = mod.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)
    sched = cfg["lr_scheduler"]["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler)


def test_configure_optimizers_excludes_frozen_params() -> None:
    """Freezing a prefix removes those parameters from the optimizer (stage-1 warm-up).

    Mirrors the research/06 §4.7 two-stage transfer recipe: stage 1 freezes the flow
    ``encoder`` and trains only the synthesis ``head``. We use a two-submodule dummy so
    freezing one prefix still leaves trainable parameters (the realistic case).
    """
    from frameflow.train.module import VFIModule

    class _TwoPartVFI(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Conv2d(2, 4, 3, padding=1)  # "flow encoder" (frozen in stage 1)
            self.head = nn.Conv2d(3, 1, 3, padding=1)  # "synthesis head" (trained)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            tt = t.view(t.shape[0], 1, 1, 1).to(I0.dtype)
            blend = (1.0 - tt) * I0 + tt * I1
            _ = self.encoder(torch.cat([I0, I1], dim=1))  # exercised but its grads are frozen
            return blend + self.head(torch.cat([I0, I1, blend], dim=1))

    model = _TwoPartVFI()
    mod = VFIModule(model=model, freeze_prefixes=["encoder"])
    opt = mod.configure_optimizers()["optimizer"]
    opt_params = {id(p) for grp in opt.param_groups for p in grp["params"]}
    # Encoder params are frozen → excluded; head params remain trainable → included.
    for p in model.encoder.parameters():
        assert id(p) not in opt_params
    for p in model.head.parameters():
        assert id(p) in opt_params


# ===========================================================================
# 5. End-to-end training — Trainer.fit completes and overfits one batch
# ===========================================================================
def _build_trainer(callbacks: list) -> pl.Trainer:
    """A tiny CPU Trainer for the overfit-one-batch sanity test (max 4 steps)."""
    return pl.Trainer(
        max_steps=4,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        overfit_batches=1,  # reuse the SAME batch every step ⇒ a true overfit sanity check
        callbacks=callbacks,
    )


class _LossRecorder(pl.Callback):
    """Records the scalar training loss reported at the end of each training batch."""

    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        value = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.losses.append(float(value))


def test_vfimodule_fit_completes_and_overfits_one_batch() -> None:
    """A real ``Trainer.fit`` completes and the last training loss is below the first.

    Wraps the in-test :class:`_DummyVFI` in :class:`~frameflow.train.module.VFIModule`,
    feeds ~16 random triplets through :class:`~frameflow.train.datamodule.VFIDataModule`, and
    trains for four steps on a single repeated batch. The dummy model is initialised to the
    identity blend, so a successful optimisation step must reduce the combined loss — the
    overfit-one-batch sanity signal.
    """
    from frameflow.train.datamodule import VFIDataModule
    from frameflow.train.module import VFIModule

    pl.seed_everything(0, workers=True)
    ds = _TripletDataset(_make_random_triplets(n=16, hw=24, seed=0))
    dm = VFIDataModule(dataset=ds, batch_size=4, num_workers=0)
    module = VFIModule(model=_DummyVFI(), lr=1e-2, normalized_io=True)

    # Snapshot the loss on a fixed batch BEFORE training (robust before/after check).
    dm.setup()
    fixed_batch = next(iter(dm.train_dataloader()))
    module.eval()
    with torch.no_grad():
        i0, i1, it, tt = module._unpack_batch(fixed_batch)
        loss_before = float(module.criterion(module(i0, i1, tt), it)[0])

    recorder = _LossRecorder()
    trainer = _build_trainer([recorder])
    trainer.fit(module, datamodule=dm)

    # 1) fit actually ran four steps.
    assert trainer.global_step == 4
    assert len(recorder.losses) == 4
    assert all(np.isfinite(recorder.losses)), "training produced a non-finite loss"

    # 2) literal requirement: the last training-step loss is below the first.
    assert recorder.losses[-1] < recorder.losses[0], (
        f"overfit sanity failed: first={recorder.losses[0]:.5f} "
        f"last={recorder.losses[-1]:.5f}"
    )

    # 3) robust corroboration: the loss on the fixed batch dropped after training.
    module.eval()
    with torch.no_grad():
        loss_after = float(module.criterion(module(i0, i1, tt), it)[0])
    assert loss_after < loss_before, (
        f"overfit sanity failed: before={loss_before:.5f} after={loss_after:.5f}"
    )


def test_vfimodule_fit_runs_validation_with_fixed_range_metrics() -> None:
    """``Trainer.fit`` with a val loader logs fixed-range PSNR/SSIM without error (P1)."""
    from frameflow.train.datamodule import VFIDataModule
    from frameflow.train.module import VFIModule

    pl.seed_everything(0, workers=True)
    ds = _TripletDataset(_make_random_triplets(n=16, hw=16, seed=1))
    dm = VFIDataModule(dataset=ds, batch_size=4, num_workers=0)
    module = VFIModule(model=_DummyVFI(), lr=1e-2, normalized_io=True)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    trainer.fit(module, datamodule=dm)
    metrics = trainer.callback_metrics
    assert "val_psnr" in metrics and "val_ssim" in metrics
    assert torch.isfinite(torch.as_tensor(metrics["val_psnr"])).item()
    assert torch.isfinite(torch.as_tensor(metrics["val_ssim"])).item()


def test_run_training_smoke(tmp_path) -> None:  # noqa: ANN001
    """The plain ``run_training`` entrypoint fits an injected model+dataset end-to-end."""
    from frameflow.config import FrameFlowConfig
    from frameflow.train.cli import run_training

    cfg = FrameFlowConfig()
    cfg.train.batch_size = 8
    cfg.data.num_workers = 0
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.ckpt_dir = str(tmp_path / "runs" / "ckpt")

    ds = _TripletDataset(_make_random_triplets(n=16, hw=16, seed=0))
    ckpt = run_training(
        cfg,
        model=_DummyVFI(),
        dataset=ds,
        accelerator="cpu",
        devices=1,
        trainer_kwargs={
            "max_steps": 2,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
        },
    )
    # With checkpointing disabled, the entrypoint returns the (stable) ckpt-dir handle.
    from pathlib import Path

    assert isinstance(ckpt, Path)

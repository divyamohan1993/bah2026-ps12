"""FrameFlow training LightningModule — :class:`VFIModule`.

Wraps a VFI backbone (the MODELS team's :class:`frameflow.models.base.VFIModel`, or any
``nn.Module`` with a ``forward(I0, I1, t) -> It`` signature) in a PyTorch Lightning module
that:

    * runs the forward pass on a :class:`frameflow.contracts.Sample` batch,
    * computes the :class:`~frameflow.train.losses.CombinedLoss`
      (Charbonnier + census + MS-SSIM + gradient) and logs each component,
    * computes **validation PSNR & SSIM in physical Kelvin using the FIXED shared
      ``data_range`` :data:`frameflow.constants.BT_DATA_RANGE_K`** — never per-image min/max
      (code-review correction P1),
    * configures AdamW + a cosine LR schedule (with optional linear warm-up),
    * supports fine-tuning from a pretrained checkpoint and optionally freezing early blocks.

BF16 mixed precision is selected at the *Trainer* level (``precision="bf16-mixed"``); this
module is precision-agnostic. See :mod:`frameflow.train.cli` / :mod:`frameflow.train.trainer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch

from .. import config as _config
from .. import constants as C
from .losses import CombinedLoss

if TYPE_CHECKING:  # type-checkers only
    from torch import Tensor


__all__ = ["VFIModule"]


# Backbone working range for normalized [0, 1] model I/O. MS-SSIM in the *training* loss
# uses this; the *reported* validation metrics use the Kelvin range (P1) below.
_MODEL_WORKING_RANGE: float = 1.0


class _TinyBlendModel(torch.nn.Module):
    """A minimal, real VFI backbone used only as a default when none is supplied.

    It produces a learnable refinement of the linear t-blend of the two inputs:
    ``out = blend + conv(cat[I0, I1, blend])`` where ``blend = (1-t)*I0 + t*I1``. It is a
    genuine ``nn.Module`` with trainable parameters and the contract
    ``forward(I0, I1, t) -> It`` (``t`` shaped ``(B, 1)``), so :class:`VFIModule` is usable
    end-to-end even before the MODELS team lands a real backbone. Real training should pass
    a proper :class:`frameflow.models.base.VFIModel`.
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.refine = torch.nn.Conv2d(3 * channels, channels, kernel_size=3, padding=1)
        # Start near identity so the initial output is ~the linear blend.
        torch.nn.init.zeros_(self.refine.weight)
        torch.nn.init.zeros_(self.refine.bias)

    def forward(self, I0: Tensor, I1: Tensor, t: Tensor) -> Tensor:
        # t: (B, 1) -> (B, 1, 1, 1) for broadcasting against (B, C, H, W).
        tt = t.view(t.shape[0], 1, 1, 1).to(I0.dtype)
        blend = (1.0 - tt) * I0 + tt * I1
        feat = torch.cat([I0, I1, blend], dim=1)
        return blend + self.refine(feat)


class VFIModule(pl.LightningModule):
    """LightningModule for fine-tuning a video-frame-interpolation backbone on TIR.

    Args:
        model: a VFI backbone ``nn.Module`` with ``forward(I0, I1, t) -> It`` (``t`` shape
            ``(B, 1)``). If ``None``, the backbone named ``model_name`` is lazy-imported from
            :func:`frameflow.models.registry.get_model`; if that registry is unavailable a
            small built-in :class:`_TinyBlendModel` is used so the module still runs.
        model_name: registry name of the backbone to build when ``model is None``.
        loss_weights: weights for :class:`~frameflow.train.losses.CombinedLoss` — a
            :class:`frameflow.config.TrainConfig`, a plain dict, or ``None`` for defaults.
        lr: peak learning rate for AdamW.
        lr_min: floor learning rate for the cosine schedule.
        weight_decay: AdamW weight decay.
        max_epochs: total epochs (drives the cosine schedule period). May be overridden from
            the Trainer at ``configure_optimizers`` time.
        warmup_epochs: linear LR warm-up length (epochs) before cosine decay.
        bt_data_range_k: FIXED Kelvin range for validation PSNR/SSIM (P1). Defaults to
            :data:`frameflow.constants.BT_DATA_RANGE_K`.
        bt_vmin_k / bt_vmax_k: the physical Kelvin bounds the metric range spans (P1); used
            to de-normalize predictions back to Kelvin when inputs are normalized ``[0,1]``.
        normalized_io: whether the model consumes/produces normalized ``[0, 1]`` tensors (so
            validation must de-normalize to Kelvin before computing fixed-range metrics).
        pretrained: optional path to a checkpoint/state-dict to warm-start the backbone from
            (transfer learning; research/06 §4.7).
        freeze_prefixes: optional list of parameter-name prefixes to freeze (e.g. the flow
            encoder) for stage-1 warm-up fine-tuning; ``None`` trains everything.
    """

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        *,
        model_name: str = "ifnet",
        loss_weights: _config.TrainConfig | dict[str, float] | None = None,
        lr: float = 2e-4,
        lr_min: float = 1e-5,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        warmup_epochs: int = 0,
        bt_data_range_k: float = C.BT_DATA_RANGE_K,
        bt_vmin_k: float = C.BT_METRIC_VMIN_K,
        bt_vmax_k: float = C.BT_METRIC_VMAX_K,
        normalized_io: bool = True,
        pretrained: str | None = None,
        freeze_prefixes: list[str] | None = None,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        # Don't try to hash/pickle a possibly-huge injected model into hparams.
        self.save_hyperparameters(ignore=["model"])

        self.model = model if model is not None else self._build_model(model_name, in_channels)
        self.criterion = CombinedLoss(loss_weights, data_range=_MODEL_WORKING_RANGE)

        self.lr = float(lr)
        self.lr_min = float(lr_min)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(warmup_epochs)

        self.bt_data_range_k = float(bt_data_range_k)
        self.bt_vmin_k = float(bt_vmin_k)
        self.bt_vmax_k = float(bt_vmax_k)
        self.normalized_io = bool(normalized_io)

        # Try to use the VALIDATE team's fixed-range metric if present; else inline ours.
        self._fixed_metrics_fn = self._load_validate_metrics()

        if pretrained:
            self.load_pretrained(pretrained)
        if freeze_prefixes:
            self.freeze_blocks(freeze_prefixes)

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _build_model(model_name: str, in_channels: int) -> torch.nn.Module:
        """Lazy-import the requested backbone; fall back to a tiny built-in blend model."""
        try:
            from ..models.registry import get_model  # lazy; owned by Team MODELS
        except ImportError:
            return _TinyBlendModel(channels=in_channels)
        try:
            return get_model(model_name)  # type: ignore[no-any-return]
        except Exception:
            # Registry exists but couldn't build the named model — stay runnable.
            return _TinyBlendModel(channels=in_channels)

    def _load_validate_metrics(self) -> Any:
        """Return the VALIDATE team's per-frame metric fn if importable, else ``None``."""
        try:
            from ..validate import metrics as _vm  # lazy; owned by Team INFER+VALIDATE
        except ImportError:
            return None
        return getattr(_vm, "per_frame_metrics", None)

    @classmethod
    def from_config(
        cls,
        cfg: _config.FrameFlowConfig,
        *,
        model: torch.nn.Module | None = None,
    ) -> VFIModule:
        """Build a :class:`VFIModule` from a composed :class:`frameflow.config.FrameFlowConfig`.

        Pulls LR/schedule/loss-weight/precision-independent settings from ``cfg.train`` and
        the fixed Kelvin metric range from ``cfg.validate`` (P1), and the pretrained-weights
        path + channel count from ``cfg.model``.

        Args:
            cfg: the composed top-level config.
            model: optional pre-built backbone (overrides ``cfg.model.name``).

        Returns:
            A configured :class:`VFIModule`.
        """
        t = cfg.train
        # Stage-1 warm-up freezes the flow encoder per research/06 §4.7; expose the warmup
        # length so configure_optimizers builds the warm-up→cosine schedule.
        return cls(
            model=model,
            model_name=cfg.model.name,
            loss_weights=t,
            lr=t.lr,
            lr_min=t.lr_min,
            weight_decay=t.weight_decay,
            max_epochs=t.epochs,
            warmup_epochs=t.warmup_epochs,
            bt_data_range_k=cfg.validate.bt_data_range_k,
            bt_vmin_k=cfg.validate.bt_vmin_k,
            bt_vmax_k=cfg.validate.bt_vmax_k,
            normalized_io=(cfg.data.norm_mode != "none"),
            pretrained=(cfg.model.pretrained or None),
            in_channels=cfg.model.in_channels,
        )

    # ------------------------------------------------------------------ transfer learning
    def load_pretrained(self, path: str, *, strict: bool = False) -> None:
        """Warm-start the backbone from a checkpoint / state-dict (transfer learning).

        Accepts a raw ``state_dict``, a Lightning checkpoint (``{"state_dict": ...}`` with
        ``model.`` prefixes), or a plain ``{"model"/"net"/"params": state_dict}`` wrapper.
        Loads non-strictly by default so single-channel-adapted backbones (whose ``conv1``
        differs from the RGB-pretrained one) still load every compatible tensor
        (research/06 §4.7).

        Args:
            path: filesystem path to the weights.
            strict: passed to ``load_state_dict`` (default ``False`` for transfer learning).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = self._extract_state_dict(ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            # Informational only — expected when adapting RGB→1ch (conv1) etc.
            print(
                f"[VFIModule] loaded pretrained weights from {path}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected tensors."
            )

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> dict[str, Tensor]:
        """Normalize the many checkpoint container shapes into a backbone ``state_dict``."""
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "net", "params", "weights"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        if not isinstance(ckpt, dict):
            raise ValueError("Unrecognized checkpoint format for load_pretrained().")
        # Strip a leading "model." (Lightning) prefix if present.
        out: dict[str, Tensor] = {}
        for k, v in ckpt.items():
            nk = k[len("model.") :] if k.startswith("model.") else k
            out[nk] = v
        return out

    def freeze_blocks(self, prefixes: list[str]) -> None:
        """Freeze backbone parameters whose names start with any of ``prefixes``.

        Used for stage-1 fine-tuning (freeze the flow-estimation encoder, train only the
        synthesis/refine head; research/06 §4.7). Frozen parameters get
        ``requires_grad=False`` and are skipped by the optimizer.

        Args:
            prefixes: list of parameter-name prefixes to freeze (matched against
                ``model.<name>``).
        """
        n_frozen = 0
        for name, p in self.model.named_parameters():
            if any(name.startswith(pref) for pref in prefixes):
                p.requires_grad_(False)
                n_frozen += 1
        print(f"[VFIModule] froze {n_frozen} parameter tensors matching {prefixes}.")

    # ------------------------------------------------------------------ forward / batch
    def forward(self, I0: Tensor, I1: Tensor, t: Tensor) -> Tensor:
        """Forward through the backbone: ``(I0, I1, t) -> It`` with ``t`` shape ``(B, 1)``.

        The backbone's raw output is normalized to the predicted-frame tensor: the MODELS
        team's :class:`frameflow.models.base.VFIModel` backbones (e.g. ``ifnet``) return a
        dict ``{"pred", "flow", "mask"}`` rather than a bare tensor, while a plain
        ``nn.Module`` (and the built-in :class:`_TinyBlendModel`) returns the tensor directly.
        :meth:`_extract_pred` handles both so :class:`VFIModule` works with either contract.
        """
        return self._extract_pred(self.model(I0, I1, t))

    @staticmethod
    def _extract_pred(out: Any) -> Tensor:
        """Normalize a backbone output into the predicted-frame tensor.

        Accepts a bare tensor, a ``dict`` carrying the prediction under one of the common
        keys (``pred``/``I_t``/``It``/``out``/``output``/``img``), or a ``tuple``/``list``
        whose first element is the prediction (e.g. ``(It, flow)`` from ``forward_with_flow``).

        Args:
            out: the raw backbone output.

        Returns:
            The predicted-frame :class:`~torch.Tensor`.

        Raises:
            TypeError: if no prediction tensor can be located in ``out``.
        """
        if torch.is_tensor(out):
            return out
        if isinstance(out, dict):
            for key in ("pred", "I_t", "It", "out", "output", "img", "mid"):
                if key in out and torch.is_tensor(out[key]):
                    return out[key]
            raise TypeError(
                f"backbone returned a dict without a known prediction key; got {list(out)}"
            )
        if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
            return out[0]
        raise TypeError(f"could not extract a prediction tensor from backbone output {type(out)!r}")

    def _unpack_batch(self, batch: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract ``(I0, I1, It, t)`` tensors from a Sample-style batch.

        Supports the :class:`frameflow.contracts.Sample` dict layout (keys ``I0``/``I1``/
        ``It``/``t``) produced by the default collate, and a tuple/list fallback
        ``(I0, I1, It, t)``. ``t`` is normalized to shape ``(B, 1)`` (P2-ONNX convention).
        """
        if isinstance(batch, dict):
            I0, I1, It, t = batch["I0"], batch["I1"], batch["It"], batch["t"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 4:
            I0, I1, It, t = batch[0], batch[1], batch[2], batch[3]
        else:
            raise TypeError(
                "VFIModule expects a Sample dict (I0/I1/It/t) or a (I0,I1,It,t) tuple; "
                f"got {type(batch)!r}."
            )

        I0 = self._as_float(I0)
        I1 = self._as_float(I1)
        It = self._as_float(It)
        t = self._as_float(t)
        # Normalize t to shape (B, 1).
        if t.dim() == 0:
            t = t.view(1, 1).expand(I0.shape[0], 1).contiguous()
        elif t.dim() == 1:
            t = t.view(-1, 1)
        elif t.dim() > 2:
            t = t.reshape(t.shape[0], -1)[:, :1]
        return I0, I1, It, t

    def _as_float(self, x: Any) -> Tensor:
        """Coerce an array/tensor to a float tensor on the module's device."""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        return x.to(device=self.device, dtype=torch.float32) if x.dtype != torch.float32 else x.to(self.device)

    # ------------------------------------------------------------------ training
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """One optimization step: forward, combined loss, log components, return loss."""
        I0, I1, It, t = self._unpack_batch(batch)
        pred = self(I0, I1, t)
        loss, parts = self.criterion(pred, It)

        bs = I0.shape[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        for name, value in parts.items():
            self.log(f"train_{name}", value, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    # ------------------------------------------------------------------ validation
    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        """Compute val loss + PSNR/SSIM in FIXED-range Kelvin (P1)."""
        I0, I1, It, t = self._unpack_batch(batch)
        pred = self(I0, I1, t)
        loss, parts = self.criterion(pred, It)

        # ---- FIXED-RANGE metrics (P1) ----------------------------------------------------
        # De-normalize to Kelvin if the model works in [0, 1], then compute PSNR/SSIM with
        # the SHARED, FIXED Kelvin data_range — NEVER per-image min/max (code-review P1).
        pred_k = self._to_kelvin(pred)
        gt_k = self._to_kelvin(It)
        psnr, ssim = self._fixed_range_psnr_ssim(pred_k, gt_k)

        bs = I0.shape[0]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=bs)
        for name, value in parts.items():
            self.log(f"val_{name}", value, on_epoch=True, batch_size=bs)
        self.log("val_psnr", psnr, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val_ssim", ssim, on_epoch=True, prog_bar=True, batch_size=bs)
        return {"val_loss": loss, "val_psnr": psnr, "val_ssim": ssim}

    def _to_kelvin(self, x: Tensor) -> Tensor:
        """Map a model-space tensor to physical Kelvin for metric computation.

        If the pipeline uses normalized ``[0, 1]`` I/O, invert the fixed-range normalization
        ``K = vmin + x * (vmax - vmin)``; otherwise assume the tensor is already Kelvin.
        """
        if not self.normalized_io:
            return x
        span = self.bt_vmax_k - self.bt_vmin_k
        return self.bt_vmin_k + x * span

    def _fixed_range_psnr_ssim(self, pred_k: Tensor, gt_k: Tensor) -> tuple[Tensor, Tensor]:
        """PSNR & SSIM in Kelvin using the FIXED shared ``data_range`` (P1).

        Prefers the VALIDATE team's :func:`frameflow.validate.metrics.per_frame_metrics`
        (which is contractually fixed-range, P1) when available; otherwise computes a correct
        fixed-range PSNR/SSIM inline. In BOTH paths ``data_range`` is the fixed Kelvin span
        :data:`frameflow.constants.BT_DATA_RANGE_K`, never per-image min/max.

        Returns:
            ``(psnr, ssim)`` as scalar tensors on the module's device.
        """
        dr = self.bt_data_range_k

        if self._fixed_metrics_fn is not None:
            psnr_v, ssim_v = self._metrics_via_validate(pred_k, gt_k, dr)
            if psnr_v is not None and ssim_v is not None:
                return (
                    torch.as_tensor(psnr_v, dtype=torch.float32, device=self.device),
                    torch.as_tensor(ssim_v, dtype=torch.float32, device=self.device),
                )

        # ---- inline, correct fixed-range PSNR/SSIM ---------------------------------------
        psnr = self._psnr_fixed_range(pred_k, gt_k, data_range=dr)
        ssim = self._ssim_fixed_range(pred_k, gt_k, data_range=dr, band_floor=self.bt_vmin_k)
        return psnr, ssim

    def _metrics_via_validate(
        self, pred_k: Tensor, gt_k: Tensor, dr: float
    ) -> tuple[float | None, float | None]:
        """Average PSNR/SSIM over the batch via the VALIDATE per-frame metric (fixed-range)."""
        import numpy as np  # lazy

        pred_np = pred_k.detach().to(torch.float32).cpu().numpy()
        gt_np = gt_k.detach().to(torch.float32).cpu().numpy()
        psnrs: list[float] = []
        ssims: list[float] = []
        try:
            for b in range(pred_np.shape[0]):
                # Squeeze channel dim → (H, W) per the per_frame_metrics(pred_k, true_k) API.
                rec = self._fixed_metrics_fn(
                    pred_np[b, 0], gt_np[b, 0], data_range_k=dr
                )
                rec_d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
                if rec_d.get("psnr") is not None:
                    psnrs.append(float(rec_d["psnr"]))
                if rec_d.get("ssim") is not None:
                    ssims.append(float(rec_d["ssim"]))
        except Exception:
            return None, None
        if not psnrs or not ssims:
            return None, None
        return float(np.mean(psnrs)), float(np.mean(ssims))

    @staticmethod
    def _psnr_fixed_range(pred: Tensor, gt: Tensor, *, data_range: float) -> Tensor:
        r"""PSNR with a FIXED peak ``data_range`` (P1): ``10*log10(R^2 / MSE)``.

        The peak ``R`` is the fixed physical Kelvin span, NOT each frame's own extremes, so a
        uniform warm/cold bias actually lowers PSNR (the whole point of P1).
        """
        mse = torch.mean((pred - gt) ** 2)
        eps = torch.finfo(torch.float32).eps
        mse = torch.clamp(mse, min=eps)
        r2 = torch.as_tensor(data_range * data_range, dtype=mse.dtype, device=mse.device)
        return 10.0 * torch.log10(r2 / mse)

    @staticmethod
    def _ssim_fixed_range(
        pred: Tensor, gt: Tensor, *, data_range: float, band_floor: float
    ) -> Tensor:
        r"""SSIM with a FIXED ``data_range`` (P1), via piq when usable, else a Gaussian SSIM.

        ``piq.ssim`` asserts inputs lie within ``[0, data_range]``, so we shift the Kelvin
        values down by the fixed band floor ``band_floor`` (== ``bt_vmin_k``): since
        ``data_range == bt_vmax_k - bt_vmin_k``, subtracting ``bt_vmin_k`` maps the physical
        band ``[vmin, vmax]`` exactly onto ``[0, data_range]`` with NO contrast rescaling
        (clip only guards against out-of-band model output). The SSIM constants ``C1/C2``
        stay tied to the fixed ``data_range``, so this is a true fixed-range SSIM (P1) — not a
        per-image-normalized one. Falls back to an inline Gaussian SSIM with the same fixed
        ``data_range`` if piq is unavailable or the patch is too small for piq's minimum.

        Args:
            pred: predicted frame in Kelvin, shape ``(B, C, H, W)``.
            gt: ground-truth frame in Kelvin, same shape.
            data_range: the fixed Kelvin span (``bt_vmax_k - bt_vmin_k``).
            band_floor: the fixed band minimum in Kelvin (``bt_vmin_k``).
        """
        # Exact, contrast-preserving shift of the FIXED band [vmin, vmax] -> [0, data_range].
        p = torch.clamp(pred - band_floor, 0.0, data_range)
        g = torch.clamp(gt - band_floor, 0.0, data_range)
        min_hw = min(p.shape[-2], p.shape[-1])
        try:
            import piq  # lazy

            ks = 11 if min_hw >= 11 else max(3, ((min_hw - 1) // 2) * 2 + 1)
            val = piq.ssim(p, g, kernel_size=ks, data_range=float(data_range), reduction="mean")
            return val.to(torch.float32)
        except Exception:
            return VFIModule._gaussian_ssim(p, g, data_range=float(data_range))

    @staticmethod
    def _gaussian_ssim(pred: Tensor, gt: Tensor, *, data_range: float, window: int = 11, sigma: float = 1.5) -> Tensor:
        """Inline single-scale Gaussian SSIM with a FIXED ``data_range`` (fallback path)."""
        import torch.nn.functional as F  # lazy

        c = pred.shape[1]
        win = min(window, pred.shape[-2], pred.shape[-1])
        if win % 2 == 0:
            win -= 1
        win = max(win, 3)
        coords = torch.arange(win, dtype=pred.dtype, device=pred.device) - (win - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g1d = g1d / g1d.sum()
        kernel = (g1d[:, None] @ g1d[None, :]).expand(c, 1, win, win).contiguous()
        pad = win // 2

        def filt(x: Tensor) -> Tensor:
            return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=c)

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        mu_x, mu_y = filt(pred), filt(gt)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sig_x2 = filt(pred * pred) - mu_x2
        sig_y2 = filt(gt * gt) - mu_y2
        sig_xy = filt(pred * gt) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2))
        return ssim_map.mean().to(torch.float32)

    # ------------------------------------------------------------------ optimizers
    def configure_optimizers(self) -> Any:
        """AdamW + cosine LR schedule (with optional linear warm-up).

        Builds an AdamW over the *trainable* parameters (so frozen blocks are excluded), then
        a cosine annealing schedule from ``lr`` to ``lr_min`` over ``max_epochs`` (preferring
        the Trainer's ``max_epochs`` if set). When ``warmup_epochs > 0`` a linear warm-up is
        chained in front via :class:`~torch.optim.lr_scheduler.SequentialLR` (research/06 §0,
        §4.7 two-stage transfer recipe).

        Returns:
            The Lightning optimizer/scheduler config dict.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # Prefer the Trainer's configured epoch budget when available. Use the private
        # ``_trainer`` attribute: the public ``self.trainer`` property RAISES (rather than
        # returning ``None``) when the module is not attached to a Trainer, which would break
        # calling ``configure_optimizers()`` standalone.
        max_epochs = self.max_epochs
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and getattr(trainer, "max_epochs", None):
            te = int(trainer.max_epochs)
            if te > 0:
                max_epochs = te
        max_epochs = max(1, max_epochs)
        warmup = max(0, min(self.warmup_epochs, max_epochs - 1)) if max_epochs > 1 else 0

        cosine_T = max(1, max_epochs - warmup)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_T, eta_min=self.lr_min
        )
        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup
            )
            scheduler: Any = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup]
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

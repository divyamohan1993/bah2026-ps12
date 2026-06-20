"""FrameFlow training entrypoints — Hydra-driven ``train(cfg)`` + plain ``run_training(...)``.

This module wires together the :class:`~frameflow.train.datamodule.VFIDataModule`, the
:class:`~frameflow.train.module.VFIModule`, and a :class:`pytorch_lightning.Trainer`, then
runs ``fit``. It exposes two callables:

    * :func:`run_training` — a plain Python function (no Hydra dependency) that takes typed
      config objects and/or an injected model/dataset and returns the best-checkpoint path.
      This is what ``scripts`` / tests / the Typer CLI call.
    * :func:`train` — a Hydra ``@hydra.main`` entrypoint (``train(cfg)``) that composes the
      YAMLs in ``configs/`` into a :class:`frameflow.config.FrameFlowConfig` and delegates to
      :func:`run_training`. Hydra/OmegaConf are imported lazily, so this module imports fine
      even where Hydra is not installed.

Precision (BF16): the BF16 mixed-precision recipe (research/06 §0, §4.1) is honoured by
passing ``precision="bf16-mixed"`` to the Trainer; on CPU we coerce to ``32-true`` so tests
and CPU smoke-runs work. Checkpointing writes ``best_ssim.ckpt`` (monitor ``val_ssim``) and
``last.ckpt``; logging is TensorBoard by default with an optional, lazy W&B hook.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl

from .. import config as _config
from .datamodule import VFIDataModule
from .module import VFIModule

if TYPE_CHECKING:  # type-checkers only
    import torch


__all__ = ["run_training", "train", "build_trainer"]


# ---------------------------------------------------------------------------
# Precision / accelerator resolution
# ---------------------------------------------------------------------------
def _resolve_precision(requested: str, accelerator: str) -> str:
    """Pick a Lightning precision string that is valid for the chosen accelerator.

    BF16-mixed needs a CUDA (Ampere+) or supported CPU build; for plain CPU runs (tests,
    smoke runs) we fall back to ``32-true`` so nothing errors. On CUDA we honour the request
    (``bf16-mixed`` by default), letting Lightning fall back internally if unsupported.

    Args:
        requested: the requested precision (e.g. ``"bf16-mixed"``).
        accelerator: ``"cpu"``, ``"gpu"``/``"cuda"``, or ``"auto"``.

    Returns:
        A precision string accepted by :class:`pytorch_lightning.Trainer`.
    """
    acc = accelerator.lower()
    if acc in ("cpu",):
        # BF16 autocast on CPU is brittle/slow; use full precision for CPU runs.
        return "32-true"
    return requested or "bf16-mixed"


def _resolve_accelerator(device: str) -> str:
    """Map a config device string to a Lightning accelerator, defaulting to availability."""
    import torch  # lazy

    d = (device or "auto").lower()
    if d in ("cuda", "gpu"):
        return "gpu" if torch.cuda.is_available() else "cpu"
    if d == "cpu":
        return "cpu"
    # auto
    return "gpu" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Logger / callbacks
# ---------------------------------------------------------------------------
def _build_logger(tracker: str, out_dir: str, project: str) -> Any:
    """Build a Lightning logger from the tracker name (lazy, optional W&B).

    ``tensorboard`` (default, offline-safe) uses Lightning's TensorBoardLogger; ``wandb``
    lazily imports the W&B logger and silently degrades to TensorBoard if W&B is missing;
    ``none``/``False`` disables logging. (research/06 §6 MLOps.)

    Args:
        tracker: ``"tensorboard"`` | ``"wandb"`` | ``"mlflow"`` | ``"none"``.
        out_dir: directory for logger output.
        project: project/run name.

    Returns:
        A Lightning logger instance, or ``False`` to disable logging.
    """
    name = (tracker or "tensorboard").lower()
    if name in ("none", "false", ""):
        return False
    if name == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger  # lazy

            return WandbLogger(project=project, save_dir=out_dir)
        except Exception:
            # W&B not installed / offline — fall back to TensorBoard rather than fail.
            pass
    if name == "mlflow":
        try:
            from pytorch_lightning.loggers import MLFlowLogger  # lazy

            return MLFlowLogger(experiment_name=project, save_dir=out_dir)
        except Exception:
            pass
    try:
        from pytorch_lightning.loggers import TensorBoardLogger  # lazy

        return TensorBoardLogger(save_dir=out_dir, name=project)
    except Exception:  # pragma: no cover - extremely unlikely
        return False


def _build_callbacks(ckpt_dir: str) -> list[pl.Callback]:
    """Build checkpoint + LR-monitor callbacks.

    Saves the best checkpoint by ``val_ssim`` (max) as ``best_ssim`` and always keeps
    ``last`` (research/06 §6: keep ``best_ssim.ckpt`` and ``last.ckpt``). A
    :class:`~pytorch_lightning.callbacks.LearningRateMonitor` logs the cosine schedule.

    Args:
        ckpt_dir: directory to write checkpoints into.

    Returns:
        A list of Lightning callbacks.
    """
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_ssim",
        monitor="val_ssim",
        mode="max",
        save_last=True,  # writes last.ckpt
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    lr_mon = LearningRateMonitor(logging_interval="epoch")
    return [ckpt, lr_mon]


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------
def build_trainer(
    train_cfg: _config.TrainConfig,
    *,
    accelerator: str = "auto",
    devices: Any = "auto",
    logger: Any = None,
    callbacks: list[pl.Callback] | None = None,
    enable_checkpointing: bool = True,
    project: str = "FrameFlow",
    **trainer_kwargs: Any,
) -> pl.Trainer:
    """Construct a :class:`pytorch_lightning.Trainer` from a :class:`TrainConfig`.

    Sets BF16 mixed precision (coerced to 32-true on CPU), gradient accumulation, the cosine
    epoch budget, and (by default) checkpoint + LR-monitor callbacks and a TensorBoard logger.

    Args:
        train_cfg: training configuration (epochs, precision, grad_accum, dirs, tracker).
        accelerator: ``"auto"`` | ``"cpu"`` | ``"gpu"``.
        devices: number of devices or ``"auto"``.
        logger: explicit logger (``None`` → built from ``train_cfg.tracker``;
            ``False`` → disabled).
        callbacks: explicit callbacks (``None`` → checkpoint + LR monitor when
            ``enable_checkpointing``).
        enable_checkpointing: whether to write checkpoints.
        project: project/run name for the logger.
        **trainer_kwargs: extra keyword args forwarded to :class:`pl.Trainer` (e.g.
            ``max_steps``, ``limit_*_batches`` for tests).

    Returns:
        A configured (un-launched) :class:`pytorch_lightning.Trainer`.
    """
    acc = _resolve_accelerator(accelerator) if accelerator == "auto" else accelerator
    precision = _resolve_precision(train_cfg.precision, acc)

    if logger is None:
        logger = _build_logger(train_cfg.tracker, train_cfg.out_dir, project)

    if callbacks is None:
        callbacks = _build_callbacks(train_cfg.ckpt_dir) if enable_checkpointing else []

    return pl.Trainer(
        accelerator=acc,
        devices=devices,
        precision=precision,
        max_epochs=train_cfg.epochs,
        accumulate_grad_batches=train_cfg.grad_accum,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        **trainer_kwargs,
    )


# ---------------------------------------------------------------------------
# Plain (non-Hydra) training entrypoint
# ---------------------------------------------------------------------------
def run_training(
    cfg: _config.FrameFlowConfig | None = None,
    *,
    model: torch.nn.Module | None = None,
    datamodule: VFIDataModule | None = None,
    dataset: Any = None,
    accelerator: str = "auto",
    devices: Any = "auto",
    trainer_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Build datamodule + module + Trainer and run ``fit``; return the best-checkpoint path.

    This is the plain, Hydra-free entrypoint. Pass a :class:`frameflow.config.FrameFlowConfig`
    (or rely on defaults), and optionally inject a ready ``model`` and/or ``dataset`` to
    decouple from the MODELS/DATA teams (e.g. for tests). The train/val split is time-based
    (no leakage; see :class:`~frameflow.train.datamodule.VFIDataModule`), validation metrics
    use the FIXED Kelvin ``data_range`` (P1), and BF16 mixed precision is used on GPU.

    Args:
        cfg: composed top-level config (``None`` → :class:`FrameFlowConfig` defaults).
        model: optional pre-built backbone (overrides ``cfg.model.name``).
        datamodule: optional pre-built :class:`VFIDataModule` (overrides ``dataset``/cube).
        dataset: optional injected dataset for the datamodule.
        accelerator: ``"auto"`` | ``"cpu"`` | ``"gpu"``.
        devices: device count or ``"auto"``.
        trainer_kwargs: extra kwargs for :class:`pl.Trainer` (e.g. ``max_steps``,
            ``logger=False``, ``enable_checkpointing=False`` for fast tests).

    Returns:
        Path to the best checkpoint (``best_ssim.ckpt``); falls back to ``last.ckpt`` or the
        checkpoint directory when checkpointing is disabled.
    """
    cfg = cfg or _config.FrameFlowConfig()
    pl.seed_everything(cfg.seed, workers=True)

    # --- DataModule (time-based split) ----------------------------------------------------
    if datamodule is None:
        datamodule = VFIDataModule(
            cube_path=cfg.data.cube_path,
            dataset=dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            patch_size=cfg.train.patch_size_start,
            normalized=(cfg.data.norm_mode != "none"),
            train_frac=cfg.data.train_frac,
            val_frac=cfg.data.val_frac,
        )

    # --- LightningModule ------------------------------------------------------------------
    module = VFIModule.from_config(cfg, model=model)
    # Stage-1 warm-up: research/06 §4.7 suggests freezing the flow encoder first. We keep
    # this opt-in via cfg.model.pretrained being set (transfer learning); freezing prefixes
    # are backbone-specific, so we leave them to the caller / from_config defaults.

    # --- Trainer --------------------------------------------------------------------------
    tk = dict(trainer_kwargs or {})
    enable_ckpt = tk.pop("enable_checkpointing", True)
    logger = tk.pop("logger", None)
    trainer = build_trainer(
        cfg.train,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        enable_checkpointing=enable_ckpt,
        project=cfg.project,
        **tk,
    )

    trainer.fit(module, datamodule=datamodule)

    return _best_ckpt_path(trainer, cfg.train.ckpt_dir)


def _best_ckpt_path(trainer: pl.Trainer, ckpt_dir: str) -> Path:
    """Resolve the best checkpoint path from the Trainer's ModelCheckpoint, with fallbacks."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    for cb in trainer.callbacks:  # type: ignore[attr-defined]
        if isinstance(cb, ModelCheckpoint):
            if cb.best_model_path:
                return Path(cb.best_model_path)
            if cb.last_model_path:
                return Path(cb.last_model_path)
    # No checkpointing (e.g. tests) → return the directory as a stable handle.
    return Path(ckpt_dir)


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------
def _to_frameflow_config(cfg: Any) -> _config.FrameFlowConfig:
    """Coerce a Hydra/OmegaConf ``DictConfig`` (or a FrameFlowConfig) into FrameFlowConfig.

    When OmegaConf is available we ``OmegaConf.to_object`` / merge into the structured
    :class:`FrameFlowConfig`; otherwise we accept an already-typed config or fall back to
    defaults. This keeps :func:`train` importable without Hydra installed.
    """
    if isinstance(cfg, _config.FrameFlowConfig):
        return cfg
    try:
        from omegaconf import OmegaConf  # lazy

        if OmegaConf.is_config(cfg):
            base = OmegaConf.structured(_config.FrameFlowConfig())
            merged = OmegaConf.merge(base, cfg)
            obj = OmegaConf.to_object(merged)
            if isinstance(obj, _config.FrameFlowConfig):
                return obj
    except Exception:
        pass
    return _config.FrameFlowConfig()


def train(cfg: Any = None) -> Path:
    """Hydra entrypoint: compose config, build everything, fit, return best ckpt path.

    Intended to be decorated by ``@hydra.main`` in a thin runner script::

        import hydra
        from frameflow.train.cli import train as _train

        @hydra.main(version_base=None, config_path="configs", config_name="config")
        def main(cfg):
            _train(cfg)

    Called directly with a :class:`FrameFlowConfig` (or ``None`` for defaults) it works
    without Hydra installed. The actual training is delegated to :func:`run_training`.

    Args:
        cfg: a Hydra ``DictConfig``, a :class:`frameflow.config.FrameFlowConfig`, or ``None``.

    Returns:
        Path to the best checkpoint.
    """
    ff_cfg = _to_frameflow_config(cfg)
    return run_training(ff_cfg)


def _hydra_main() -> None:  # pragma: no cover - exercised only with Hydra installed + CLI
    """Build and invoke the ``@hydra.main`` wrapper at runtime (lazy Hydra import)."""
    import hydra  # lazy

    config_path = os.path.relpath(
        Path(__file__).resolve().parents[2] / "configs",
        Path(__file__).resolve().parent,
    )

    @hydra.main(version_base=None, config_path=config_path, config_name="config")
    def _entry(cfg: Any) -> None:
        train(cfg)

    _entry()


if __name__ == "__main__":  # pragma: no cover
    _hydra_main()

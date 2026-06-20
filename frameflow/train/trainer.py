"""FrameFlow training runner — the CONTRACTS §5.2 ``run(config, overrides)`` entrypoint.

The top-level :mod:`frameflow.cli` ``train`` subcommand lazy-imports this module and calls
``trainer.run(config="configs/config.yaml", overrides=[...])``. This is a thin adapter that
composes the Hydra config (when Hydra/OmegaConf are installed) into a typed
:class:`frameflow.config.FrameFlowConfig` and delegates the actual training to
:func:`frameflow.train.cli.run_training`.

If Hydra/OmegaConf are not installed, it degrades gracefully: it loads the YAML directly with
``yaml`` if possible (applying simple ``key=value`` overrides), else falls back to the typed
config defaults — so ``frameflow train`` never crashes merely because Hydra is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import config as _config
from .cli import run_training


__all__ = ["run"]


def run(config: str = "configs/config.yaml", overrides: list[str] | None = None) -> Path:
    """Fine-tune the model per the composed config; return the best-checkpoint path.

    Honours :class:`frameflow.config.TrainConfig` (BF16 mixed precision, cosine schedule,
    loss weights) and a TIME-BASED train/val split (research/06 §6). Called by
    ``frameflow.cli train``.

    Args:
        config: path to the root Hydra/YAML config (``configs/config.yaml``).
        overrides: optional Hydra-style ``key=value`` overrides (e.g. ``["train.epochs=1"]``).

    Returns:
        :class:`pathlib.Path` to the best checkpoint (``runs/ckpt/best_ssim.ckpt``).
    """
    cfg = _compose_config(config, overrides or [])
    return run_training(cfg)


def _compose_config(config: str, overrides: list[str]) -> _config.FrameFlowConfig:
    """Compose a typed :class:`FrameFlowConfig`, preferring Hydra, then YAML, then defaults."""
    # 1) Preferred: Hydra compose API (handles the defaults list + group overrides).
    cfg = _try_hydra_compose(config, overrides)
    if cfg is not None:
        return cfg

    # 2) Fallback: plain YAML of the root file + simple dotted key=value overrides.
    cfg = _try_yaml(config, overrides)
    if cfg is not None:
        return cfg

    # 3) Last resort: typed defaults with dotted overrides applied.
    base = _config.FrameFlowConfig()
    _apply_dotted_overrides(base, overrides)
    return base


def _try_hydra_compose(config: str, overrides: list[str]) -> _config.FrameFlowConfig | None:
    """Compose via Hydra's initialize/compose API; return ``None`` if Hydra is unavailable."""
    try:
        from hydra import compose, initialize_config_dir  # lazy
        from omegaconf import OmegaConf  # lazy
    except Exception:
        return None

    cfg_path = Path(config).resolve()
    config_dir = str(cfg_path.parent)
    config_name = cfg_path.stem
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            composed = compose(config_name=config_name, overrides=list(overrides))
        base = OmegaConf.structured(_config.FrameFlowConfig())
        merged = OmegaConf.merge(base, composed)
        obj = OmegaConf.to_object(merged)
        if isinstance(obj, _config.FrameFlowConfig):
            return obj
    except Exception:
        return None
    return None


def _try_yaml(config: str, overrides: list[str]) -> _config.FrameFlowConfig | None:
    """Load the root YAML directly (ignoring Hydra defaults-list groups) + dotted overrides."""
    try:
        import yaml  # lazy
    except Exception:
        return None
    p = Path(config)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        return None

    base = _config.FrameFlowConfig()
    # Apply only top-level scalar keys present in the root config (e.g. project, seed); the
    # per-group YAMLs are composed by Hydra in the preferred path, so here we keep defaults
    # for the nested dataclasses and just honour explicit dotted overrides.
    if isinstance(raw, dict):
        for key in ("seed", "project"):
            if key in raw and not isinstance(raw[key], (dict, list)):
                setattr(base, key, raw[key])
    _apply_dotted_overrides(base, overrides)
    return base


def _apply_dotted_overrides(cfg: _config.FrameFlowConfig, overrides: list[str]) -> None:
    """Apply simple ``a.b=value`` overrides onto a typed config in place (best-effort).

    Only dotted ``key=value`` assignments into the nested dataclasses are handled (group
    overrides like ``model=ifrnet`` require Hydra and are ignored here). Values are coerced to
    the existing attribute's type where possible.
    """
    for ov in overrides:
        if "=" not in ov:
            continue
        dotted, _, value = ov.partition("=")
        parts = dotted.strip().split(".")
        obj: Any = cfg
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            leaf = parts[-1]
            if not hasattr(obj, leaf):
                continue
            current = getattr(obj, leaf)
            setattr(obj, leaf, _coerce(value.strip(), current))
        except AttributeError:
            continue


def _coerce(value: str, like: Any) -> Any:
    """Coerce a string override to the type of the existing value (bool/int/float/str)."""
    if isinstance(like, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(like, int) and not isinstance(like, bool):
        try:
            return int(value)
        except ValueError:
            return like
    if isinstance(like, float):
        try:
            return float(value)
        except ValueError:
            return like
    return value

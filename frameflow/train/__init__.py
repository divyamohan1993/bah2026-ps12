"""FrameFlow training subpackage (Team TRAIN) — losses, datamodule, module, runner.

Public API (see ``CONTRACTS.md`` §5):
    * :mod:`frameflow.train.losses` — single-channel TIR losses + :class:`CombinedLoss`
      (Charbonnier + census + MS-SSIM + gradient; VGG/LPIPS deliberately omitted).
    * :mod:`frameflow.train.datamodule` — :class:`VFIDataModule` with a TIME-BASED split.
    * :mod:`frameflow.train.module` — :class:`VFIModule` (Lightning), fixed-range val metrics.
    * :mod:`frameflow.train.cli` — :func:`run_training` / Hydra :func:`train`.
    * :mod:`frameflow.train.trainer` — :func:`run` (the CONTRACTS §5.2 entrypoint).

Submodules are imported lazily via :pep:`562` ``__getattr__`` so importing this package does
not pull in heavy deps (torch/lightning/piq) until something is actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "losses",
    "datamodule",
    "module",
    "cli",
    "trainer",
    "CombinedLoss",
    "VFIDataModule",
    "VFIModule",
    "run_training",
    "train",
    "run",
]

# Map exported names to their defining submodule for lazy resolution.
_LAZY: dict[str, str] = {
    "losses": "losses",
    "datamodule": "datamodule",
    "module": "module",
    "cli": "cli",
    "trainer": "trainer",
    "CombinedLoss": "losses",
    "VFIDataModule": "datamodule",
    "VFIModule": "module",
    "run_training": "cli",
    "train": "cli",
    "run": "trainer",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    """Lazily import submodules / public symbols on first access."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'frameflow.train' has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{target}", __name__)
    return mod if name == target else getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)

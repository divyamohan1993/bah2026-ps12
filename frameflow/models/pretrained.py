"""Pretrained-weight loading + single-channel adaptation for VFI backbones.

PS-12 wants **easy fine-tuning on single-channel brightness temperature**
(``research/01_vfi_models.md`` §0, §7.1, §9). This module provides:

    * :func:`load_pretrained` — return a VFI model, optionally initialized from a checkpoint
      of RIFE / IFRNet-style weights when one is provided or discoverable, otherwise a
      randomly-initialized :class:`~frameflow.models.ifnet.IFNet` (with a warning). It is
      deliberately tolerant: an exactly-matching checkpoint loads strictly; a partially
      matching one loads the overlapping tensors and reports the rest.
    * :func:`adapt_to_single_channel` — convert a model whose first conv expects 3-channel
      RGB into a single-channel (C=1) model by **averaging** (or duplicating) the first
      conv's input-channel weights — the standard grayscale fine-tuning trick.

Where to get MIT-licensed RIFE weights (for the operational deliverable):
    * **Practical-RIFE** (MIT) — https://github.com/hzwer/Practical-RIFE (release "model"
      zips, e.g. ``v4.25``/``v4.26``: a ``flownet.pkl`` / ``train_log/flownet.pkl``).
    * **ECCV2022-RIFE** (MIT) — https://github.com/hzwer/ECCV2022-RIFE (``RIFE_trained_model``).
    Both LICENSE files are MIT (verified in ``research/01`` §0/§9), so the weights are safe
    for ISRO/government deployment and downstream commercialization. Other comparators:
    IFRNet (MIT, https://github.com/ltkong218/IFRNet), EMA-VFI (Apache-2.0).

The official RIFE checkpoints target RIFE's *exact* IFNet variant (which differs in block
widths/counts from this clean re-implementation), so a raw ``state_dict`` will generally
NOT key-match this :class:`IFNet` 1:1. :func:`load_pretrained` therefore loads what it can
and warns about the rest; for a faithful port, instantiate this :class:`IFNet` with
matching ``scales``/``hidden`` or fine-tune from the partial load. The function never
raises on a shape/key mismatch — it degrades to a usable (partly/ randomly) initialized
model so the pipeline keeps running.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ifnet import IFNet

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

__all__ = ["load_pretrained", "adapt_to_single_channel", "PRETRAINED_SOURCES"]

#: Human-readable pointers to MIT/Apache VFI weights (documented, not auto-downloaded).
PRETRAINED_SOURCES: dict[str, str] = {
    "rife": "https://github.com/hzwer/Practical-RIFE (MIT) — flownet.pkl",
    "rife-eccv22": "https://github.com/hzwer/ECCV2022-RIFE (MIT) — RIFE_trained_model",
    "ifrnet": "https://github.com/ltkong218/IFRNet (MIT)",
    "ema-vfi": "https://github.com/MCG-NJU/EMA-VFI (Apache-2.0)",
}


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """Pull a ``{param_name: tensor}`` mapping out of a loaded checkpoint object.

    Handles the common layouts: a raw ``state_dict``; a dict wrapping it under
    ``"state_dict"`` / ``"model"`` / ``"flownet"`` / ``"net"``; and strips a leading
    ``"module."`` (DataParallel) prefix.
    """
    import torch

    sd = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "flownet", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
    if not isinstance(sd, dict):
        raise ValueError("checkpoint does not contain a state_dict-like mapping")

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def adapt_to_single_channel(model: nn.Module, *, mode: str = "average") -> nn.Module:
    """Adapt a model's FIRST conv to accept a single input channel (in place).

    Finds the first :class:`torch.nn.Conv2d` whose ``in_channels`` is a multiple of 3 (an
    RGB stem, possibly stacking two RGB frames → 6) and collapses each group of 3 input
    channels to 1, so the layer consumes single-channel (grayscale / brightness-temperature)
    frames. This is the standard trick for fine-tuning RGB-pretrained nets on grayscale
    (``research/01`` §7.1).

    Args:
        model: the module to adapt (modified in place; also returned).
        mode: ``"average"`` (default) — set the new 1-ch weight to the MEAN of the 3 RGB
            weights (preserves output scale; recommended); ``"duplicate"`` — use the first
            (red) channel's weights; ``"sum"`` — sum the 3 (scales activations up by ~3×).

    Returns:
        The same ``model`` with its first conv reshaped to single-channel input.

    Raises:
        ValueError: if ``mode`` is invalid, or no RGB-like first conv is found.
    """
    import torch
    import torch.nn as nn

    if mode not in ("average", "duplicate", "sum"):
        raise ValueError(f"mode must be 'average'|'duplicate'|'sum', got {mode!r}")

    target: nn.Conv2d | None = None
    target_name = ""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels % 3 == 0 and mod.in_channels >= 3:
            target = mod
            target_name = name
            break
    if target is None:
        raise ValueError(
            "no Conv2d with in_channels divisible by 3 found — model does not look like it "
            "has an RGB stem to adapt"
        )

    groups_in = target.in_channels // 3  # e.g. 2 if two stacked RGB frames (6 ch)
    w = target.weight.data  # (out, in, kh, kw)
    out_c, in_c, kh, kw = w.shape
    w_grouped = w.view(out_c, groups_in, 3, kh, kw)
    if mode == "average":
        new_w = w_grouped.mean(dim=2)  # (out, groups_in, kh, kw)
    elif mode == "sum":
        new_w = w_grouped.sum(dim=2)
    else:  # duplicate -> take the first (red) channel of each group
        new_w = w_grouped[:, :, 0, :, :]

    new_conv = nn.Conv2d(
        in_channels=groups_in,
        out_channels=out_c,
        kernel_size=(kh, kw),
        stride=target.stride,
        padding=target.padding,
        dilation=target.dilation,
        groups=target.groups,
        bias=target.bias is not None,
        padding_mode=target.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(new_w.contiguous())
        if target.bias is not None:
            new_conv.bias.copy_(target.bias.data)

    # Replace the module in its parent.
    if target_name:
        parent = model
        *parents, leaf = target_name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, leaf, new_conv)
    else:  # the model itself is the conv (unusual)
        return new_conv  # type: ignore[return-value]
    return model


def load_pretrained(
    name: str = "ifnet",
    ckpt_path: str | None = None,
    *,
    in_channels: int = 1,
    strict: bool = False,
    **model_kwargs: Any,
) -> nn.Module:
    """Return a VFI model, optionally initialized from a checkpoint.

    Behavior:
        * Always constructs an :class:`~frameflow.models.ifnet.IFNet` (our clean RIFE-style
          backbone) with ``in_channels`` and any extra ``model_kwargs``.
        * If ``ckpt_path`` is given and exists, loads it (CPU) and copies every tensor whose
          name AND shape match the model; mismatches are reported (and, with
          ``strict=True``, raise). If the checkpoint is missing/unreadable, warns and returns
          the randomly-initialized model so the pipeline still runs.
        * If ``ckpt_path`` is ``None``, returns the randomly-initialized model with an
          informational warning pointing at :data:`PRETRAINED_SOURCES` for real weights.

    Args:
        name: model family — ``"ifnet"``/``"rife"`` build an :class:`IFNet` (the only
            backbone implemented here). Other names are accepted but also map to IFNet with a
            warning (so callers/configs don't crash).
        ckpt_path: path to a ``.pkl``/``.pth``/``.ckpt`` checkpoint, or ``None``.
        in_channels: input channels for the constructed model (1 for TIR).
        strict: if ``True``, raise on any missing/unexpected/shape-mismatched key.
        **model_kwargs: forwarded to :class:`IFNet` (e.g. ``hidden``, ``scales``).

    Returns:
        A ready-to-use (and trainable) :class:`torch.nn.Module`.
    """
    import torch

    fam = name.lower()
    if fam not in ("ifnet", "rife"):
        warnings.warn(
            f"load_pretrained: backbone {name!r} is not implemented here; "
            "constructing an IFNet (RIFE-style) instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    model = IFNet(in_channels=in_channels, **model_kwargs)

    if ckpt_path is None:
        warnings.warn(
            "load_pretrained: no ckpt_path given — returning a RANDOMLY-INITIALIZED IFNet "
            "(untrained: forward is ~the average blend). For real MIT-licensed RIFE weights "
            f"see frameflow.models.pretrained.PRETRAINED_SOURCES ({PRETRAINED_SOURCES['rife']}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return model

    path = Path(ckpt_path)
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"checkpoint not found: {path}")
        warnings.warn(
            f"load_pretrained: checkpoint {path} not found — returning randomly-initialized "
            "IFNet.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        ckpt_sd = _extract_state_dict(obj)
    except Exception as exc:  # noqa: BLE001 - tolerate any malformed checkpoint
        if strict:
            raise
        warnings.warn(
            f"load_pretrained: failed to read checkpoint {path} ({exc!r}) — returning "
            "randomly-initialized IFNet.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model

    model_sd = model.state_dict()
    to_copy: dict[str, torch.Tensor] = {}
    mismatched: list[str] = []
    for k, v in ckpt_sd.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            to_copy[k] = v
        else:
            mismatched.append(k)
    missing = [k for k in model_sd if k not in to_copy]

    if strict and (mismatched or missing):
        raise RuntimeError(
            f"strict load failed: {len(missing)} missing, {len(mismatched)} unexpected/"
            f"shape-mismatched keys"
        )

    model_sd.update(to_copy)
    model.load_state_dict(model_sd)

    if to_copy:
        warnings.warn(
            f"load_pretrained: loaded {len(to_copy)}/{len(model_sd)} tensors from {path} "
            f"({len(missing)} left at init, {len(mismatched)} checkpoint keys ignored). "
            "RIFE's official IFNet differs from this clean re-implementation, so a partial "
            "load is expected — fine-tune from here.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            f"load_pretrained: NO tensors in {path} matched this IFNet by name+shape "
            "(likely a different architecture) — returning randomly-initialized IFNet. "
            "Construct IFNet with matching scales/hidden for a faithful port.",
            RuntimeWarning,
            stacklevel=2,
        )
    return model

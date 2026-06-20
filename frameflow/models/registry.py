"""Model registry: the ``VFIModel`` protocol + a ``get_model`` factory.

This is the single entry point the rest of FrameFlow uses to obtain an interpolation model
by name, without importing each backbone directly (``CONTRACTS.md`` §4). Every object the
factory returns satisfies the :class:`VFIModel` protocol — a callable / ``forward`` taking
``(I0, I1, t)`` and returning a dict with at least a ``"pred"`` key — so the learned
network and the classical baselines are drop-in interchangeable for inference and
validation.

Names:
    * ``"ifnet"`` / ``"rife"``  → :class:`~frameflow.models.ifnet.IFNet` (the primary
      learned engine; returns ``{"pred","flow","mask"}``).
    * ``"linear"``              → linear cross-fade baseline.
    * ``"farneback"``           → classical Farnebäck optical-flow baseline.
    * ``"tvl1"``                → classical TV-L1 optical-flow baseline.

The baseline wrappers adapt the numpy-array functions in
:mod:`frameflow.models.baselines` to the same tensor-in / dict-out interface as
:class:`IFNet`, so a caller can iterate over methods uniformly (``research/01`` §8: "show
AI beats traditional optical flow").

P2-ONNX (``CONTRACTS.md`` §0.3): ``t`` is accepted as a ``(B, 1)`` tensor (preferred) or a
python float (broadcast). The learned model keeps the batched-``t`` contract for ONNX/Triton
export; the baselines simply read a per-sample scalar from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from . import baselines as _baselines
from .ifnet import IFNet

if TYPE_CHECKING:
    import torch

__all__ = ["VFIModel", "get_model", "ClassicalBaseline", "MODEL_NAMES"]

#: All names :func:`get_model` understands.
MODEL_NAMES: tuple[str, ...] = ("ifnet", "rife", "linear", "farneback", "tvl1")


@runtime_checkable
class VFIModel(Protocol):
    """Structural interface every FrameFlow interpolation model implements.

    A ``VFIModel`` is any object that, called as ``model(I0, I1, t)`` (or via its
    ``forward``), maps two single-channel frame tensors ``(B, 1, H, W)`` and a timestep
    ``t`` to a result dict containing at least ``"pred"`` — the interpolated frame
    ``(B, 1, H, W)``. Learned backbones additionally return ``"flow"`` and ``"mask"``.

    NOTE (P2-ONNX, ``CONTRACTS.md`` §0.3): ``t`` SHOULD be a ``(B, 1)`` tensor with a
    leading batch dim (a python float is also accepted and broadcast).

    This is a :func:`typing.runtime_checkable` :class:`typing.Protocol`, so
    ``isinstance(obj, VFIModel)`` checks for a callable ``forward`` at runtime. Both
    :class:`~frameflow.models.ifnet.IFNet` (an ``nn.Module``) and
    :class:`ClassicalBaseline` satisfy it.
    """

    def forward(
        self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor | float
    ) -> dict[str, torch.Tensor]:
        """Interpolate the frame at fraction ``t``; return a dict with key ``"pred"``."""
        ...

    def __call__(
        self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor | float
    ) -> dict[str, torch.Tensor]:
        """Callable form, equivalent to :meth:`forward`."""
        ...


class ClassicalBaseline:
    """Tensor-in / dict-out wrapper around a classical numpy interpolation baseline.

    Adapts the numpy single-channel functions in :mod:`frameflow.models.baselines`
    (``linear`` / ``farneback`` / ``tvl1``) to the :class:`VFIModel` interface so they can
    be used interchangeably with the learned :class:`~frameflow.models.ifnet.IFNet` in the
    inference/validation loops. Operates per batch element (classical flow is not batched),
    converting each ``(1, H, W)`` slice to numpy, interpolating, and stacking back to a
    tensor on the input's device/dtype.

    Args:
        method: one of :data:`frameflow.models.baselines.CLASSICAL_METHODS`.
    """

    def __init__(self, method: str) -> None:
        key = method.lower()
        if key not in _baselines.CLASSICAL_METHODS:
            raise ValueError(
                f"unknown classical method {method!r}; "
                f"choose from {_baselines.CLASSICAL_METHODS}"
            )
        self.method = key

    # not an nn.Module, but mirror its trivial attributes for uniformity
    params_m: float = 0.0

    def _scalar_t(self, t: torch.Tensor | float, i: int, batch: int) -> float:
        """Read the scalar timestep for batch element ``i`` from a float or ``(B,1)`` tensor."""
        if isinstance(t, (float, int)):
            return float(t)
        tt = t.reshape(-1)
        if tt.numel() == 1:
            return float(tt.item())
        if tt.numel() == batch:
            return float(tt[i].item())
        # Fall back to the first entry if the shape is unexpected.
        return float(tt[0].item())

    def forward(
        self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor | float = 0.5
    ) -> dict[str, torch.Tensor]:
        """Interpolate per batch element with the classical method; return ``{"pred": ...}``.

        Args:
            I0: ``(B, 1, H, W)`` earlier frames (tensor).
            I1: ``(B, 1, H, W)`` later frames (tensor).
            t: timestep — ``(B, 1)`` tensor or python float.

        Returns:
            ``{"pred": (B, 1, H, W)}`` on ``I0``'s device/dtype.
        """
        import numpy as np
        import torch

        if I0.shape != I1.shape:
            raise ValueError(f"I0 {tuple(I0.shape)} and I1 {tuple(I1.shape)} must match")
        if I0.dim() != 4:
            raise ValueError(f"expected (B,1,H,W) tensors; got {tuple(I0.shape)}")
        b = I0.shape[0]
        a_np = I0.detach().cpu().numpy()
        b_np = I1.detach().cpu().numpy()
        preds = []
        for i in range(b):
            ti = self._scalar_t(t, i, b)
            # each element is (C, H, W); baselines handle (1,H,W) directly.
            out_i = _baselines.classical_interpolate(a_np[i], b_np[i], ti, method=self.method)
            preds.append(np.asarray(out_i, dtype=np.float32))
        pred = torch.from_numpy(np.stack(preds, axis=0)).to(device=I0.device, dtype=I0.dtype)
        return {"pred": pred}

    def __call__(
        self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor | float = 0.5
    ) -> dict[str, torch.Tensor]:
        return self.forward(I0, I1, t)

    def __repr__(self) -> str:
        return f"ClassicalBaseline(method={self.method!r})"


def get_model(name: str, **kwargs: Any) -> VFIModel:
    """Factory: return a :class:`VFIModel` by name.

    Args:
        name: one of :data:`MODEL_NAMES` (case-insensitive):
            ``"ifnet"`` / ``"rife"`` → :class:`~frameflow.models.ifnet.IFNet`;
            ``"linear"`` / ``"farneback"`` / ``"tvl1"`` → a :class:`ClassicalBaseline`.
        **kwargs: for the learned model, forwarded to :class:`IFNet`
            (e.g. ``in_channels``, ``hidden``, ``scales``). Ignored for baselines.

    Returns:
        An object satisfying the :class:`VFIModel` protocol (``forward``/``__call__`` →
        dict with ``"pred"``).

    Raises:
        ValueError: if ``name`` is unknown.
    """
    key = name.lower()
    if key in ("ifnet", "rife"):
        return IFNet(**kwargs)
    if key in _baselines.CLASSICAL_METHODS:  # linear / farneback / tvl1
        return ClassicalBaseline(key)
    raise ValueError(f"unknown model name {name!r}; choose from {MODEL_NAMES}")

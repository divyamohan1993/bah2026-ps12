"""FrameFlow models — VFI backbones, classical baselines, and flow utilities.

Team **MODELS** (see ``CONTRACTS.md`` §4). This subpackage implements the
video-frame-interpolation (VFI) engine FrameFlow uses to synthesize an intermediate
geostationary Thermal-IR frame from two bracketing frames, plus the classical
comparators the problem statement requires ("show AI beats traditional optical flow",
``research/01_vfi_models.md`` §1, §8).

Public modules:
    * :mod:`frameflow.models.warp` — differentiable backward warping
      (:func:`~frameflow.models.warp.backward_warp`).
    * :mod:`frameflow.models.ifnet` — the primary engine, a RIFE / Practical-RIFE style
      :class:`~frameflow.models.ifnet.IFNet` (coarse-to-fine IFBlocks estimating
      bidirectional intermediate flow + a fusion mask, conditioned on timestep ``t``).
    * :mod:`frameflow.models.flow` — optical-flow visualization utilities
      (Middlebury color, sparse vectors) + an optional RAFT wrapper.
    * :mod:`frameflow.models.baselines` — classical comparators (linear blend, Farnebäck,
      TV-L1) on single-channel numpy frames.
    * :mod:`frameflow.models.pretrained` — checkpoint loading + single-channel adaptation.
    * :mod:`frameflow.models.registry` — the :class:`~frameflow.models.registry.VFIModel`
      protocol and :func:`~frameflow.models.registry.get_model` factory.

Design choices are grounded in ``research/01_vfi_models.md``: RIFE/IFNet is the PRIMARY
deliverable (fastest flow-based VFI, MIT license, native arbitrary-time ``t``, tiny
~10 M params → trivial single-channel fine-tuning). To keep imports light (the top-level
package never imports submodules), heavy/optional deps (``cv2``, ``torchvision`` RAFT)
are imported lazily inside the functions that need them.

P2-ONNX (``CONTRACTS.md`` §0.3): every model takes the timestep ``t`` as a tensor with a
leading BATCH dimension, shape ``(B, 1)`` (a python float is also accepted and broadcast),
so the model exports cleanly to ONNX/Triton with ``t`` in ``dynamic_axes``.
"""

from __future__ import annotations

from .ifnet import IFNet
from .registry import VFIModel, get_model
from .warp import backward_warp

__all__ = [
    "IFNet",
    "VFIModel",
    "get_model",
    "backward_warp",
]

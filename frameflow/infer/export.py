"""Model export — ONNX (P2-ONNX) and TensorRT notes for serving.

The serving design (R6 §1-3) exports the trained VFI model to **ONNX (opset >= 17)** and
then optionally to a **TensorRT FP16** engine for the on-demand endpoint, while the primary
dashboard path is precomputed/CDN (O(1)).

THE P2-ONNX RULE (CONTRACTS.md §0.3, baked in here):
    The timestep input ``t`` MUST have shape ``[B, 1]`` (a leading BATCH dimension) and be
    listed in ``dynamic_axes`` together with the two image inputs and the output. Only then
    does NVIDIA Triton **dynamic batching** (``max_batch_size > 0``) work — Triton forms a
    batch by concatenating requests along axis 0 of *every* input, so a scalar ``t`` (rank
    0/without a batch axis) cannot be batched and forces ``max_batch_size = 0``.

    Reference ``dynamic_axes``::

        {"img0": {0: "B", 2: "H", 3: "W"},
         "img1": {0: "B", 2: "H", 3: "W"},
         "t":    {0: "B"},
         "mid":  {0: "B", 2: "H", 3: "W"}}

grid_sample caveat (RIFE/IFRNet use ``F.grid_sample`` for backward warping):
    * 2D ``grid_sample`` exports cleanly at **opset >= 16** (default here is 17).
    * **TensorRT >= 8.5/8.6** imports 2D ``GridSample`` natively; older TRT needs a
      graph-surgeon plugin rename (only relevant for 3D, which VFI does not use).

Quantization verdict (R6 §1.4): **FP16/BF16 only. DO NOT use INT8 for VFI** — naive INT8
PTQ measured **-0.89 dB (RIFE flow)** to **-4.38 dB (IFRNet frame mode)** degradation
(ANVIL). See :func:`to_tensorrt`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # typing only
    import torch


__all__ = ["to_onnx", "to_tensorrt", "ONNX_DYNAMIC_AXES", "triton_config_pbtxt"]


#: The reference ``dynamic_axes`` — batch dim on ALL THREE inputs AND the output, so Triton
#: dynamic batching works (P2-ONNX). ``t`` carries a leading batch axis.
ONNX_DYNAMIC_AXES: dict[str, dict[int, str]] = {
    "img0": {0: "B", 2: "H", 3: "W"},
    "img1": {0: "B", 2: "H", 3: "W"},
    "t": {0: "B"},
    "mid": {0: "B", 2: "H", 3: "W"},
}


class _TForwardWrapper:  # pragma: no cover - tiny shim, exercised indirectly
    """Wrap a model so ``forward(img0, img1, t)`` returns ONLY the interpolated frame.

    Some models' ``forward`` returns ``(It, flow)`` (``forward_with_flow`` style). ONNX
    export wants a single tensor output named ``mid``; this shim selects element 0.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, img0: Any, img1: Any, t: Any) -> Any:
        out = self._model(img0, img1, t)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _make_export_module(model: Any) -> "torch.nn.Module":
    """Return an ``nn.Module`` whose forward yields a single ``mid`` tensor for export."""
    import torch  # lazy

    class _ExportModule(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, img0: "torch.Tensor", img1: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
            out = self.inner(img0, img1, t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

    return _ExportModule(model)


def to_onnx(
    model: Any,
    path: str | Path,
    sample_hw: tuple[int, int] = (256, 256),
    opset: int = 17,
    *,
    channels: int = 1,
    batch: int = 1,
    device: str = "cpu",
    verify: bool = False,
) -> Path:
    """Export a VFI model to ONNX with the P2-ONNX batched-``t`` contract.

    Builds example inputs ``img0``, ``img1`` of shape ``(batch, channels, H, W)`` and ``t``
    of shape ``(batch, 1)`` — the leading BATCH dim on ``t`` is the whole point (P2-ONNX) —
    and exports with :data:`ONNX_DYNAMIC_AXES` so the batch axis is dynamic on all three
    inputs and the output. ``opset >= 16`` is enforced because RIFE/IFRNet's
    ``F.grid_sample`` only exports from opset 16.

    Args:
        model: the trained VFI model (an ``nn.Module`` with ``forward(img0, img1, t)``).
        path: destination ``.onnx`` path (parent dirs created).
        sample_hw: ``(H, W)`` of the example inputs (dynamic at runtime via ``dynamic_axes``).
        opset: ONNX opset version (>= 16 required for ``grid_sample``; default 17).
        channels: input channel count (1 for single-channel TIR; 3 if the model replicates).
        batch: example batch size (the axis is dynamic regardless).
        device: device to trace on.
        verify: if True, load the exported graph with ``onnx`` and run a shape/structure
            check (``onnx.checker``); skipped silently if ``onnx`` is not installed.

    Returns:
        The :class:`pathlib.Path` of the written ``.onnx`` file.

    Raises:
        ValueError: if ``opset < 16`` (would break ``grid_sample`` export).
    """
    import torch  # lazy

    if opset < 16:
        raise ValueError(
            f"opset must be >= 16 for grid_sample (RIFE/IFRNet backward warp); got {opset}"
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    export_mod = _make_export_module(model).to(dev).eval()

    h, w = int(sample_hw[0]), int(sample_hw[1])
    img0 = torch.randn(batch, channels, h, w, device=dev)
    img1 = torch.randn(batch, channels, h, w, device=dev)
    # P2-ONNX: t is (B, 1), NEVER a scalar.
    t = torch.full((batch, 1), 0.5, dtype=torch.float32, device=dev)

    # Prefer the stable TorchScript-based exporter (``dynamo=False``): it honours the
    # ``dynamic_axes`` dict verbatim (so ``t``'s leading batch axis is preserved exactly as
    # ``ONNX_DYNAMIC_AXES`` declares) and only needs the ``onnx`` package, not the heavier
    # ``onnxscript`` the newer dynamo exporter pulls in. ``dynamo`` was added in recent torch;
    # fall back to the default export path on older versions that lack the kwarg.
    export_kwargs: dict[str, Any] = dict(
        input_names=["img0", "img1", "t"],
        output_names=["mid"],
        opset_version=opset,
        dynamic_axes=ONNX_DYNAMIC_AXES,
        do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(export_mod, (img0, img1, t), str(out), dynamo=False, **export_kwargs)
        except TypeError:  # pragma: no cover - older torch without the dynamo kwarg
            torch.onnx.export(export_mod, (img0, img1, t), str(out), **export_kwargs)

    if verify:  # pragma: no cover - depends on optional onnx
        try:
            import onnx

            graph = onnx.load(str(out))
            onnx.checker.check_model(graph)
        except ImportError:
            pass

    return out


def to_tensorrt(
    onnx_path: str | Path,
    engine_path: str | Path = "runs/export/vfi_fp16.plan",
    *,
    fp16: bool = True,
    min_hw: tuple[int, int] = (256, 256),
    opt_hw: tuple[int, int] = (512, 512),
    max_hw: tuple[int, int] = (512, 512),
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 8,
    channels: int = 1,
) -> str:
    """Return the ``trtexec`` command to build a TensorRT engine from the ONNX file.

    This does **not** shell out (TensorRT/CUDA are not present in this CPU environment and
    must not be assumed); it returns the exact, copy-pasteable ``trtexec`` command — the
    documented build path (R6 §8.2) — so the serving step is reproducible.

    QUANTIZATION POLICY — **DO NOT use INT8 for VFI.**
        Naive INT8 PTQ collapses VFI quality: the ANVIL study measured **-0.89 dB
        (RIFE flow)** up to **-4.38 dB (IFRNet frame mode)**. The task is graded on
        PSNR/SSIM, so an INT8 engine would directly tank the headline numbers. **FP16 is
        fine** (negligible accuracy impact) and is the default; BF16 likewise. If INT8 is
        ever mandated for an edge target, it requires full quantization-aware training
        (QAT) to recover accuracy — never plain PTQ.

    Args:
        onnx_path: the exported ``.onnx`` file.
        engine_path: destination ``.plan`` engine path.
        fp16: build an FP16 engine (recommended). INT8 is intentionally NOT an option.
        min_hw / opt_hw / max_hw: dynamic spatial-shape profile for ``trtexec``.
        min_batch / opt_batch / max_batch: dynamic batch profile (``max_batch > 0`` relies on
            the P2-ONNX batched-``t`` export so the engine can be dynamically batched).
        channels: input channel count.

    Returns:
        The ``trtexec`` command string (and the same INT8 warning applies if anyone edits it).
    """
    o = Path(onnx_path)
    e = Path(engine_path)
    c = int(channels)

    def shape(b: int, hw: tuple[int, int]) -> str:
        return f"img0:{b}x{c}x{hw[0]}x{hw[1]},img1:{b}x{c}x{hw[0]}x{hw[1]},t:{b}x1"

    precision = "--fp16" if fp16 else ""
    # DO NOT use INT8 for VFI: -0.9..-4.4 dB PSNR collapse (ANVIL). FP16 only.
    cmd = (
        f"trtexec --onnx={o} {precision} --saveEngine={e} "
        f"--minShapes={shape(min_batch, min_hw)} "
        f"--optShapes={shape(opt_batch, opt_hw)} "
        f"--maxShapes={shape(max_batch, max_hw)}"
    ).replace("  ", " ").strip()
    return cmd


def triton_config_pbtxt(
    *,
    max_batch_size: int = 16,
    channels: int = 1,
    platform: str = "tensorrt_plan",
    name: str = "vfi",
) -> str:
    """Return a Triton ``config.pbtxt`` for the exported model with dynamic batching.

    Because the ONNX export gives ``t`` a leading batch axis (P2-ONNX), Triton can form a
    server-side batch by concatenating requests along axis 0 of ``img0``/``img1``/``t``.
    With ``max_batch_size > 0`` Triton prepends the batch dim itself, so the per-input
    ``dims`` here are the **non-batch** dims (``t`` -> ``[1]``). Set ``max_batch_size = 0``
    to disable server-side batching (the documented alternative) — then ``dims`` would carry
    the explicit batch axis instead.

    Args:
        max_batch_size: Triton dynamic-batch cap (``0`` disables server-side batching).
        channels: input channel count.
        platform: ``"tensorrt_plan"`` (FP16 engine) or set backend ``"onnxruntime"``.
        name: model name.

    Returns:
        The ``config.pbtxt`` text.
    """
    c = int(channels)
    if max_batch_size > 0:
        # Non-batch dims; Triton prepends [batch]. t is [1] (its (B,1) shape minus batch).
        body = (
            f'name: "{name}"\n'
            f'platform: "{platform}"\n'
            f"max_batch_size: {max_batch_size}\n"
            f'input  [ {{ name: "img0" data_type: TYPE_FP16 dims: [{c},-1,-1] }},\n'
            f'         {{ name: "img1" data_type: TYPE_FP16 dims: [{c},-1,-1] }},\n'
            f'         {{ name: "t"    data_type: TYPE_FP32 dims: [1] }} ]\n'
            f'output [ {{ name: "mid"  data_type: TYPE_FP16 dims: [{c},-1,-1] }} ]\n'
            f"instance_group [ {{ count: 1 kind: KIND_GPU }} ]\n"
            f"dynamic_batching {{ preferred_batch_size: [4, 8] max_queue_delay_microseconds: 2000 }}\n"
        )
    else:
        # No server-side batching: carry the explicit batch axis in dims.
        body = (
            f'name: "{name}"\n'
            f'platform: "{platform}"\n'
            f"max_batch_size: 0\n"
            f'input  [ {{ name: "img0" data_type: TYPE_FP16 dims: [-1,{c},-1,-1] }},\n'
            f'         {{ name: "img1" data_type: TYPE_FP16 dims: [-1,{c},-1,-1] }},\n'
            f'         {{ name: "t"    data_type: TYPE_FP32 dims: [-1,1] }} ]\n'
            f'output [ {{ name: "mid"  data_type: TYPE_FP16 dims: [-1,{c},-1,-1] }} ]\n'
            f"instance_group [ {{ count: 1 kind: KIND_GPU }} ]\n"
        )
    return body

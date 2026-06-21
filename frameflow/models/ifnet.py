"""IFNet — a clean RIFE / Practical-RIFE style network for arbitrary-time VFI.

This is FrameFlow's PRIMARY video-frame-interpolation engine
(``research/01_vfi_models.md`` §0, §3.2, §9). It follows the **IFNet** design of RIFE
(Huang et al., ECCV 2022) / Practical-RIFE v4.x:

    * **Intermediate-flow estimation.** Instead of estimating flow between the two inputs
      and then reversing it, IFNet directly estimates the *intermediate* bidirectional
      flows ``F_{t->0}`` and ``F_{t->1}`` (from the unknown middle frame back to each
      input). This is why RIFE is fast and why it natively outputs the in-between frame.
    * **Coarse-to-fine IFBlocks.** A stack of residual flow blocks runs at increasing
      resolution. Each block warps the inputs by the current flow estimate, looks at the
      residual, and predicts a flow *update* plus a soft fusion ``mask``. Coarse scales
      capture large cloud motion; fine scales refine edges (multi-scale flow is exactly
      what helped on convective satellite IR — ``research/01`` §1, §7.3).
    * **Arbitrary timestep ``t``.** The desired time fraction ``t ∈ (0, 1)`` is injected as
      a constant feature channel into every block, so a single trained model interpolates
      at t = 0.25 / 0.5 / 0.75 → the 30→15→7.5 min densification PS-12 needs
      (``research/01`` §3.2, §7.8).

P2-ONNX (``CONTRACTS.md`` §0.3): :meth:`IFNet.forward` takes ``t`` as a tensor of shape
``(B, 1)`` (leading BATCH dim) so the model exports to ONNX/Triton with ``t`` in
``dynamic_axes`` and server-side dynamic batching works. A python ``float`` is also
accepted and broadcast to ``(B, 1)`` for convenience.

Single-channel by design: ``in_channels`` defaults to 1 (brightness temperature). The net
is intentionally small so an untrained CPU forward on 64–128 px runs in well under a
second, and so it fine-tunes cheaply on single-channel TIR (``research/01`` §7.1, §9).

Initialization: the final flow/mask head is zero-initialized, so an **untrained** forward
produces ~zero flow and ``mask = sigmoid(0) = 0.5`` → the output is the simple average of
the two inputs (a reasonable blend), and training only has to learn the *residual* motion.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp import backward_warp

__all__ = ["IFNet", "IFBlock"]


def _conv(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """A ``conv -> PReLU`` block (RIFE uses PReLU throughout its IFBlocks)."""
    padding = ((kernel - 1) // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation=dilation, bias=True),
        nn.PReLU(out_ch),
    )


class IFBlock(nn.Module):
    """One coarse-to-fine RIFE IFBlock: predicts a flow *update* and a fusion mask.

    The block consumes the two (already coarsely warped) frames, the current bidirectional
    flow estimate, and a constant ``t`` channel, downsamples by ``scale`` for speed/receptive
    field, runs a small conv trunk, and upsamples a 5-channel head: 4 channels of flow
    residual (``F_{t->0}`` and ``F_{t->1}``) and 1 channel of (pre-sigmoid) fusion mask.

    Args:
        in_planes: number of input channels fed to the block (frames + flow + t).
        hidden: width of the conv trunk.
        scale: internal downsampling factor (coarse blocks use a larger scale). The block
            downsamples its input by ``scale``, predicts at that resolution, then upsamples
            the flow back (scaling the flow magnitude by ``scale`` since flow is in pixels).
        zero_flow_init: if ``True``, initialize the flow head with VERY small (but nonzero)
            weights so this block contributes ~zero flow at init while still receiving
            gradients. Used for the FINAL block so the untrained network's applied flow is
            negligible (→ output is the average blend) WITHOUT killing the gradient path into
            this block's feature trunk. (An EXACT zero on the final ``flow_head`` weight would
            zero the gradient w.r.t. the trunk features — ``d feat`` flows back through
            ``flow_head.weight`` — and leave that block's stem/trunk untrained; using a tiny
            std avoids that while keeping the init blend ~average.) Earlier blocks use a
            slightly larger small init. Both choices keep the untrained blend ~average AND
            guarantee live gradients into every parameter from the first step (the model is
            trainable end-to-end; verified in ``tests/test_models.py``).
    """

    #: Std of the (tiny) random init for the FINAL block's flow head — small enough that the
    #: untrained applied flow is ~zero (output ≈ average blend) but nonzero so gradients flow.
    _FINAL_FLOW_INIT_STD: float = 1e-4
    #: Std of the small random init for EARLIER (coarse) blocks' flow heads.
    _COARSE_FLOW_INIT_STD: float = 1e-3

    def __init__(
        self, in_planes: int, hidden: int = 64, scale: int = 1, *, zero_flow_init: bool = True
    ) -> None:
        super().__init__()
        self.scale = int(scale)
        # Downsample-by-2 stem (kept even for scale==1 the input is pre-downsampled by
        # `scale` outside via interpolation, then this halves once more for the trunk).
        self.stem = nn.Sequential(
            _conv(in_planes, hidden, kernel=3, stride=2),
            _conv(hidden, hidden, kernel=3, stride=1),
        )
        self.trunk = nn.Sequential(
            _conv(hidden, hidden, kernel=3, stride=1),
            _conv(hidden, hidden, kernel=3, stride=1),
            _conv(hidden, hidden, kernel=3, stride=1),
        )
        # Two upsampling heads (x2 transpose conv to undo the stride-2 stem):
        #   flow_head -> 4 ch (F_t->0 dx,dy ; F_t->1 dx,dy)
        #   mask_head -> 1 ch (pre-sigmoid fusion logit)
        self.flow_head = nn.ConvTranspose2d(hidden, 4, kernel_size=4, stride=2, padding=1, bias=True)
        self.mask_head = nn.ConvTranspose2d(hidden, 1, kernel_size=4, stride=2, padding=1, bias=True)
        # Mask head: small init -> mask logit ~0 -> sigmoid ~0.5 (≈equal blend) at init, while
        # keeping a live gradient into this block's mask path (an EXACT zero would also work for
        # the final block, whose mask is used directly, but a tiny std keeps intermediate
        # blocks' mask heads trainable too once gradients flow back through the next block).
        nn.init.normal_(self.mask_head.weight, mean=0.0, std=self._FINAL_FLOW_INIT_STD)
        nn.init.zeros_(self.mask_head.bias)
        nn.init.zeros_(self.flow_head.bias)
        # Flow head: tiny (NOT exact-zero) weights so the applied flow is ~0 at init (output ≈
        # average blend) but gradients still reach the trunk/stem (see class docstring). The
        # final/finest block uses an even smaller std so its untrained contribution is minimal.
        std = self._FINAL_FLOW_INIT_STD if zero_flow_init else self._COARSE_FLOW_INIT_STD
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=std)

    def forward(
        self, x: torch.Tensor, flow: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the block at its internal scale.

        Args:
            x: concatenated input ``(B, in_planes, H, W)`` (frames + t channel, and if
                ``flow`` is given it is expected to ALREADY be concatenated into ``x`` by
                the caller; this signature keeps ``flow`` only for an optional residual add).
            flow: optional current flow ``(B, 4, H, W)`` to add the predicted residual to.

        Returns:
            ``(flow_update, mask)`` where ``flow_update`` is ``(B, 4, H, W)`` (the new total
            flow if ``flow`` was provided, else the residual) and ``mask`` is the
            pre-sigmoid fusion logit ``(B, 1, H, W)`` — both at the ORIGINAL ``H, W``.
        """
        b, _c, h, w = x.shape
        s = self.scale
        # Downsample to the block's working resolution (coarse blocks see more context).
        if s != 1:
            xs = F.interpolate(x, scale_factor=1.0 / s, mode="bilinear", align_corners=False)
        else:
            xs = x
        feat = self.stem(xs)
        feat = self.trunk(feat)
        flow_out = self.flow_head(feat)  # (B, 4, ~h/s, ~w/s)
        mask_out = self.mask_head(feat)  # (B, 1, ~h/s, ~w/s)

        # Upsample heads back to the original resolution. Flow is in pixels, so when we
        # upsample by `s` we must also scale its magnitude by `s`.
        if flow_out.shape[-2:] != (h, w):
            flow_out = F.interpolate(flow_out, size=(h, w), mode="bilinear", align_corners=False)
            mask_out = F.interpolate(mask_out, size=(h, w), mode="bilinear", align_corners=False)
        flow_res = flow_out * float(s)

        if flow is not None:
            flow_res = flow_res + flow
        return flow_res, mask_out


class IFNet(nn.Module):
    """RIFE-style IFNet for arbitrary-time single-channel frame interpolation.

    Coarse-to-fine stack of :class:`IFBlock` s estimating bidirectional intermediate flow
    ``(F_{t->0}, F_{t->1})`` and a fusion mask, conditioned on the timestep ``t``. The two
    inputs are warped toward the intermediate frame with the final flow and blended by the
    mask. See the module docstring for the architecture rationale (``research/01`` §3.2).

    Args:
        in_channels: channels per input frame. Default 1 (brightness temperature). The PS
            is single-channel; set 3 only to consume RGB / pretrained-RGB adaptations.
        hidden: base conv width of the IFBlocks (controls capacity/speed). Default 64.
        scales: per-block internal downsampling factors, coarse → fine. Default
            ``(4, 2, 1)`` (3 scales): block 0 sees a 1/4-res view for large motion, block 2
            refines at full res. 2 scales (e.g. ``(2, 1)``) also work for tiny CPU tests.

    Shapes:
        Input  ``I0, I1`` : ``(B, in_channels, H, W)``.
        Input  ``t``      : ``(B, 1)`` tensor (or python float, broadcast) in ``(0, 1)``.
        Output dict:
            ``"pred"`` : ``(B, in_channels, H, W)`` interpolated frame.
            ``"flow"`` : ``(B, 4, H, W)`` bidirectional flow (``F_{t->0}`` then ``F_{t->1}``).
            ``"mask"`` : ``(B, 1, H, W)`` fusion weight in ``(0, 1)`` (weight on the I0 warp).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden: int = 64,
        scales: tuple[int, ...] = (4, 2, 1),
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("scales must be a non-empty tuple of downsampling factors")
        self.in_channels = int(in_channels)
        self.scales = tuple(int(s) for s in scales)

        # Per-block input channels:
        #   block 0 : I0, I1, t                       -> 2*C + 1
        #   block k : I0, I1, warp(I0), warp(I1), t, flow(4), mask(1)
        #             -> 4*C + 1 + 4 + 1
        first_in = 2 * self.in_channels + 1
        later_in = 4 * self.in_channels + 1 + 4 + 1
        n_blocks = len(self.scales)
        blocks: list[IFBlock] = []
        for i, s in enumerate(self.scales):
            in_planes = first_in if i == 0 else later_in
            # Only the FINAL (finest) block is zero-flow-initialized, so the untrained
            # applied flow is exactly zero (average blend) yet all blocks get gradients.
            is_last = i == (n_blocks - 1)
            blocks.append(IFBlock(in_planes, hidden=hidden, scale=s, zero_flow_init=is_last))
        self.blocks = nn.ModuleList(blocks)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_t_map(t: torch.Tensor | float, batch: int, h: int, w: int,
                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Broadcast the timestep ``t`` into a ``(B, 1, H, W)`` constant feature map.

        Accepts either a ``(B, 1)`` tensor (P2-ONNX batched form) or a python ``float``
        (broadcast across the batch). Any tensor shape that broadcasts to ``(B, 1)`` is
        reshaped; a scalar tensor or float fills the whole batch.
        """
        if isinstance(t, (float, int)):
            t_col = torch.full((batch, 1), float(t), device=device, dtype=dtype)
        else:
            t_t = t.to(device=device, dtype=dtype)
            if t_t.dim() == 0:  # scalar tensor
                t_col = t_t.reshape(1, 1).expand(batch, 1)
            elif t_t.dim() == 1:  # (B,) -> (B,1)
                t_col = t_t.reshape(batch, 1)
            else:  # (B,1) or broadcastable
                t_col = t_t.reshape(t_t.shape[0], -1)[:, :1].expand(batch, 1)
        return t_col.reshape(batch, 1, 1, 1).expand(batch, 1, h, w)

    # ------------------------------------------------------------------ forward
    def forward(
        self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor | float
    ) -> dict[str, torch.Tensor]:
        """Interpolate the frame at fraction ``t`` between ``I0`` and ``I1``.

        Args:
            I0: earlier frame ``(B, in_channels, H, W)``.
            I1: later frame ``(B, in_channels, H, W)``.
            t: timestep in ``(0, 1)`` as a ``(B, 1)`` tensor (preferred, P2-ONNX) or a
                python float (broadcast across the batch).

        Returns:
            ``{"pred": (B,C,H,W), "flow": (B,4,H,W), "mask": (B,1,H,W)}``.
        """
        if I0.shape != I1.shape:
            raise ValueError(f"I0 {tuple(I0.shape)} and I1 {tuple(I1.shape)} must match")
        b, _c, h, w = I0.shape
        t_map = self._as_t_map(t, b, h, w, I0.device, I0.dtype)

        flow: torch.Tensor | None = None
        mask: torch.Tensor | None = None
        warped0, warped1 = I0, I1

        for i, block in enumerate(self.blocks):
            if i == 0:
                x = torch.cat([I0, I1, t_map], dim=1)
                flow, mask = block(x, flow=None)
            else:
                # Warp inputs toward the intermediate frame with the running flow estimate.
                warped0 = backward_warp(I0, flow[:, 0:2])
                warped1 = backward_warp(I1, flow[:, 2:4])
                x = torch.cat([I0, I1, warped0, warped1, t_map, flow, mask], dim=1)
                flow, mask = block(x, flow=flow)

        assert flow is not None and mask is not None  # at least one block always runs

        # Final warp + mask blend toward the intermediate frame.
        warped0 = backward_warp(I0, flow[:, 0:2])
        warped1 = backward_warp(I1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        pred = m * warped0 + (1.0 - m) * warped1

        return {"pred": pred, "flow": flow, "mask": m}

    # ------------------------------------------------------------------ convenience
    def forward_pred(
        self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor | float
    ) -> torch.Tensor:
        """Return only the predicted frame (``forward(...)["pred"]``)."""
        return self.forward(I0, I1, t)["pred"]

    @property
    def params_m(self) -> float:
        """Total parameter count in millions (for the model card / manifest)."""
        return sum(p.numel() for p in self.parameters()) / 1.0e6

    def num_parameters(self) -> int:
        """Total number of trainable + buffer parameters (exact integer count)."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:  # nicer print()
        return (
            f"in_channels={self.in_channels}, scales={self.scales}, "
            f"params_m={self.params_m:.3f}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # typing aid
        return super().__call__(*args, **kwargs)

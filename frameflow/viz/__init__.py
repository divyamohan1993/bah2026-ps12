"""FrameFlow visualization package (SERVE+VIZ ownership).

This subpackage turns brightness-temperature (BT) fields and optical-flow into the
web-dashboard artifacts described in :mod:`frameflow.contracts` and ``CONTRACTS.md`` §7:

    * :mod:`frameflow.viz.colormap` — fixed-range IR colormaps + ``bt_to_rgba``.
    * :mod:`frameflow.viz.render`   — full-frame WebP/PNG raster rendering.
    * :mod:`frameflow.viz.tiles`    — PER-FRAME XYZ WebP tile pyramids (+ optional PMTiles).
    * :mod:`frameflow.viz.video`    — all-intra MP4/WebM encoding (O(1) seek).
    * :mod:`frameflow.viz.flow_overlay` — deck.gl ``LineLayer`` flow-vector JSON.
    * :mod:`frameflow.viz.manifest` — assemble + validate a :class:`contracts.Manifest`.

Everything honours the two code-review corrections baked into the shared contracts:
    * P1: the FIXED physical Kelvin value range
      (:data:`frameflow.constants.BT_METRIC_VMIN_K` ..
      :data:`frameflow.constants.BT_METRIC_VMAX_K`) is used for *all* colorization and
      metric ranges — never per-image min/max.
    * P2: each timeline frame gets its OWN tile source (per-frame XYZ pyramid and/or
      per-frame PMTiles), so the slider can switch the active source by timestamp.

Heavy dependencies (numpy/matplotlib/PIL/imageio) are imported lazily inside functions so
importing this package stays cheap and side-effect free.
"""

from __future__ import annotations

__all__ = [
    "colormap",
    "render",
    "tiles",
    "video",
    "flow_overlay",
    "manifest",
]

"""IR brightness-temperature colormaps + ``bt_to_rgba`` (SERVE+VIZ, P1).

This module builds the perceptually-reasonable Thermal-IR colormaps named in
:data:`frameflow.constants.COLORMAP_NAMES` and colorizes a BT field to an 8-bit RGBA
image using the **FIXED** physical value range
(:data:`frameflow.constants.BT_METRIC_VMIN_K` .. :data:`frameflow.constants.BT_METRIC_VMAX_K`).

CODE-REVIEW CORRECTION P1 (honoured here): colorization uses the shared fixed Kelvin range,
exactly matching the metric ``data_range`` — never per-image min/max. A per-image stretch
would make two frames with different temperature extremes look identical and would hide a
warm/cold bias, so the legend/colorbar shown on the dashboard must map a *fixed* physical
span. This keeps the displayed colours comparable across frames, satellites and methods.

The three colormaps (R4 §7):
    * ``ir_clouds``    — enhanced meteorological IR ramp; COLD cloud tops are bright/saturated
      and the warm surface is dark, built from :data:`constants.IR_CLOUDS_COLORMAP_STOPS`.
    * ``greyscale_ir`` — the classic INVERTED-grey IR (cold = white, warm = black).
    * ``turbo``        — perceptually-uniform rainbow, inverted so cold tops sit at the
      bright/red end (meteorological convention).

NaN BT pixels (off-disk / space) map to a FULLY TRANSPARENT RGBA pixel (alpha = 0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import constants as C

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from matplotlib.colors import Colormap

# Cache built matplotlib colormaps so repeated frame rendering is cheap.
_CMAP_CACHE: dict[str, Any] = {}


def _build_ir_clouds() -> Colormap:
    """Build the enhanced IR-cloud colormap from the shared anchor stops.

    The stops in :data:`constants.IR_CLOUDS_COLORMAP_STOPS` are ``(bt_kelvin, "#rrggbb")``
    pairs spanning the fixed metric range; coldest tops are white, warmest surface black.
    They are normalized onto ``[0, 1]`` (the colormap's domain) by the fixed range so a
    value of ``vmin`` maps to position 0 and ``vmax`` to position 1.
    """
    from matplotlib.colors import LinearSegmentedColormap

    stops = C.IR_CLOUDS_COLORMAP_STOPS
    span = C.BT_METRIC_VMAX_K - C.BT_METRIC_VMIN_K
    entries: list[tuple[float, str]] = []
    for bt_k, hex_color in stops:
        pos = (float(bt_k) - C.BT_METRIC_VMIN_K) / span
        pos = min(max(pos, 0.0), 1.0)
        entries.append((pos, hex_color))
    # Guarantee anchors exactly at 0.0 and 1.0 so the ramp is well-defined end-to-end.
    if entries[0][0] > 0.0:
        entries.insert(0, (0.0, entries[0][1]))
    if entries[-1][0] < 1.0:
        entries.append((1.0, entries[-1][1]))
    return LinearSegmentedColormap.from_list("ir_clouds", entries, N=256)


def _build_greyscale_ir() -> Colormap:
    """Classic INVERTED-grey IR colormap (cold = white, warm = black)."""
    from matplotlib.colors import LinearSegmentedColormap

    # position 0 (cold/vmin) -> white, position 1 (warm/vmax) -> black.
    return LinearSegmentedColormap.from_list(
        "greyscale_ir", [(0.0, "#ffffff"), (1.0, "#000000")], N=256
    )


def _build_turbo() -> Colormap:
    """Perceptually-uniform turbo, REVERSED so cold tops sit at the bright/red end."""
    import matplotlib as mpl

    # matplotlib's turbo runs blue(0)->red(1); reverse so cold (position 0) is red/bright.
    return mpl.colormaps["turbo"].reversed()


_BUILDERS = {
    "ir_clouds": _build_ir_clouds,
    "greyscale_ir": _build_greyscale_ir,
    "turbo": _build_turbo,
}


def get_colormap(name: str = "ir_clouds") -> Colormap:
    """Return a (cached) matplotlib :class:`~matplotlib.colors.Colormap` by name.

    Args:
        name: one of :data:`frameflow.constants.COLORMAP_NAMES`
            (``"ir_clouds"``, ``"greyscale_ir"``, ``"turbo"``).

    Returns:
        A matplotlib ``Colormap`` over the normalized ``[0, 1]`` domain. The colormap's
        bad-value (NaN) colour is set to fully transparent so masked pixels render clear.

    Raises:
        ValueError: if ``name`` is not a known FrameFlow colormap.
    """
    if name in _CMAP_CACHE:
        return _CMAP_CACHE[name]
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"unknown colormap {name!r}; expected one of {C.COLORMAP_NAMES}"
        )
    cmap = builder()
    # Copy so we can set the 'bad' (NaN) colour without mutating a shared registry object.
    cmap = cmap.copy()
    cmap.set_bad(color=(0.0, 0.0, 0.0, 0.0))  # NaN -> fully transparent
    _CMAP_CACHE[name] = cmap
    return cmap


def bt_to_rgba(
    bt: np.ndarray,
    cmap: str = "ir_clouds",
    vmin: float = C.BT_METRIC_VMIN_K,
    vmax: float = C.BT_METRIC_VMAX_K,
) -> np.ndarray:
    """Colorize a brightness-temperature field to an 8-bit RGBA image (P1 fixed range).

    The BT field is normalized with the FIXED physical Kelvin range ``[vmin, vmax]``
    (defaults: the shared metric range), clamped to that range, looked up through the named
    colormap, and returned as ``uint8`` RGBA. NaN pixels (off-disk / space) become fully
    transparent (alpha = 0); all valid pixels are fully opaque (alpha = 255).

    Args:
        bt: brightness-temperature array, shape ``(H, W)`` (Kelvin); NaN = masked/space.
        cmap: colormap name (see :func:`get_colormap`).
        vmin: lower bound of the fixed display range (Kelvin). Defaults to
            :data:`constants.BT_METRIC_VMIN_K`.
        vmax: upper bound of the fixed display range (Kelvin). Defaults to
            :data:`constants.BT_METRIC_VMAX_K`.

    Returns:
        ``np.ndarray`` of shape ``(H, W, 4)``, dtype ``uint8`` (RGBA).

    Raises:
        ValueError: if ``vmax <= vmin``.
    """
    import numpy as np

    if vmax <= vmin:
        raise ValueError(f"vmax ({vmax}) must be > vmin ({vmin})")

    arr = np.asarray(bt, dtype=np.float64)
    nan_mask = ~np.isfinite(arr)

    # Normalize to [0, 1] on the FIXED physical range, then clamp. Masked pixels are set to
    # 0.0 first only to avoid NaN propagation through the colormap; their alpha is zeroed
    # afterwards so the (arbitrary) colour underneath is invisible.
    norm = (arr - float(vmin)) / (float(vmax) - float(vmin))
    norm = np.where(nan_mask, 0.0, norm)
    norm = np.clip(norm, 0.0, 1.0)

    colormap = get_colormap(cmap)
    rgba = colormap(norm, bytes=True)  # -> (H, W, 4) uint8
    rgba = np.array(rgba, dtype=np.uint8, copy=True)
    rgba[nan_mask, 3] = 0  # NaN -> fully transparent
    return rgba


__all__ = ["get_colormap", "bt_to_rgba"]

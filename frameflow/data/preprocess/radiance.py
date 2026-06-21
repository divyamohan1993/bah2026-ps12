"""Radiance / count -> brightness-temperature (Kelvin) conversions.

Two physical conversions live here, both grounded in research/02 §A3 and research/03 §5.1:

* :func:`radiance_to_bt` — the GOES/Himawari/GK-2A *Planck inverse*. Geostationary
  longwave-IR L1b products store spectral radiance; the file ships per-band Planck
  coefficients (``planck_fk1``/``fk2`` and the band-correction ``planck_bc1``/``bc2``) so
  the inverse Planck function recovers brightness temperature in Kelvin::

      Tb = ( fk2 / ln(fk1 / Rad + 1) - bc1 ) / bc2

* :func:`count_to_bt` — the INSAT-3DS/3DR path. INSAT-3D L1B Imager files store integer
  *counts* plus, for the thermal channels, a count->temperature lookup table
  (``IMG_TIR1_TEMP``). When the LUT is present BT is a direct gather ``lut[counts]``; when
  only radiance-calibration (quadratic gain/offset) coefficients are present we recover
  radiance first and then apply the Planck inverse (research/03 §5.1).

Functions accept and return plain ``numpy.ndarray`` (or any array supporting the ufuncs);
the convenience wrappers preserve :class:`xarray.DataArray` metadata when given one. NaNs
and non-physical/zero radiances are handled so off-disk ("space") pixels stay NaN rather
than producing warnings or ``inf``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


__all__ = [
    "radiance_to_bt",
    "count_to_bt",
    "radiance_to_bt_dataarray",
]


def radiance_to_bt(
    radiance: "np.ndarray | Any",
    fk1: float,
    fk2: float,
    bc1: float,
    bc2: float,
) -> "np.ndarray":
    r"""Convert spectral radiance to brightness temperature via the inverse Planck function.

    This is the standard GOES-R ABI (and, with the equivalent coefficients, Himawari AHI /
    GK-2A AMI) L1b conversion (research/02 §A3)::

        Tb = ( fk2 / ln(fk1 / Rad + 1) - bc1 ) / bc2     [Kelvin]

    where ``fk1`` (``planck_fk1``) and ``fk2`` (``planck_fk2``) are the forward-Planck
    constants for the band's central wavenumber and ``bc1`` (``planck_bc1``) /
    ``bc2`` (``planck_bc2``) are the band-correction offset and slope. The coefficients are
    read straight from the source NetCDF for the channel in question.

    Args:
        radiance: spectral radiance array (ABI units ``mW m-2 sr-1 (cm-1)-1``). May contain
            NaN (space pixels); non-positive values are masked to NaN to avoid ``log`` of a
            non-positive argument.
        fk1: forward-Planck constant ``planck_fk1``.
        fk2: forward-Planck constant ``planck_fk2``.
        bc1: band-correction offset ``planck_bc1``.
        bc2: band-correction slope ``planck_bc2``.

    Returns:
        ``numpy.ndarray`` of brightness temperature in Kelvin, float64, NaN where the input
        was NaN or non-positive.

    Raises:
        ValueError: if ``fk1`` or ``bc2`` is zero (degenerate coefficients).
    """
    import numpy as np  # lazy

    fk1 = float(fk1)
    fk2 = float(fk2)
    bc1 = float(bc1)
    bc2 = float(bc2)
    if fk1 == 0.0:
        raise ValueError("radiance_to_bt: planck_fk1 must be non-zero")
    if bc2 == 0.0:
        raise ValueError("radiance_to_bt: planck_bc2 must be non-zero")

    rad = np.asarray(radiance, dtype=np.float64)
    # Mask non-physical radiance (<=0) and propagate existing NaNs so log() is always valid.
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(rad) & (rad > 0.0)
    safe_rad = np.where(valid, rad, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        bt = (fk2 / np.log(fk1 / safe_rad + 1.0) - bc1) / bc2

    bt = np.where(valid, bt, np.nan)
    return bt


def count_to_bt(
    counts: "np.ndarray | Any",
    *,
    lut: "np.ndarray | Any | None" = None,
    fk1: float | None = None,
    fk2: float | None = None,
    bc1: float | None = None,
    bc2: float | None = None,
    gain: float | None = None,
    offset: float | None = None,
    quadratic: float | None = None,
    fill_value: float | None = None,
) -> "np.ndarray":
    r"""Convert integer detector *counts* to brightness temperature (Kelvin) — INSAT path.

    INSAT-3D/3DR/3DS L1B Imager thermal channels store counts. Two recovery routes are
    supported (research/03 §5.1, MOSDAC INSAT3D product format):

    1. **Direct count->temperature LUT** (preferred when present). The HDF5 ships a
       ``IMG_TIR1_TEMP`` lookup table indexed by count value; BT is the gather
       ``lut[counts]``. This is the most accurate, vendor-supplied path.
    2. **Quadratic radiance calibration + Planck inverse**. When only radiance-calibration
       coefficients are present, radiance is recovered as
       ``Rad = quadratic*c^2 + gain*c + offset`` (``quadratic`` defaults to 0 for the
       common linear case) and then :func:`radiance_to_bt` is applied with the Planck
       coefficients.

    Args:
        counts: integer count array (any shape). NaN / out-of-range entries become NaN.
        lut: optional 1-D count->temperature lookup table (Kelvin). If given, route 1 is
            used and the radiance coefficients are ignored.
        fk1, fk2, bc1, bc2: Planck coefficients for route 2.
        gain, offset: linear radiance-calibration slope/intercept for route 2.
        quadratic: optional quadratic radiance-calibration term (default 0).
        fill_value: optional sentinel count value to treat as missing (-> NaN).

    Returns:
        ``numpy.ndarray`` of brightness temperature in Kelvin (float64), NaN where invalid.

    Raises:
        ValueError: if neither a LUT nor a usable (gain + Planck) coefficient set is given.
    """
    import numpy as np  # lazy

    c = np.asarray(counts)
    finite = np.isfinite(c) if np.issubdtype(c.dtype, np.floating) else np.ones(c.shape, dtype=bool)
    if fill_value is not None:
        finite = finite & (c != fill_value)

    # ---- Route 1: direct LUT gather ----------------------------------------------------
    if lut is not None:
        lut_arr = np.asarray(lut, dtype=np.float64)
        idx = np.asarray(c, dtype=np.int64)
        in_range = (idx >= 0) & (idx < lut_arr.shape[0])
        safe_idx = np.where(in_range, idx, 0)
        bt = lut_arr[safe_idx]
        bt = np.where(in_range & finite, bt, np.nan)
        # A LUT may itself encode "missing" as a non-physical temperature; clamp absurd values.
        with np.errstate(invalid="ignore"):
            bt = np.where((bt > 0.0) & np.isfinite(bt), bt, np.nan)
        return bt

    # ---- Route 2: quadratic radiance calibration -> Planck inverse ---------------------
    if gain is None or fk1 is None or fk2 is None or bc1 is None or bc2 is None:
        raise ValueError(
            "count_to_bt: provide either a count->temperature `lut`, or radiance "
            "calibration (`gain`, `offset`) together with Planck coefficients "
            "(`fk1`, `fk2`, `bc1`, `bc2`)."
        )
    cf = np.asarray(c, dtype=np.float64)
    q = 0.0 if quadratic is None else float(quadratic)
    off = 0.0 if offset is None else float(offset)
    rad = q * cf * cf + float(gain) * cf + off
    rad = np.where(finite, rad, np.nan)
    return radiance_to_bt(rad, fk1, fk2, bc1, bc2)


def radiance_to_bt_dataarray(da: "Any", **coeffs: float) -> "Any":
    """Convert an :class:`xarray.DataArray` of radiance to a BT DataArray (Kelvin).

    Thin wrapper over :func:`radiance_to_bt` that preserves dims/coords and updates the
    ``units``/``long_name`` attributes. Planck coefficients are read from ``coeffs`` if
    given, else from the DataArray's own attributes (keys ``planck_fk1`` etc.), matching the
    flat ``frameflow.data.preprocess.radiance_to_bt(da, **coeffs)`` contract entry.

    Args:
        da: radiance :class:`xarray.DataArray`.
        **coeffs: optional ``fk1``/``fk2``/``bc1``/``bc2`` overrides.

    Returns:
        A brightness-temperature :class:`xarray.DataArray` in Kelvin.
    """
    import xarray as xr  # lazy

    def _coef(*names: str) -> float:
        for n in names:
            if n in coeffs:
                return float(coeffs[n])
            if n in da.attrs:
                return float(da.attrs[n])
        raise ValueError(
            f"radiance_to_bt_dataarray: missing Planck coefficient (tried {names}); "
            "pass it explicitly or attach it to the DataArray attrs."
        )

    fk1 = _coef("fk1", "planck_fk1")
    fk2 = _coef("fk2", "planck_fk2")
    bc1 = _coef("bc1", "planck_bc1")
    bc2 = _coef("bc2", "planck_bc2")

    bt = radiance_to_bt(da.values, fk1, fk2, bc1, bc2)
    out = xr.DataArray(
        bt.astype("float32"),
        dims=da.dims,
        coords=da.coords,
        attrs={**da.attrs, "units": "K", "long_name": "brightness_temperature"},
        name="bt",
    )
    return out

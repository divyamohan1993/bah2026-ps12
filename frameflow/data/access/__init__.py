"""Per-satellite access drivers + the flat ``access`` contract surface (CONTRACTS.md §3.1).

Each concrete driver (:class:`GOESSource`, :class:`HimawariSource`, :class:`GK2ASource`,
:class:`INSATSource`) subclasses :class:`~frameflow.data.access.base.SatelliteSource` and
implements ``list_files`` / ``download`` / ``read_frame``. This module additionally exposes
the two flat functions CONTRACTS.md §3.1 names — :func:`list_frames` and :func:`open_frame`
— as thin dispatchers over the right driver, so callers can use either the granular driver
API or the contract's ``frameflow.data.access.<fn>`` names.

All network access is lazy: nothing touches S3/MOSDAC until you call a listing/read, and any
offline/credential failure raises a clear :class:`~frameflow.data.access.base.OfflineError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import OfflineError, SatelliteSource, parse_time
from .gk2a import GK2ASource
from .goes import GOESSource
from .himawari import HimawariSource
from .insat import INSATSource

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr


__all__ = [
    "SatelliteSource",
    "OfflineError",
    "parse_time",
    "GOESSource",
    "HimawariSource",
    "GK2ASource",
    "INSATSource",
    "get_source",
    "list_frames",
    "open_frame",
]


# Map of canonical source keys -> (driver class, constructor kwargs). Accepts a few aliases
# so callers can say "GOES-19", "goes", "himawari", "insat", etc.
def get_source(source: str, *, anon: bool = True, **kw: Any) -> SatelliteSource:
    """Construct the :class:`SatelliteSource` driver for a satellite/source key.

    Args:
        source: a satellite key such as ``"GOES-19"``, ``"GOES-18"``, ``"Himawari-9"``,
            ``"GK-2A"``, ``"INSAT-3DS"``, ``"INSAT-3DR"`` (case/separator insensitive; the
            aliases ``"goes"``, ``"himawari"``, ``"gk2a"``, ``"insat"`` also work).
        anon: anonymous (no-credential) access for the public S3 buckets.
        **kw: extra driver-specific constructor arguments (e.g. ``product="CMIPF"``).

    Returns:
        A ready (but not-yet-connected) :class:`SatelliteSource` driver.

    Raises:
        ValueError: if ``source`` is not a recognized satellite/source key.
    """
    key = source.strip().lower().replace("_", "-").replace(" ", "-")
    if key in {"goes-19", "goes19", "goes", "g19"}:
        return GOESSource("GOES-19", anon=anon, **kw)
    if key in {"goes-18", "goes18", "g18"}:
        return GOESSource("GOES-18", anon=anon, **kw)
    if key in {"himawari-9", "himawari9", "himawari", "h09", "ahi"}:
        return HimawariSource(anon=anon, **kw)
    if key in {"gk-2a", "gk2a", "ami"}:
        return GK2ASource(anon=anon, **kw)
    if key in {"insat-3ds", "insat3ds", "insat", "3ds"}:
        return INSATSource("INSAT-3DS", anon=anon, **kw)
    if key in {"insat-3dr", "insat3dr", "3dr"}:
        return INSATSource("INSAT-3DR", anon=anon, **kw)
    raise ValueError(
        f"Unknown source {source!r}. Known: GOES-19, GOES-18, Himawari-9, GK-2A, "
        f"INSAT-3DS, INSAT-3DR."
    )


def list_frames(
    source: str,
    start: str,
    end: str,
    *,
    channel: str = "C13",
    anon: bool = True,
) -> list[str]:
    """Return source URLs/keys for frames in ``[start, end]`` (CONTRACTS.md §3.1).

    Dispatches to the right :class:`SatelliteSource` driver for ``source`` and lists its
    thermal-IR clean-window frames in the UTC interval (e.g. GOES-19 ABI C13 on S3). The
    ``channel`` is passed through to the driver, which interprets it appropriately (GOES
    ``C13``, Himawari band 13, GK-2A ``ir105``, INSAT ``TIR1``).

    Args:
        source: satellite/source key (see :func:`get_source`).
        start: inclusive start time, ISO-8601 (e.g. ``"2025-06-20T00:00:00Z"``).
        end: inclusive end time, ISO-8601.
        channel: TIR channel/band identifier (default ``"C13"``).
        anon: anonymous access for the public buckets.

    Returns:
        A sorted list of source identifiers (``s3://`` URIs / catalog ids / file paths).

    Raises:
        OfflineError: if listing requires a network/credentialed call that fails.
        ValueError: if ``source`` is unknown.
    """
    drv = get_source(source, anon=anon)
    return drv.list_files(start, end, channel=channel)


def open_frame(url: str, *, source: str | None = None, anon: bool = True) -> "xr.DataArray":
    """Open one source frame lazily and return its TIR brightness-temperature ``DataArray`` (K).

    Applies the appropriate radiance->BT (ABI/AHI/AMI Planck inverse) or count->BT (INSAT
    LUT) conversion (research/02 §A3, research/03 §5.1). When ``source`` is not given it is
    inferred from the file/URL name; pass it explicitly when the name is ambiguous.

    Args:
        url: a local path or ``s3://`` URI to one source frame.
        source: optional satellite/source key to force a specific driver.
        anon: anonymous access for public buckets.

    Returns:
        A brightness-temperature :class:`xarray.DataArray` named ``"bt"`` in Kelvin, with
        ``lat``/``lon`` coordinates attached where the driver can derive them.

    Raises:
        OfflineError: if opening requires a network call that fails.
        ValueError: if the source driver cannot be inferred from ``url``.
    """
    key = source if source is not None else _infer_source(url)
    drv = get_source(key, anon=anon)
    return drv.read_frame(url)


def _infer_source(url: str) -> str:
    """Best-effort satellite inference from a file/URL name (for :func:`open_frame`)."""
    u = url.lower()
    if "goes19" in u or "noaa-goes19" in u or "_g19_" in u:
        return "GOES-19"
    if "goes18" in u or "noaa-goes18" in u or "_g18_" in u:
        return "GOES-18"
    if "himawari" in u or "hs_h09" in u or "_h09_" in u or "ahi" in u:
        return "Himawari-9"
    if "gk2a" in u or "gk-2a" in u or "ami" in u or "ir105" in u:
        return "GK-2A"
    if "3dr" in u or "insat-3dr" in u or "3rimg" in u:
        return "INSAT-3DR"
    if "insat" in u or "3ds" in u or "3simg" in u or u.endswith(".h5"):
        return "INSAT-3DS"
    raise ValueError(
        f"Could not infer the satellite source from {url!r}; pass source=... explicitly "
        f"(e.g. source='GOES-19')."
    )

"""``SatelliteSource`` — the abstract base class every sensor access driver implements.

A :class:`SatelliteSource` is the uniform interface the ingest layer uses to pull thermal-IR
frames from any satellite archive (GOES on S3, Himawari on S3/JAXA, GK-2A on S3, INSAT on
MOSDAC/EUMETSAT). Concrete drivers (``goes.py``, ``himawari.py``, ``gk2a.py``, ``insat.py``)
override three methods:

* :meth:`list_files` — enumerate source keys/URLs in a UTC time range (lazy; network only
  when called).
* :meth:`download`   — fetch keys to a local directory (or pass through already-local paths).
* :meth:`read_frame` — open one file and return its TIR brightness temperature as an
  :class:`xarray.DataArray` in **Kelvin** with ``lat``/``lon`` coordinates.

Network access is always lazy and any offline/credential failure raises a clear, actionable
:class:`OfflineError` (a subclass of :class:`RuntimeError`) rather than a cryptic socket
error, so the pipeline degrades gracefully when run without connectivity.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr


__all__ = ["SatelliteSource", "OfflineError", "parse_time"]


class OfflineError(RuntimeError):
    """Raised when a network/credentialed source cannot be reached or authenticated.

    Carries an actionable message (which bucket/portal, what credentials, how to retry) so a
    user running offline understands exactly what to do, instead of seeing a raw connection
    or botocore error.
    """


def parse_time(value: "str | datetime") -> datetime:
    """Parse a flexible time input into a timezone-aware UTC :class:`datetime`.

    Accepts a :class:`datetime` (naive treated as UTC) or an ISO-8601 string, including the
    trailing ``Z`` form used throughout the contracts (e.g. ``"2025-06-20T00:10:00Z"``).

    Args:
        value: a datetime or ISO-8601 string.

    Returns:
        A tz-aware UTC :class:`datetime`.

    Raises:
        ValueError: if a string cannot be parsed.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse time {value!r}; use a datetime or ISO-8601 string "
            f"(e.g. '2025-06-20T00:10:00Z')."
        ) from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class SatelliteSource(abc.ABC):
    """Abstract access driver for one satellite's thermal-IR clean-window channel.

    Subclasses set :attr:`satellite` / :attr:`default_channel` / :attr:`wavelength_um` and
    implement :meth:`list_files`, :meth:`download`, and :meth:`read_frame`.

    Attributes:
        satellite: canonical satellite key (matches ``constants.SATELLITE_BANDS``).
        default_channel: the default TIR channel/band identifier (e.g. ``"C13"``).
        wavelength_um: central wavelength of that channel in micrometres.
        anon: whether anonymous (no-credential) access is used (S3 public buckets).
    """

    satellite: str = ""
    default_channel: str = ""
    wavelength_um: float = 0.0

    def __init__(self, *, anon: bool = True) -> None:
        """Initialize the driver.

        Args:
            anon: use anonymous access for public buckets (default True).
        """
        self.anon = anon

    # -- interface ---------------------------------------------------------------------
    @abc.abstractmethod
    def list_files(
        self, start: "str | datetime", end: "str | datetime", **kw: Any
    ) -> list[str]:
        """Return source URLs/keys for frames in the UTC interval ``[start, end]``.

        Args:
            start: inclusive start time (datetime or ISO-8601 string).
            end: inclusive end time.
            **kw: driver-specific options (e.g. ``channel``, ``domain``).

        Returns:
            A sorted list of source identifiers (S3 URIs / file paths / catalog ids).

        Raises:
            OfflineError: if listing requires a network/credentialed call that fails.
        """

    @abc.abstractmethod
    def download(self, keys: list[str], out_dir: str | Path) -> list[Path]:
        """Fetch ``keys`` into ``out_dir`` and return the local file paths.

        Implementations should skip files already present locally and pass through inputs
        that are already local paths.

        Args:
            keys: source identifiers from :meth:`list_files`.
            out_dir: destination directory (created if missing).

        Returns:
            Local :class:`pathlib.Path` objects in the same order as ``keys``.

        Raises:
            OfflineError: if a download requires a network call that fails.
        """

    @abc.abstractmethod
    def read_frame(self, path: str | Path) -> "xr.DataArray":
        """Open one source file and return TIR brightness temperature (Kelvin).

        The returned :class:`xarray.DataArray` is named ``"bt"``, carries ``units="K"`` and
        ``lat``/``lon`` coordinates, and has any radiance->BT (Planck) or count->BT (LUT)
        conversion already applied.

        Args:
            path: a local file path (or an ``s3://`` URI a driver can open lazily).

        Returns:
            A brightness-temperature :class:`xarray.DataArray` in Kelvin.
        """

    # -- shared helpers ----------------------------------------------------------------
    def list_and_download(
        self,
        start: "str | datetime",
        end: "str | datetime",
        out_dir: str | Path,
        **kw: Any,
    ) -> list[Path]:
        """Convenience: :meth:`list_files` then :meth:`download` in one call."""
        keys = self.list_files(start, end, **kw)
        return self.download(keys, out_dir)

    @staticmethod
    def _s3_filesystem(anon: bool):
        """Build an ``s3fs.S3FileSystem`` with a clear error if s3fs/network is unavailable.

        Raises:
            OfflineError: if ``s3fs`` cannot be imported.
        """
        try:
            import s3fs  # lazy
        except ImportError as exc:  # pragma: no cover - s3fs is installed in this env
            raise OfflineError(
                "s3fs is required to access public satellite S3 buckets but is not "
                "installed. Install it (`pip install s3fs`) or use a local file path."
            ) from exc
        return s3fs.S3FileSystem(anon=anon)

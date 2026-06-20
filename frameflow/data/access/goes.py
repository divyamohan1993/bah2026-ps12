"""GOES-19 ABI Channel 13 (10.3 µm "clean" longwave window) — PRIMARY training source.

Access is via the public, no-auth NOAA Open Data bucket on AWS (research/02 §1):

    s3://noaa-goes19/ABI-L1b-RadF/{year}/{doy:03d}/{hour:02d}/   (Full Disk, 10-min)
    s3://noaa-goes19/ABI-L2-CMIPF/{year}/{doy:03d}/{hour:02d}/   (BT-ready CMI, Kelvin)

We list Channel-13 files by UTC time range (the bucket is partitioned by year / day-of-year
/ hour), download them with anonymous :class:`s3fs.S3FileSystem`, and read them into
brightness temperature:

* ``ABI-L1b-RadF`` stores spectral radiance ``Rad`` plus the per-band Planck coefficients
  (``planck_fk1``/``fk2``/``bc1``/``bc2``); we delegate the inverse Planck to
  :func:`frameflow.data.preprocess.radiance.radiance_to_bt` (research/02 §A3).
* ``ABI-L2-CMIPF`` already provides brightness temperature in the ``CMI`` variable, so no
  conversion is needed — handy as a faster path.

The ABI fixed-grid geostationary projection is converted to lat/lon and attached as
``lat``/``lon`` coordinates so downstream regridding (research/03 §4) has georeferencing.
All network calls are lazy and surface a clear :class:`OfflineError` when offline.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import constants as C
from .base import OfflineError, SatelliteSource, parse_time

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr


__all__ = ["GOESSource"]


# Parse the scan-start stamp from an ABI filename: ..._sYYYYJJJHHMMSSs_...
_ABI_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


class GOESSource(SatelliteSource):
    """GOES-19 (default) / GOES-18 ABI Channel-13 access driver over anonymous S3.

    Args:
        satellite: ``"GOES-19"`` (default) or ``"GOES-18"``.
        product: ``"RadF"`` (L1b radiance, default) or ``"CMIPF"`` (L2 BT-ready), and the
            CONUS variants ``"RadC"``/``"CMIPC"``.
        anon: anonymous S3 access (default True).
    """

    default_channel = "C13"

    def __init__(
        self,
        satellite: str = "GOES-19",
        *,
        product: str = "RadF",
        anon: bool = True,
    ) -> None:
        super().__init__(anon=anon)
        self.satellite = satellite
        self.product = product
        spec = C.SATELLITE_BANDS.get(satellite, C.SATELLITE_BANDS["GOES-19"])
        self.wavelength_um = float(spec["wavelength_um"])
        self._bucket = "noaa-goes19" if satellite == "GOES-19" else "noaa-goes18"
        self._sat_short = "G19" if satellite == "GOES-19" else "G18"

    # -- listing -----------------------------------------------------------------------
    def list_files(
        self, start: "str | datetime", end: "str | datetime", **kw: Any
    ) -> list[str]:
        """List GOES ABI C13 files in ``[start, end]`` (UTC) for the configured product.

        Args:
            start: inclusive start time.
            end: inclusive end time.
            **kw: ``channel`` (default ``"C13"``), ``product`` (override the instance default,
                e.g. ``"RadC"`` for the 5-min CONUS scan).

        Returns:
            Sorted list of ``s3://`` URIs whose scan-start time falls in the interval.

        Raises:
            OfflineError: if the S3 listing cannot be performed (offline / s3fs missing).
        """
        channel = kw.get("channel", self.default_channel)
        product = kw.get("product", self.product)
        t0 = parse_time(start)
        t1 = parse_time(end)
        if t1 < t0:
            t0, t1 = t1, t0

        fs = self._s3_filesystem(self.anon)
        prefix = f"ABI-L1b-{product}" if product.startswith("Rad") else f"ABI-L2-{product}"

        results: list[str] = []
        for year, doy, hour in _iter_year_doy_hour(t0, t1):
            key = f"{self._bucket}/{prefix}/{year}/{doy:03d}/{hour:02d}/"
            try:
                listing = fs.ls(key)
            except FileNotFoundError:
                continue
            except Exception as exc:  # network/credential failure
                raise OfflineError(
                    f"Failed to list GOES files at s3://{key} (anon={self.anon}). "
                    f"This needs network access to the public NOAA bucket. "
                    f"Underlying error: {exc}"
                ) from exc
            for f in listing:
                name = f.rsplit("/", 1)[-1]
                if f"{channel}_" not in name and not name.split("-")[-1].startswith(channel):
                    if channel not in name:
                        continue
                stamp = _abi_scan_start(name)
                if stamp is not None and t0 <= stamp <= t1:
                    results.append(f"s3://{f}" if not f.startswith("s3://") else f)
        return sorted(results)

    # -- download ----------------------------------------------------------------------
    def download(self, keys: list[str], out_dir: str | Path) -> list[Path]:
        """Download S3 ``keys`` to ``out_dir`` (skipping files already present).

        Args:
            keys: ``s3://`` URIs from :meth:`list_files`.
            out_dir: destination directory (created if missing).

        Returns:
            Local file paths in the same order as ``keys``.

        Raises:
            OfflineError: if a download fails (offline / s3fs missing).
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fs = self._s3_filesystem(self.anon)
        paths: list[Path] = []
        for key in keys:
            src = key[len("s3://"):] if key.startswith("s3://") else key
            local = out / src.rsplit("/", 1)[-1]
            if not local.exists():
                try:
                    fs.get(src, str(local))
                except Exception as exc:
                    raise OfflineError(
                        f"Failed to download s3://{src} to {local}. Needs network access to "
                        f"the public NOAA bucket. Underlying error: {exc}"
                    ) from exc
            paths.append(local)
        return paths

    # -- read --------------------------------------------------------------------------
    def read_frame(self, path: str | Path) -> "xr.DataArray":
        """Open a GOES ABI file and return Channel-13 brightness temperature (Kelvin).

        Handles both ``RadF``/``RadC`` (radiance -> Planck inverse) and ``CMIPF``/``CMIPC``
        (``CMI`` already in Kelvin), and attaches ``lat``/``lon`` coordinates derived from
        the ABI fixed-grid geostationary projection.

        Args:
            path: local file path, or an ``s3://`` URI to open lazily.

        Returns:
            A brightness-temperature :class:`xarray.DataArray` (``"bt"``, Kelvin) with
            ``lat``/``lon`` coords.
        """
        import xarray as xr  # lazy

        from ..preprocess.radiance import radiance_to_bt

        ds = self._open_dataset(path)
        try:
            if "CMI" in ds.data_vars:
                bt_vals = ds["CMI"].values.astype("float32")
                dims = ds["CMI"].dims
            elif "Rad" in ds.data_vars:
                rad = ds["Rad"]
                fk1 = float(ds["planck_fk1"].values)
                fk2 = float(ds["planck_fk2"].values)
                bc1 = float(ds["planck_bc1"].values)
                bc2 = float(ds["planck_bc2"].values)
                bt_vals = radiance_to_bt(rad.values, fk1, fk2, bc1, bc2).astype("float32")
                dims = rad.dims
            else:
                raise ValueError(
                    f"GOES file {path} has neither 'CMI' nor 'Rad' — not an ABI L1b/L2 "
                    f"Channel imagery file."
                )

            lats, lons = _abi_latlon(ds)
            da = xr.DataArray(
                bt_vals,
                dims=dims,
                coords={"lat": (dims, lats), "lon": (dims, lons)},
                attrs={
                    "units": "K",
                    "long_name": "brightness_temperature",
                    "satellite": self.satellite,
                    "channel": self.default_channel,
                    "wavelength_um": self.wavelength_um,
                },
                name="bt",
            )
            return da
        finally:
            ds.close()

    # -- internals ---------------------------------------------------------------------
    def _open_dataset(self, path: str | Path) -> "Any":
        import xarray as xr  # lazy

        p = str(path)
        if p.startswith("s3://"):
            fs = self._s3_filesystem(self.anon)
            try:
                return xr.open_dataset(fs.open(p[len("s3://"):]))
            except Exception as exc:
                raise OfflineError(
                    f"Failed to open {p} from S3. Needs network access to the public NOAA "
                    f"bucket. Underlying error: {exc}"
                ) from exc
        return xr.open_dataset(p)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _iter_year_doy_hour(t0: datetime, t1: datetime):
    """Yield ``(year, day_of_year, hour)`` tuples covering ``[t0, t1]`` (hourly)."""
    cur = t0.replace(minute=0, second=0, microsecond=0)
    while cur <= t1:
        yield cur.year, cur.timetuple().tm_yday, cur.hour
        cur += timedelta(hours=1)


def _abi_scan_start(filename: str) -> datetime | None:
    """Parse the scan-start UTC time from an ABI filename, or None if it doesn't match."""
    m = _ABI_START_RE.search(filename)
    if not m:
        return None
    year, doy, hh, mm, ss = (int(g) for g in m.groups())
    base = datetime(year, 1, 1, tzinfo=parse_time("2000-01-01T00:00:00Z").tzinfo)
    return base + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)


def _abi_latlon(ds: "Any") -> "tuple[Any, Any]":
    """Convert the ABI fixed-grid (GEOS) scan angles to latitude/longitude arrays.

    Uses the ``goes_imager_projection`` metadata (satellite height, sweep axis,
    sub-satellite longitude) and the ``x``/``y`` scan-angle coordinates, applying the closed
    -form GEOS->geodetic transform (GOES-R PUG vol.3). Off-disk pixels become NaN.

    Args:
        ds: an open ABI :class:`xarray.Dataset`.

    Returns:
        ``(lat2d, lon2d)`` float64 arrays matching the image grid.
    """
    import numpy as np  # lazy

    proj = ds["goes_imager_projection"]
    r_eq = float(proj.attrs["semi_major_axis"])
    r_pol = float(proj.attrs["semi_minor_axis"])
    h_sat = float(proj.attrs["perspective_point_height"]) + r_eq
    lon_0 = float(proj.attrs["longitude_of_projection_origin"])

    x = np.asarray(ds["x"].values, dtype=np.float64)  # E-W scan angle (radians)
    y = np.asarray(ds["y"].values, dtype=np.float64)  # N-S scan angle (radians)
    xx, yy = np.meshgrid(x, y)

    lambda_0 = np.deg2rad(lon_0)
    sin_x = np.sin(xx)
    cos_x = np.cos(xx)
    sin_y = np.sin(yy)
    cos_y = np.cos(yy)

    a = sin_x ** 2 + cos_x ** 2 * (cos_y ** 2 + (r_eq ** 2 / r_pol ** 2) * sin_y ** 2)
    b = -2.0 * h_sat * cos_x * cos_y
    c = h_sat ** 2 - r_eq ** 2

    with np.errstate(invalid="ignore"):
        discriminant = b ** 2 - 4.0 * a * c
        r_s = (-b - np.sqrt(discriminant)) / (2.0 * a)
        sx = r_s * cos_x * cos_y
        sy = -r_s * sin_x
        sz = r_s * cos_x * sin_y

        lat = np.rad2deg(
            np.arctan((r_eq ** 2 / r_pol ** 2) * (sz / np.sqrt((h_sat - sx) ** 2 + sy ** 2)))
        )
        lon = np.rad2deg(lambda_0 - np.arctan(sy / (h_sat - sx)))

    off_disk = ~np.isfinite(discriminant) | (discriminant < 0)
    lat = np.where(off_disk, np.nan, lat)
    lon = np.where(off_disk, np.nan, lon)
    return lat, lon

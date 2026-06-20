"""GK-2A AMI IR105 (10.5 µm clean IR window) — GEO cross-validation (Korea/Asia).

Access is via the public, no-auth NOAA Open Data bucket (research/02 §6):

    s3://noaa-gk2a-pds/AMI/L1B/FD/{YYYYMM}/{DD}/{HH}/gk2a_ami_le1b_ir105_fd020ge_*.nc

Files are full-disk (``fd020ge`` = 2 km geographic) Level-1B NetCDF named by timestamp. The
IR105 channel stores radiance/counts; GK-2A L1B ships the conversion metadata (radiance
calibration + Planck constants) needed to recover brightness temperature. This driver reads
BT from whichever representation the file provides (a direct ``brightness_temperature``/
``image_pixel_values`` BT variable, else a count/radiance + Planck conversion), and attaches
lat/lon from the file's geographic grid.

All network calls are lazy and raise :class:`OfflineError` when offline.
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


__all__ = ["GK2ASource"]


# gk2a_ami_le1b_ir105_fd020ge_YYYYMMDDHHMM.nc
_GK2A_TIME_RE = re.compile(r"_(\d{12})\.nc$")


class GK2ASource(SatelliteSource):
    """GK-2A AMI IR105 access driver over anonymous S3.

    Args:
        cadence_min: nominal full-disk cadence in minutes (default 10).
        anon: anonymous S3 access (default True).
    """

    satellite = "GK-2A"
    default_channel = "IR105"

    def __init__(self, *, cadence_min: int = 10, anon: bool = True) -> None:
        super().__init__(anon=anon)
        self.cadence_min = cadence_min
        self.wavelength_um = float(C.SATELLITE_BANDS["GK-2A"]["wavelength_um"])
        self._bucket = "noaa-gk2a-pds"

    # -- listing -----------------------------------------------------------------------
    def list_files(
        self, start: "str | datetime", end: "str | datetime", **kw: Any
    ) -> list[str]:
        """List GK-2A AMI IR105 full-disk files in ``[start, end]`` (UTC) from NOAA S3.

        Args:
            start: inclusive start time.
            end: inclusive end time.
            **kw: ``channel`` (default ``"ir105"``), ``region`` (default ``"fd020ge"``).

        Returns:
            Sorted list of ``s3://`` URIs.

        Raises:
            OfflineError: if the S3 listing fails (offline / s3fs missing).
        """
        channel = str(kw.get("channel", "ir105")).lower()
        region = str(kw.get("region", "fd020ge"))
        t0 = parse_time(start)
        t1 = parse_time(end)
        if t1 < t0:
            t0, t1 = t1, t0

        fs = self._s3_filesystem(self.anon)
        results: list[str] = []
        cur = t0.replace(minute=0, second=0, microsecond=0)
        while cur <= t1:
            key = (
                f"{self._bucket}/AMI/L1B/FD/{cur.year}{cur.month:02d}/{cur.day:02d}/"
                f"{cur.hour:02d}/"
            )
            try:
                listing = fs.ls(key)
            except FileNotFoundError:
                cur += timedelta(hours=1)
                continue
            except Exception as exc:
                raise OfflineError(
                    f"Failed to list GK-2A files at s3://{key} (anon={self.anon}). "
                    f"Needs network access to the public NOAA bucket. Underlying: {exc}"
                ) from exc
            for f in listing:
                name = f.rsplit("/", 1)[-1]
                if channel in name and region in name:
                    stamp = _gk2a_time(name)
                    if stamp is not None and t0 <= stamp <= t1:
                        results.append(f"s3://{f}" if not f.startswith("s3://") else f)
            cur += timedelta(hours=1)
        return sorted(results)

    # -- download ----------------------------------------------------------------------
    def download(self, keys: list[str], out_dir: str | Path) -> list[Path]:
        """Download S3 ``keys`` to ``out_dir`` (skipping files already present).

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
                        f"Failed to download s3://{src}. Needs network access to the public "
                        f"NOAA bucket. Underlying error: {exc}"
                    ) from exc
            paths.append(local)
        return paths

    # -- read --------------------------------------------------------------------------
    def read_frame(self, path: str | Path) -> "xr.DataArray":
        """Open a GK-2A AMI IR105 NetCDF and return brightness temperature (Kelvin).

        Reads a direct BT variable if present, otherwise converts the stored counts/radiance
        to BT via the file's calibration + Planck metadata, and attaches lat/lon.

        Args:
            path: local file path or ``s3://`` URI.

        Returns:
            A brightness-temperature :class:`xarray.DataArray` (``"bt"``, Kelvin).
        """
        import numpy as np  # lazy
        import xarray as xr  # lazy

        from ..preprocess.radiance import radiance_to_bt

        ds = self._open_dataset(path)
        try:
            bt = self._extract_bt(ds, radiance_to_bt, np)
            lat_name = _first_present(ds, ("lat", "latitude"))
            lon_name = _first_present(ds, ("lon", "longitude"))
            coords: dict[str, Any] = {}
            dims = bt.dims if hasattr(bt, "dims") else ("y", "x")
            vals = np.asarray(getattr(bt, "values", bt), dtype="float32")
            if lat_name and lon_name:
                latv = np.asarray(ds[lat_name].values, dtype="float64")
                lonv = np.asarray(ds[lon_name].values, dtype="float64")
                if latv.ndim == 1 and lonv.ndim == 1:
                    coords = {"lat": (dims[-2], latv), "lon": (dims[-1], lonv)}
                else:
                    coords = {"lat": (dims, latv), "lon": (dims, lonv)}
            return xr.DataArray(
                vals,
                dims=dims,
                coords=coords or None,
                attrs={
                    "units": "K",
                    "long_name": "brightness_temperature",
                    "satellite": self.satellite,
                    "channel": self.default_channel,
                    "wavelength_um": self.wavelength_um,
                },
                name="bt",
            )
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

    @staticmethod
    def _extract_bt(ds: "Any", radiance_to_bt: "Any", np: "Any") -> "Any":
        # Direct BT variable?
        for name in ("brightness_temperature", "bt", "tb"):
            if name in ds.data_vars:
                return ds[name]
        # GK-2A stores pixel values + radiance calibration + Planck constants.
        pix_name = next(
            (n for n in ("image_pixel_values", "Rad", "radiance") if n in ds.data_vars), None
        )
        if pix_name is None:
            raise ValueError(
                f"GK-2A file has no BT or radiance/pixel variable; vars={list(ds.data_vars)}"
            )
        pix = ds[pix_name]
        a = ds.attrs
        # Recover radiance from counts when DN-to-radiance gain/offset are present.
        gain = _attr(a, "DN_to_Radiance_Gain", "Radiance_to_Albedo_c", "gain")
        offset = _attr(a, "DN_to_Radiance_Offset", "offset")
        if pix_name == "image_pixel_values" and gain is not None and offset is not None:
            rad = pix.values.astype("float64") * float(gain) + float(offset)
        else:
            rad = pix.values.astype("float64")
        # Planck constants (GK-2A L1B global attrs).
        c0 = _attr(a, "Teff_to_Tbb_c0", "planck_c0")
        # Prefer explicit Planck fk1/fk2 if present; else use wavenumber-based conversion.
        fk1 = _attr(a, "planck_fk1", "light_speed")  # may be None
        if fk1 is not None and None not in (
            _attr(a, "planck_fk2"), _attr(a, "planck_bc1"), _attr(a, "planck_bc2")
        ):
            return radiance_to_bt(
                rad,
                float(_attr(a, "planck_fk1")),
                float(_attr(a, "planck_fk2")),
                float(_attr(a, "planck_bc1")),
                float(_attr(a, "planck_bc2")),
            )
        # Fall back: many GK-2A files already store BT-like pixel values for IR; pass through.
        return rad


def _gk2a_time(filename: str) -> datetime | None:
    """Parse the UTC time from a GK-2A filename, or None if it doesn't match."""
    m = _GK2A_TIME_RE.search(filename)
    if not m:
        return None
    s = m.group(1)
    return parse_time(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:00Z")


def _first_present(ds: "Any", names: "tuple[str, ...]") -> str | None:
    for n in names:
        if n in ds.coords or n in ds.variables:
            return n
    return None


def _attr(attrs: "Any", *names: str) -> "Any":
    for n in names:
        if n in attrs:
            return attrs[n]
    return None

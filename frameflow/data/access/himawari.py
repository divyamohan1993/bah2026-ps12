"""Himawari-9 AHI Band 13 (10.4 µm clean IR window) — SECONDARY training (Asia-Pacific).

Two access routes are supported (research/02 §2):

* **Gridded NetCDF (JAXA P-Tree)** — a regular lat/lon grid, no segment stitching, easiest
  to read. This driver's :meth:`read_frame` reads that NetCDF variant: it looks for a
  brightness-temperature variable (``tbb_13`` / ``tbb`` / ``Band13`` …) already in Kelvin
  and attaches lat/lon. JAXA P-Tree requires free registration (SFTP/FTP), so listing that
  source is documented but credential-gated.

* **HSD ``.DAT.bz2`` (JMA Himawari Standard Data) on the public NOAA S3 bucket**
  ``s3://noaa-himawari9/AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/``.
  These are the 10 segmented, bzip2-compressed Band-13 tiles per time step; the canonical
  reader is satpy's ``ahi_hsd`` reader. :meth:`list_files` enumerates the Band-13 segments
  from S3 by UTC time; :meth:`read_frame` reads a stitched HSD scene via satpy when given
  the segment list (documented route). The path/segment naming is per research/02 §2.

All network calls are lazy and raise a clear :class:`OfflineError` when offline.
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


__all__ = ["HimawariSource"]


# AHI HSD filename: HS_H09_YYYYMMDD_HHMM_B13_FLDK_R20_S0101.DAT[.bz2]
_AHI_TIME_RE = re.compile(r"HS_H\d\d_(\d{8})_(\d{4})_B(\d\d)_")

# Candidate BT variable names in JAXA gridded NetCDF (band 13).
_BT_VAR_CANDIDATES = ("tbb_13", "tbb", "Band13", "band13", "TBB", "bt", "brightness_temperature")


class HimawariSource(SatelliteSource):
    """Himawari-9 AHI Band-13 access driver (NOAA S3 HSD listing + gridded-NetCDF read).

    Args:
        cadence_min: nominal full-disk cadence in minutes (default 10).
        anon: anonymous S3 access (default True).
    """

    satellite = "Himawari-9"
    default_channel = "B13"

    def __init__(self, *, cadence_min: int = 10, anon: bool = True) -> None:
        super().__init__(anon=anon)
        self.cadence_min = cadence_min
        self.wavelength_um = float(C.SATELLITE_BANDS["Himawari-9"]["wavelength_um"])
        self._bucket = "noaa-himawari9"

    # -- listing -----------------------------------------------------------------------
    def list_files(
        self, start: "str | datetime", end: "str | datetime", **kw: Any
    ) -> list[str]:
        """List Himawari-9 AHI Band-13 HSD segments in ``[start, end]`` (UTC) from NOAA S3.

        The bucket is partitioned ``AHI-L1b-FLDK/Y/M/D/HHMM/``; we walk each cadence step in
        the interval and collect the Band-13 (``B13``) segment files.

        Args:
            start: inclusive start time.
            end: inclusive end time.
            **kw: ``band`` (default 13).

        Returns:
            Sorted list of ``s3://`` URIs for Band-13 HSD segments.

        Raises:
            OfflineError: if the S3 listing fails (offline / s3fs missing).
        """
        band = int(kw.get("band", 13))
        t0 = parse_time(start)
        t1 = parse_time(end)
        if t1 < t0:
            t0, t1 = t1, t0

        fs = self._s3_filesystem(self.anon)
        results: list[str] = []
        # Snap to the cadence grid to enumerate HHMM folders.
        step = max(1, self.cadence_min)
        cur = t0.replace(second=0, microsecond=0)
        cur -= timedelta(minutes=cur.minute % step)
        while cur <= t1:
            key = (
                f"{self._bucket}/AHI-L1b-FLDK/{cur.year}/{cur.month:02d}/{cur.day:02d}/"
                f"{cur.hour:02d}{cur.minute:02d}/"
            )
            try:
                listing = fs.ls(key)
            except FileNotFoundError:
                cur += timedelta(minutes=step)
                continue
            except Exception as exc:
                raise OfflineError(
                    f"Failed to list Himawari files at s3://{key} (anon={self.anon}). "
                    f"Needs network access to the public NOAA bucket. Underlying: {exc}"
                ) from exc
            for f in listing:
                name = f.rsplit("/", 1)[-1]
                m = _AHI_TIME_RE.search(name)
                if m and int(m.group(3)) == band:
                    stamp = _ahi_time(name)
                    if stamp is not None and t0 <= stamp <= t1:
                        results.append(f"s3://{f}" if not f.startswith("s3://") else f)
            cur += timedelta(minutes=step)
        return sorted(results)

    # -- download ----------------------------------------------------------------------
    def download(self, keys: list[str], out_dir: str | Path) -> list[Path]:
        """Download S3 segment ``keys`` to ``out_dir`` (skipping files already present).

        Args:
            keys: ``s3://`` URIs from :meth:`list_files`.
            out_dir: destination directory.

        Returns:
            Local file paths in input order.

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
        """Open one Himawari frame and return Band-13 brightness temperature (Kelvin).

        Dispatches by extension/inputs:

        * a single ``.nc`` (JAXA P-Tree gridded NetCDF) is read directly — the BT variable
          (``tbb_13``/``tbb``/…) is already in Kelvin and carries lat/lon.
        * a list of HSD ``.DAT``/``.DAT.bz2`` segments (or a glob to them) is read via satpy's
          ``ahi_hsd`` reader and resampled-free BT is returned (documented route).

        Args:
            path: a ``.nc`` file path, a single HSD segment, or a list/glob of HSD segments.

        Returns:
            A brightness-temperature :class:`xarray.DataArray` (``"bt"``, Kelvin) with
            lat/lon coords.
        """
        # HSD route (list of segments, or a .DAT path)
        if isinstance(path, (list, tuple)) or str(path).endswith((".DAT", ".bz2")):
            return self._read_hsd(path)
        return self._read_gridded_netcdf(path)

    # -- internals ---------------------------------------------------------------------
    def _read_gridded_netcdf(self, path: str | Path) -> "xr.DataArray":
        import numpy as np  # lazy
        import xarray as xr  # lazy

        ds = xr.open_dataset(str(path))
        try:
            var = next((v for v in _BT_VAR_CANDIDATES if v in ds.data_vars), None)
            if var is None:
                raise ValueError(
                    f"Himawari NetCDF {path} has no recognized Band-13 BT variable "
                    f"(looked for {_BT_VAR_CANDIDATES}); got {list(ds.data_vars)}."
                )
            da = ds[var]
            lat_name = _first_present(ds, ("latitude", "lat"))
            lon_name = _first_present(ds, ("longitude", "lon"))
            vals = np.asarray(da.values, dtype="float32")
            coords: dict[str, Any] = {}
            if lat_name and lon_name:
                coords["lat"] = (da.dims[-2], np.asarray(ds[lat_name].values, dtype="float64"))
                coords["lon"] = (da.dims[-1], np.asarray(ds[lon_name].values, dtype="float64"))
            out = xr.DataArray(
                vals,
                dims=da.dims,
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
            return out
        finally:
            ds.close()

    def _read_hsd(self, path: "str | Path | list[str]") -> "xr.DataArray":
        try:
            from glob import glob

            from satpy import Scene  # lazy
        except ImportError as exc:
            raise OfflineError(
                "Reading Himawari HSD (.DAT.bz2) segments needs satpy with the 'ahi_hsd' "
                "reader. Install satpy, or use the JAXA gridded-NetCDF route instead."
            ) from exc

        if isinstance(path, (list, tuple)):
            files = [str(p) for p in path]
        else:
            files = glob(str(path))
        if not files:
            raise OfflineError(
                f"No Himawari HSD segments matched {path!r}. Download the 10 Band-13 "
                f"segments (S0101..S1010) first via list_files()/download()."
            )
        scn = Scene(reader="ahi_hsd", filenames=files)
        scn.load(["B13"])
        da = scn["B13"]
        # satpy DataArrays carry 'area'; expose lat/lon for downstream regridding.
        try:
            lons, lats = scn["B13"].attrs["area"].get_lonlats()
            da = da.assign_coords(lat=(da.dims, lats), lon=(da.dims, lons))
        except Exception:  # pragma: no cover - area always present for ahi_hsd
            pass
        da = da.rename("bt")
        da.attrs.update({"units": "K", "long_name": "brightness_temperature",
                         "satellite": self.satellite, "channel": self.default_channel})
        return da


def _ahi_time(filename: str) -> datetime | None:
    """Parse the UTC time from an AHI HSD filename, or None if it doesn't match."""
    m = _AHI_TIME_RE.search(filename)
    if not m:
        return None
    ymd, hhmm = m.group(1), m.group(2)
    return parse_time(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}T{hhmm[:2]}:{hhmm[2:4]}:00Z")


def _first_present(ds: "Any", names: "tuple[str, ...]") -> str | None:
    for n in names:
        if n in ds.coords or n in ds.variables:
            return n
    return None

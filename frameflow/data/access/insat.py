"""INSAT-3DS / INSAT-3DR Imager TIR1 (10.8 µm window) — DEPLOYMENT TARGET.

INSAT has no public S3 bucket; two scriptable, credentialed routes exist (research/02 §3):

* **MOSDAC** (primary, ``mosdac.gov.in``) — the ISRO portal. Bulk download uses the MOSDAC
  Data Download API (``mdapi.py``) with username/password credentials and a ``datasetId``:

      INSAT-3DS L1B Standard : ``3SIMG_L1B_STD``   (constants.INSAT3DS_MOSDAC_DATASET_ID)
      INSAT-3DR L1B Standard : ``3RIMG_L1B_STD``   (constants.INSAT3DR_MOSDAC_DATASET_ID)

  Registration is required; the API takes credentials in a ``config.json`` (NOT an API key),
  limits 5000 files/day/user. Files are HDF5, e.g. ``3RIMG_20JUN2025_0600_L1B_STD_V01R00.h5``.

* **EUMETSAT** (redundancy) — INSAT-3DS L1C is mirrored on the EUMETSAT Data Store
  (collection ``EO:EUM:DAT:INSAT:INSAT3D-L1C``, constants.INSAT_EUMETSAT_COLLECTION),
  pullable with the ``eumdac`` client using a consumer key/secret.

Credentials are read from environment variables; if they are missing, :meth:`list_files`
raises a clear :class:`OfflineError` explaining exactly which variables to set. The HDF5
reader (:meth:`read_frame`, ``h5py``) returns ``IMG_TIR1`` as brightness temperature in
Kelvin, applying the count->BT lookup table (``IMG_TIR1_TEMP``) when present.

Required environment variables:
    MOSDAC route   : ``MOSDAC_USERNAME``, ``MOSDAC_PASSWORD``
    EUMETSAT route : ``EUMETSAT_CONSUMER_KEY``, ``EUMETSAT_CONSUMER_SECRET``
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import constants as C
from .base import OfflineError, SatelliteSource, parse_time

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr


__all__ = ["INSATSource"]


# 3RIMG_20JUN2025_0600_L1B_STD_V01R00.h5  ->  date + HHMM
_INSAT_TIME_RE = re.compile(r"_(\d{2}[A-Z]{3}\d{4})_(\d{4})_")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


class INSATSource(SatelliteSource):
    """INSAT-3DS/3DR TIR1 access driver (MOSDAC primary, EUMETSAT redundancy).

    Args:
        satellite: ``"INSAT-3DS"`` (default) or ``"INSAT-3DR"``.
        route: ``"mosdac"`` (default) or ``"eumetsat"``.
        anon: unused (INSAT always needs credentials); kept for interface symmetry.
    """

    default_channel = "TIR1"

    def __init__(
        self,
        satellite: str = "INSAT-3DS",
        *,
        route: str = "mosdac",
        anon: bool = False,
    ) -> None:
        super().__init__(anon=anon)
        self.satellite = satellite
        self.route = route
        spec = C.SATELLITE_BANDS.get(satellite, C.SATELLITE_BANDS["INSAT-3DS"])
        self.wavelength_um = float(spec["wavelength_um"])
        self.dataset_id = (
            C.INSAT3DS_MOSDAC_DATASET_ID
            if satellite == "INSAT-3DS"
            else C.INSAT3DR_MOSDAC_DATASET_ID
        )

    # -- listing -----------------------------------------------------------------------
    def list_files(
        self, start: "str | datetime", end: "str | datetime", **kw: Any
    ) -> list[str]:
        """Query MOSDAC (or EUMETSAT) for INSAT TIR1 granules in ``[start, end]`` (UTC).

        This is a credentialed network operation. Credentials are read from environment
        variables; when they are absent (the offline default), a clear :class:`OfflineError`
        names the variables and the registration portal so the user can fix it.

        Args:
            start: inclusive start time.
            end: inclusive end time.
            **kw: ``bbox`` ``(W,S,E,N)`` spatial filter (default the Indian domain),
                ``count`` max results.

        Returns:
            List of granule identifiers/URLs (for :meth:`download`).

        Raises:
            OfflineError: if required credentials are not set, or the query fails.
        """
        t0 = parse_time(start)
        t1 = parse_time(end)
        bbox = kw.get("bbox", C.DEFAULT_GRID_BBOX)

        if self.route == "eumetsat":
            return self._list_eumetsat(t0, t1, **kw)
        return self._list_mosdac(t0, t1, bbox, **kw)

    def _list_mosdac(
        self, t0: datetime, t1: datetime, bbox: "tuple[float, float, float, float]", **kw: Any
    ) -> list[str]:
        user = os.environ.get("MOSDAC_USERNAME")
        pw = os.environ.get("MOSDAC_PASSWORD")
        if not user or not pw:
            raise OfflineError(
                "INSAT via MOSDAC needs credentials. Set the environment variables "
                "MOSDAC_USERNAME and MOSDAC_PASSWORD (register at https://mosdac.gov.in). "
                f"Then this lists datasetId={self.dataset_id} over bbox={bbox} for "
                f"{t0.isoformat()}..{t1.isoformat()} via the MOSDAC Data Download API "
                "(mdapi). Offline: provide local .h5 paths to read_frame() directly."
            )
        # With credentials present, the MOSDAC mdapi flow would post a search request:
        #   POST https://mosdac.gov.in/apios/... {datasetId, startTime, endTime, boundingBox}
        # returning granule ids/urls. The HTTP client is intentionally not invoked here in
        # the offline test environment; raise if the network call cannot be made.
        try:  # pragma: no cover - requires live MOSDAC + valid credentials
            import requests  # type: ignore

            w, s, e, n = bbox
            payload = {
                "datasetId": self.dataset_id,
                "startTime": t0.strftime("%Y-%m-%d"),
                "endTime": t1.strftime("%Y-%m-%d"),
                "boundingBox": f"{w},{s},{e},{n}",
                "count": str(kw.get("count", 50)),
            }
            resp = requests.post(
                "https://mosdac.gov.in/apios/search",
                json={"user_credentials": {"username": user, "password": pw},
                      "search_parameters": payload},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["download_url"] for item in data.get("results", [])]
        except Exception as exc:  # pragma: no cover
            raise OfflineError(
                f"MOSDAC query failed (network/credentials). Underlying error: {exc}. "
                "Verify MOSDAC_USERNAME/PASSWORD and connectivity to mosdac.gov.in."
            ) from exc

    def _list_eumetsat(self, t0: datetime, t1: datetime, **kw: Any) -> list[str]:
        key = os.environ.get("EUMETSAT_CONSUMER_KEY")
        secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
        if not key or not secret:
            raise OfflineError(
                "INSAT via EUMETSAT needs credentials. Set EUMETSAT_CONSUMER_KEY and "
                "EUMETSAT_CONSUMER_SECRET (free EO Portal account). Then this searches "
                f"collection {C.INSAT_EUMETSAT_COLLECTION} via the eumdac client."
            )
        try:  # pragma: no cover - requires live EUMETSAT + valid credentials
            import eumdac  # type: ignore

            token = eumdac.AccessToken((key, secret))
            store = eumdac.DataStore(token)
            coll = store.get_collection(C.INSAT_EUMETSAT_COLLECTION)
            products = coll.search(dtstart=t0, dtend=t1)
            return [str(p) for p in products]
        except Exception as exc:  # pragma: no cover
            raise OfflineError(
                f"EUMETSAT query failed (network/credentials). Underlying error: {exc}."
            ) from exc

    # -- download ----------------------------------------------------------------------
    def download(self, keys: list[str], out_dir: str | Path) -> list[Path]:
        """Download INSAT granules to ``out_dir`` (credentialed; passes through local paths).

        Already-local paths in ``keys`` are returned as-is (so a user with manually-fetched
        ``.h5`` files can feed them straight to :meth:`read_frame`).

        Args:
            keys: granule identifiers/URLs from :meth:`list_files`, or local file paths.
            out_dir: destination directory.

        Returns:
            Local file paths in input order.

        Raises:
            OfflineError: if a remote download is needed but credentials/network are absent.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for key in keys:
            local_candidate = Path(key)
            if local_candidate.exists():
                paths.append(local_candidate)
                continue
            raise OfflineError(
                f"Downloading INSAT granule {key!r} requires the MOSDAC/EUMETSAT client and "
                "valid credentials (see list_files docstring). For offline work, pass the "
                "path to an already-downloaded .h5 file."
            )
        return paths

    # -- read --------------------------------------------------------------------------
    def read_frame(self, path: str | Path) -> "xr.DataArray":
        """Open an INSAT L1B HDF5 file and return TIR1 brightness temperature (Kelvin).

        Reads ``IMG_TIR1`` with ``h5py``; if a count->temperature LUT (``IMG_TIR1_TEMP``) is
        present, BT is the LUT gather, otherwise the stored values are returned as BT (some
        L1C products already store Kelvin). Geolocation is attached from the file's
        ``Latitude``/``Longitude`` datasets when present (applying their scale_factor).

        Args:
            path: local ``.h5`` file path.

        Returns:
            A brightness-temperature :class:`xarray.DataArray` (``"bt"``, Kelvin) with
            lat/lon coords when available.

        Raises:
            OfflineError: if ``h5py`` is unavailable.
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if no ``IMG_TIR1`` dataset is found.
        """
        import numpy as np  # lazy
        import xarray as xr  # lazy

        try:
            import h5py  # lazy
        except ImportError as exc:  # pragma: no cover - h5py installed in this env
            raise OfflineError(
                "Reading INSAT HDF5 needs h5py (`pip install h5py`)."
            ) from exc

        from ..preprocess.radiance import count_to_bt

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"INSAT HDF5 not found: {p}")

        with h5py.File(str(p), "r") as f:
            tir1_name = _find_dataset(f, ("IMG_TIR1", "IMG_TIR1_TB", "TIR1"))
            if tir1_name is None:
                raise ValueError(
                    f"INSAT file {p} has no IMG_TIR1 dataset; keys={list(f.keys())}"
                )
            raw = np.asarray(f[tir1_name][:])
            raw = np.squeeze(raw)

            lut_name = _find_dataset(f, ("IMG_TIR1_TEMP", "IMG_TIR1_TEMPERATURE"))
            if lut_name is not None:
                lut = np.asarray(f[lut_name][:]).ravel()
                bt = count_to_bt(raw, lut=lut)
            elif np.issubdtype(raw.dtype, np.integer) or raw.max() > 1000:
                # Counts with no LUT in-file: cannot calibrate without coefficients.
                raise ValueError(
                    f"INSAT file {p} stores counts but no IMG_TIR1_TEMP LUT and no "
                    "calibration coefficients were provided; cannot derive BT."
                )
            else:
                bt = raw.astype("float64")

            lat, lon = _insat_latlon(f, raw.shape, np)

        coords: dict[str, Any] = {}
        dims = ("y", "x")
        if lat is not None and lon is not None:
            if lat.ndim == 1 and lon.ndim == 1:
                coords = {"lat": ("y", lat), "lon": ("x", lon)}
            else:
                coords = {"lat": (dims, lat), "lon": (dims, lon)}

        return xr.DataArray(
            bt.astype("float32"),
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _find_dataset(group: "Any", names: "tuple[str, ...]") -> str | None:
    """Return the first dataset name present in an h5py group (case-insensitive fallback)."""
    keys = list(group.keys())
    lower = {k.lower(): k for k in keys}
    for n in names:
        if n in group:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _insat_latlon(f: "Any", shape: "tuple[int, ...]", np: "Any") -> "tuple[Any, Any]":
    """Extract lat/lon from an INSAT HDF5 file, applying scale_factor where present."""
    lat_name = _find_dataset(f, ("Latitude", "latitude", "lat"))
    lon_name = _find_dataset(f, ("Longitude", "longitude", "lon"))
    if lat_name is None or lon_name is None:
        return None, None

    def _scaled(name: str) -> "Any":
        dset = f[name]
        arr = np.squeeze(np.asarray(dset[:])).astype("float64")
        sf = dset.attrs.get("scale_factor")
        if sf is not None:
            arr = arr * float(np.asarray(sf).ravel()[0])
        fill = dset.attrs.get("_FillValue")
        if fill is not None:
            arr = np.where(arr == float(np.asarray(fill).ravel()[0]) * (float(np.asarray(sf).ravel()[0]) if sf is not None else 1.0), np.nan, arr)
        return arr

    try:
        lat = _scaled(lat_name)
        lon = _scaled(lon_name)
    except Exception:  # pragma: no cover - defensive
        return None, None
    return lat, lon


def _insat_time(filename: str) -> datetime | None:
    """Parse the UTC time from an INSAT filename (e.g. ``3RIMG_20JUN2025_0600_...``)."""
    m = _INSAT_TIME_RE.search(filename)
    if not m:
        return None
    datestr, hhmm = m.group(1), m.group(2)
    day = int(datestr[:2])
    mon = _MONTHS.get(datestr[2:5].upper())
    year = int(datestr[5:9])
    if mon is None:
        return None
    return parse_time(f"{year:04d}-{mon:02d}-{day:02d}T{hhmm[:2]}:{hhmm[2:4]}:00Z")

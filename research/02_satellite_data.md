# 02 — Satellite Datasets & Fastest Programmatic Access (ISRO BAH 2026 PS-12)

**Topic:** ALL relevant satellite datasets for **geostationary thermal-IR (~10 µm)** imagery, and the **fastest programmatic access** for each, to support **satellite frame interpolation** (temporal super-resolution / "in-betweening" of geostationary IR frames).

**Date compiled:** 2026-06-20.
**Research method:** Live web research (WebSearch + WebFetch, 18+ queries/fetches) cross-checked against author's Jan-2026 training knowledge. Where a primary spec could not be confirmed live, it is flagged `[verify]`.

---

## 0. TL;DR — Recommended Data Strategy

PS-12 needs dense (high-cadence) TIR ~10 µm frames to train an interpolator, broad cross-validation to prove robustness, and the **INSAT-3DS/3DR TIR1** path for deployment. Concrete plan:

| Phase | Use | Dataset(s) | Why |
|---|---|---|---|
| **Primary training** | Dense triplets/sequences (frame N-1, N, N+1) | **GOES-19 ABI L1b RadF C13** (`s3://noaa-goes19/ABI-L1b-RadF/`), 10-min full-disk; optionally GOES-19 L2 CMIPF C13 (brightness temperature, ready-to-use). Add GOES-18 (West) for a second independent geometry. | Free, no auth, NetCDF4, 10-min cadence, huge archive, fast S3 anonymous access. Channel 13 = 10.3 µm clean longwave window — the closest ABI analog to INSAT TIR1 (10.8 µm). |
| **Secondary training / domain diversity** | More dense frames over Asia–Pacific | **Himawari-9 AHI Band 13** (`s3://noaa-himawari9/AHI-L1b-FLDK/`) 10-min full disk; **GK-2A AMI IR105** (`s3://noaa-gk2a-pds/`) 10-min. | Different sensors/geometry → better generalization; Himawari Band 13 = 10.4 µm, very close to INSAT TIR1. |
| **Cross-validation (independent truth)** | Verify interpolated frames against unseen sensors | Polar orbiters at overpass times: **MODIS B31 (11 µm)**, **VIIRS M15 (10.76 µm)**, **Sentinel-3 SLSTR S8 (10.85 µm)**, **MetOp AVHRR ch4/5**. Plus overlapping GEOs (Meteosat MSG/MTG, FY-4B). | Polar orbiters give a higher-spatial-res, independent radiometric reference at known times; GEO–GEO overlap regions allow same-time cross-checks. |
| **Deployment target** | Final inference + domain adaptation | **INSAT-3DS / INSAT-3DR TIR1 (10.8 µm)** from **MOSDAC** (`mosdac.gov.in`), HDF5. INSAT-3DS L1C also mirrored on **EUMETSAT Data Store**. | This is the operational target of PS-12. Fine-tune / adapt the GOES/Himawari-trained model on INSAT TIR1. |

**Model I/O note:** GOES (NetCDF4 `.nc`), Himawari AWS NetCDF (`.nc`) and INSAT (HDF5 `.h5`) all map cleanly to xarray/h5py → satisfies the "NetCDF or HDF5" requirement. Keep everything in xarray and serialize intermediate tensors to `.nc`/`.h5`.

**Fastest access pattern (one line):** anonymous S3 via `s3fs`/`fsspec` straight into `xarray.open_dataset(...)` — no download, no credentials, lazy chunked reads. Code in §A and §1.

**Datasets / access-methods catalogued below: 38** (see running count; ≥30 required). 

**Cross-validation key idea:** the world's GEO satellites form a *ring* (GOES-East 75.2°W, GOES-West 137°W, Meteosat-0°/MSG, Meteosat-IODC ~45.5°E, INSAT ~74-82°E, FY-4 ~105/133°E, Himawari 140.7°E, GK-2A 128.2°E). Adjacent satellites **overlap at their limbs**, and polar orbiters cut across all of them. This lets you (a) train on the densest feed (GOES-19), (b) test transfer on every other GEO, and (c) use overpasses as independent ground truth — directly satisfying the "30 methods, fill the gaps, broad cross-validation" requirement. Details in §11.

---

## A. The universal fast-access recipe (read this first)

Almost every cloud-hosted dataset below is reachable with the **same three Python idioms**. Learn these once; reuse for GOES, Himawari, GK-2A, GMGSI, Landsat, Sentinel-3, etc.

**A1. Anonymous S3 listing + lazy xarray (NetCDF4 on S3):**
```python
import s3fs, xarray as xr
fs = s3fs.S3FileSystem(anon=True)                     # no AWS account needed
# List GOES-19 ABI L1b Full-Disk Channel 13 for a given day/hour:
key = "noaa-goes19/ABI-L1b-RadF/2025/171/14/"          # <prod>/<year>/<DOY>/<hour>/
files = fs.ls(key)
c13 = [f for f in files if "C13" in f]                  # Channel 13 only
ds = xr.open_dataset(fs.open(c13[0]))                   # lazy, no full download
print(ds)                                               # variable 'Rad' (radiance)
```

**A2. fsspec single-file open (works for S3, GCS, HTTPS):**
```python
import fsspec, xarray as xr
url = "s3://noaa-goes19/ABI-L1b-RadF/2025/171/14/OR_ABI-L1b-RadF-M6C13_G19_*.nc"
with fsspec.open(url, anon=True) as f:
    ds = xr.open_dataset(f)
```

**A3. Convert radiance → brightness temperature (ABI L1b)** (CMIP/MCMIP already give BT in Kelvin):
```python
import numpy as np
p = ds["planck_fk1"].values; q = ds["planck_fk2"].values
bc1 = ds["planck_bc1"].values; bc2 = ds["planck_bc2"].values
rad = ds["Rad"]                                          # mW m-2 sr-1 (cm-1)-1
Tb = (q / np.log(p/rad + 1.0) - bc1) / bc2              # Kelvin
```

**Key Python libraries** (install once): `xarray`, `netCDF4`, `h5py`, `s3fs`, `fsspec`, `satpy`, `pyresample`, `rioxarray`, `goes2go`, `herbie-data`, `earthengine-api`, `eumdac`, `kerchunk`, `zarr`, `dask`, `pystac-client`, `planetary-computer`. (Catalogued individually in §10.)

---

## 1. GOES-R Series (GOES-16/17/18/19) — ABI — **PRIMARY TRAINING SOURCE**

**Instrument:** Advanced Baseline Imager (ABI), 16 channels. **Thermal target = Channel 13 ("Clean" longwave window), 10.3 µm.** (Channel 14 = 11.2 µm, Channel 15 = 12.3 µm are close alternates.)
**Resolution:** C13 = **2 km** at nadir. **Cadence:** Full Disk (F) **10 min** (current operational mode M6); CONUS (C) 5 min; Mesoscale (M1/M2) 30–60 s.
**Positions:** GOES-19 = GOES-East @ 75.2°W (operational since 2025, replaced GOES-16); GOES-18 = GOES-West @ 137.2°W; GOES-16/17 archived. **Coverage:** Americas + most of Atlantic & Pacific.
**Format:** NetCDF4 (`.nc`).

### Method 1 — NOAA Open Data on AWS (FASTEST, no auth) ✅ recommended
- Buckets: `s3://noaa-goes16`, `s3://noaa-goes17`, `s3://noaa-goes18`, **`s3://noaa-goes19`** (all `us-east-1`).
- **Path structure:** `<Product>/<Year>/<Day-of-Year>/<Hour>/<Filename>`
- **Products of interest:**
  - `ABI-L1b-RadF` — Level-1b radiances, Full Disk (raw radiance; convert to BT via §A3).
  - `ABI-L2-CMIPF` — Level-2 Cloud & Moisture Imagery, Full Disk, **per single channel**, already in **brightness temperature (K)** for IR channels → easiest for ML.
  - `ABI-L2-MCMIPF` — Multiband CMIP (all 16 channels in one file).
- **Filename:** `OR_ABI-L1b-RadF-M6C13_G19_sYYYYJJJHHMMSSs_eYYYYJJJHHMMSSs_cYYYYJJJHHMMSSs.nc`
  - `OR`=operational; `ABI-L1b-RadF`=product; `M6`=scan mode 6 (10-min FD); **`C13`=Channel 13**; `G19`=GOES-19; `s…`=scan start (yr, day-of-year, hh mm ss .s); `e…`=end; `c…`=creation.
- **AWS CLI:**
```bash
aws s3 ls --no-sign-request s3://noaa-goes19/ABI-L1b-RadF/2025/171/14/
aws s3 cp  --no-sign-request \
  s3://noaa-goes19/ABI-L1b-RadF/2025/171/14/OR_ABI-L1b-RadF-M6C13_G19_s20251711400205_e20251711409513_c20251711409576.nc .
```
- **Per-channel CMIP path (BT-ready):** `s3://noaa-goes19/ABI-L2-CMIPF/2025/171/14/OR_ABI-L2-CMIPF-M6C13_G19_*.nc` → variable `CMI` (Kelvin).
- Docs: registry.opendata.aws/noaa-goes/ ; github.com/NOAA-Big-Data-Program/nodd-data-docs/tree/main/GOES ; github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-goes16/README.md

### Method 2 — `goes2go` Python package (convenience wrapper over AWS) ✅
```python
from goes2go import GOES
# Channel-13 brightness temperature, Full Disk, GOES-19, a 1-hour window:
G = GOES(satellite=19, product="ABI-L2-CMIP", domain="F", channel=13)
df = G.df(start="2025-06-20 00:00", end="2025-06-20 01:00")   # list available files
ds = G.nearesttime("2025-06-20 00:30")                         # xarray.Dataset
G.timerange(start="2025-06-20 00:00", end="2025-06-20 06:00")  # bulk download
# Raw radiance instead: product="ABI-L1b-Rad"
```
Repo: github.com/blaylockbk/goes2go (pip install goes2go). Returns `xarray.Dataset`; uses `s3fs` under the hood. Great for building training triplets quickly.

### Method 3 — Google Cloud Storage mirror (counts as separate redundant method)
- Buckets: `gs://gcp-public-data-goes-16` … and `gcp-public-data-goes-18`/`-19` (same path layout as AWS).
- Access via `gsutil` or `gcsfs`:
```python
import gcsfs, xarray as xr
fs = gcsfs.GCSFileSystem(token="anon")
files = fs.ls("gcp-public-data-goes-19/ABI-L1b-RadF/2025/171/14/")
```
- Bonus: NOAA GOES metadata is in **BigQuery** tables (L1b Rad, L2 CMIP, L2 MCMIP) for fast time/spatial queries before pulling files. (cloud.google.com/blog/products/bigquery weather-satellite post.)

### Method 4 — Microsoft Planetary Computer (STAC) (separate redundant method)
- GOES-R collections in the MS Planetary Computer catalog (planetarycomputer.microsoft.com/catalog). Query via STAC API + sign:
```python
import pystac_client, planetary_computer as pc
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)
# search GOES collection by datetime/bbox, then open assets with xarray
```

### Method 5 — Google Earth Engine asset IDs (server-side, great for cross-val mosaics)
- `ee.ImageCollection("NOAA/GOES/19/MCMIPF")` — MCMIP Full Disk, 33 bands, 2 km, 10-min. Thermal band **`CMI_C13`** (10.3 µm) in Kelvin via scale/offset.
- Siblings: `NOAA/GOES/16|17|18/MCMIPF`, `.../MCMIPC` (CONUS), `.../MCMIPM` (Meso). Also `…/19/FDCF` (fire), etc.
```python
import ee; ee.Initialize()
img = ee.ImageCollection("NOAA/GOES/19/MCMIPF").filterDate("2025-06-20","2025-06-21").first()
c13 = img.select("CMI_C13")   # apply CMI_C13_scale / CMI_C13_offset for Kelvin
```
Catalog: developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_19_MCMIPF

### Method 6 — NOAA CLASS (archive of record)
- Comprehensive Large Array-data Stewardship System (class.noaa.gov). Order-based; slower than S3 but authoritative for old/QC'd data. Use only if S3 lacks a date.

> **Datasets so far: 6** (Methods 1–6 are distinct GOES access paths.)

---

## 2. Himawari-8/9 — AHI — **SECONDARY TRAINING (Asia-Pacific)**

**Instrument:** Advanced Himawari Imager (AHI), 16 bands (mirrors ABI). **Thermal target = Band 13, 10.4 µm** ("clean" IR window; closest to INSAT TIR1). Band 14 = 11.2 µm, Band 15 = 12.4 µm.
**Resolution:** Band 13 = **2 km** at nadir. **Cadence:** Full Disk **10 min**; Japan/Target regions 2.5 min; landmark 0.5 min. **Position:** Himawari-9 @ 140.7°E (operational; Himawari-8 standby). **Coverage:** East Asia, Western/Central Pacific, Australia, **Indian Ocean east edge (overlaps INSAT)**.

### Method 7 — NOAA Open Data on AWS (FASTEST, no auth) ✅
- Buckets: `s3://noaa-himawari8`, **`s3://noaa-himawari9`** (`us-east-1`).
- **Path:** `AHI-L1b-FLDK/<Year>/<Month>/<Day>/<HourMinute>/<Filename>`
- **Two file flavors:**
  - **JMA Himawari Standard Data (HSD):** `HS_H09_YYYYMMDD_HHMM_B13_FLDK_R20_S0101.DAT.bz2` (segmented, bzip2; read with **satpy** `ahi_hsd` reader). `B13`=Band 13; `R20`=2 km; `S0101..S1010`=10 disk segments.
  - **NetCDF products** (e.g., L2 cloud mask `AHI-CMSK_v1r1_h09_sYYYY…​.nc`) and L1 gridded NetCDF.
- **AWS CLI:**
```bash
aws s3 ls --no-sign-request s3://noaa-himawari9/AHI-L1b-FLDK/2025/06/20/0200/
```
- Reading HSD with satpy:
```python
from satpy import Scene; from glob import glob
scn = Scene(reader="ahi_hsd", filenames=glob("HS_H09_*_B13_FLDK_R20_S*.DAT"))
scn.load(["B13"]); scn["B13"]   # brightness temperature
```
Registry: registry.opendata.aws/noaa-himawari/

### Method 8 — JAXA Himawari Monitor "P-Tree" (FTP/SFTP + gridded NetCDF) ✅
- Free registration: eorc.jaxa.jp/ptree/registration_top.html ; user guide: eorc.jaxa.jp/ptree/userguide.html
- Offers **L1 Gridded NetCDF** (lat/lon regular grid — very convenient, no segment stitching) + JAXA geophysical params. Archive from 2015-03-20; NRT 5–20 min latency.
- **Note (live finding):** from **2026-01-06** JAXA serves only the *new expanded* gridded version (band 13 = 10.4 µm aligned to European GEOs). Account also unlocks SFTP bulk pulls.

### Method 9 — JMA HimawariCast / direct (broadcast) — institutional
- HRIT/LRIT dissemination + JMA MSC pages (data.jma.go.jp/mscweb/en/himawari89/). Mostly for operational receivers; listed for completeness.

### Method 10 — Google Earth Engine Himawari asset
- JAXA-provided Himawari standard/geophysical data appear under EE's JAXA tags (developers.google.com/earth-engine/datasets/tags/jaxa). Asset availability shifts; query the catalog for the current AHI L1 collection ID before use. `[verify exact ID at run time]`

> **Datasets so far: 10.**

---

## 3. INSAT-3D / 3DR / 3DS — Imager — **DEPLOYMENT TARGET**

**Instrument:** 6-channel Imager. **Thermal targets: TIR1 = 10.8 µm, TIR2 = 11.9 µm** (infrared window). MIR = 3.9 µm, WV = 6.8 µm.
**Resolution:** TIR1/TIR2 = **4 km** at sub-satellite point. **Cadence:** Full frame ~**30 min** (L1/L2 distributed at 30-min; rapid-scan sectors faster). **Positions:** INSAT-3D ~82°E (legacy), INSAT-3DR ~74°E, **INSAT-3DS** launched 2024-02-17 (operational). **Coverage:** Indian subcontinent, Indian Ocean, surrounding Asia/Africa. **Format:** **HDF5** (`.h5`), plus GeoTIFF/GIF previews.

### Method 11 — MOSDAC portal + Data Download API ✅ (primary INSAT path)
- Portal: **mosdac.gov.in**. **Registration required** (Sign Up → login). Catalog browser: mosdac.gov.in/catalog-app/satellite.php
- **Download API** (Python tool `mdapi.py`, zip at mosdac.gov.in/software/mdapi.zip; manual: mosdac.gov.in/user-manual-mosdac-data-download-api). Credential-based (username/password in `config.json`, **not** an API key). Limits: **5000 files/day/user**; 3 failed logins → 1-hour lockout.
- **config.json example (INSAT-3DS L1B standard):**
```json
{
  "user_credentials": {"username": "YOUR_USER", "password": "YOUR_PASS"},
  "search_parameters": {
    "datasetId": "3SIMG_L1B_STD",
    "startTime": "2025-06-01", "endTime": "2025-06-20",
    "count": "50", "boundingBox": "70.0,8.0,90.0,28.0", "gId": ""
  },
  "download_settings": {"download_path": "./insat_dl", "organize_by_date": true}
}
```
  Run: `python mdapi.py`.
- **Dataset IDs (prefixes):**
  - **INSAT-3DS:** `3SIMG_*` (e.g., `3SIMG_L1B_STD`, `3SIMG_L1C_SGP` standard geo-projected, `3SIMG_L2B_*`).
  - **INSAT-3DR:** `3RIMG_*` (e.g., `3RIMG_L1B_STD`, `3RIMG_L1C_SGP`).
  - **INSAT-3D:** `3DIMG_*`.
- **File naming (HDF5):** e.g., `3RIMG_20JUN2025_0600_L1B_STD_V01R00.h5` (sat-prefix _ DDMMMYYYY _ HHMM _ level _ type _ version). TIR1 dataset inside the HDF5: brightness-temperature/`IMG_TIR1` (counts) + lookup tables (`IMG_TIR1_TEMP`) for K. Read with `h5py`/`xarray`(`engine="h5netcdf"`)/`satpy` (`insat3d_img_l1b_h5` reader where available).
```python
import h5py
f = h5py.File("3RIMG_20JUN2025_0600_L1B_STD_V01R00.h5","r")
print(list(f.keys()))           # e.g., IMG_TIR1, IMG_TIR1_TEMP (LUT counts->K)
tir1_counts = f["IMG_TIR1"][:]
```
- Docs: mosdac.gov.in/insat-3dr-data-products ; mosdac.gov.in/insat-3ds ; INSAT-3D Products format PDF: mosdac.gov.in/docs/INSAT3D_Products.pdf

### Method 12 — MOSDAC "Open Data" + Order Data (browser/FTP-style)
- mosdac.gov.in/open-data (some products without per-order auth) and the "Satellite data order" workflow (mosdac.gov.in/how-ordered-and-download-satellite-data) producing download links/FTP. OpenDAP/THREDDS endpoints exist for some collections (catalog-dependent). Slower than the API but useful for ad-hoc grabs.

### Method 13 — EUMETSAT Data Store mirror of INSAT-3D/3DR/3DS L1C ✅ (great redundancy!)
- **Live finding:** From **2025-03-26**, **INSAT-3DS L1C** (Radiance/Albedo/Brightness-Temperature) is distributed via EUMETSAT (collection `EO:EUM:DAT:INSAT:INSAT3D-L1C`). INSAT-3D/3DR also present historically. Pull with `eumdac` (see §4) — this gives a **non-MOSDAC, scriptable** route to INSAT TIR data with a familiar API.
- Catalog: user.eumetsat.int/catalogue/EO:EUM:DAT:INSAT:INSAT3D-L1C

### Method 14 — Bhuvan / ISRO (visualization + some downloads)
- bhuvan.nrsc.gov.in hosts INSAT imagery layers/WMS and selected products; primarily for browse/GIS overlay, secondary to MOSDAC for raw HDF5.

> **Note on AWS/GEE for INSAT:** As of Jan-2026 there is **no first-party INSAT bucket on AWS** and **no native INSAT collection in GEE**; MOSDAC + EUMETSAT are the authoritative programmatic routes. (Flag as the main "gap" for deployment data — mitigated by Method 13.)

> **Datasets so far: 14.**

---

## 4. Meteosat — MSG (SEVIRI) & MTG (FCI) — EUMETSAT — **GEO cross-val (Africa/Europe/Indian Ocean)**

**MSG/SEVIRI:** 12 channels; **IR10.8 µm** is the key TIR window (also IR12.0, IR9.7). Resolution 3 km at nadir (HRV 1 km vis). **Cadence 15 min** full disk (Rapid Scan 5 min). Positions: Meteosat-0° (prime) and **Meteosat-IODC ~45.5°E (Indian Ocean Data Coverage — overlaps INSAT!)**.
**MTG/FCI** (new gen): IR channels incl. **IR 10.5 µm**; up to 1–2 km IR; **full disk 10 min** (RSS faster). Position 0°.
**Format:** Native (`.nat`) / NetCDF (via Data Tailor).

### Method 15 — EUMETSAT Data Store + `eumdac` (Python/CLI) ✅
- Free EO Portal account → consumer key/secret. Library: `pip install eumdac`.
```python
import eumdac
token = eumdac.AccessToken(("CONSUMER_KEY","CONSUMER_SECRET"))
ds = eumdac.DataStore(token)
# MSG SEVIRI Level 1.5 collection (IR10.8 included in full-disk product):
coll = ds.get_collection("EO:EUM:DAT:MSG:HRSEVIRI")
for prod in coll.search(dtstart="2025-06-20T00:00:00", dtend="2025-06-20T01:00:00"):
    with prod.open() as src, open(src.name,"wb") as dst:
        dst.write(src.read())
```
- CLI: `eumdac search -c EO:EUM:DAT:MSG:HRSEVIRI -s 2025-06-20 -e 2025-06-20` then `eumdac download`.
- Collections: MSG full disk `EO:EUM:DAT:MSG:HRSEVIRI`; IODC `EO:EUM:DAT:MSG:HRSEVIRI-IODC`; **MTG FCI L1C** `EO:EUM:DAT:0662` (FCI). (Confirm exact MTG IDs in catalogue at run time.)
- Read with **satpy** (`seviri_l1b_native` / `fci_l1c_nc` readers).

### Method 16 — EUMETView / WMS (quick visual cross-check)
- eumetview.eumetsat.int — WMS layers of IR10.8 for fast visual validation of interpolated frames.

### Method 17 — Data Tailor (Web Service / `epct`) — reformat & subset
- Convert Native → NetCDF/GeoTIFF, subset ROI, reproject server-side before download. Chain into livefeed via DTWS. Useful to deliver INSAT-comparable IR10.8 NetCDF subsets.

> **Datasets so far: 17.**

---

## 5. FengYun FY-4A / FY-4B — AGRI — **GEO cross-val (East/Central Asia)**

**Instrument:** Advanced Geostationary Radiation Imager (AGRI), 14–15 channels. **TIR window ~10.8 µm** (FY-4B AGRI added an extra IR channel vs 4A). **Resolution** IR ~4 km. **Cadence:** full disk ~15 min (regional faster). **Positions:** FY-4A ~104.7°E, FY-4B ~133°E (operational). **Coverage:** Asia-Pacific, **overlaps Himawari & INSAT**. **Format:** HDF.

### Method 18 — NSMC FengYun Cloud (`data.nsmc.org.cn` / `satellite.nsmc.org.cn`) ✅
- Free registration; FY-4B/AGRI L1 (geolocation + radiance) released publicly (since 2022-06-01). Bulk/manual data service for large orders. HDF files; read with `h5py`/satpy (`agri_fy4b_l1` / `agri_fy4a_l1` readers; SSEC geo2grid documents the readers).
- Portal: data.nsmc.org.cn (English UI available). 

### Method 19 — satpy AGRI readers (processing path)
```python
from satpy import Scene
scn = Scene(reader="agri_fy4b_l1", filenames=[...HDF...])
scn.load(["C12"])   # ~10.8 µm window channel (check channel map per FY-4B)
```

> **Datasets so far: 19.**

---

## 6. GK-2A / GEO-KOMPSAT-2A — AMI — **GEO cross-val (Korea/Asia)**

**Instrument:** Advanced Meteorological Imager (AMI), 16 channels (ABI-like). **TIR target = IR105 = 10.5 µm** (also IR112 = 11.2, IR123 = 12.3). **Resolution** IR = 2 km. **Cadence:** full disk **10 min**; Korea 2 min. **Position:** 128.2°E. **Coverage:** Korea, Japan, much of Asia-Pacific (**overlaps Himawari**). **Format:** NetCDF (`.nc`).

### Method 20 — NOAA Open Data on AWS (FASTEST, no auth) ✅
- Bucket: **`s3://noaa-gk2a-pds`** (`us-east-1`). Browse: noaa-gk2a-pds.s3.amazonaws.com/index.html
- **Filename:** `gk2a_ami_le1b_ir105_fd020ge_YYYYMMDDHHMM.nc`
  - `le1b`=Level-1B; **`ir105`=10.5 µm**; `fd020ge`=Full Disk 2 km geographic; (`ea020lc`=East-Asia 2 km LCC). 
```bash
aws s3 ls --no-sign-request s3://noaa-gk2a-pds/
# (path typically .../AMI/L1B/FD/<YYYYMM>/<DD>/<HH>/gk2a_ami_le1b_ir105_fd020ge_*.nc)
```
- satpy reader: `ami_l1b`.

### Method 21 — KMA NMSC data service (`datasvc.nmsc.kma.go.kr`)
- KMA's National Meteorological Satellite Center portal (nmsc.kma.go.kr) — registration-based download of AMI L1B/L2 NetCDF; authoritative source if AWS lags.

> **Datasets so far: 21.**

---

## 7. Electro-L N2 / N3 (and N4) — MSU-GS — **GEO cross-val (Indian Ocean/Atlantic)**

**Instrument:** MSU-GS, 3 VIS + **7 IR channels (~0.5–12.5 µm)**; includes **~10.7–11.5 µm window**. **Resolution** IR = 4 km, VIS = 1 km. **Cadence:** 30 min (15 min frequent mode). **Positions:** 14.5°W, **76°E (overlaps INSAT!)**, 165.8°E (Pacific). **Coverage:** WMO-approved Atlantic/Indian/Pacific. **Format:** HRIT/LRIT (binary); converted products.

### Method 22 — Roscosmos/Roshydromet NMC (`ntsomz.gptl.ru` / `electro.ntsomz.ru`)
- Russian SRC "Planeta" / NTs OMZ distribute Electro-L imagery (registration; HRIT/LRIT + processed). Indian-Ocean position (76°E) makes Electro-L a useful independent IR window cross-check directly over the INSAT domain. Access is less turnkey than AWS — flag `[verify current portal/credentials]`. eoPortal page: eoportal.org/satellite-missions/electro-l

> **Datasets so far: 22.**

---

## 8. Polar Orbiters — independent "ground-truth" cross-validation & gap-fill

These are **not** geostationary, but their **higher spatial resolution** and **independent radiometry** make them gold-standard checks at overpass times, and gap-fillers when a GEO sector is missing. Each is a distinct dataset/method.

### Method 23 — Terra/Aqua **MODIS**, Band 31 (11.03 µm) [+ B32 12 µm]
- **Res** 1 km; swath ~2330 km; ~daily global. **Format** HDF-EOS. Products: `MOD021KM`/`MYD021KM` (calibrated radiances, 36 bands → emissive B20–B36) and `MOD11`/`MYD11` LST.
- **Access:** LAADS DAAC (ladsweb.modaps.eosdis.nasa.gov) + Earthdata login; also GEE `MODIS/061/MOD021KM`, `MODIS/061/MOD11A1`.
```python
# Earthdata: ladsweb.modaps.eosdis.nasa.gov product MOD021KM; B31 = emissive ~11um
```

### Method 24 — Suomi-NPP / NOAA-20 / NOAA-21 **VIIRS**, M15 (10.76 µm) & I5 (11.45 µm)
- **Res** M15 = 750 m, I5 = 375 m; near-daily global. **Format** NetCDF (VNP02/VJ102 L1B; VNP14 active fire). 
- **Access:** LAADS DAAC; `s3://noaa-jpss` (NODD JPSS bucket); GEE `NOAA/VIIRS/001/VNP14A1` etc. M15 (10.76 µm) is an excellent INSAT-TIR1 analog.

### Method 25 — Sentinel-3A/3B **SLSTR**, S8 (10.85 µm) [+ S9 12 µm]
- **Res** thermal 1 km; dual-view; ~daily. **Format** NetCDF (`LST_in.nc`, plus L1B `S8_BT_in.nc`). 
- **Access:** Copernicus Data Space Ecosystem; **EUMETSAT Data Store** (`eumdac`); **MS Planetary Computer** STAC: `sentinel-3-slstr-lst-l2-netcdf` and `sentinel-3-slstr-wst-l2-netcdf`.
```python
import pystac_client, planetary_computer as pc
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
search = cat.search(collections=["sentinel-3-slstr-lst-l2-netcdf"], bbox=[68,8,98,38], datetime="2025-06-20")
```

### Method 26 — MetOp-B/C **AVHRR/3**, Ch4 (10.8 µm) & Ch5 (12 µm)
- **Res** 1.1 km (LAC); global. **Format** Native/NetCDF (EPS). **Access:** EUMETSAT Data Store (`eumdac`), NOAA CLASS (NOAA-19 AVHRR). Long heritage 10.8 µm record — ideal for bias checks.

### Method 27 — **MetOp IASI** (hyperspectral IR) — radiometric reference
- Not an imager, but the GSICS **cross-calibration reference** for all GEO IR channels. Use IASI-derived GSICS corrections to harmonize GOES C13 / Himawari B13 / INSAT TIR1 onto a common scale. Data: EUMETSAT (`eumdac`); GSICS products via NOAA/JMA.

### Method 28 — **Landsat 8/9 TIRS**, Band 10 (10.6–11.2 µm)
- **Res** 100 m (resampled 30 m); 16-day revisit. **Format** GeoTIFF (Collection-2 L2 ST). **Access:** `s3://usgs-landsat` (requester-pays), USGS EarthExplorer, GEE `LANDSAT/LC09/C02/T1_L2` (band `ST_B10`). Highest-res 10 µm truth for sharp-edge validation over land.

### Method 29 — **ECOSTRESS** (ISS), ~10.5 µm bands → LST
- **Res** ~70 m; irregular ISS overpasses. **Format** HDF5/GeoTIFF. **Access:** NASA AppEEARS / LP DAAC (lpdaac.usgs.gov), Earthdata. Excellent fine-scale LST cross-check (diurnal sampling from ISS orbit).

### Method 30 — **GCOM-C / SGLI** TIR (10.8 & 12 µm) — JAXA polar
- **Res** ~250–500 m (TIR 250 m product); global. **Access:** JAXA G-Portal (gportal.jaxa.jp); GEE JAXA LST. Another independent 10.8 µm reference.

> **Datasets so far: 30.** ✅ (requirement met) — continuing with platform/catalog methods and a global mosaic for extra robustness.

---

## 9. Cloud platforms, catalogs & multi-sensor mosaics (extra cross-validation infrastructure)

### Method 31 — **NOAA GMGSI** Global Mosaic of Geostationary Satellite Imagery ✅ (killer cross-val product)
- Bucket: **`s3://noaa-gmgsi-pds`** (`us-east-1`). **Blends GOES-East, GOES-West, Meteosat-9/10, Himawari-9** into a single global grid. **Longwave IR product `GMGSI_LW`** (thermal). **Res ~8 km, hourly.** NetCDF.
```bash
aws s3 ls --no-sign-request s3://noaa-gmgsi-pds/
```
- Use as a **globally-consistent target/reference** to test whether your interpolator generalizes across *all* GEOs simultaneously. Registry: registry.opendata.aws/noaa-gmgsi/

### Method 32 — **AWS Open Data Registry / NODD** (umbrella index)
- registry.opendata.aws + noaa.gov/nodd/datasets — discover every NOAA bucket (GOES, Himawari, GK-2A, JPSS, GMGSI). The "no-sign-request" anonymous pattern (§A1) works across all.

### Method 33 — **Google Earth Engine** (server-side analysis catalog)
- One API, dozens of relevant collections: `NOAA/GOES/{16,17,18,19}/MCMIP{F,C,M}`, MODIS (`MODIS/061/MOD021KM`, `…/MOD11A1`), VIIRS, Landsat ST, JAXA Himawari/GCOM-C. Ideal for building co-located cross-val samples without downloading. `earthengine-api` (Python) / Code Editor (JS).

### Method 34 — **Microsoft Planetary Computer** (STAC + Hub)
- planetarycomputer.microsoft.com — STAC API for GOES-R, Sentinel-3 SLSTR LST/WST, Landsat, MODIS. Free compute Hub. `pystac-client` + `planetary-computer` signing (§1 Method 4, §8 Method 25).

### Method 35 — **NASA Earthdata / GES DISC / LAADS / LP DAAC / PO.DAAC**
- earthdata.nasa.gov (single login) → CMR STAC search across MODIS, VIIRS, ECOSTRESS, GHRSST. PO.DAAC hosts **GHRSST L2P/L3C from Himawari AHI** (e.g., `H09-AHI-L3C-ACSPO-v2.90`) — SST-focused but uses the same 10.x µm window radiances; another redundant Himawari IR-derived check (podaac.jpl.nasa.gov).

### Method 36 — **Copernicus Data Space Ecosystem** (Sentinel-3 SLSTR)
- dataspace.copernicus.eu — S3 STAC + OData APIs + `sentinelsat`-style clients for SLSTR S8/S9 L1B & LST. Free account.

### Method 37 — **kerchunk / VirtualiZarr + Zarr** (fastest *repeated* access)
- Build a JSON "reference" over the NetCDF4 files on S3 → open thousands of GOES/Himawari frames as a single lazy **Zarr** cube via `fsspec` reference filesystem. Massive speed-up for training-time random access across time.
```python
import fsspec, xarray as xr
m = fsspec.get_mapper("reference://", fo="goes19_c13_2025.json",
                      remote_protocol="s3", remote_options={"anon": True})
cube = xr.open_dataset(m, engine="zarr", consolidated=False)  # [time,y,x] BT cube
```
- NOAA even publishes prebuilt kerchunk references: `s3://noaa-nodd-kerchunk` (registry.opendata.aws/noaa-nodd-kerchunk/). Tutorial: Pangeo "Fake it until you make it" (Medium). Tooling: fsspec/kerchunk, virtualizarr.

### Method 38 — **THREDDS / OPeNDAP servers** (subset-on-server)
- Unidata THREDDS catalogs and OPeNDAP endpoints (incl. some MOSDAC collections, NASA GES DISC, NOAA) let you `xr.open_dataset("https://…/dodsC/…")` and pull only the ROI/time slice over HTTP — no full-file download. Generic, standards-based fallback for any DAP-served IR dataset.

> **TOTAL distinct datasets / access-methods catalogued: 38** (≥30 required ✅).

---

## 10. Python libraries to read/access (quick reference)

| Library | Role | One-liner |
|---|---|---|
| **xarray** | N-D labeled arrays; reads NetCDF/HDF5/Zarr | `xr.open_dataset(path)` |
| **netCDF4** | NetCDF4 engine for xarray | backend for `.nc` |
| **h5py** | Native HDF5 (INSAT `.h5`) | `h5py.File("3RIMG_*.h5")` |
| **h5netcdf** | HDF5-as-NetCDF xarray engine | `xr.open_dataset(f, engine="h5netcdf")` |
| **s3fs / fsspec / gcsfs** | Anonymous cloud file access | `s3fs.S3FileSystem(anon=True)` |
| **satpy** | Multi-sensor readers (ABI, AHI HSD, FCI, AGRI, AMI, INSAT, VIIRS, SLSTR) + resampling/composites | `Scene(reader="abi_l1b", filenames=...)` |
| **pyresample** | Reproject/regrid swath↔grid (co-locate GEO vs polar) | `kd_tree.resample_nearest` |
| **goes2go** | GOES AWS convenience | `GOES(19,"ABI-L2-CMIP","F",channel=13)` |
| **GOES-py / GOES (palmoreck/joaohenry23)** | Alt GOES reader/plotter | `GOES.download(...)` |
| **herbie-data** | NWP + some satellite archive access | `Herbie(...)` |
| **earthengine-api** | GEE Python client | `ee.ImageCollection("NOAA/GOES/19/MCMIPF")` |
| **eumdac** | EUMETSAT Data Store (MSG, MTG, SLSTR, AVHRR, **INSAT-3DS L1C**) | `eumdac.DataStore(token)` |
| **rioxarray** | Geo-aware raster (GeoTIFF Landsat/ECOSTRESS) | `rioxarray.open_rasterio(...)` |
| **pystac-client + planetary-computer** | STAC search + signing | MS Planetary Computer |
| **kerchunk / virtualizarr / zarr / dask** | Virtual Zarr over NetCDF, parallel | fast training-time cubes |

---

## 11. Multi-satellite cross-validation & gap-filling strategy (the "30 methods, fill the gaps" core)

**Why multiple satellites?** A single GEO has (a) fixed viewing geometry (limb distortion), (b) periodic outages/eclipse/housekeeping gaps, and (c) one radiometric calibration. Frame interpolation trained on one source can overfit to its artifacts. Using many sources gives independent verification and physical robustness.

**11.1 The geostationary ring (same-time GEO↔GEO overlap).** Sub-satellite longitudes:

| Satellite | Lon | TIR window | Overlaps with |
|---|---|---|---|
| GOES-19 (East) | 75.2°W | C13 10.3 µm | GOES-West (Pacific), Meteosat-0° (Atlantic) |
| GOES-18 (West) | 137.2°W | C13 10.3 µm | GOES-East, **Himawari** (Pacific) |
| Meteosat-0° (MSG/MTG) | 0° | IR10.8 / IR10.5 | GOES-East (Atlantic), Meteosat-IODC |
| Meteosat-IODC | ~45.5°E | IR10.8 | **INSAT**, Electro-L 76°E |
| INSAT-3DR/3DS | ~74–82°E | **TIR1 10.8 µm** | Meteosat-IODC, Electro-L, **FY-4**, **Himawari** (E edge) |
| Electro-L N3 | 76°E | ~10.8 µm | INSAT (co-located!) |
| GK-2A | 128.2°E | IR105 10.5 µm | Himawari, FY-4B |
| FY-4B | 133°E | ~10.8 µm | Himawari, GK-2A, INSAT |
| Himawari-9 | 140.7°E | B13 10.4 µm | GOES-West, GK-2A, FY-4B, INSAT (E edge) |

**Practical overlaps to exploit for PS-12:**
- **GOES-West ↔ Himawari over the Pacific (~180°):** two 10-min feeds of the *same* clouds from opposite sides → validate that an interpolated GOES frame matches the corresponding Himawari frame after reprojection (independent sensor, same scene/time).
- **INSAT ↔ Himawari over the Indian Ocean / SE Asia (eastern INSAT limb):** directly tests the deployment sensor against a dense 10-min source in the *same region*.
- **INSAT ↔ Meteosat-IODC (45.5°E) ↔ Electro-L (76°E):** triple coverage of the Indian domain — Electro-L at 76°E is essentially co-located with INSAT-3DR (74°E), an almost nadir-matched independent IR window check.
- **INSAT ↔ FY-4B (133°E):** eastern overlap for Asian convection.

**11.2 Polar overpasses = independent ground truth (higher res).** At each LEO overpass time, co-locate the GEO frame with MODIS B31 / VIIRS M15 / SLSTR S8 / AVHRR ch4 / Landsat-TIRS B10 / ECOSTRESS. Because LEOs are 0.75–1 km (down to 70–100 m for Landsat/ECOSTRESS), they validate **spatial sharpness** that 2–4 km GEO interpolation should preserve, and provide an absolute BT reference.

**11.3 Temporal-interpolation validation protocol (recommended):**
1. **Hold-out-the-middle:** from a dense GOES-19 C13 sequence (t-1, t, t+1 at 10 min), train to predict frame *t* from t-1 & t+1; evaluate PSNR/SSIM/MAE-in-Kelvin on held-out *t*. This is the core self-supervised signal (no external labels needed).
2. **Cross-sensor transfer test:** apply the GOES-trained model to Himawari B13 and GK-2A IR105 sequences (same 10-min cadence) → measures generalization across sensors/geometry.
3. **Independent-truth test:** where a LEO overpass falls between two GEO frames, interpolate the GEO to the overpass time and compare to the co-located, reprojected MODIS/VIIRS/SLSTR BT (after spectral adjustment ~10.3↔10.8 µm and GSICS bias correction).
4. **Operational test:** fine-tune/evaluate on **INSAT-3DR/3DS TIR1** 30-min frames — does the model upsample 30-min→finer cadence with fidelity verified against Himawari/Meteosat-IODC overlap?

**11.4 Harmonization before mixing sensors:**
- **Spectral:** C13 (10.3) ≈ Himawari B13 (10.4) ≈ GK-2A IR105 (10.5) ≈ Meteosat IR10.8 ≈ INSAT TIR1 (10.8) ≈ MODIS B31 (11.0). Small spectral offsets → apply a learned or RTTOV-based BT adjustment when using one as truth for another.
- **Radiometric:** use **GSICS** IR inter-calibration (IASI/CrIS-anchored) corrections so all sensors sit on a common scale (gsics; JMA/NOAA monitoring pages).
- **Geometric:** reproject everything to a common grid with **pyresample**/satpy (e.g., regular lat/lon over the INSAT domain) before computing cross-sensor metrics.

**11.5 Gap-filling:** when GOES has an eclipse/housekeeping gap, substitute the same-time frame from an overlapping GEO (Himawari/Meteosat) reprojected into the grid; when no GEO covers a sector at a needed instant, use the nearest LEO overpass as a sparse anchor. GMGSI (`noaa-gmgsi-pds`, Method 31) already does a global blend hourly and can serve as a coarse gap-fill backbone.

---

## 12. Concrete pull plan with exact paths (copy-paste starting points)

**Training (dense, primary) — GOES-19 ABI C13, Full Disk, 10-min:**
```python
import s3fs, xarray as xr, numpy as np
fs = s3fs.S3FileSystem(anon=True)
day = "noaa-goes19/ABI-L1b-RadF/2025/171"          # DOY 171
for hr in range(0, 24):
    for f in fs.ls(f"{day}/{hr:02d}/"):
        if "C13" in f:
            ds = xr.open_dataset(fs.open(f))
            # -> stack consecutive frames into (t-1, t, t+1) triplets for training
```
*Faster BT-ready alternative:* swap product to `ABI-L2-CMIPF` and read variable `CMI` (Kelvin) — no Planck conversion.

**Training (secondary) — Himawari-9 B13 gridded NetCDF (JAXA P-Tree) or HSD (AWS):**
- AWS HSD: `s3://noaa-himawari9/AHI-L1b-FLDK/2025/06/20/<HHMM>/HS_H09_*_B13_FLDK_R20_S*.DAT.bz2` → satpy `ahi_hsd`.
- JAXA P-Tree gridded NetCDF (register) → direct lat/lon grid, no stitching.

**Training (tertiary) — GK-2A IR105:** `s3://noaa-gk2a-pds/.../gk2a_ami_le1b_ir105_fd020ge_*.nc`.

**Validation (independent) — pick overpass times, pull:**
- MODIS `MOD021KM`/`MYD021KM` (B31) from LAADS or GEE; VIIRS `VNP02MOD` (M15); Sentinel-3 SLSTR LST/L1B from MS Planetary Computer STAC.

**Deployment (target) — INSAT-3DS/3DR TIR1:**
- MOSDAC API: `datasetId="3SIMG_L1B_STD"` (3DS) or `"3RIMG_L1B_STD"` (3DR), bbox `70,8,98,38`, run `python mdapi.py`. Inside HDF5: `IMG_TIR1` (+ `IMG_TIR1_TEMP` LUT → Kelvin).
- Redundant route: EUMETSAT `eumdac` collection `EO:EUM:DAT:INSAT:INSAT3D-L1C` (INSAT-3DS L1C BT, NetCDF/native).

---

## 13. Sources (key URLs)

- NOAA GOES on AWS — registry.opendata.aws/noaa-goes/ ; github.com/NOAA-Big-Data-Program/nodd-data-docs/tree/main/GOES ; github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-goes16/README.md
- goes2go — github.com/blaylockbk/goes2go
- GOES GCS mirror / BigQuery — cloud.google.com/blog/products/bigquery (weather satellite) ; console.cloud.google.com/storage/browser/gcp-public-data-goes-16
- GEE GOES-19 MCMIPF — developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_19_MCMIPF
- Himawari on AWS — registry.opendata.aws/noaa-himawari/
- JAXA Himawari Monitor (P-Tree) — eorc.jaxa.jp/ptree/ ; userguide.html ; registration_top.html ; 2026-01 change notice eorc.jaxa.jp/ptree/information/20180207_info.html
- MOSDAC — mosdac.gov.in ; user-manual-mosdac-data-download-api ; insat-3dr-data-products ; insat-3ds ; docs/INSAT3D_Products.pdf ; catalog-app/satellite.php
- INSAT-3DS overview — eoportal.org/satellite-missions/insat-3d ; en.wikipedia.org/wiki/INSAT-3DS
- INSAT-3DS L1C on EUMETSAT — user.eumetsat.int/catalogue/EO:EUM:DAT:INSAT:INSAT3D-L1C
- EUMETSAT Data Store / eumdac — user.eumetsat.int/resources/user-guides/data-store-detailed-guide ; eumetsat-data-access-client-eumdac-guide ; mtg-data-access-guide
- FengYun NSMC — data.nsmc.org.cn ; FY-4B AGRI reader ssec.wisc.edu/software/geo2grid/readers/agri_fy4b_l1.html ; eoportal.org/satellite-missions/fy-4
- GK-2A on AWS — registry.opendata.aws/noaa-gk2a-pds/ ; noaa-gk2a-pds.s3.amazonaws.com/index.html ; KMA NMSC nmsc.kma.go.kr / datasvc.nmsc.kma.go.kr
- Electro-L — eoportal.org/satellite-missions/electro-l
- MODIS/VIIRS — ladsweb.modaps.eosdis.nasa.gov (MOD021KM) ; earthdata.nasa.gov
- Sentinel-3 SLSTR — planetarycomputer.microsoft.com/dataset/sentinel-3-slstr-lst-l2-netcdf ; sentiwiki.copernicus.eu/web/s3-slstr-instrument ; user.eumetsat.int (SLSTR L1/L2 guides)
- MetOp AVHRR / IASI / GSICS — eumetsat.int/metop-sg-instruments ; ds.data.jma.go.jp/mscweb/data/monitoring/gsics/ir/techinfo_geoleoir.html
- Landsat — registry.opendata.aws (usgs-landsat) ; GEE LANDSAT/LC09/C02/T1_L2
- ECOSTRESS — lpdaac.usgs.gov ; AppEEARS
- NOAA GMGSI — registry.opendata.aws/noaa-gmgsi/ (bucket s3://noaa-gmgsi-pds)
- NODD index — noaa.gov/nodd/datasets ; registry.opendata.aws
- kerchunk/Zarr — fsspec.github.io/kerchunk/ ; github.com/fsspec/kerchunk ; registry.opendata.aws/noaa-nodd-kerchunk/ ; Pangeo Medium "Fake it until you make it"
- MS Planetary Computer — planetarycomputer.microsoft.com/catalog
- GHRSST Himawari (PO.DAAC) — podaac.jpl.nasa.gov/dataset/H09-AHI-L3C-ACSPO-v2.90

---

### Verification flags
- `[verify]` GEE Himawari AHI exact L1 collection ID (catalog shifts).
- `[verify]` exact MTG/FCI L1C EUMETSAT collection IDs at run time.
- `[verify]` Electro-L current public download portal/credentials (Roscosmos/SRC Planeta).
- INSAT HDF5 internal dataset names (`IMG_TIR1`, `IMG_TIR1_TEMP`) per the MOSDAC INSAT-3D Products format PDF — confirm against an actual file for 3DS vs 3DR.
- GK-2A exact S3 subfolder hierarchy (`AMI/L1B/FD/<YYYYMM>/<DD>/<HH>/`) — confirm with `aws s3 ls --no-sign-request` before scripting.

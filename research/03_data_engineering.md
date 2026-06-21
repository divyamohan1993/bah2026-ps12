# 03 — High-Performance Geospatial Data Engineering for Satellite Frame Interpolation

**Project:** ISRO BAH 2026 PS-12 — "Fill in the Frames Seamlessly" (AI/ML optical-flow video frame interpolation, VFI, for geostationary TIR satellite imagery)
**Scope of this doc:** FASTEST platform + O(1) / constant-time data access & preprocessing techniques, end-to-end (ingest → preprocess → train → serve).
**Date:** 2026-06-20
**Author:** data-engineering research agent

> Research method: 16+ web searches + targeted page fetches (June 2026). All sources cited inline and in the Sources section. Where a claim is from January-2026 background knowledge rather than a fetched source, it is marked `[bg]`.

---

## 0. TL;DR — Recommended Fast Data Stack

**The honest framing of "O(1)".** True algorithmic O(1) (constant time independent of dataset size) is achievable for *random sample/chunk/tile access* when the address of the byte range you need is computable directly from the request (no scan, no index walk that grows with N). It is **not** achievable for the heavy compute steps (warp/regrid, radiance→BT, running the VFI network). The winning strategy is therefore: **do all the expensive, super-linear work ONCE, offline, and persist the results in formats whose read path is a single direct-addressed range request.** Then both training-sample access and dashboard frame/tile delivery become O(1).

Two distinct O(1) access paths, two different formats:

| Stage | Format | Why it is O(1) | Key tech |
|---|---|---|---|
| **Ingest / archival access** | **VirtualiZarr + Icechunk** (virtual Zarr over the original `.nc`/`.h5`) | chunk key → precomputed `(file, offset, length)` → one HTTP range GET; no copy | `virtualizarr`, `icechunk`, `fsspec`/`obstore` |
| **Analysis-ready cube** | **Zarr v3** (BT, normalized, on a common grid), sharded, `blosc(zstd)+shuffle` | `array[t, y, x]` → integer chunk-coord → exactly one shard/chunk object | `zarr`, `xarray`, `dask`, `numcodecs` |
| **Training sample access** | **Zarr v3 chunk = sample tile** (or WebDataset/FFCV/LMDB shard) | sample index → chunk coord (Zarr) or byte offset (FFCV/LMDB) = constant | `tensorstore`/`kvikIO` for GPU-direct reads, `DALI` |
| **Serving (the real O(1) part)** | **Precomputed tile pyramid (PNG/WebP XYZ) + per-clip MP4/HLS**, content-addressed, on a **CDN** | `(z,x,y,t)` → deterministic object key → CDN edge cache hit, constant time | static tiles + CDN; TiTiler only as a dynamic fallback |

**One-line architecture:**
`AWS noaa-goes19 .nc + MOSDAC INSAT .h5` → *(VirtualiZarr virtual refs, no copy)* → *(Dask: warp to common lat/lon grid with **cached pyresample KDTree**, convert radiance→BT, normalize)* → **Zarr v3 analysis cube (sharded, zstd+shuffle)** → *train VFI (tensorstore/DALI O(1) sample reads)* → *(batch-infer intermediate frames)* → **render to precomputed XYZ tile pyramid + MP4 per region/day** → **S3 + CDN (Cache-Control: immutable, content-addressed keys)** → web dashboard does **O(1) tile/clip GETs**.

---

## 1. Cloud-Optimized Array Formats & Virtual Access

### 1.1 Zarr (v2 / v3) — the core analysis-ready format

**What makes Zarr near-O(1) for random access.** A Zarr array is split into equally-sized **chunks**; *traditionally each chunk maps to exactly one object/key in storage* (a file on disk, or an S3 key). To read `array[t, y0:y1, x0:x1]` the library computes the integer chunk coordinates directly by arithmetic (`floor(index / chunk_size)`), so locating the chunk(s) is **constant time** — there is no index B-tree to walk whose depth grows with the number of chunks. ([cloudnativegeo Zarr guide](https://guide.cloudnativegeo.org/zarr/intro.html), [Earthmover "What is Zarr"](https://www.earthmover.io/blog/what-is-zarr/)). If your read aligns to a single chunk, it is exactly **one** GET → O(1).

**v2 vs v3.**
- v3 replaces the scattered `.zarray`/`.zattrs` with a single `zarr.json`, adds a **codec pipeline** (chain transforms per chunk), variable-length dtypes, and crucially the **sharding codec**. ([zarr-specs v3 core](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html), [Zarr-Python 3 release](https://zarr.dev/blog/zarr-python-3-release/)).
- The **sharding codec** stores many inner chunks ("inner chunks") inside one storage object (a "shard"), with an index at the end of the shard. This decouples *chunk size* from *object count* — you can keep chunks small (good for fine-grained, low-latency reads and browser viz) without producing millions of tiny S3 objects (which S3/GCS/Lustre handle poorly). Read of one inner chunk = one ranged GET into the shard using the shard's footer index → still O(1). ([sharding-indexed spec](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html), [ZEP 2](https://zarr.dev/zeps/accepted/ZEP0002.html)).

> **Why we want v3 + sharding here:** GOES-19 full-disk at 2 km is ~5424×5424; a long time series of 2D TIR fields with small spatial chunks would otherwise explode object counts. Sharding keeps it manageable and keeps per-tile reads cheap.

**Compression (chunk-level).** Data is compressed *per chunk*, so within-chunk partial reads are not possible (you read+decompress the whole chunk) — another reason chunk shape matters. ([cloudnativegeo Zarr guide](https://guide.cloudnativegeo.org/zarr/intro.html)). Recommended codecs below in §1.5.

**Example: create a v3 sharded, compressed analysis cube**
```python
import zarr
import numpy as np
from zarr.codecs import BloscCodec, BloscShuffle

# TIR brightness-temperature cube: (time, y, x) on a common lat/lon grid
shape  = (4032, 2816, 2816)     # ~4 weeks @ 10 min for a regional GOES crop, e.g.
# chunk = a "tile" sized for both ML patches AND map tiles (multiple of 256)
chunks = (8, 512, 512)          # ~8 frames x 512x512 ; ~8 MB raw (float32) -> good
shards = (32, 1024, 1024)       # 4x2x2 = 16 chunks per shard -> few large objects

z = zarr.create_array(
    store="s3://my-bucket/cubes/goes19_tir_bt.zarr",  # or local path
    shape=shape, chunks=chunks, shards=shards, dtype="float32",
    compressors=[BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
    fill_value=np.nan,           # space pixels = NaN
)
# z[t0:t1, y0:y1, x0:x1] -> arithmetic -> exactly the needed shard/chunk objects
```
Chunk-size guidance: chunks of **≥1 MB uncompressed** give better Blosc performance; for 2D arrays accessed along one axis, chunk full-width on the perpendicular axis; for balanced access use moderate square chunks (e.g. `(1000,1000)`). ([zarr performance guide](https://zarr.readthedocs.io/en/latest/user-guide/performance/)).

### 1.2 kerchunk / VirtualiZarr — virtual Zarr over the original NetCDF/HDF5 (zero-copy O(1))

This is the single most important technique for *ingest* in this project, because the rules require `.nc`/`.h5` I/O and the source archives (NOAA GOES on S3, MOSDAC INSAT) are already NetCDF4/HDF5.

**Idea.** Every NetCDF4/HDF5 file is internally chunked. **kerchunk** scans a file once and records, for each variable's internal chunk, its **byte offset, length, and compression** inside the original file. That metadata becomes a "reference set" presented to Zarr/xarray as a *virtual Zarr store*. Reading a chunk then becomes a **single HTTP range GET** of `[offset, offset+length)` in the untouched original file — **no copy, no translation**, and the address is precomputed → **O(1)**. ([kerchunk docs](https://fsspec.github.io/kerchunk/), [Pangeo/Marsh tutorial](https://medium.com/pangeo/accessing-netcdf-and-grib-file-collections-as-cloud-native-virtual-datasets-using-kerchunk-625a2d0a9191), [fsspec/kerchunk repo](https://github.com/fsspec/kerchunk)).

**VirtualiZarr** is the modern, Zarr-native, xarray-friendly successor: it builds the same virtual references but as `ManifestArray` objects you manipulate with the normal xarray API, then export to kerchunk JSON/parquet **or** to an Icechunk repo. "It merely creates an in-memory lookup table that points to the location of chunks in the original netCDF when data is needed later on." ([VirtualiZarr docs](https://virtualizarr.readthedocs.io/en/latest/index.html), [usage](https://virtualizarr.readthedocs.io/en/latest/usage.html), [VEDA CMIP6 example](https://docs.openveda.cloud/user-guide/notebooks/veda-operations/generate-cmip6-virtual-zarr-historical.html)). Supported: NetCDF3, NetCDF4/HDF5, GRIB2, TIFF/COG.

**Example: virtualize a GOES-19 `.nc` collection (no copy) and aggregate over time**
```python
import xarray as xr
from obstore.store import from_url
from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.parsers import HDFParser            # netCDF4 == HDF5
from obspec_utils.registry import ObjectStoreRegistry

bucket = "s3://noaa-goes19"
store  = from_url(bucket, region="us-east-1", skip_signature=True)  # public bucket
registry = ObjectStoreRegistry({bucket: store})

urls = [f"{bucket}/ABI-L2-CMIPF/2025/170/00/OR_ABI-L2-CMIPF-M6C13_G19_s...nc",
        f"{bucket}/ABI-L2-CMIPF/2025/170/00/OR_ABI-L2-CMIPF-M6C13_G19_s...nc"]

vds = open_virtual_mfdataset(
    urls, registry=registry, parser=HDFParser(),
    combine="by_coords", combine_attrs="drop_conflicts",
    loadable_variables=["t", "x", "y"],   # tiny coords get loaded; big data stays virtual
)
# Persist the reference set; reads later are single range GETs into the original .nc
vds.vz.to_kerchunk("refs/goes19_c13_day170.parquet", format="parquet")

# Consume virtually with xarray (lazy, O(1) per chunk)
ds = xr.open_dataset("refs/goes19_c13_day170.parquet", engine="kerchunk")
```
Save references as **parquet** for large reference sets (faster to load than giant JSON). ([VirtualiZarr usage](https://virtualizarr.readthedocs.io/en/latest/usage.html)).

### 1.3 Icechunk — transactional, multi-writer storage engine for Zarr (incl. virtual)

**Icechunk 1.0 (July 2025)** sits *under* Zarr and adds: serializable-isolation transactions, time-travel/snapshots (cheap revert), and — key for us — **virtual chunk containers** so a single Icechunk repo can mix *native* Zarr chunks with *virtual* references into original `.nc`/`.h5` byte ranges. ([Icechunk 1.0 blog](https://www.earthmover.io/blog/icechunk-1-0-production-grade-cloud-native-array-storage-is-here/), [Icechunk repo](https://github.com/earth-mover/icechunk), [virtual datasets guide](https://icechunk.io/en/latest/virtual/)).

Performance: Icechunk has its **own async multithreaded I/O pipeline** and is "at least as fast as the existing Zarr/Dask/fsspec stack and in many cases achieves significantly lower latency and higher throughput, *without using Dask*." ([Pangeo announcement](https://discourse.pangeo.io/t/icechunk-a-new-cloud-native-transactional-storage-engine-for-zarr/4610), [Earthmover I/O-maxing](https://www.earthmover.io/blog/i-o-maxing-tensors-in-the-cloud/)). The transaction/snapshot model is ideal for an **append-on-new-frame** ingest loop and for reproducible model training (pin a snapshot ID).

**Example: commit virtual refs + native data to Icechunk**
```python
import icechunk
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
    url_prefix="s3://noaa-goes19/",
    store=icechunk.s3_store(region="us-east-1", anonymous=True)))
repo = icechunk.Repository.create(icechunk.s3_storage(
    bucket="my-bucket", prefix="icechunk/goes19"), config)

s = repo.writable_session("main")
vds.vz.to_icechunk(s.store)                 # virtual refs (no copy)
snap = s.commit("ingest GOES-19 C13 day170")
# later, append the next time step in O(1) metadata work:
s = repo.writable_session("main")
vds_next.vz.to_icechunk(s.store, append_dim="time")
s.commit("append next frame")
```

### 1.4 Other formats — when (not) to use them here

- **Cloud-Optimized GeoTIFF (COG).** A normal GeoTIFF laid out for HTTP range GETs: internal **tiling matches web-map tile structure so "only one HTTP range request needs to be performed to access any tile"**, plus **overviews** (downsampled pyramids) so a zoomed-out view fetches a tiny overview, not full res; all metadata (incl. tile-offset table) is at the front and readable in a single ~16 KB request. ([cogeo in-depth](https://cogeo.org/in-depth.html), [OGC COG standard](https://docs.ogc.org/is/21-026/21-026.html), [GDAL COG driver](https://gdal.org/en/stable/drivers/raster/cog.html)). **Use COG as the per-frame rendered raster** (one COG per timestamp per channel) feeding TiTiler, and/or as the *output* `.nc`-companion. COG = O(1) tile read. Not ideal as the *training cube* (per-file overhead; Zarr handles the N-D time axis better).
- **Parquet / GeoParquet.** Columnar; great for *tabular* metadata catalogs (frame index, per-frame stats, QC flags, tile manifests) and for **kerchunk reference sets**. Row-group + column statistics give predicate pushdown (fast filtering, not strictly O(1) but near-constant for point lookups with a sorted key). Use Parquet for the **frame/tile catalog** the dashboard queries. `[bg]`
- **HDF5 vs NetCDF4 chunking.** NetCDF4 *is* HDF5 underneath. Both support internal chunking + compression; the catch is HDF5's chunk **B-tree index** and the fact that, read naively over the network, the library may issue many small reads walking metadata. kerchunk/VirtualiZarr fixes this by pre-extracting the chunk offset table once → afterwards reads are direct range GETs. **Always set sensible internal chunking when you (re)write `.nc` outputs** (e.g. `chunksizes=(1, 512, 512)`), or downstream virtual access will be coarse. ([kerchunk repo](https://github.com/fsspec/kerchunk), [Pangeo HYCOM example](https://medium.com/pangeo/using-kerchunk-with-uncompressed-netcdf-64-bit-offset-files-cloud-optimized-access-to-hycom-ocean-9008ba6d0d67)).

### 1.5 Compression: zstd / blosc / lz4 — speed vs ratio

- **Blosc** is a *meta-compressor*: it shards a chunk into blocks, runs them multi-threaded, and applies **shuffle / bitshuffle** byte/bit reordering that dramatically improves both ratio and speed on numeric arrays. ([blosc.org](https://blosc.org/pages/)).
- **LZ4**: fastest decompression at every level (6–8× faster than zlib); best when you are I/O- or latency-bound and want minimal CPU. ([LZ4 wiki](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)), [FCBench](https://arxiv.org/pdf/2312.10301)).
- **Zstd**: best ratio among the fast codecs at "reasonably fast" speed — the best *balance*. ([blosc.org](https://blosc.org/), holotomography benchmark [arXiv:2503.18037](https://arxiv.org/pdf/2503.18037)).

**Recommendation for TIR BT (smooth, spatially correlated float fields):**
- **Analysis cube (cold-ish, read many times in training):** `Blosc(cname="zstd", clevel=5, shuffle=SHUFFLE)`. Great ratio, decompress fast enough.
- **GPU-direct / latency-critical path:** `Blosc(cname="lz4", shuffle)` or LZ4 — nvCOMP can GPU-decompress LZ4 (see §3). ([kvikIO Zarr docs](https://docs.rapids.ai/api/kvikio/stable/zarr/)).
- Apply **bit-rounding / quantization** before compression (e.g. round BT to 0.01 K) to boost ratio massively with negligible model impact. `[bg]`

---

## 2. Chunking & Tiling Strategy

### 2.1 Chunk shape for a time-series of 2D fields (the central decision)

The cube is `(time, y, x)`. Access patterns differ by stage:
- **Training a VFI model:** you read **short temporal windows of small spatial patches** — e.g. frames `t, t+1, t+2` over a 256–512 px patch. → chunk should bundle a few time steps with a spatial tile: `chunks=(≈4–8, 512, 512)`.
- **Dashboard rendering:** you read **one full frame at a coarse zoom** or a **tile at native zoom** → spatial tiling (256/512) is again right; time chunk can stay small.

General rules confirmed by sources:
- Aim for **~100 MB Dask chunks** built from **≥1 MB storage chunks**; too-small time chunks are inefficient if you compute along time. ([Dask array best practices](https://docs.dask.org/en/latest/array-best-practices.html), [zarr performance](https://zarr.readthedocs.io/en/latest/user-guide/performance/)).
- A practical satellite example: `{'lat':1000,'lon':3000,'time':10}` or `{'lat':1000,'lon':3000,'time':10}` style. ([PO.DAAC dask tutorial](https://podaac.github.io/tutorials/notebooks/Advanced_cloud/basic_dask.html)).
- **Align Dask chunks to storage chunks** (each Dask chunk dim = integer multiple of the Zarr chunk dim) or you re-read data repeatedly. ([Dask chunks docs](https://docs.dask.org/en/latest/array-chunks.html)).
- **`rechunker`** does out-of-core rechunking without blowing memory — use it to produce a *training-optimized* copy if your ingest layout differs. ([Abernathey rechunker](https://medium.com/pangeo/rechunker-the-missing-link-for-chunked-array-analytics-5b2359e9dc11)).

> **Concrete choice for PS-12:** store the cube with `chunks=(8, 512, 512)` float32 (~8 MB) + `shards=(32,1024,1024)`. This single layout serves *both* training patches and map tiles (512 is a multiple of 256). One training window or one map tile ≈ one chunk read = **O(1)**.

### 2.2 Spatial tile pyramids (XYZ / TMS / WMTS) + overviews — the serving substrate

For the dashboard, the O(1) win is a **precomputed tile pyramid**: render each output frame to a multi-resolution set of 256×256 (or 512×512) tiles addressed by `(z, x, y)` in the standard slippy-map (XYZ/WMTS) scheme. A client request `(z,x,y)` maps by pure arithmetic to a tile object key → **constant-time lookup**, and the tile is already an encoded PNG/WebP. Overviews/pyramids mean zoomed-out views fetch few small tiles instead of full-res. (COG provides exactly this layout internally; a static tile tree provides it as discrete files.) ([cogeo in-depth](https://cogeo.org/in-depth.html), [Kyle Barron COG mosaic](https://kylebarron.dev/blog/cog-mosaic/overview/)).

Pyramid math (why it is O(1)): tile at zoom `z` covering lon/lat is `x = floor((lon+180)/360 * 2^z)`, `y = floor(...)`. No search.

### 2.3 Precomputed tiles for O(1) CDN serving vs dynamic tiling

"**Static tiling will always be faster to load than dynamic tiling, but a cache layer can be set up in front of the dynamic tiler.**" ([TiTiler dynamic tiling](https://developmentseed.org/titiler/user_guide/dynamic_tiling/)). For an animation dashboard where the set of frames is **finite and known after inference**, **precompute everything** → pure static O(1) serving. Keep TiTiler/rio-tiler as a *fallback* for ad-hoc exploration of the source COGs.

---

## 3. Fast I/O Libraries

| Library | Role | Speed / O(1) note |
|---|---|---|
| **xarray + dask** | lazy N-D labeled arrays, parallel out-of-core compute | Lazy slicing reads only needed chunks; parallelizes warp/BT over chunks. Not O(1) for compute, but reads are chunk-direct. ([xarray-dask docs](https://examples.dask.org/xarray.html)) |
| **fsspec / s3fs + caching** | remote filesystem + transparent cache | `simplecache` (whole-file, thread/proc-safe), `blockcache` (fixed MB blocks, LRU, sparse-file, great for partial Zarr reads), `filecache`. Warm cache → local O(1) re-reads. ([fsspec features](https://filesystem-spec.readthedocs.io/en/latest/features.html), [Anaconda fsspec caching](https://www.anaconda.com/blog/fsspec-remote-caching)) |
| **kvikIO + GPUDirect Storage (GDS)** | read Zarr chunks **straight into GPU memory** | `kvikio.zarr.GDSStore` reads → CuPy arrays via cuFile, bypassing the CPU bounce buffer; GPU-decompress LZ4 via **nvCOMP**. Lower latency; GDS needs ext4/xfs etc. ([kvikIO Zarr](https://docs.rapids.ai/api/kvikio/stable/zarr/), [xarray+kvikIO](https://xarray.dev/blog/xarray-kvikio), [zarr-benchmark GDS](https://github.com/zarr-developers/zarr-benchmark/discussions/14)) |
| **NVIDIA DALI** | GPU data-loading/augmentation pipeline | Offloads decode/crop/resize/normalize to GPU; async prefetch + parallel exec removes the CPU dataloader bottleneck. ([DALI blog](https://developer.nvidia.com/blog/fast-ai-data-preprocessing-with-nvidia-dali/), [DALI repo](https://github.com/NVIDIA/DALI)) |
| **tensorstore** | high-perf C++/Py array I/O (Zarr/N5) | Async API, read/write caching, optimistic-concurrency; **"faster than Zarr-Python in every case except loading a single 1 GB chunk"**, near-linear scaling with concurrency, matches `fio`. Great for the **training read path**. ([TensorStore](https://google.github.io/tensorstore/), [research.google](https://research.google/blog/tensorstore-for-high-performance-scalable-array-storage/), [zarr-benchmark #25](https://github.com/zarr-developers/zarr-benchmark/discussions/25)) |
| **rioxarray / rasterio** | GeoTIFF/COG read + warp | Range-GET COG reads; GDAL warp for reprojection. |
| **numcodecs** | codec implementations (blosc, zstd, lz4, shuffle, delta…) | Underlies Zarr compression. |

**Memory-mapping / lazy loading.** Local Zarr/HDF5 + `mmap`/lazy xarray means the OS pages in only the chunks touched → effectively O(1) per access once the layout is direct. FFCV/LMDB (below) mmap their stores too.

**GPU pipeline pattern (Earth-science specific):** Zarr (LZ4) → kvikIO GDS read to GPU → nvCOMP decompress on GPU → DALI augment on GPU → model. ([xarray GPU pipeline blog](https://xarray.dev/blog/gpu-pipeline), [FOSS4G 2025 GPU-native Zarr](https://talks.osgeo.org/foss4g-2025/talk/TSVGYJ/)).

---

## 4. Reprojection / Regridding of Geostationary Data (do it once, cache the kernel)

**The problem.** GOES-19 ABI and INSAT-3DS are on **GEOS (geostationary) fixed-grid** projections from *different* sub-satellite longitudes (GOES-East 75.2°W; INSAT ~74–82°E), Himawari from ~140°E. To co-register them (and to make a clean lat/lon cube for the dashboard) you must **resample/regrid** to a common target grid (a regional lat/lon `AreaDefinition`). ([GOES on AWS registry](https://registry.opendata.aws/noaa-goes/); INSAT TIR1 4 km, full-disk every 30 min ([MOSDAC INSAT3D product doc](https://www.mosdac.gov.in/docs/INSAT3D_Products.pdf), [SST ATBD](https://www.mosdac.gov.in/look/DOCS/ATBD_INSAT-3D_SST_REV_V1.1.pdf))).

**The O(1)/fast trick: cache the resampling kernel (KDTree / weights).** For a *fixed* source grid → *fixed* target grid (exactly the geostationary case, since the satellite and grid don't move), the nearest-neighbor/bilinear **mapping is identical for every time step**. Compute the KDTree neighbor indices + weights **once**, cache to disk, then every subsequent frame's resample is just a **gather** (index + weighted sum) — no tree build. This turns per-frame regridding from O(N log N) into a near-O(N) gather with a precomputed kernel (and O(1) "setup" since the kernel is loaded, not rebuilt).

- **pyresample** uses `pykdtree` (fast KDTree) for NN/bilinear. ([pyresample README](https://github.com/pytroll/pyresample/blob/v1.20.0/README.md)).
- **satpy** wraps it: `KDTreeResampler` with **on-disk caching via `cache_dir`** — explicitly noted as "most beneficial with geostationary satellite data where the locations of the source data and the target pixels don't change over time … significant performance improvements on consecutive resampling." ([satpy resampling](https://satpy.readthedocs.io/en/stable/resample.html)).

**Example: build kernel once, reuse for all frames (pyresample)**
```python
from pyresample import geometry, kd_tree
import numpy as np

src = geometry.AreaDefinition(...)   # GOES-19 GEOS fixed grid (from .nc metadata)
tgt = geometry.AreaDefinition(...)   # common regional lat/lon grid

# Build neighbour info ONCE (the expensive KDTree step):
valid_in, valid_out, idx, dist = kd_tree.get_neighbour_info(
    src, tgt, radius_of_influence=6000, neighbours=1)
np.savez("kernel_goes19_to_grid.npz", valid_in=valid_in, valid_out=valid_out,
         idx=idx, dist=dist)            # cache it

# Per-frame: just gather (cheap, reused for every timestamp)
def resample_frame(bt_2d):
    return kd_tree.get_sample_from_neighbour_info(
        'nn', tgt.shape, bt_2d, valid_in, valid_out, idx)
```
Or with satpy (kernel auto-cached):
```python
from satpy import Scene
scn = Scene(reader="abi_l2_nc", filenames=[...])   # GOES-19 ABI L2 reader
scn.load(["C13"])
local = scn.resample("my_area_def", resampler="nearest", cache_dir="/cache/resample")
# 2nd..Nth call with same areas reuses cached KDTree -> fast
```
([satpy ABI L2 reader/geo2grid](https://www.ssec.wisc.edu/software/geo2grid/readers/abi_l2_nc.html)).

**Alternatives.** `rioxarray.reproject` / rasterio `warp` (GDAL) for COG-style warps; for many fixed warps, GDAL can also reuse transformer setups but pyresample/satpy caching is the cleanest for swath-style geostationary NN. For INSAT vs GOES co-registration, resample **both** to the *same* target `AreaDefinition` (then they're pixel-aligned for training/transfer).

---

## 5. Preprocessing for ML

### 5.1 Radiance → Brightness Temperature (BT)

- **GOES-19 ABI**: the **L2 CMIP** product already provides Cloud-and-Moisture Imagery; for C13 (10.3 µm) you can use CMI directly, or convert L1b radiance with the Planck coefficients (`planck_fk1, fk2, bc1, bc2`) in the file. ([GOES-R Beginner's Guide](https://www.goes-r.gov/downloads/resources/documents/Beginners_Guide_to_GOES-R_Series_Data.pdf), [ABI L2 reader](https://www.ssec.wisc.edu/software/geo2grid/readers/abi_l2_nc.html)).
- **INSAT-3DS TIR1 (10.8 µm)**: L1C stores counts, radiance **and** BT; **"Quadratic, Gain and Offset coefficients are provided for calculating radiances directly from the HDF5 files"**, then a LUT/Planck gives BT. ([MOSDAC INSAT3D product format](https://www.mosdac.gov.in/docs/INSAT3D_Products.pdf), [EUMETSAT INSAT-3DS L1C](https://user.eumetsat.int/catalogue/EO:EUM:DAT:INSAT:INSAT3D-L1C)).

Do this conversion **once**, store BT in the Zarr cube → training/serving never re-derive it.

### 5.2 Normalization (single-channel TIR)

Standard practice: **normalize using channel statistics computed over the training period** (z-score), which "ensures stable and balanced model optimization." ([Nature 3D U-Net IR nowcasting](https://www.nature.com/articles/s41598-025-34207-9)). Practical recipe:
- Compute global `mean`, `std` (or robust min/max, e.g. physical BT range ~180–330 K) once over the training split; persist as cube attrs.
- Normalize `x' = (BT - mean)/std`. For VFI, a fixed physical-range min-max (e.g. map [180 K, 330 K] → [0,1]) is reproducible across GOES/INSAT and makes transfer cleaner. `[bg]`
- Store normalization constants in `zarr.json` attrs / catalog so inference is deterministic.

### 5.3 NaN / space-pixel masking

Geostationary full-disk has off-Earth (space) pixels. Use `fill_value=NaN`; carry a boolean **valid mask** (from the resample `valid_out`, §4) as a companion array. The VFI loss must ignore NaNs; feed a mask channel or fill with the per-frame mean. `[bg]`

### 5.4 Patch extraction + sample storage: on-the-fly vs precomputed

**Precompute wins for O(1).** Two good options:

1. **Zarr-as-dataset (recommended, single source of truth).** With `chunks=(8,512,512)`, a training sample = a contiguous `(window, 512, 512)` slab = **one chunk read → O(1)**. Use `tensorstore`/`kvikIO` for fast/GPU-direct reads. No extra copy of data.

2. **Sharded sample formats** when you want max dataloader throughput and explicit shuffling:
   - **WebDataset**: tar shards of samples; streams + shards beautifully for large-scale, but **sequential** — *no native random access to individual samples* (shuffle is buffer/shard-level). ([FFCV CVPR supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Leclerc_FFCV_Accelerating_Training_CVPR_2023_supplemental.pdf)).
   - **FFCV** (`.beton`): "**eliminates random read penalties by making it easy to read data in large chunks**" and **significantly outperforms PyTorch DataLoader, WebDataset, and DALI**; supports quasi-random + memory-mapped, near-O(1) sample access. Best raw training speed. ([FFCV paper](https://arxiv.org/pdf/2306.12517), [FFCV benchmarks](https://docs.ffcv.io/benchmarks.html), [FFCV repo](https://github.com/libffcv/ffcv/)).
   - **LMDB**: memory-mapped B+tree key→value; **the pick for heavy random access** (true O(1)-ish keyed lookups), simple, battle-tested. ([HackerNoon dataloader landscape](https://hackernoon.com/an-overview-of-the-data-loader-landscape-experimental-setup)).

> **PS-12 recommendation:** keep the **Zarr cube as the canonical store**; if profiling shows the dataloader is the bottleneck, **bake an FFCV `.beton`** (or LMDB if you need keyed random access) of pre-extracted `(prev, mid_gt, next)` triplets for the GOES/Himawari training split. Triplet sampling for VFI = "frames at 00:00 and 00:20 predict 00:10" — precompute these index triplets in a Parquet catalog.

---

## 6. Serving Layer for O(1) Frame / Animation Delivery (the real O(1))

**Design principle:** the dashboard must *never* run the model or GDAL on the request path. After inference, the full set of frames is finite → **precompute outputs into directly-addressable, CDN-cacheable artifacts.**

### 6.1 Two artifact types

1. **Animation playback → per-clip video.** For each region/day/variant (original vs interpolated), render an **MP4 (H.264/H.265)** or **HLS** clip with `ffmpeg`. The dashboard `<video>`/`hls.js` requests segments; the **CDN caches segments at edge** so playback is constant-time per segment, scales to many users. ([Cloudflare HLS](https://www.cloudflare.com/learning/video/what-is-http-live-streaming/), [Mux mp4→HLS](https://www.mux.com/articles/how-to-convert-mp4-to-hls-format-with-ffmpeg-a-step-by-step-guide)). For short loops, a single MP4/animated WebP is simplest.
2. **Pan/zoom map → precomputed XYZ tile pyramid** (PNG/WebP) per timestamp, key `tiles/{var}/{t}/{z}/{x}/{y}.webp`. Client `(z,x,y,t)` → deterministic key → CDN edge → **O(1)**.

### 6.2 Content-addressed keys + immutable caching = guaranteed O(1) at the edge

Put a **content hash (or fixed `(t,z,x,y)`)** in the URL and serve with **`Cache-Control: public, max-age=31536000, immutable`**. "Only use `immutable` on content-addressed resources (files with a hash in the URL)"; it "nullifies the need for browsers to perform conditional revalidation." ([KeyCDN immutable](https://www.keycdn.com/blog/cache-control-immutable), [MDN Cache-Control](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cache-Control), [wrujel HTTP caching](https://blog.wrujel.com/http-caching-cache-control-etags-cdn-strategies-cfce0a)). Result: after first edge fill, every tile/clip read is a **constant-time cache hit**, no origin round-trip, no revalidation.

### 6.3 Caching tiers (all O(1) lookups)

- **CDN edge** (CloudFront/Cloudflare/Fastly): primary, geo-distributed constant-time GET.
- **Redis** (optional): cache the *frame/tile catalog* and any small JSON the SPA needs; keyed `GET` is O(1). A redis-backed fsspec cache also exists. ([redis-fsspec-cache](https://github.com/mpiannucci/redis-fsspec-cache)).
- **HTTP cache / browser**: immutable assets cached locally.

### 6.4 Dynamic fallback (only for exploration)

For ad-hoc inspection of source COGs, run **TiTiler** (FastAPI + rio-tiler) behind the CDN: it serves XYZ tiles dynamically from COGs (`/cog/tiles/{z}/{x}/{y}`), with rescaling/colormap on the fly. Put a cache in front since "static tiling will always be faster." ([TiTiler repo](https://github.com/developmentseed/titiler), [dynamic tiling](https://developmentseed.org/titiler/user_guide/dynamic_tiling/), [TiTiler explained](https://www.geowgs84.ai/post/titiler-explained-fast-cloud-native-raster-tile-generation-for-gis)). **Not on the hot path** for the deliverable animations.

### 6.5 Serving artifact generation example
```bash
# Per-frame COG (for TiTiler fallback + as raster of record)
gdal_translate frame_t.tif frame_t_cog.tif -of COG \
   -co COMPRESS=DEFLATE -co BLOCKSIZE=512 -co OVERVIEWS=AUTO

# Precompute XYZ tiles for a frame (gdal2tiles / rio-tiler)
gdal2tiles.py --xyz --processes=8 -z 3-8 frame_t_cog.tif tiles/var/t/

# Interpolated animation clip (CDN-cached, content-hashed name)
ffmpeg -framerate 12 -pattern_type glob -i 'frames/interp/*.png' \
   -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
   clips/interp_region_day170_<sha256>.mp4
```

---

## 7. Pipeline Orchestration

The pipeline has a **batch backfill** part (build the cube + train) and a **near-real-time append** part (new GOES/INSAT frame → warp/BT → append to Icechunk → infer → render tiles).

- **Airflow**: heaviest ops (scheduler+DB+workers; ~0.5–1 FTE at scale), biggest operator ecosystem; good for static, predictable DAGs. ([ZenML showdown](https://www.zenml.io/blog/orchestration-showdown-dagster-vs-prefect-vs-airflow), [Bruin 2026](https://getbruin.com/blog/best-data-pipeline-tools-2026/)).
- **Prefect**: **lightest** self-hosted, dynamic flows, fast dev iteration, hybrid execution — best for a small team / new platform. ([ZenML showdown](https://www.zenml.io/blog/orchestration-showdown-dagster-vs-prefect-vs-airflow), [UK Data Services 2026](https://ukdataservices.co.uk/blog/articles/python-data-pipeline-tools-2025)).
- **Dagster**: **asset-first**, strong typing/lineage/observability, built-in asset checks — best if you treat the cube/frames/tiles as **versioned data assets** (which pairs naturally with Icechunk snapshots). ([ZenML showdown](https://www.zenml.io/blog/orchestration-showdown-dagster-vs-prefect-vs-airflow)).

> **PS-12 recommendation:** For a hackathon/prototype, **Prefect** (minimal overhead, dynamic) — or **Dagster** if you want clean asset lineage (raw `.nc` → virtual refs → cube snapshot → tiles) for the demo/report. Airflow only if a heavy production scheduler is mandated. **Streaming vs batch:** batch-backfill historical GOES/Himawari for training; **micro-batch/append** for live INSAT frames (Icechunk `append_dim="time"` per arrival = O(1) metadata write, then trigger infer+tile).

---

## 8. Recommended Concrete Architecture (the answer)

### 8.1 Storage formats
- **Ingest/archival:** original GOES `.nc` (S3 `noaa-goes19`) + INSAT `.h5` (MOSDAC) accessed **virtually** via **VirtualiZarr → Icechunk** (zero-copy, O(1) chunk reads, append-friendly, snapshot-versioned). Satisfies the "input must be `.nc`/`.h5`" rule.
- **Analysis-ready cube:** **Zarr v3**, dims `(time, y, x)`, **`chunks=(8,512,512)` float32**, **`shards=(32,1024,1024)`**, compressor **`Blosc(zstd, clevel=5, shuffle)`** (or LZ4 for GPU-decode path), `fill_value=NaN`, BT-converted + normalized, on a **common regional lat/lon grid** (so GOES & INSAT are co-registered).
- **Serving:** per-frame **COG** + precomputed **XYZ tile pyramid (WebP)** + per-clip **MP4/HLS**; a **Parquet** frame/tile catalog.

### 8.2 Preprocessing path (offline, ONCE)
1. `VirtualiZarr.open_virtual_mfdataset` over the file collection → virtual refs (no copy).
2. Dask map over chunks: **radiance→BT** (file coeffs/Planck) → **resample with cached pyresample KDTree** to common grid (kernel built once, reused) → **normalize** (precomputed stats) → mask space pixels (NaN).
3. Write to **Zarr v3 cube** (sharded, zstd+shuffle); commit an **Icechunk snapshot** (pin for reproducible training).
4. Build a **Parquet catalog** of frame timestamps + VFI training **triplets** (t−Δ, t, t+Δ).

### 8.3 Access path during TRAINING (O(1) per sample)
- Sample index → triplet → `cube[t:t+w, y0:y0+512, x0:x0+512]` = **one chunk** → **tensorstore** async read (or **kvikIO GDS** straight to GPU) → **DALI** GPU augment → model. If dataloader-bound, swap in a precomputed **FFCV `.beton`** (or **LMDB** for keyed random access) of triplet patches — each sample = one memory-mapped/large-chunk read ≈ **O(1)**.

### 8.4 Access path during SERVING (the O(1) part)
- After batch inference, **render once**: interpolated frames → **XYZ tiles (WebP)** + **MP4/HLS clips**, named with **content hashes / fixed `(t,z,x,y)` keys**, uploaded to **S3 + CDN** with **`Cache-Control: immutable`**.
- Dashboard request `(z,x,y,t)` or clip id → **deterministic key** → **CDN edge cache hit** → constant-time bytes. No model, no GDAL, no DB scan on the request path. **Redis** caches the small JSON catalog (O(1) keyed gets). TiTiler is a dynamic *fallback* for exploring source COGs only.

### 8.5 Orchestration
- **Prefect** (light) or **Dagster** (asset lineage). Batch-backfill for training; **micro-batch append** (Icechunk `append_dim`) + render trigger for live INSAT frames.

### 8.6 Text architecture diagram
```
                         ┌──────────────────────────── INGEST (zero-copy, O(1) reads) ───────────────────────────┐
  NOAA GOES-19 (.nc, S3)─┤  VirtualiZarr ── ManifestArrays (byte-range refs) ── Icechunk repo (snapshots, append) │
  MOSDAC INSAT-3DS (.h5)─┤                  (kerchunk parquet refs as portable alt)                               │
  Himawari (.nc) ────────┘                                                                                        │
                                                   │ (lazy chunk reads)                                           │
                                                   ▼                                                              │
            ┌──────────────── PREPROCESS ONCE (Dask over chunks) ──────────────────────────────────────────────┘
            │  radiance→BT (file coeffs)  →  resample to common lat/lon  →  normalize  →  NaN-mask
            │  RESAMPLE KERNEL (pyresample/pykdtree KDTree) BUILT ONCE, CACHED, reused every frame  (fast)
            ▼
   ┌─────────────────────────── ANALYSIS CUBE ───────────────────────────┐
   │  Zarr v3 (time,y,x)  chunks=(8,512,512) shards=(32,1024,1024)        │   array[t,y,x] -> chunk coord (arithmetic)
   │  Blosc(zstd, shuffle)  fill=NaN   +  Parquet frame/triplet catalog   │   -> exactly one shard/chunk object = O(1)
   └───────────────┬───────────────────────────────────┬─────────────────┘
                   │ O(1) sample reads                  │ batch inference (VFI: RIFE/FILM-style)
   tensorstore / kvikIO(GDS)+nvCOMP / DALI ──► TRAIN    └──► interpolated frames
   (optional FFCV .beton / LMDB for max throughput)                       │ render ONCE
                                                                          ▼
                          ┌──────────────── SERVE (precomputed, O(1)) ───────────────────────┐
                          │  XYZ tiles (WebP)  +  MP4/HLS clips  +  per-frame COG (fallback)  │
                          │  content-addressed keys, Cache-Control: immutable                │
                          │  S3 ──► CDN edge (constant-time GET)  + Redis(catalog) + TiTiler  │
                          └───────────────────────────────┬──────────────────────────────────┘
                                                           ▼
                                           Web dashboard: GET (z,x,y,t)/clip-id -> edge hit -> O(1)
```

### 8.7 Where it is genuinely O(1) vs not (be precise)
- **O(1):** Zarr/virtual chunk lookup (index arithmetic → 1 object/range GET); training sample read (1 chunk / 1 mmap offset); tile/clip serving (deterministic key → CDN edge hit); Redis catalog gets; resample *application* setup (kernel preloaded, not rebuilt).
- **NOT O(1) (so do offline, once):** warp/regrid compute, radiance→BT, normalization stats, VFI inference, tile/clip rendering, KDTree *construction*. Amortize all of these into the persisted artifacts above.

---

## Sources
Cloud-optimized arrays / Zarr / virtual:
- Zarr cloud-native guide — https://guide.cloudnativegeo.org/zarr/intro.html
- Earthmover "What is Zarr" — https://www.earthmover.io/blog/what-is-zarr/
- Zarr v3 core spec — https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
- Zarr-Python 3 release — https://zarr.dev/blog/zarr-python-3-release/
- Zarr sharding-indexed codec spec — https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
- ZEP 2 (sharding) — https://zarr.dev/zeps/accepted/ZEP0002.html
- Zarr performance guide — https://zarr.readthedocs.io/en/latest/user-guide/performance/
- Zarr vs Tiff perf paper — https://arxiv.org/pdf/2411.11291
- kerchunk docs — https://fsspec.github.io/kerchunk/
- kerchunk repo — https://github.com/fsspec/kerchunk
- Pangeo kerchunk tutorial (Marsh) — https://medium.com/pangeo/accessing-netcdf-and-grib-file-collections-as-cloud-native-virtual-datasets-using-kerchunk-625a2d0a9191
- Pangeo HYCOM kerchunk (Signell) — https://medium.com/pangeo/using-kerchunk-with-uncompressed-netcdf-64-bit-offset-files-cloud-optimized-access-to-hycom-ocean-9008ba6d0d67
- VirtualiZarr docs — https://virtualizarr.readthedocs.io/en/latest/index.html
- VirtualiZarr usage — https://virtualizarr.readthedocs.io/en/latest/usage.html
- VEDA CMIP6 virtual zarr — https://docs.openveda.cloud/user-guide/notebooks/veda-operations/generate-cmip6-virtual-zarr-historical.html
- Icechunk 1.0 — https://www.earthmover.io/blog/icechunk-1-0-production-grade-cloud-native-array-storage-is-here/
- Icechunk repo — https://github.com/earth-mover/icechunk
- Icechunk virtual datasets — https://icechunk.io/en/latest/virtual/
- Pangeo Icechunk announcement — https://discourse.pangeo.io/t/icechunk-a-new-cloud-native-transactional-storage-engine-for-zarr/4610
- Earthmover I/O-maxing tensors — https://www.earthmover.io/blog/i-o-maxing-tensors-in-the-cloud/

COG / tiling / compression:
- COG in-depth — https://cogeo.org/in-depth.html
- OGC COG standard — https://docs.ogc.org/is/21-026/21-026.html
- GDAL COG driver — https://gdal.org/en/stable/drivers/raster/cog.html
- Kyle Barron COG mosaic — https://kylebarron.dev/blog/cog-mosaic/overview/
- Blosc — https://blosc.org/pages/ ; https://blosc.org/
- LZ4 — https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)
- FCBench floating-point compression — https://arxiv.org/pdf/2312.10301
- Holotomography OME-Zarr compression — https://arxiv.org/pdf/2503.18037

Fast I/O / GPU:
- fsspec features (caching) — https://filesystem-spec.readthedocs.io/en/latest/features.html
- Anaconda fsspec remote caching — https://www.anaconda.com/blog/fsspec-remote-caching
- redis-fsspec-cache — https://github.com/mpiannucci/redis-fsspec-cache
- Dask array best practices — https://docs.dask.org/en/latest/array-best-practices.html
- Dask chunks — https://docs.dask.org/en/latest/array-chunks.html
- xarray+dask examples — https://examples.dask.org/xarray.html
- PO.DAAC dask tutorial — https://podaac.github.io/tutorials/notebooks/Advanced_cloud/basic_dask.html
- rechunker (Abernathey) — https://medium.com/pangeo/rechunker-the-missing-link-for-chunked-array-analytics-5b2359e9dc11
- kvikIO Zarr — https://docs.rapids.ai/api/kvikio/stable/zarr/ ; repo https://github.com/rapidsai/kvikio
- xarray + kvikIO — https://xarray.dev/blog/xarray-kvikio
- xarray GPU pipeline (DALI/nvcomp) — https://xarray.dev/blog/gpu-pipeline
- zarr-benchmark kvikIO/GDS — https://github.com/zarr-developers/zarr-benchmark/discussions/14
- FOSS4G 2025 GPU-native Zarr — https://talks.osgeo.org/foss4g-2025/talk/TSVGYJ/
- NVIDIA DALI blog — https://developer.nvidia.com/blog/fast-ai-data-preprocessing-with-nvidia-dali/ ; repo https://github.com/NVIDIA/DALI
- TensorStore — https://google.github.io/tensorstore/ ; https://research.google/blog/tensorstore-for-high-performance-scalable-array-storage/
- zarr-benchmark TensorStore — https://github.com/zarr-developers/zarr-benchmark/discussions/25

Reprojection / geostationary:
- satpy resampling — https://satpy.readthedocs.io/en/stable/resample.html
- pyresample README — https://github.com/pytroll/pyresample/blob/v1.20.0/README.md
- geo2grid ABI L2 reader — https://www.ssec.wisc.edu/software/geo2grid/readers/abi_l2_nc.html
- GOES on AWS registry — https://registry.opendata.aws/noaa-goes/
- GOES-R Beginner's Guide — https://www.goes-r.gov/downloads/resources/documents/Beginners_Guide_to_GOES-R_Series_Data.pdf
- MOSDAC INSAT-3D product format — https://www.mosdac.gov.in/docs/INSAT3D_Products.pdf
- MOSDAC INSAT-3D SST ATBD — https://www.mosdac.gov.in/look/DOCS/ATBD_INSAT-3D_SST_REV_V1.1.pdf
- EUMETSAT INSAT-3DS L1C — https://user.eumetsat.int/catalogue/EO:EUM:DAT:INSAT:INSAT3D-L1C

ML preprocessing / dataloaders:
- Nature 3D U-Net IR BT nowcasting (normalization) — https://www.nature.com/articles/s41598-025-34207-9
- FFCV paper — https://arxiv.org/pdf/2306.12517 ; benchmarks https://docs.ffcv.io/benchmarks.html ; repo https://github.com/libffcv/ffcv/
- FFCV CVPR supplemental (random access vs WebDataset) — https://openaccess.thecvf.com/content/CVPR2023/supplemental/Leclerc_FFCV_Accelerating_Training_CVPR_2023_supplemental.pdf
- Dataloader landscape — https://hackernoon.com/an-overview-of-the-data-loader-landscape-experimental-setup
- Deep Lake (lakehouse for DL) — https://arxiv.org/pdf/2209.10785

Serving / caching / orchestration:
- TiTiler repo — https://github.com/developmentseed/titiler
- TiTiler dynamic tiling — https://developmentseed.org/titiler/user_guide/dynamic_tiling/
- TiTiler explained — https://www.geowgs84.ai/post/titiler-explained-fast-cloud-native-raster-tile-generation-for-gis
- Cloudflare HLS — https://www.cloudflare.com/learning/video/what-is-http-live-streaming/
- Mux mp4→HLS — https://www.mux.com/articles/how-to-convert-mp4-to-hls-format-with-ffmpeg-a-step-by-step-guide
- KeyCDN Cache-Control immutable — https://www.keycdn.com/blog/cache-control-immutable
- MDN Cache-Control — https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cache-Control
- HTTP caching strategies — https://blog.wrujel.com/http-caching-cache-control-etags-cdn-strategies-cfce0a
- Orchestration showdown (Dagster/Prefect/Airflow) — https://www.zenml.io/blog/orchestration-showdown-dagster-vs-prefect-vs-airflow
- Best pipeline tools 2026 — https://getbruin.com/blog/best-data-pipeline-tools-2026/
- Python pipeline tools 2026 — https://ukdataservices.co.uk/blog/articles/python-data-pipeline-tools-2025

`[bg]` = January-2026 background knowledge (network sources not separately fetched for that specific point).

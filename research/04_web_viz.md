# 04 — Web Visualization: Fastest Stack for Ground-Truth vs AI-Interpolated Satellite Time-Lapse Dashboard

**Project:** ISRO BAH 2026 PS-12 — "Fill in the Frames Seamlessly" (AI/ML optical-flow temporal interpolation of INSAT-3DS/3DR satellite imagery).
**This document covers Step 2 of the deliverable:** the web dashboard showing *original vs interpolated satellite animations* (time-lapse for BOTH), a scrubbable time slider, and SSIM / MSE / PSNR / FSIM comparison plots.
**Judging criteria that this targets:** "Web GUI Design" and "Visual Quality of the INSAT-3DS interpolation images."
**Author goal:** FASTEST platform, O(1) tile/frame serving, impressive modern UI.
**Date:** 2026-06-20. Research method: 15 WebSearch queries + 6 WebFetch deep-dives (network was up; all claims sourced below).

---

## 0. TL;DR — The Recommended Stack (read this first)

> **One-line recommendation:** Build a **Vite + React (TypeScript)** single-page app rendering with **MapLibre GL JS + deck.gl** (`BitmapLayer` + `TileLayer`), serve all imagery as **raster PMTiles archives on Cloudflare R2 fronted by the Cloudflare CDN** (true O(1) HTTP range-request frame seek, zero egress fees), drive metric charts with **uPlot**, and deploy the static front-end to **Cloudflare Pages**. Use a **`@maplibre/maplibre-gl-compare` swipe slider** to put ground-truth and interpolated side-by-side, with one shared time-slider scrubbing both.

| Concern | Choice | Why (1-liner) |
|---|---|---|
| App framework | **Vite + React + TypeScript** | Fastest dev HMR + smallest config for a pure SPA dashboard; no SSR needed because all data is static artifacts. |
| Deployment | **Cloudflare Pages** (+ R2 + optional Worker) | Free CDN egress, sub-10 ms edge cache, range-request friendly, team already has CF tooling. |
| Map/render engine | **MapLibre GL JS** (basemap + geo plumbing) **+ deck.gl** (`BitmapLayer`/`TileLayer` GPU raster) | Open-source (no Mapbox token), WebGL2 today + WebGPU-ready, fastest for animated raster over a map. |
| Tile / frame serving | **Raster PMTiles on R2** (one `.pmtiles` per dataset) | Single-file archive, HTTP range requests = **O(1) frame/tile seek** from static storage, no tile server. |
| Animation strategy | **Pre-decoded frame textures swapped per timestep** (PMTiles raster), with **MP4/WebM hardware-decoded video via WebCodecs** as the "hero" fast-path for the headline loop | Scrubbable + smooth 60 fps; PMTiles gives random seek, video gives buttery playback. |
| Metric charts | **uPlot** | Smallest + fastest time-series plotter; 10% CPU / 12 MB RAM for 3,600 pts @ 60 fps vs Plotly's much higher cost. |
| Compare UX | **`@maplibre/maplibre-gl-compare`** (swipe) + custom timeline | Synced pan/zoom, draggable swipe handle = instant "wow." |

This is the **O(1) design**: every frame/tile is a fixed byte-range inside a single immutable file on a CDN edge. Seeking to time *t* is a constant-time `Range:` GET that is cache-hit after first view. No database, no dynamic tiler in the hot path, no per-request compute billing.

---

## 1. Map / Geo Rendering Engines — comparison

The dashboard's core rendering job is: **draw a sequence of georeferenced raster frames (TIR brightness-temperature imagery) over a basemap, animate them at 15-min cadence, and let the user scrub.** This is a *raster-animation-over-a-map* problem, not a vector-data or 3D-point-cloud problem.

### Candidates

| Engine | Renderer | Raster animation fit | Notes |
|---|---|---|---|
| **deck.gl** | WebGL2 today, **WebGPU-ready** (v9 / luma.gl v9) | ★★★★★ `BitmapLayer` (single georeferenced image, accepts `HTMLVideoElement` / `ImageBitmap` / texture) + `TileLayer` (XYZ pyramid). "Smooth visualization of large datasets at 60 FPS." | **Best for animated raster overlays.** Pairs with MapLibre as a base layer. Per-frame animation by updating layer props each frame. |
| **MapLibre GL JS** | WebGL2, **WebGPU API in progress** | ★★★★ Native `raster` source + `raster` layer; can host deck.gl as a custom layer; built-in compare/swipe plugins. | Open-source fork of Mapbox GL (post-license-change). "Remarkable speed rendering lightweight MVT data," best FCP/CLS for vector. Our recommended **base** + plumbing layer. |
| **Mapbox GL JS** | WebGL2 | ★★★★ Same lineage as MapLibre | **Avoid:** requires access token + commercial license; MapLibre is the free, equivalent drop-in. |
| **CesiumJS** | WebGL2 | ★★★ Great 3D globe / 3D Tiles / time-dynamic (`CZML`, `ImageryProvider` with clock) | Wins for **point clouds / 3D Tiles** (FOSS4G 2025: CesiumJS excels at 3D Tiles; deck.gl+MapLibre crushed it on point-cloud TBT: 3 ms vs 21,357 ms). Overkill + heavier for a flat 2-D raster time-lapse, but a **globe view is a legit "wow" secondary tab**. |
| **OpenLayers** | Canvas + WebGL | ★★★ Solid raster/`ImageStatic`/XYZ, has WebGL tile renderer | Very capable for raster, but UX polish + GPU animation ergonomics trail deck.gl. |
| **Leaflet** | DOM/Canvas (no WebGL core) | ★★ Tile swap works but not GPU-accelerated; janky for fast frame animation | **Avoid for animation** — fine for a static fallback only. |

### Verdict
**MapLibre GL JS (base + UI plumbing) + deck.gl (GPU raster animation).** This combo is the documented performance leader for "large-scale GPU rendering over a map" and is fully open-source (no token). deck.gl's `BitmapLayer` is purpose-built for georeferenced imagery and now even accepts a live `HTMLVideoElement`, which is the killer feature for hardware-decoded video time-lapse. Keep **CesiumJS as an optional 3D-globe tab** for extra judge dazzle, not as the primary engine.

Sources:
- deck.gl What's New / BitmapLayer / TileLayer: https://deck.gl/docs/whats-new , https://deck.gl/docs/api-reference/layers/bitmap-layer , https://deck.gl/docs/api-reference/geo-layers/tile-layer
- deck.gl v9 WebGPU-ready announcement (CARTO): https://carto.com/blog/announcing-deck-gl-v9-webgpu-ready-with-typescript-support/
- deck.gl WebGPU dev guide (still WIP, not production): https://deck.gl/docs/developer-guide/webgpu
- MapLibre GL JS: https://maplibre.org/projects/gl-js/ ; WebGPU progress: https://maplibre.org/news/2025-10-04-maplibre-newsletter-september-2025/
- FOSS4G 2025 perf study (CesiumJS vs MapLibre+deck.gl): https://talks.osgeo.org/foss4g-2025/talk/9FVHFA/ , https://isprs-archives.copernicus.org/articles/XLVIII-4-W20-2025/83/2026/
- Mapping libraries practical comparison: https://giscarta.com/blog/mapping-libraries-a-practical-comparison

---

## 2. WebGPU vs WebGL, WebCodecs, Texture Streaming

### WebGPU vs WebGL (mid-2026 reality)
- **deck.gl v9 / luma.gl v9 are "WebGPU-ready"** — the core API is portable across WebGL2 and WebGPU — **but WebGPU support is still landing layer-by-layer and is NOT production-ready.** WebGL2 is feature-complete and well-tested.
- **MapLibre** has a WebGPU API "in progress" (and a WebGPU backend contributed to MapLibre Native), also not yet the default.
- **Decision: ship on WebGL2 now.** It is universally supported, fast enough for 60 fps raster animation, and the *same code* will ride the WebGPU upgrade for free when deck.gl flips the backend. Mention "WebGPU-ready architecture" in the pitch — it's true and impressive — but do not depend on it.

Sources: https://deck.gl/docs/developer-guide/webgpu , https://luma.gl/docs/whats-new , https://openjsf.org/blog/deckgl-v9

### WebCodecs — hardware video decode for the headline loop
- **`VideoDecoder` gives hardware-accelerated, per-frame decode with frame-accurate control.** Each decoded `VideoFrame` can be uploaded straight into a deck.gl `BitmapLayer` (it accepts WebGL2 texture sources). Frame must render within ~16 ms to feel responsive on scrub — exactly WebCodecs' niche.
- **Caveat (important for O(1) seek):** inter-frame (delta) frames depend on the previous keyframe, so seeking requires decoding from the nearest preceding **keyframe**. **Mitigation: encode the time-lapse with a very short GOP / all-intra (e.g., keyframe every frame, or every 2-4 frames).** Time-lapses are short (dozens-to-hundreds of frames) so all-intra is cheap and makes every frame an O(1) seek target.
- **Browser status (2026):** Chrome/Edge lead; Safari + Firefox made big progress in 2025. Provide a **PMTiles image-frame fallback** for any browser without solid WebCodecs (covered below) — that fallback is itself O(1).

Sources: https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API , https://developer.chrome.com/docs/web-platform/best-practices/webcodecs , https://www.w3.org/TR/webcodecs/ , https://vidstudio.app/blog/webcodecs-vs-ffmpeg-wasm

### Texture streaming / GPU memory
- **WebGL2 texture arrays (`sampler2DArray`)** store N frames as slices of one texture object → one bind, batched draw calls, far fewer texture switches → smooth playback. Ideal when the whole loop fits in VRAM.
- **Watch GPU memory:** images are stored *uncompressed* on the GPU, so a full-res 100-frame loop can blow VRAM. Strategy: (a) animate at a sensible display resolution, (b) for large loops prefer hardware **video decode** (one decoded frame in flight) over holding every frame as a texture, (c) use texture arrays only for the small "always-resident" hero clip.

Sources: https://medium.com/better-programming/how-to-use-texture-arrays-in-webgl-921dff1c22d8 , https://www.oreilly.com/library/view/webgl-up-and/9781449326487/ch04.html

---

## 3. Animation / Streaming Approaches — tradeoffs

| Approach | Smooth playback | O(1) scrub/seek | VRAM | Build complexity | Verdict for us |
|---|---|---|---|---|---|
| **(a) Pre-rendered video (MP4 H.264 / WebM VP9/AV1) + WebCodecs/`<video>` HW decode** | ★★★★★ buttery 60 fps | ★★★★ if **short GOP / all-intra** (else decode-from-keyframe) | ★ (1 frame in flight) | medium | **Use as the headline "play" experience.** Encode all-intra for cheap exact seek. Range-served from R2. |
| **(b) Image tile sequences swapped per timestep (PMTiles raster pyramid)** | ★★★★ (preload neighbors) | ★★★★★ **true random-access O(1)** — every (z,x,y,frame) is a byte range | medium | low–medium | **Use as the scrub/precision path + universal fallback.** Best for the time-slider. |
| **(c) GPU texture array (all frames resident)** | ★★★★★ | ★★★★★ (just change slice index) | ★★ (limited # frames) | medium | **Use for the small hero loop / thumbnails** where frame count is bounded. |

### Recommended hybrid (the pragmatic winner)
1. **Scrubbing / precision compare → image frames from PMTiles** (random-access, deck.gl `BitmapLayer` per frame, preload ±2 neighbors). This is what makes the slider feel instant and gives perfect side-by-side parity between GT and interpolated.
2. **"Play" button → hardware-decoded video** (all-intra MP4/WebM) for the smoothest possible cinematic loop and the strongest visual-quality impression.
3. **Thumbnails / mini-loop → texture array.**

This way the *scrub* path is provably O(1) (range GET per frame) and the *play* path is GPU-smooth. Both come off the same CDN.

**Accessibility:** any `<video>` element on the play path must ship a captions track:
```html
<video src="clip.mp4" controls>
  <track kind="captions" srclang="en" label="English" src="captions.vtt" default />
</video>
<!-- NOTE: every <video> ships a captions <track> for WCAG 1.2.2. -->
```

deck.gl animation pattern (authoritative): "the most powerful way to create animations is to manage data/settings externally and update the layers' props on every frame… deck.gl is designed to handle layer updates very efficiently at high frame rate." Keep stable layer `id`s; don't recreate layers. (See §8 code.)

Sources: https://deck.gl/docs/developer-guide/animations-and-transitions , https://deck.gl/docs/api-reference/layers/bitmap-layer , WebCodecs sources above.

---

## 4. Tile Serving — the O(1) decision

### Options

| Option | O(1)? | Hot-path compute | Egress cost | Fit |
|---|---|---|---|---|
| **PMTiles (single-file archive, HTTP range requests) on R2 + CDN** | ✅ **Yes** — each tile/frame is a fixed byte range; client requests only the bytes it needs (compact Hilbert-curve layout); cache-hit after first view | **None** (static file) | **Free egress on R2** (per-request only) | ★★★★★ **Recommended.** |
| Static XYZ tile pyramid (millions of small files) on CDN | ✅ Yes (1 file = 1 tile) | None | CDN-dependent | ★★★★ Works, but millions of tiny files are painful to deploy/manage vs one `.pmtiles`. |
| **TiTiler** (dynamic COG tiling, FastAPI) | ❌ **No** — generates tiles on the fly; ≥2 GET to the COG per tile; "will never be faster than serving local tiles" | **Yes** (gdal/rasterio per request) | server + egress | ★★ Great for exploratory/ad-hoc, **wrong for a judged demo's hot path.** Optional: use it *offline* to pre-bake, or behind a cache. |
| Vector tiles / Protomaps basemap | ✅ Yes (also PMTiles) | None | Free on R2 | ★★★★ **Use for the dark basemap** (Protomaps basemap as PMTiles). |

### Verdict: **Raster PMTiles on Cloudflare R2, fronted by the Cloudflare CDN.**
- **Why O(1):** "MapLibre requests byte ranges over HTTP from a single hosted file, requiring no tile server or dynamic backend. The format stores tiles in a compact layout so the client can request only the byte ranges it needs." Seek = one `Range:` GET = constant time, edge-cached.
- **PMTiles is NOT vector-only.** It is "a general format for tiled data addressed by Z/X/Y… cartographic basemap vector tiles, remote sensing observations, JPEG images, or more." Confirmed raster path:
  - **Build:** `rio-pmtiles` (Rasterio plugin) exports a GeoTIFF/COG → PMTiles v3, with auto-reprojection + concurrent tile generation (good up to ~1 GB sources). For our INSAT/GOES frames, produce **one PMTiles per frame** *or* pack a time dimension via per-frame layers/archives.
  - **Serve:** MapLibre `type: 'raster'` source with a `pmtiles://` URL (see §8). deck.gl `TileLayer` can also read the same XYZ endpoint the PMTiles protocol exposes.
- **R2 specifics:** recommended PMTiles store; **no bandwidth fees, per-request GET billing only**; needs CORS allowing `range`/`if-match` headers + `etag` exposed (config in §8). Note R2 origin latency can be ~500 ms cold — **the CDN cache in front is what delivers the O(1)/sub-10 ms hot path**, so set long-lived immutable `Cache-Control` and (optionally) a tiny Worker to normalize range caching.
- **Optional Worker pattern:** a Cloudflare Worker can parse `z/x/y`, fetch the byte range from R2, cache it, and return it — useful to guarantee edge cache + add CORS, but for a demo the static R2-public-bucket + CDN is enough.

Sources:
- PMTiles concepts / general (incl. raster, JPEG): https://docs.protomaps.com/pmtiles/ , https://github.com/protomaps/PMTiles
- Cloud storage / R2 + CORS + range: https://docs.protomaps.com/pmtiles/cloud-storage
- Cloudflare integration + Worker: https://docs.protomaps.com/deploy/cloudflare , https://github.com/thomasgauvin/protomaps-on-cloudflare
- R2 chunking/batch range strategies + latency: https://github.com/protomaps/PMTiles/discussions/465
- rio-pmtiles (raster → PMTiles): https://pypi.org/project/rio-pmtiles/ , https://gist.github.com/JesseCrocker/4fee23a262cdd454d14e95f2fb25137f
- MapLibre PMTiles protocol patterns (raster/vector/raster-dem): https://github.com/maplibre/maplibre-agent-skills/blob/main/skills/maplibre-pmtiles-patterns/SKILL.md
- TiTiler dynamic tiling tradeoffs: https://developmentseed.org/titiler/user_guide/dynamic_tiling/ , https://github.com/developmentseed/titiler/discussions/377

---

## 5. App Framework & Deployment — fastest platform

### Framework
| Option | Verdict |
|---|---|
| **Vite + React + TS** | ✅ **Recommended.** This dashboard is a pure client-side SPA over *static* artifacts (PMTiles, video, JSON metrics). No SSR/RSC value, no DB. Vite = fastest HMR, smallest config, instant builds; React = huge ecosystem (deck.gl `@deck.gl/react`, react-map-gl/MapLibre, uPlot wrappers, shadcn/ui). |
| Next.js (App Router/RSC) | Overkill here — RSC/SSR help data-fetching pages, but our data is static files served by CDN. Adds 200–300 KB baseline JS + server complexity for no benefit. (If the team already standardizes on Next + Vercel, a static export works too — see deployment note.) |
| SvelteKit | Smallest bundles (15–100 KB vs Next 200–300 KB) and great perf, but **React's geospatial ecosystem (deck.gl/react-map-gl) is richer** and we want velocity for the demo. Svelte is a fine alt if the team prefers it. |

> Build-tool note: Create React App was deprecated (Feb 2025); the ecosystem standardized on **Vite**. Use Vite.

### Deployment — Cloudflare vs Vercel (the team has both)
| | **Cloudflare Pages (+R2 +Workers/KV)** | Vercel |
|---|---|---|
| Static asset edge cache | ✅ served from cache w/o per-request compute; static asset requests **not billed** | ✅ fast edge CDN |
| Egress / bandwidth | ✅ **no egress fees on R2**; cache-hit delivery effectively ~$0.002/GB; KV hot reads 0.5–10 ms | ❌ ~$0.15/GB overage (reports of ~$550/TB); not ideal for big tile/video payloads |
| Range-request / large binary tiles & video | ✅ ideal for PMTiles range GETs + MP4; R2 = co-located origin, free egress | ⚠️ fine functionally, but bandwidth-metered; CF self-serve also limits heavy non-HTML on the base CDN — that's exactly why **R2** is the right home for our heavy assets |
| Edge KV for O(1) metadata | ✅ Workers KV (0.5–10 ms hot, tiered cache, 3× faster after 2025 rework) | KV/Edge Config exists |

**Verdict:** **Deploy the front-end to Cloudflare Pages; put PMTiles + video on R2; front them with the CDN; optional Worker for range/CORS normalization; optional KV for the manifest/index.** This is the cheapest *and* fastest for our heavy, cacheable, range-served payloads, and the team already has CF tooling. (Vercel remains a perfectly good host for the *app shell* if preferred, but **keep the heavy tiles/video on R2** to avoid Vercel bandwidth costs.)

Sources:
- Vite-standardization / CRA deprecation + framework bundle sizes: https://calmops.com/programming/javascript/javascript-framework-comparison/ , https://markaicode.com/vs/sveltekit-vs-nextjs/ , https://prismic.io/blog/sveltekit-vs-nextjs
- CF Pages vs Workers static serving: https://www.justaftermidnight247.com/insights/cloudflare-pages-vs-workers-which-one-should-you-use/ , https://architectingoncloudflare.com/chapter-04/
- R2 no-egress / KV latency / 2025 KV rework: https://medium.com/@kaushalsinh73/7-r2-kv-cache-plays-for-edge-native-speed-603c2c6b9927 , https://developers.cloudflare.com/kv/concepts/how-kv-works/ , https://blog.cloudflare.com/rearchitecting-workers-kv-for-redundancy/
- CF vs Vercel bandwidth pricing: https://blog.blazingcdn.com/en-us/cdn-cloudflare-pricing-2025-guide-plans-hidden-fees-roi , https://flexprice.io/blog/vercel-pricing-breakdown , https://www.rodyvansambeek.com/blog/optimizing-costs-and-performance-of-vercel-edge-request-pricing

---

## 6. Charts / Plots for Metrics (SSIM / MSE / PSNR / FSIM)

We need time-series of each metric per frame, ideally with a cursor that **syncs to the map's current frame** (move the slider → marker moves on the chart, and vice-versa).

| Lib | Speed | Size | Verdict |
|---|---|---|---|
| **uPlot** | ★★★★★ 3,600 pts @60 fps = **10% CPU / 12.3 MB RAM**; cold-start 166,650 pts in 25 ms; ~100k pts/ms after | **~50 KB**, no WebGL/WASM | ✅ **Recommended primary.** Perfect for per-frame metric lines + a synced cursor; trivially handles all our metric series at 60 fps. |
| ECharts | ★★★★ strong canvas perf | larger | ✅ Good alt if you want polished built-in interactions / heatmaps out of the box (e.g., per-pixel error heatmap). |
| Plotly.js | ★★★ feature-rich but heavy/slow vs uPlot | large | Use only if you want quick scientific niceties; not for the hot path. |
| Recharts / visx | ★★ (SVG/React) | medium | Fine for small static summary cards, not for dense/animated series. |
| Observable Plot | ★★★ concise grammar | medium | Nice for quick exploratory static charts. |

**Recommendation:** **uPlot** for the live metric strip (SSIM/PSNR/MSE/FSIM lines with a frame cursor synced to the slider). Add **ECharts** *only if* you want a fancy **per-pixel error heatmap** overlay or radial gauges for extra polish. For per-frame error *maps* (spatial heatmap of |GT−interp|), render them as just another deck.gl `BitmapLayer` toggle — no chart lib needed.

Sources: https://github.com/leeoniya/uPlot , https://leeoniya.github.io/uPlot/ , https://cprimozic.net/notes/posts/my-thoughts-on-the-uplot-charting-library/ , SciChart bench: https://www.scichart.com/blog/chart-bench-compare-javascript-chart-libraries/

---

## 7. UX Patterns for a Judge-Impressing "Wow" GUI

Target aesthetic: **satellite-ops / mission-control dark theme** (à la Windy / Zoom Earth / NASA Worldview), responsive, dense but clean.

**Must-have interactions**
1. **Side-by-side synced compare** — `@maplibre/maplibre-gl-compare` swipe handle (drag to wipe GT ↔ interpolated) **with synced pan/zoom/rotate** (`mapbox-gl-sync-move` under the hood). Offer a toggle between **swipe** and **true side-by-side** (two map panes) — `maplibre-gl-compare-plus` supports both modes.
2. **Unified timeline** — one play/pause/scrub control driving *both* panes and the metric cursor. Frame-step buttons (⏮⏭), speed control (0.5×–4×), loop toggle. Show timestamp + "interpolated" badge on synthetic frames.
3. **Region selector** — draw/box or preset AOIs (cyclone, fire, flood case studies). Use deck.gl `EditableGeoJsonLayer` or a simple bbox picker; switching region = swap PMTiles URL (still O(1)).
4. **Layer toggles** — Basemap / GT raster / Interpolated raster / **Error heatmap** (|GT−interp| as a colored BitmapLayer) / Optical-flow vectors (deck.gl `LineLayer`/`TripsLayer` for motion vectors = strong visual storytelling of the optical-flow model).
5. **Metric HUD** — uPlot strip + live numeric readout of SSIM/PSNR/MSE/FSIM for the current frame; small "GT vs Interp" delta badges.
6. **3D globe "wow" tab (optional)** — CesiumJS or deck.gl `GlobeView` showing the INSAT disk with the time-lapse on the sphere.

**Visual polish**
- Dark palette (#0a0e14 bg, cyan/amber accents), glassy panels, subtle grid, monospace for telemetry numbers, smooth transitions. Use **shadcn/ui + Tailwind** for fast, modern components; **Framer Motion** for panel/slider micro-animations.
- Color-ramp the TIR brightness temperature with a perceptually-uniform scientific colormap (e.g., turbo/IR) applied in the BitmapLayer (or bake into tiles); show a legend/colorbar.
- Responsive: collapse to stacked panes + bottom-sheet controls on narrow screens.

Sources: https://github.com/maplibre/maplibre-gl-compare , https://libraries.io/npm/maplibre-gl-compare-plus , https://github.com/opengeos/maplibre-gl-swipe , dark weather-dashboard UI refs: https://www.devoq.io/modern-weather-website-ui-ux-design-guide-for-real-time-forecast-platforms/

---

## 8. Concrete Component Architecture + Code Snippets

### 8.1 Architecture (diagram-as-text)

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                         BUILD-TIME (offline, Python)                            │
 │  INSAT-3DS/3DR (.nc/.h5)  ──►  AI interpolation model (RIFE/optical-flow)       │
 │      │ ground-truth frames        │ interpolated frames                         │
 │      ▼                            ▼                                             │
 │  colormap + georeference (rasterio)                                             │
 │      │                                                                          │
 │      ├─► rio-pmtiles  ──►  gt.pmtiles        (raster pyramid, 1 archive)        │
 │      ├─► rio-pmtiles  ──►  interp.pmtiles    (raster pyramid, 1 archive)        │
 │      ├─► rio-pmtiles  ──►  error.pmtiles     (|GT−interp| heatmap, optional)    │
 │      ├─► ffmpeg (all-intra)  ──►  gt.mp4 / interp.mp4  (HW-decode hero loops)   │
 │      └─► metrics.json   (per-frame SSIM/MSE/PSNR/FSIM + flow vectors)           │
 └───────────────────────────────────────┬──────────────────────────────────────┘
                                          │  upload (wrangler)
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │            CLOUDFLARE R2 (origin)  +  CLOUDFLARE CDN (edge cache)               │
 │   *.pmtiles  *.mp4  metrics.json   ── immutable, Cache-Control: max-age=1yr     │
 │   [optional Worker]: parse Range / add CORS / cache; [optional KV]: manifest    │
 │   ► O(1) frame seek = one HTTP Range GET, edge-cached after first hit           │
 └───────────────────────────────────────┬──────────────────────────────────────┘
                                          │  HTTPS range requests
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │     FRONT-END  (Vite + React + TS)  served from CLOUDFLARE PAGES                │
 │                                                                                │
 │  <AppShell>  (shadcn/ui + Tailwind, dark satellite-ops theme)                   │
 │   ├─ <CompareDeck>                                                              │
 │   │    ├─ MapLibre base (Protomaps dark basemap, PMTiles)                       │
 │   │    ├─ deck.gl BitmapLayer  ◄── current GT frame      (left of swipe)        │
 │   │    ├─ deck.gl BitmapLayer  ◄── current Interp frame  (right of swipe)       │
 │   │    ├─ deck.gl BitmapLayer  ◄── error heatmap (toggle)                       │
 │   │    ├─ deck.gl LineLayer    ◄── optical-flow vectors (toggle)                │
 │   │    └─ maplibre-gl-compare swipe handle (synced pan/zoom)                    │
 │   ├─ <Timeline>  play/pause/scrub/step/speed/loop  ── drives `currentFrame`     │
 │   ├─ <MetricPanel> uPlot SSIM/PSNR/MSE/FSIM + cursor synced to currentFrame     │
 │   ├─ <LayerToggles> / <RegionSelector> / <ColorbarLegend>                       │
 │   └─ (optional) <GlobeTab> CesiumJS / deck.gl GlobeView                          │
 │                                                                                │
 │  STATE:  currentFrame, isPlaying, region, activeLayers  (Zustand)               │
 │  PLAY path → WebCodecs VideoDecoder → VideoFrame → BitmapLayer (60 fps)          │
 │  SCRUB path → PMTiles raster frame (range GET) → BitmapLayer (O(1) seek)         │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 PMTiles + MapLibre (raster) setup

```ts
import maplibregl from 'maplibre-gl';
import * as pmtiles from 'pmtiles';
import 'maplibre-gl/dist/maplibre-gl.css';

// Register the pmtiles:// protocol ONCE before creating maps
const protocol = new pmtiles.Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({
  container: 'gt-map',
  style: {
    version: 8,
    sources: {
      // Dark vector basemap (Protomaps PMTiles on R2)
      basemap: { type: 'vector', url: 'pmtiles://https://cdn.example.com/basemap.pmtiles' },
      // Ground-truth raster frame archive (one frame per timestep)
      gt: { type: 'raster', url: 'pmtiles://https://cdn.example.com/gt.pmtiles', tileSize: 256 }
    },
    layers: [
      // ...dark basemap layers...
      { id: 'gt-raster', type: 'raster', source: 'gt', paint: { 'raster-opacity': 1 } }
    ]
  },
  center: [80, 20], zoom: 4
});

// Teardown: maplibregl.removeProtocol('pmtiles');
```
*Source: https://github.com/maplibre/maplibre-agent-skills/blob/main/skills/maplibre-pmtiles-patterns/SKILL.md*

### 8.3 R2 CORS for range requests (build-time)

```json
// cors_rules.json
{
  "allowed": {
    "origins": ["https://your-dashboard.pages.dev"],
    "methods": ["GET", "HEAD"],
    "headers": ["range", "if-match"]
  },
  "exposeHeaders": ["etag"],
  "maxAgeSeconds": 3000
}
```
```bash
wrangler r2 bucket cors set MY_BUCKET --file cors_rules.json
# Set immutable caching on upload so the CDN delivers O(1) hot hits:
#   Cache-Control: public, max-age=31536000, immutable
```
*Source: https://docs.protomaps.com/pmtiles/cloud-storage*

### 8.4 deck.gl per-frame raster animation (scrub path, O(1))

```tsx
import DeckGL from '@deck.gl/react';
import { BitmapLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';

// currentFrame comes from <Timeline> state. Frame URLs are static (R2/CDN).
function CompareDeck({ currentFrame, bounds }: { currentFrame: number; bounds: [number,number,number,number] }) {
  const gtUrl     = `https://cdn.example.com/frames/gt/${currentFrame}/{z}/{x}/{y}.png`;
  const interpUrl = `https://cdn.example.com/frames/interp/${currentFrame}/{z}/{x}/{y}.png`;

  const layers = [
    new TileLayer({
      id: 'gt-tiles',            // STABLE id — never recreate the layer identity
      data: gtUrl,
      minZoom: 0, maxZoom: 8, tileSize: 256,
      renderSubLayers: props => {
        const { boundingBox } = props.tile;
        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [boundingBox[0][0], boundingBox[0][1], boundingBox[1][0], boundingBox[1][1]]
        });
      }
    }),
    new TileLayer({ id: 'interp-tiles', data: interpUrl, /* ...same... */ })
  ];

  return <DeckGL layers={layers} initialViewState={{ longitude: 80, latitude: 20, zoom: 4 }} controller />;
}
```
*Pattern source: https://deck.gl/docs/developer-guide/animations-and-transitions , https://deck.gl/docs/api-reference/geo-layers/tile-layer , https://deck.gl/docs/api-reference/layers/bitmap-layer*

### 8.5 deck.gl + WebCodecs hardware-decoded "play" path (single georeferenced frame)

```ts
// All-intra MP4 => every frame is a keyframe => O(1) seek.
const decoder = new VideoDecoder({
  output: (frame: VideoFrame) => {
    // Upload the decoded frame straight into a BitmapLayer (accepts WebGL2 texture sources)
    deck.setProps({
      layers: [ new BitmapLayer({
        id: 'gt-video',                 // stable id
        image: frame,                   // HTMLVideoElement / VideoFrame / ImageBitmap all accepted
        bounds: [westLon, southLat, eastLon, northLat]
      }) ]
    });
    frame.close();
  },
  error: console.error
});
decoder.configure({ codec: 'avc1.640028' /* H.264 High */ });
// feed EncodedVideoChunk(s) parsed from the mp4 (e.g., via mp4box.js) on play/scrub.
```
*Sources: BitmapLayer accepts HTMLVideoElement/texture sources — https://deck.gl/docs/api-reference/layers/bitmap-layer ; WebCodecs — https://developer.chrome.com/docs/web-platform/best-practices/webcodecs*

### 8.6 MapLibre compare (swipe) for synced GT vs interpolated

```ts
import Compare from '@maplibre/maplibre-gl-compare';
import '@maplibre/maplibre-gl-compare/dist/maplibre-gl-compare.css';

// two MapLibre maps (left = GT, right = interpolated), same center/zoom
const cmp = new Compare(mapLeft, mapRight, '#compare-container', {
  mousemove: false,   // drag the handle (true = follow cursor)
  orientation: 'vertical'
});
// Pan/zoom/rotate stay synced automatically. Drive both raster sources from one currentFrame.
```
*Source: https://github.com/maplibre/maplibre-gl-compare , https://www.npmjs.com/package/@maplibre/maplibre-gl-compare*

### 8.7 uPlot metric strip with cursor synced to the timeline

```ts
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

// data: [frameIdx[], ssim[], psnr[], mse[], fsim[]]
const opts: uPlot.Options = {
  width: 900, height: 180,
  series: [
    {},
    { label: 'SSIM', stroke: '#22d3ee' },
    { label: 'PSNR', stroke: '#f59e0b' },
    { label: 'MSE',  stroke: '#ef4444' },
    { label: 'FSIM', stroke: '#a78bfa' }
  ],
  cursor: { sync: { key: 'frameSync' } }   // sync cursor across charts / to the slider
};
const u = new uPlot(opts, data, document.getElementById('metric-strip')!);

// When the slider moves, move the uPlot cursor to that frame:
function onFrame(currentFrame: number) {
  u.setCursor({ left: u.valToPos(currentFrame, 'x'), top: 0 });
}
```
*Source: https://github.com/leeoniya/uPlot , https://leeoniya.github.io/uPlot/*

### 8.8 Recommended package set
```
react react-dom typescript vite
maplibre-gl pmtiles @maplibre/maplibre-gl-compare
@deck.gl/core @deck.gl/react @deck.gl/layers @deck.gl/geo-layers
uplot
zustand framer-motion tailwindcss            # state + polish + styling
mp4box                                        # demux mp4 for WebCodecs (play path)
# build-time (python): rio-pmtiles rasterio  ; ffmpeg for all-intra mp4/webm
# deploy: wrangler (Cloudflare Pages + R2)
```

---

## 9. Why this is the FASTEST + O(1) (the pitch line for judges)

- **O(1) frame/tile seek:** every frame and tile is a deterministic byte range inside a single immutable PMTiles file on an edge CDN. Seeking to any time *t* is one constant-time `Range:` GET, cache-hit after first view — **no tile server, no database, no per-request compute** in the hot path.
- **GPU-accelerated 60 fps rendering:** deck.gl on WebGL2 (WebGPU-ready) animates georeferenced rasters by swapping textures / decoded video frames; uPlot updates metric series at 60 fps on ~10% CPU.
- **Zero-egress, sub-10 ms edge:** Cloudflare R2 (no bandwidth fees) + CDN cache + optional KV manifest gives the cheapest and lowest-latency delivery of heavy tiles/video.
- **Hardware video decode for the hero loop:** WebCodecs + all-intra MP4/WebM = cinematic, smooth, *and* exact-seek.
- **Wow UX:** synced swipe compare, unified scrubbable timeline, optical-flow vector overlay, error-heatmap toggle, dark mission-control theme, optional 3D globe.

---

## 10. Risks / gotchas

- **WebGPU not production-ready** in deck.gl/MapLibre yet → ship WebGL2; it's plenty. Advertise "WebGPU-ready," don't depend on it.
- **Video keyframe seeking** → encode **all-intra / very short GOP** so scrub stays O(1); otherwise seek decodes from the prior keyframe.
- **GPU VRAM** for many full-res frames → prefer video decode for long loops; texture arrays only for short hero loops.
- **R2 cold latency (~500 ms)** → rely on the CDN cache + immutable `Cache-Control`; optionally a Worker to normalize range caching and CORS.
- **R2 GET billing** → each range request is a billed GET, but cache hits are served from the edge (cheap) and egress is free; fine at demo scale.
- **rio-pmtiles size limit** (~1 GB/source) → split datasets / per-frame archives if larger; or pre-bake XYZ then pack.
- **CF base-CDN heavy-media policy** → that's precisely why heavy assets live on **R2** (designed for it), not the generic proxy CDN.

---

## 11. Full source list (URLs)

**Rendering engines / deck.gl / MapLibre / Cesium**
- https://deck.gl/docs/whats-new
- https://deck.gl/docs/api-reference/layers/bitmap-layer
- https://deck.gl/docs/api-reference/geo-layers/tile-layer
- https://deck.gl/docs/developer-guide/animations-and-transitions
- https://deck.gl/docs/developer-guide/webgpu
- https://carto.com/blog/announcing-deck-gl-v9-webgpu-ready-with-typescript-support/
- https://openjsf.org/blog/deckgl-v9
- https://luma.gl/docs/whats-new
- https://maplibre.org/projects/gl-js/
- https://maplibre.org/news/2025-10-04-maplibre-newsletter-september-2025/
- https://talks.osgeo.org/foss4g-2025/talk/9FVHFA/
- https://isprs-archives.copernicus.org/articles/XLVIII-4-W20-2025/83/2026/
- https://giscarta.com/blog/mapping-libraries-a-practical-comparison
- https://cybergarden.au/blog/7-powerful-open-source-webgl-data-visualization-tools-2025

**WebGPU / WebCodecs / texture streaming**
- https://www.w3.org/TR/webcodecs/
- https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API
- https://developer.chrome.com/docs/web-platform/best-practices/webcodecs
- https://vidstudio.app/blog/webcodecs-vs-ffmpeg-wasm
- https://lionkeng.medium.com/a-tutorial-webcodecs-video-scroll-synchronization-8b251e1a1708
- https://medium.com/better-programming/how-to-use-texture-arrays-in-webgl-921dff1c22d8
- https://www.oreilly.com/library/view/webgl-up-and/9781449326487/ch04.html

**Tiles / PMTiles / TiTiler**
- https://docs.protomaps.com/pmtiles/
- https://docs.protomaps.com/pmtiles/cloud-storage
- https://docs.protomaps.com/deploy/cloudflare
- https://github.com/protomaps/PMTiles
- https://github.com/protomaps/PMTiles/discussions/465
- https://github.com/thomasgauvin/protomaps-on-cloudflare
- https://thomasgauvin.com/writing/static-protomaps-on-cloudflare/
- https://github.com/maplibre/maplibre-agent-skills/blob/main/skills/maplibre-pmtiles-patterns/SKILL.md
- https://pypi.org/project/rio-pmtiles/
- https://gist.github.com/JesseCrocker/4fee23a262cdd454d14e95f2fb25137f
- https://developmentseed.org/titiler/user_guide/dynamic_tiling/
- https://github.com/developmentseed/titiler
- https://github.com/developmentseed/titiler/discussions/377
- https://www.geowgs84.ai/post/titiler-explained-fast-cloud-native-raster-tile-generation-for-gis

**Framework / deployment / Cloudflare vs Vercel**
- https://calmops.com/programming/javascript/javascript-framework-comparison/
- https://markaicode.com/vs/sveltekit-vs-nextjs/
- https://prismic.io/blog/sveltekit-vs-nextjs
- https://www.justaftermidnight247.com/insights/cloudflare-pages-vs-workers-which-one-should-you-use/
- https://architectingoncloudflare.com/chapter-04/
- https://medium.com/@kaushalsinh73/7-r2-kv-cache-plays-for-edge-native-speed-603c2c6b9927
- https://developers.cloudflare.com/kv/concepts/how-kv-works/
- https://blog.cloudflare.com/rearchitecting-workers-kv-for-redundancy/
- https://blog.blazingcdn.com/en-us/cdn-cloudflare-pricing-2025-guide-plans-hidden-fees-roi
- https://flexprice.io/blog/vercel-pricing-breakdown
- https://www.rodyvansambeek.com/blog/optimizing-costs-and-performance-of-vercel-edge-request-pricing

**Charts**
- https://github.com/leeoniya/uPlot
- https://leeoniya.github.io/uPlot/
- https://cprimozic.net/notes/posts/my-thoughts-on-the-uplot-charting-library/
- https://www.scichart.com/blog/chart-bench-compare-javascript-chart-libraries/

**Compare UX / UI**
- https://github.com/maplibre/maplibre-gl-compare
- https://www.npmjs.com/package/@maplibre/maplibre-gl-compare
- https://libraries.io/npm/maplibre-gl-compare-plus
- https://github.com/opengeos/maplibre-gl-swipe
- https://www.devoq.io/modern-weather-website-ui-ux-design-guide-for-real-time-forecast-platforms/

---
*End of 04_web_viz.md*

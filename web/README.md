# FrameFlow — Web Dashboard

**ISRO BAH 2026 · PS-12 — "Fill in the Frames Seamlessly"**
A premium, fast, "mission-control" dashboard that visualises **ground-truth vs
AI-interpolated** geostationary **Thermal-IR** satellite animations side-by-side,
with a scrubbable timeline, per-frame image-quality metrics, and an optical-flow
overlay.

> This is deliverable **D2** of the FrameFlow system (see `../ARCHITECTURE.md`
> §9 and `../research/04_web_viz.md`). It renders static, precomputed artifacts
> (the "O(1)" serving model) and ships with a self-contained mock dataset so it
> runs standalone with no backend.

---

## Quick start

```bash
cd web
npm install          # install dependencies
npm run mock         # generate the self-contained demo dataset (public/data/…)
npm run dev          # start the Vite dev server  → http://localhost:5173
```

For a production build / preview:

```bash
npm run build        # type-check (tsc -b) + bundle (vite build) → dist/
npm run preview      # serve the production build locally
```

> **`npm run mock` is required before first run** — it writes the manifests and
> frame images the app loads. If the data is missing, the app shows a friendly
> "Telemetry unavailable" state telling you to run it.

---

## Tech stack

| Concern        | Choice |
|----------------|--------|
| Framework      | **Vite + React 18 + TypeScript** |
| Map / render   | **MapLibre GL JS** (dark base) + **deck.gl v9** (`BitmapLayer` / `TileLayer`, GPU raster) |
| Compare UX     | **`@maplibre/maplibre-gl-compare`** swipe (synced pan/zoom) + a two-pane "split" mode |
| Charts         | **uPlot** (per-frame metric strip with a cursor synced to the timeline) |
| UI / styling   | **Tailwind CSS** + hand-rolled shadcn-style components |
| Animation      | **Framer Motion** (subtle panel/modal transitions) |
| Icons          | **lucide-react** |
| State          | tiny `useSyncExternalStore` store (Zustand-flavoured API), no extra dep |

No Mapbox token is required — the basemap uses free CARTO dark raster tiles with
an **offline inline fallback** if the CDN is blocked.

---

## Features

1. **Split compare view** — two synced map panes (Ground Truth · Interpolated)
   via maplibre-gl-compare swipe, both driven by the **same current frame**, with
   synced pan/zoom. Toggle between *swipe* and *side-by-side split*.
2. **Unified scrubbable timeline** — play/pause, step, speed (0.5–4×), loop,
   frame ticks marking **observed (solid)** vs **interpolated (hollow)** frames,
   a live UTC readout, and full **keyboard** support (←/→ step, Shift+←/→ jump 5,
   Home/End, Space play/pause, `l` loop).
3. **Metric strip (uPlot)** — PSNR / SSIM / MS-SSIM across interpolated frames
   with a cursor synced to the current frame, click-to-seek, a live numeric
   readout, summary numbers, and a **baseline comparison** (linear blend, TV-L1,
   persistence) with delta badges.
4. **Optical-flow overlay** — motion vectors from each frame's `flow_overlay`
   rendered with a deck.gl `LineLayer` + arrowhead `IconLayer`.
5. **Error-heatmap toggle** — a `|GT − interp|` difference layer (synthetic diff
   image in the mock) with an opacity slider.
6. **Scene selector** — dropdown across multiple scenes (mock ships a GOES-19
   Atlantic cyclone demo and a synthetic Bay-of-Bengal demo).
7. **Hero "play" mode** — an HTML5 `<video>` modal that plays `videos.*` MP4s
   (with a **captions track**) and degrades gracefully if the MP4 is absent.
8. **Mission-control header** — title, satellite/channel badges, temporal-
   resolution readout ("30 min → 7.5 min · 4× via optical-flow AI"), and an
   **about / metrics-methodology** slide-over.
9. **Responsive, polished dark theme** — loading + error/empty states, subtle
   Framer Motion transitions, an IR colorbar legend, and tabular telemetry.

---

## How the rendering works (Codex review P2)

A single raster source **cannot** be addressed by timestamp — **each frame has
its own source**. The timeline slider selects the **active frame index**, and
`src/lib/deckLayers.ts → buildRasterLayer()` (re)builds that frame's raster
layer on every change:

- If `frame.tiles_url_template` is present → a deck.gl **`TileLayer`** over that
  frame's XYZ pyramid (the production O(1) path). The layer `id` encodes the
  frame index, so each frame is a **distinct source**.
- Otherwise → a deck.gl **`BitmapLayer`** using `frame.image` over the manifest
  `bbox` (the **mock-data path**). The layer `id` also encodes the frame index,
  so the texture swaps as the slider scrubs.

The **left pane always shows real ground truth**: for an interpolated frame it
displays the lower-`bracket` observed frame; the right pane shows the active
(possibly interpolated) frame.

## Accessibility (AccessLint WCAG 1.2.2)

Every `<video>` element includes:

```html
<track kind="captions" srclang="en" label="English" src="…/captions.vtt" default>
```

`npm run mock` writes a `captions.vtt` per scene describing the animation. The
app also provides alt text on images, `aria-label`s on all controls, a
keyboard-operable timeline (`role="slider"` + arrow keys), `role="switch"`
toggles, focus-visible rings, and a high-contrast dark palette.

---

## The manifest contract

The app loads `public/data/<scene>/manifest.json`. The TypeScript types live in
`src/types/manifest.ts` and a validating loader in `src/lib/manifest.ts`. The
mock generator (`scripts/gen_mock.mjs`) and the real precompute pipeline
(`frameflow.serve.precompute`) both emit this exact schema. Key points:

- `frames[]` interleave `observed` and `interpolated` frames; interpolated
  frames carry `t ∈ (0,1)` and a `bracket` of the two observed frame indices.
- Each frame has `image` (+ `thumb`) and optional `tiles_url_template` /
  `pmtiles` (the mock sets these to `null` and uses `image`).
- `metrics.per_frame[]` holds PSNR/SSIM/MS-SSIM/FSIM/GMSD/BT-RMSE/… per
  interpolated frame; `metrics.summary` + `metrics.baselines` drive the
  comparison panel.

---

## Project structure

```
web/
├── index.html
├── package.json
├── vite.config.ts            # aliases @/* → src/*, manual chunks
├── tailwind.config.ts        # mission-control dark theme
├── tsconfig*.json
├── scripts/
│   ├── gen_mock.mjs          # `npm run mock` — generates the demo dataset
│   └── png.mjs               # dependency-free PNG encoder (no native canvas)
├── public/
│   ├── favicon.svg
│   └── data/                 # GENERATED by `npm run mock`
│       ├── scenes.json
│       └── <scene>/{manifest.json,frames/,thumbs/,diff/,flow/,captions.vtt,videos/}
└── src/
    ├── main.tsx · App.tsx · index.css
    ├── types/{manifest.ts, shims.d.ts}
    ├── lib/{manifest.ts, deckLayers.ts, colormap.ts, basemap.ts, utils.ts}
    ├── hooks/{useStore.ts, useManifest.ts, usePlayback.ts}
    └── components/
        ├── Header.tsx · SceneSelector.tsx
        ├── ComparePanes.tsx · MapPane.tsx · PaneLabel.tsx
        ├── Timeline.tsx · MetricStrip.tsx
        ├── LayerToggles.tsx · ColorbarLegend.tsx · SummaryPanel.tsx
        ├── VideoModal.tsx · AboutPanel.tsx · States.tsx
        └── ui/{Button,Badge,Panel,Switch,Select}.tsx
```

---

## Mock data

`npm run mock` renders, per scene, an enhanced-IR animation of a cold cloud blob
(a rotating cyclone + a growing/decaying convective cell) advecting over a warm
background, plus realistic synthetic metrics (PSNR ~38–46 dB, SSIM ~0.90–0.97,
sub-2 K BT-RMSE), a flow overlay for one frame, synthetic error/diff heatmaps,
and a `captions.vtt`. It has **no third-party dependencies** (a tiny built-in PNG
encoder is used instead of the native `canvas` package). Re-run it any time to
regenerate `public/data/`.

The mock does **not** produce binary MP4 video; the hero-loop modal degrades
gracefully. The real `frameflow.serve.video` step writes all-intra MP4/WebM into
each scene's `videos/` directory.

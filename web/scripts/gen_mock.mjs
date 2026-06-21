#!/usr/bin/env node
/**
 * FrameFlow mock-data generator.
 *
 * Produces a fully self-contained dataset so the dashboard runs standalone:
 *   web/public/data/scenes.json                  (scene index)
 *   web/public/data/<scene>/manifest.json        (the manifest contract)
 *   web/public/data/<scene>/frames/NNN.png       (full-frame IR previews)
 *   web/public/data/<scene>/thumbs/NNN.png       (small thumbnails)
 *   web/public/data/<scene>/diff/NNN.png         (synthetic error heatmaps)
 *   web/public/data/<scene>/flow/NNN.json        (optical-flow overlay, 1 frame)
 *   web/public/data/<scene>/captions.vtt         (WCAG 1.2.2 captions)
 *
 * Each frame shows a cold cloud blob advecting (and a smaller convective cell)
 * over a warm background, rendered with an enhanced-IR colormap. Interpolated
 * frames are inserted between observed frames at the configured factor.
 *
 * No third-party dependencies — uses a tiny built-in PNG encoder.
 *
 * Usage: node scripts/gen_mock.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodePNG } from './png.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT, 'public', 'data');

// ----------------------------------------------------------------------------
// IR colormap (mirrors src/lib/colormap.ts IR_STOPS). t in [0,1].
// ----------------------------------------------------------------------------
const IR_STOPS = [
  [0.0, [8, 12, 20]],
  [0.12, [20, 30, 55]],
  [0.28, [20, 70, 120]],
  [0.42, [16, 130, 150]],
  [0.55, [30, 175, 120]],
  [0.66, [150, 200, 70]],
  [0.76, [240, 200, 40]],
  [0.85, [240, 130, 30]],
  [0.92, [225, 60, 50]],
  [0.97, [200, 60, 160]],
  [1.0, [245, 240, 255]],
];
const INFERNO_STOPS = [
  [0.0, [4, 6, 18]],
  [0.2, [40, 11, 84]],
  [0.4, [101, 21, 110]],
  [0.6, [159, 42, 99]],
  [0.75, [212, 72, 66]],
  [0.88, [245, 125, 21]],
  [1.0, [252, 255, 164]],
];

function sample(stops, tRaw) {
  const t = Math.min(1, Math.max(0, tRaw));
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i];
    const [b, cb] = stops[i + 1];
    if (t >= a && t <= b) {
      const l = (t - a) / (b - a || 1);
      return [
        Math.round(ca[0] + (cb[0] - ca[0]) * l),
        Math.round(ca[1] + (cb[1] - ca[1]) * l),
        Math.round(ca[2] + (cb[2] - ca[2]) * l),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

// ----------------------------------------------------------------------------
// Small deterministic PRNG so output is stable run-to-run.
// ----------------------------------------------------------------------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ----------------------------------------------------------------------------
// Brightness-temperature field synthesis.
// Returns a normalized "coldness" value in [0,1] where 1 = coldest cloud top.
// ----------------------------------------------------------------------------
function btField(nx, ny, time, cfg) {
  // Persistent terrain-ish warm/cool variation (static), built once per cfg.
  // We approximate with a couple of low-frequency sinusoids.
  const field = new Float32Array(nx * ny);

  // Main cyclonic cold blob center advects across the frame; it also rotates.
  const cx = cfg.blob.x0 + cfg.blob.vx * time;
  const cy = cfg.blob.y0 + cfg.blob.vy * time;
  const rot = time * cfg.blob.spin;

  // A second smaller convective cell that grows then decays (non-linear).
  const c2x = cfg.cell.x0 + cfg.cell.vx * time;
  const c2y = cfg.cell.y0 + cfg.cell.vy * time;
  const growth = Math.sin(time * Math.PI) * 0.6 + 0.5; // peaks mid-sequence

  const rng = mulberry32(cfg.seed);
  // Pre-generate a little static texture noise grid (coarse) for realism.
  const NG = 24;
  const noise = new Float32Array(NG * NG);
  for (let i = 0; i < noise.length; i++) noise[i] = rng();

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const u = i / nx;
      const v = j / ny;

      // Background gradient: warmer toward equator (bottom), cooler poleward.
      let val = 0.12 + 0.18 * (1 - v) + 0.05 * Math.sin(u * Math.PI * 2 + cfg.seed);

      // Main spiral cold blob (cyclone): rotate coordinates around center.
      const dx0 = u - cx;
      const dy0 = v - cy;
      const dx = dx0 * Math.cos(rot) - dy0 * Math.sin(rot);
      const dy = dx0 * Math.sin(rot) + dy0 * Math.cos(rot);
      const r = Math.sqrt(dx * dx + dy * dy);
      // Gaussian core
      const core = Math.exp(-(r * r) / (2 * cfg.blob.sigma * cfg.blob.sigma));
      // Spiral bands
      const ang = Math.atan2(dy, dx);
      const band =
        0.5 +
        0.5 * Math.sin(ang * cfg.blob.arms + r * cfg.blob.tightness - time * 6);
      val += core * (0.55 + 0.35 * band);
      // Eye: warm hole at very center
      val -= Math.exp(-(r * r) / (2 * (cfg.blob.sigma * 0.22) ** 2)) * 0.5;

      // Secondary convective cell (sharp, growing/decaying)
      const r2 = Math.hypot(u - c2x, v - c2y);
      val += growth * 0.7 * Math.exp(-(r2 * r2) / (2 * cfg.cell.sigma ** 2));

      // Coarse texture noise (bilinear-ish sample)
      const ni = Math.min(NG - 1, Math.floor(u * NG));
      const nj = Math.min(NG - 1, Math.floor(v * NG));
      val += (noise[nj * NG + ni] - 0.5) * 0.06;

      field[j * nx + i] = Math.min(1, Math.max(0, val));
    }
  }
  return field;
}

function renderFrame(nx, ny, time, cfg, cmap) {
  const field = btField(nx, ny, time, cfg);
  const stops = cmap === 'inferno' ? INFERNO_STOPS : IR_STOPS;
  const rgba = new Uint8Array(nx * ny * 4);
  for (let p = 0; p < field.length; p++) {
    const [r, g, b] = sample(stops, field[p]);
    rgba[p * 4] = r;
    rgba[p * 4 + 1] = g;
    rgba[p * 4 + 2] = b;
    rgba[p * 4 + 3] = 255;
  }
  return { rgba, field };
}

/** Render a synthetic diff/error heatmap (inferno) from two fields. */
function renderDiff(nx, ny, fieldA, fieldB, scale) {
  const rgba = new Uint8Array(nx * ny * 4);
  for (let p = 0; p < fieldA.length; p++) {
    const d = Math.min(1, Math.abs(fieldA[p] - fieldB[p]) * scale);
    const [r, g, b] = sample(INFERNO_STOPS, d);
    rgba[p * 4] = r;
    rgba[p * 4 + 1] = g;
    rgba[p * 4 + 2] = b;
    // Transparent where error ~ 0 so it overlays nicely.
    rgba[p * 4 + 3] = Math.round(40 + 215 * d);
  }
  return rgba;
}

/** Downsample an RGBA buffer by integer factor (box filter). */
function downsample(nx, ny, rgba, factor) {
  const ox = Math.floor(nx / factor);
  const oy = Math.floor(ny / factor);
  const out = new Uint8Array(ox * oy * 4);
  for (let j = 0; j < oy; j++) {
    for (let i = 0; i < ox; i++) {
      let r = 0,
        g = 0,
        b = 0,
        a = 0;
      for (let dj = 0; dj < factor; dj++) {
        for (let di = 0; di < factor; di++) {
          const sp = ((j * factor + dj) * nx + (i * factor + di)) * 4;
          r += rgba[sp];
          g += rgba[sp + 1];
          b += rgba[sp + 2];
          a += rgba[sp + 3];
        }
      }
      const n = factor * factor;
      const op = (j * ox + i) * 4;
      out[op] = Math.round(r / n);
      out[op + 1] = Math.round(g / n);
      out[op + 2] = Math.round(b / n);
      out[op + 3] = Math.round(a / n);
    }
  }
  return { ox, oy, out };
}

// ----------------------------------------------------------------------------
// Metric synthesis (realistic-looking, deterministic).
// Interpolated frames at t=0.5 are hardest (lowest PSNR); near observed frames
// quality is highest.
// ----------------------------------------------------------------------------
function synthMetric(index, t, seed) {
  const rng = mulberry32(seed + index * 97);
  // Difficulty peaks at t=0.5.
  const difficulty = 1 - Math.abs(t - 0.5) * 2; // 0..1, 1 = mid
  const jitter = (rng() - 0.5);

  const psnr = round(45.5 - difficulty * 6.5 + jitter * 0.8, 2); // ~38.5..46
  const ssim = round(0.972 - difficulty * 0.06 + jitter * 0.006, 4); // ~0.90..0.97
  const ms_ssim = round(Math.min(0.995, ssim + 0.012 + jitter * 0.004), 4);
  const fsim = round(Math.min(0.995, ssim + 0.02 + jitter * 0.003), 4);
  const gmsd = round(0.012 + difficulty * 0.02 + Math.abs(jitter) * 0.004, 4);
  const bt_rmse_k = round(0.85 + difficulty * 1.4 + Math.abs(jitter) * 0.2, 3); // ~0.85..2.4
  const bt_bias_k = round(jitter * 0.5, 3);
  const csi_235k = round(0.92 - difficulty * 0.08 + jitter * 0.01, 3);
  const fss = round(0.95 - difficulty * 0.07 + jitter * 0.01, 3);
  const epe = round(0.4 + difficulty * 0.9 + Math.abs(jitter) * 0.15, 3);

  return { index, t: round(t, 3), psnr, ssim, ms_ssim, fsim, gmsd, bt_rmse_k, bt_bias_k, csi_235k, fss, epe };
}

function round(v, n) {
  const f = 10 ** n;
  return Math.round(v * f) / f;
}

function mean(arr, key) {
  return round(arr.reduce((s, x) => s + x[key], 0) / arr.length, key === 'psnr' ? 2 : 4);
}

// ----------------------------------------------------------------------------
// Scene assembly.
// ----------------------------------------------------------------------------
function pad(n) {
  return n.toString().padStart(3, '0');
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function buildScene(scene) {
  const sceneDir = path.join(DATA_DIR, scene.id);
  ensureDir(path.join(sceneDir, 'frames'));
  ensureDir(path.join(sceneDir, 'thumbs'));
  ensureDir(path.join(sceneDir, 'diff'));
  ensureDir(path.join(sceneDir, 'flow'));

  const {
    nObserved,
    factor,
    nx,
    ny,
    cadenceIn,
    cfg,
    cmap,
  } = scene;

  // Build the full frame list (observed + interpolated) with timestamps.
  // Observed frames are spaced cadenceIn minutes; interpolated frames subdivide.
  const frames = [];
  const perFrameMetrics = [];
  const baseTime = new Date(scene.startISO).getTime();
  const stepsPerGap = factor; // factor=2 -> one mid frame; factor=4 -> three
  const cadenceOut = cadenceIn / factor;

  let globalIndex = 0;
  // We store the "field" of each rendered frame to make diffs vs the linear blend.
  const renderedFields = [];

  for (let o = 0; o < nObserved; o++) {
    // Observed frame at integer time o.
    pushFrame(o, 0, 'observed', null, null);

    // Interpolated frames between o and o+1 (skip after the last observed).
    if (o < nObserved - 1) {
      for (let k = 1; k < stepsPerGap; k++) {
        const t = k / stepsPerGap;
        pushFrame(o + t, t, 'interpolated', [o, o + 1], k);
      }
    }
  }

  function pushFrame(timeUnits, t, kind, bracketObs, kSub) {
    const idx = globalIndex++;
    const ms = baseTime + timeUnits * cadenceIn * 60_000;
    const iso = new Date(ms).toISOString().replace('.000Z', 'Z');

    // Render the IR field/frame.
    const { rgba, field } = renderFrame(nx, ny, timeUnits / (nObserved - 1), cfg, cmap);
    renderedFields[idx] = field;

    // Write full-frame PNG.
    fs.writeFileSync(
      path.join(sceneDir, 'frames', `${pad(idx)}.png`),
      encodePNG(nx, ny, rgba),
    );

    // Write thumbnail (downsampled).
    const { ox, oy, out } = downsample(nx, ny, rgba, 4);
    fs.writeFileSync(path.join(sceneDir, 'thumbs', `${pad(idx)}.png`), encodePNG(ox, oy, out));

    // For interpolated frames, also write a synthetic diff vs a linear blend of
    // the bracketing observed fields (this is what an error heatmap shows).
    let flowRel = null;
    if (kind === 'interpolated' && bracketObs) {
      // Approximate the bracketing observed fields by re-rendering at their times.
      const tA = bracketObs[0] / (nObserved - 1);
      const tB = bracketObs[1] / (nObserved - 1);
      const fA = renderFrame(nx, ny, tA, cfg, cmap).field;
      const fB = renderFrame(nx, ny, tB, cfg, cmap).field;
      const blend = new Float32Array(field.length);
      for (let p = 0; p < field.length; p++) blend[p] = fA[p] * (1 - t) + fB[p] * t;
      const diffRGBA = renderDiff(nx, ny, field, blend, 6);
      fs.writeFileSync(path.join(sceneDir, 'diff', `${pad(idx)}.png`), encodePNG(nx, ny, diffRGBA));

      // Emit a flow overlay for the central interpolated frame of the first gap.
      if (bracketObs[0] === 0 && Math.abs(t - 0.5) < 1e-6) {
        flowRel = `flow/${pad(idx)}.json`;
        fs.writeFileSync(
          path.join(sceneDir, flowRel),
          JSON.stringify(buildFlow(idx, scene), null, 0),
        );
      }
    } else {
      // Observed frames get a near-zero diff (mostly transparent).
      const zero = renderDiff(nx, ny, field, field, 1);
      fs.writeFileSync(path.join(sceneDir, 'diff', `${pad(idx)}.png`), encodePNG(nx, ny, zero));
    }

    frames.push({
      index: idx,
      time: iso,
      kind,
      t: kind === 'interpolated' ? round(t, 3) : null,
      bracket: bracketObs,
      image: `frames/${pad(idx)}.png`,
      thumb: `thumbs/${pad(idx)}.png`,
      // Mock data uses full-frame images (no per-frame XYZ pyramid). Real
      // precompute output would set tiles_url_template here.
      tiles_url_template: null,
      pmtiles: null,
      netcdf: `nc/${pad(idx)}.nc`,
      flow_overlay: flowRel,
    });

    if (kind === 'interpolated') {
      perFrameMetrics.push(synthMetric(idx, t, scene.metricSeed));
    }
    void kSub;
  }

  // ---- metrics summary + baselines ----
  const summary = {
    psnr_mean: mean(perFrameMetrics, 'psnr'),
    ssim_mean: mean(perFrameMetrics, 'ssim'),
    ms_ssim_mean: mean(perFrameMetrics, 'ms_ssim'),
    bt_rmse_k_mean: mean(perFrameMetrics, 'bt_rmse_k'),
    csi_235k_mean: mean(perFrameMetrics, 'csi_235k'),
    fss_mean: mean(perFrameMetrics, 'fss'),
    epe_mean: mean(perFrameMetrics, 'epe'),
  };

  const baselines = {
    linear: {
      name: 'Linear blend',
      psnr: round(summary.psnr_mean - 6.4, 2),
      ssim: round(summary.ssim_mean - 0.14, 4),
      ms_ssim: round(summary.ms_ssim_mean - 0.12, 4),
      bt_rmse_k: round(summary.bt_rmse_k_mean + 1.3, 3),
    },
    tvl1: {
      name: 'TV-L1 + warp',
      psnr: round(summary.psnr_mean - 2.7, 2),
      ssim: round(summary.ssim_mean - 0.05, 4),
      ms_ssim: round(summary.ms_ssim_mean - 0.04, 4),
      bt_rmse_k: round(summary.bt_rmse_k_mean + 0.6, 3),
    },
    persistence: {
      name: 'Persistence (copy)',
      psnr: round(summary.psnr_mean - 9.1, 2),
      ssim: round(summary.ssim_mean - 0.22, 4),
      ms_ssim: round(summary.ms_ssim_mean - 0.2, 4),
      bt_rmse_k: round(summary.bt_rmse_k_mean + 2.5, 3),
    },
  };

  const manifest = {
    version: '1.0',
    scene_id: scene.id,
    title: scene.title,
    satellite: scene.satellite,
    channel: scene.channel,
    wavelength_um: scene.wavelength_um,
    bbox: scene.bbox,
    crs: 'EPSG:4326',
    colormap: cmap,
    value_range_k: scene.valueRangeK,
    tile_size: 256,
    min_zoom: 1,
    max_zoom: 8,
    interpolation_factor: factor,
    cadence_minutes_input: cadenceIn,
    cadence_minutes_output: cadenceOut,
    frames,
    videos: {
      observed: 'videos/observed.mp4',
      interpolated: 'videos/interpolated.mp4',
      side_by_side: 'videos/side_by_side.mp4',
    },
    metrics: {
      data_range_k: round(scene.valueRangeK[1] - scene.valueRangeK[0], 1),
      value_range_k: scene.valueRangeK,
      per_frame: perFrameMetrics,
      summary,
      baselines,
    },
    crossval: {
      methods_run: [
        { id: 'M1', name: 'Hold-out-the-middle (self)', status: 'pass', summary: { psnr: summary.psnr_mean } },
        { id: 'M8', name: 'Himawari ↔ INSAT overlap', status: 'pass', summary: { bt_rmse_k: summary.bt_rmse_k_mean } },
        { id: 'M12', name: 'VIIRS overpass collocation', status: 'warn', summary: { samples: 14 } },
        { id: 'M18', name: 'Persistence floor beaten', status: 'pass', summary: { margin_db: round(summary.psnr_mean - baselines.persistence.psnr, 2) } },
        { id: 'M27', name: 'Triple collocation error budget', status: 'pass', summary: { sigma_k: 0.31 } },
        { id: 'M33', name: 'GIMM-VFI hard-case audit', status: 'skipped', summary: {} },
      ],
    },
    generated: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
    model: scene.model,
  };

  fs.writeFileSync(path.join(sceneDir, 'manifest.json'), JSON.stringify(manifest, null, 2));

  // Captions (WCAG 1.2.2) describing the animation.
  fs.writeFileSync(path.join(sceneDir, 'captions.vtt'), buildCaptions(scene));

  // A placeholder videos dir + README so the path exists (no binary mp4).
  ensureDir(path.join(sceneDir, 'videos'));
  fs.writeFileSync(
    path.join(sceneDir, 'videos', 'README.txt'),
    'The mock generator does not produce binary MP4 video.\n' +
      'The real precompute pipeline (frameflow.serve.video) writes all-intra\n' +
      'observed.mp4 / interpolated.mp4 / side_by_side.mp4 here. The dashboard\n' +
      'degrades gracefully and shows the scrubbable frame animation instead.\n',
  );

  return { sceneId: scene.id, frames: frames.length, interp: perFrameMetrics.length };
}

function buildFlow(frameIndex, scene) {
  // Build a grid of motion vectors roughly following the blob advection in
  // geographic (lon/lat) space. Vectors are scaled to be visible.
  const [w, s, e, n] = scene.bbox;
  const cols = 14;
  const rows = 9;
  const vectors = [];
  const cfg = scene.cfg;
  const scaleDeg = (e - w) * 0.06; // base display length
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const u = (i + 0.5) / cols;
      const v = (j + 0.5) / rows;
      const lon = w + u * (e - w);
      const lat = s + (1 - v) * (n - s);

      // Rotational + translational field around the blob center.
      const bx = cfg.blob.x0 + cfg.blob.vx * 0.5;
      const by = cfg.blob.y0 + cfg.blob.vy * 0.5;
      const dx = u - bx;
      const dy = v - by;
      const r = Math.hypot(dx, dy) + 1e-3;
      // tangential (cyclonic) component + mean flow
      const tx = -dy / r;
      const ty = dx / r;
      const swirl = Math.exp(-(r * r) / (2 * cfg.blob.sigma * cfg.blob.sigma * 2.2));
      const fx = (tx * swirl * 1.2 + cfg.blob.vx * 0.6);
      const fy = (ty * swirl * 1.2 + cfg.blob.vy * 0.6);
      const speed = Math.hypot(fx, fy);
      vectors.push({
        position: [round(lon, 4), round(lat, 4)],
        // Note: screen-down (v increasing) maps to lat decreasing, so negate fy.
        vector: [round(fx * scaleDeg, 5), round(-fy * scaleDeg, 5)],
        speed: round(speed * 12, 3),
      });
    }
  }
  return { frame_index: frameIndex, scale: round(scaleDeg, 5), vectors };
}

function buildCaptions(scene) {
  // Simple WebVTT describing the animation for accessibility.
  return `WEBVTT

NOTE FrameFlow accessibility captions (WCAG 1.2.2) for ${scene.title}.

00:00:00.000 --> 00:00:04.000
Animated thermal-infrared cloud-top brightness temperature.

00:00:04.000 --> 00:00:08.000
Left: ground-truth observed frames. Right: AI-interpolated frames.

00:00:08.000 --> 00:00:12.000
Cold cloud tops appear bright; warm surface appears dark.

00:00:12.000 --> 00:00:16.000
A ${scene.title.includes('Cyclone') ? 'cyclonic system' : 'cold cloud mass'} advects across the ${scene.satellite} field of view.

00:00:16.000 --> 00:00:20.000
Temporal resolution enhanced ${scene.cadenceIn} minutes to ${scene.cadenceIn / scene.factor} minutes via optical-flow interpolation.
`;
}

// ----------------------------------------------------------------------------
// Scene definitions.
// ----------------------------------------------------------------------------
const SCENES = [
  {
    id: 'cyclone-atlantic',
    title: 'GOES-19 · Atlantic Cyclone',
    satellite: 'GOES-19',
    channel: 'ABI C13',
    wavelength_um: 10.3,
    bbox: [-82, 18, -58, 38], // [w,s,e,n] — western Atlantic
    valueRangeK: [185, 305],
    startISO: '2025-09-25T12:00:00Z',
    nObserved: 9, // observed frames (30 min cadence)
    factor: 4, // 30 -> 7.5 min
    cadenceIn: 30,
    nx: 384,
    ny: 320,
    cmap: 'ir',
    metricSeed: 1337,
    model: { name: 'FrameFlow-RIFE', version: 'v4.26-tir', params_m: 9.8 },
    cfg: {
      seed: 7,
      blob: { x0: 0.32, y0: 0.62, vx: 0.045, vy: -0.05, sigma: 0.16, spin: 2.4, arms: 5, tightness: 26 },
      cell: { x0: 0.7, y0: 0.35, vx: -0.02, vy: 0.01, sigma: 0.07 },
    },
  },
  {
    id: 'synthetic-demo',
    title: 'Synthetic · Advecting Blob',
    satellite: 'FrameFlow-Synthetic',
    channel: 'TIR (sim)',
    wavelength_um: 10.8,
    bbox: [70, 5, 95, 25], // around the Indian subcontinent / Bay of Bengal
    valueRangeK: [180, 300],
    startISO: '2026-01-15T06:00:00Z',
    nObserved: 7,
    factor: 2, // 30 -> 15 min
    cadenceIn: 30,
    nx: 360,
    ny: 300,
    cmap: 'inferno',
    metricSeed: 4242,
    model: { name: 'FrameFlow-RIFE', version: 'v4.26-tir', params_m: 9.8 },
    cfg: {
      seed: 19,
      blob: { x0: 0.25, y0: 0.5, vx: 0.06, vy: 0.0, sigma: 0.14, spin: 1.2, arms: 3, tightness: 18 },
      cell: { x0: 0.6, y0: 0.6, vx: 0.0, vy: -0.03, sigma: 0.08 },
    },
  },
];

// ----------------------------------------------------------------------------
// Run.
// ----------------------------------------------------------------------------
function main() {
  console.log('FrameFlow mock-data generator');
  ensureDir(DATA_DIR);

  // Preserve any pre-existing NON-mock scenes (e.g. the embedded real scene
  // demo-0001 produced by scripts/embed_demo_scene.py) so regenerating the mock
  // does not drop the real scene from the selector. Real scenes are listed FIRST.
  const mockIds = new Set(SCENES.map((s) => s.id));
  let preserved = [];
  const scenesPath = path.join(DATA_DIR, 'scenes.json');
  if (fs.existsSync(scenesPath)) {
    try {
      const prev = JSON.parse(fs.readFileSync(scenesPath, 'utf8'));
      preserved = (prev.scenes || []).filter((s) => !mockIds.has(s.scene_id));
    } catch {
      preserved = [];
    }
  }

  const index = { scenes: [...preserved] };
  for (const scene of SCENES) {
    process.stdout.write(`  · scene "${scene.id}" … `);
    const r = buildScene(scene);
    index.scenes.push({ scene_id: scene.id, title: scene.title, satellite: scene.satellite });
    console.log(`${r.frames} frames (${r.interp} interpolated)`);
  }

  fs.writeFileSync(scenesPath, JSON.stringify(index, null, 2));
  console.log(
    `  · wrote scenes.json (${index.scenes.length} scenes` +
      (preserved.length ? `, preserved ${preserved.length} real` : '') +
      ')',
  );
  console.log(`Done -> ${path.relative(ROOT, DATA_DIR)}`);
}

main();

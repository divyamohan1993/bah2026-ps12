/**
 * FrameFlow manifest contract (v1.0).
 *
 * The dashboard loads `/data/<scene>/manifest.json`. These types mirror the
 * EXACT schema produced by the precompute pipeline (frameflow.serve.precompute)
 * and by the mock generator (scripts/gen_mock.mjs).
 *
 * See ARCHITECTURE.md §9 and research/04_web_viz.md.
 */

/** [west, south, east, north] in EPSG:4326 degrees. */
export type BBox = [west: number, south: number, east: number, north: number];

/** [min, max] inclusive range. */
export type Range2 = [min: number, max: number];

export type FrameKind = 'observed' | 'interpolated';

/**
 * One frame in the time series. Observed frames are real satellite acquisitions;
 * interpolated frames are AI-synthesized at fractional time `t` between a
 * `bracket` of two observed frames.
 */
export interface ManifestFrame {
  /** Global frame index (0-based, monotonically increasing in time). */
  index: number;
  /** ISO-8601 UTC timestamp, e.g. "2024-09-25T12:15:00Z". */
  time: string;
  kind: FrameKind;
  /**
   * Fractional position in (0,1) between the bracketing observed frames for
   * interpolated frames; null for observed frames.
   */
  t: number | null;
  /**
   * Indices of the two observed frames this interpolated frame sits between;
   * null for observed frames.
   */
  bracket: [number, number] | null;
  /** Full-frame preview image, relative to the scene dir, e.g. "frames/000.webp". */
  image: string;
  /** Small thumbnail image for the filmstrip, e.g. "thumbs/000.webp". */
  thumb: string;
  /**
   * Per-frame XYZ raster pyramid template, e.g. "tiles/000/{z}/{x}/{y}.webp".
   * When present, rendered via a deck.gl TileLayer (the O(1) production path).
   * `null` for the mock dataset, which falls back to `image` + a BitmapLayer.
   */
  tiles_url_template: string | null;
  /** Per-frame PMTiles archive, e.g. "pmtiles/000.pmtiles"; null if absent. */
  pmtiles: string | null;
  /** Source / interpolated NetCDF artifact for this frame (provenance). */
  netcdf: string;
  /**
   * Optical-flow overlay vectors for this frame, relative to scene dir,
   * e.g. "flow/030.json"; null when no overlay was produced.
   */
  flow_overlay: string | null;
}

/** Per-frame image-quality metrics (only meaningful for interpolated frames). */
export interface PerFrameMetric {
  index: number;
  /** Fractional time of the frame (mirrors ManifestFrame.t). */
  t: number;
  /** Peak signal-to-noise ratio (dB), higher is better. */
  psnr: number;
  /** Structural similarity [0,1], higher is better. */
  ssim: number;
  /** Multi-scale SSIM [0,1]. */
  ms_ssim: number;
  /** Feature similarity index [0,1]. */
  fsim: number;
  /** Gradient magnitude similarity deviation, lower is better. */
  gmsd: number;
  /** Brightness-temperature RMSE in Kelvin, lower is better. */
  bt_rmse_k: number;
  /** Brightness-temperature bias in Kelvin (signed). */
  bt_bias_k: number;
  /** Critical Success Index for the 235 K cold-cloud threshold [0,1]. */
  csi_235k: number;
  /** Fractions Skill Score [0,1]. */
  fss: number;
  /** End-point error of optical flow (px), lower is better. */
  epe: number;
}

export interface BaselineSummary {
  /** Display name, e.g. "Linear blend" or "TV-L1 + warp". */
  name: string;
  psnr: number;
  ssim: number;
  ms_ssim: number;
  bt_rmse_k: number;
  [extra: string]: number | string;
}

export interface MetricsSummary {
  psnr_mean: number;
  ssim_mean: number;
  ms_ssim_mean: number;
  bt_rmse_k_mean: number;
  csi_235k_mean: number;
  [extra: string]: number;
}

export interface ManifestMetrics {
  /** Physical data range in Kelvin used for the colormap. */
  data_range_k: number;
  value_range_k: Range2;
  per_frame: PerFrameMetric[];
  /** Aggregate stats across interpolated frames. */
  summary: MetricsSummary;
  /** Baseline methods to compare against (linear blend, TV-L1, ...). */
  baselines: Record<string, BaselineSummary>;
}

export interface CrossValMethod {
  id: string;
  name: string;
  /** e.g. "pass" | "warn" | "fail" | "skipped". */
  status: string;
  summary: Record<string, number | string>;
}

export interface ManifestCrossVal {
  methods_run: CrossValMethod[];
}

export interface ManifestVideos {
  observed: string;
  interpolated: string;
  side_by_side: string;
}

export interface ManifestModel {
  name: string;
  version: string;
  params_m: number;
}

/** The full manifest document. */
export interface Manifest {
  version: string;
  scene_id: string;
  title: string;
  satellite: string;
  channel: string;
  wavelength_um: number;
  bbox: BBox;
  crs: 'EPSG:4326';
  colormap: string;
  value_range_k: Range2;
  tile_size: number;
  min_zoom: number;
  max_zoom: number;
  interpolation_factor: number;
  cadence_minutes_input: number;
  cadence_minutes_output: number;
  frames: ManifestFrame[];
  videos: ManifestVideos;
  metrics: ManifestMetrics;
  crossval: ManifestCrossVal;
  generated: string;
  model: ManifestModel;
}

/** Lightweight scene-index entry for the scene selector. */
export interface SceneIndexEntry {
  scene_id: string;
  title: string;
  satellite: string;
}

export interface SceneIndex {
  scenes: SceneIndexEntry[];
}

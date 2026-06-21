/**
 * Ambient module declarations for packages that ship without (or with
 * incomplete) TypeScript types in our pinned versions.
 */

declare module '@maplibre/maplibre-gl-compare' {
  import type { Map as MapLibreMap } from 'maplibre-gl';

  export interface CompareOptions {
    /** If true the swipe handle follows the cursor; if false it is dragged. */
    mousemove?: boolean;
    /** 'vertical' (left/right wipe) or 'horizontal' (top/bottom wipe). */
    orientation?: 'vertical' | 'horizontal';
  }

  export default class Compare {
    constructor(
      a: MapLibreMap,
      b: MapLibreMap,
      container: string | HTMLElement,
      options?: CompareOptions,
    );
    /** Current slider position in pixels. */
    currentPosition: number;
    setSlider(x: number): void;
    on(type: 'slideend', listener: (ev: { currentPosition: number }) => void): void;
    remove(): void;
  }
}

declare module '@maplibre/maplibre-gl-compare/dist/maplibre-gl-compare.css';

/**
 * Optical-flow overlay vector. Two on-disk schemas are supported and normalized in
 * deckLayers.ts (normalizeFlowVectors):
 *   - mock generator   : { position:[lon,lat], vector:[Δlon,Δlat], speed }
 *   - real precompute  : { sourcePosition:[lon,lat], targetPosition:[lon,lat], u, v, mag }
 *     (written by frameflow.viz.flow_overlay.flow_to_overlay_json)
 */
interface FlowVector {
  /** [lon, lat] origin of the vector (mock schema). */
  position?: [number, number];
  /** [Δlon, Δlat] displacement in degrees, display-scaled (mock schema). */
  vector?: [number, number];
  /** Speed magnitude used for color/opacity ramping (mock schema). */
  speed?: number;
  /** [lon, lat] origin (precompute schema). */
  sourcePosition?: [number, number];
  /** [lon, lat] tip (precompute schema). */
  targetPosition?: [number, number];
  /** Pixel-space components + magnitude (precompute schema). */
  u?: number;
  v?: number;
  mag?: number;
}

interface FlowOverlay {
  /** Present in the mock schema; absent in the precompute schema (defaults to 0). */
  frame_index?: number;
  /** Display scale; informational. */
  scale?: number;
  /** Layer kind tag emitted by precompute ("LineLayer"); informational. */
  type?: string;
  /** [west, south, east, north] geographic extent (precompute schema); informational. */
  bbox?: number[];
  vectors: FlowVector[];
}

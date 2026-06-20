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

/** Optical-flow overlay schema produced by frameflow.flow.visualize. */
interface FlowVector {
  /** [lon, lat] origin of the vector. */
  position: [number, number];
  /** [u, v] displacement in degrees (already scaled for display). */
  vector: [number, number];
  /** Speed magnitude (px/frame), used for color/opacity ramping. */
  speed: number;
}

interface FlowOverlay {
  frame_index: number;
  /** Display scale already baked into `vector`; informational. */
  scale: number;
  vectors: FlowVector[];
}

import { BitmapLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';
import { LineLayer, IconLayer } from '@deck.gl/layers';
import type { Layer } from '@deck.gl/core';
import type { BBox, ManifestFrame } from '@/types/manifest';
import { sceneAsset } from '@/lib/utils';
import { sampleColormap } from '@/lib/colormap';

/**
 * ============================================================================
 *  CODEX REVIEW P2 — per-frame raster source switching
 * ============================================================================
 * A single raster source CANNOT be addressed by timestamp; EACH frame has its
 * OWN source (its own XYZ pyramid or its own full-frame image). The timeline
 * slider selects the ACTIVE frame index, and we (re)build the active frame's
 * raster layer here.
 *
 * Resolution order, per the manifest contract:
 *   1. If `tiles_url_template` is present  -> deck.gl TileLayer (production O(1)
 *      XYZ pyramid path). The layer `data` URL embeds the frame index, so each
 *      frame is a distinct source.
 *   2. ELSE fall back to a deck.gl BitmapLayer over the manifest `bbox` using
 *      the frame's full-frame `image` (the mock-data path).
 *
 * The deck.gl layer `id` ENCODES the frame index + role, so switching frames
 * swaps layer identity and forces deck.gl to load the new frame's source rather
 * than diffing props on a stale source. (For TileLayers we also avoid sharing a
 * tile cache across frames by keying the id with the index.)
 */

export type PaneRole = 'gt' | 'interp';

export interface RasterLayerOptions {
  role: PaneRole;
  frame: ManifestFrame;
  sceneBase: string;
  bbox: BBox;
  minZoom: number;
  maxZoom: number;
  tileSize: number;
  opacity: number;
  visible: boolean;
  /** Desaturate hint when used purely as an error-base; not used yet. */
  beforeId?: string;
}

/**
 * Build the active-frame raster layer for one pane. Returns a single deck.gl
 * layer (TileLayer or BitmapLayer) chosen per the manifest contract.
 */
export function buildRasterLayer(opts: RasterLayerOptions): Layer {
  const { role, frame, sceneBase, bbox, minZoom, maxZoom, tileSize, opacity, visible } = opts;

  // --- Path 1: per-frame XYZ pyramid (production). ---
  if (frame.tiles_url_template) {
    const dataUrl = sceneAsset(sceneBase, frame.tiles_url_template);
    return new TileLayer({
      // id encodes role + frame index => distinct source per frame (P2).
      id: `${role}-tiles-${frame.index}`,
      data: dataUrl,
      tileSize,
      minZoom,
      maxZoom,
      maxRequests: 6,
      opacity,
      visible,
      pickable: false,
      refinementStrategy: 'best-available',
      renderSubLayers: (props) => {
        // deck.gl v9 provides the tile's bounding box on props.tile.boundingBox
        // as [[west, south], [east, north]].
        const tile = props.tile as unknown as {
          boundingBox: [[number, number], [number, number]];
        };
        const { boundingBox } = tile;
        return new BitmapLayer(props, {
          data: undefined,
          image: props.data as string,
          bounds: [
            boundingBox[0][0],
            boundingBox[0][1],
            boundingBox[1][0],
            boundingBox[1][1],
          ],
        });
      },
    });
  }

  // --- Path 2: full-frame BitmapLayer over the manifest bbox (mock fallback). ---
  const imageUrl = sceneAsset(sceneBase, frame.image);
  return new BitmapLayer({
    // id encodes role + frame index => the texture swaps as the slider moves (P2).
    id: `${role}-bitmap-${frame.index}`,
    image: imageUrl,
    bounds: [bbox[0], bbox[1], bbox[2], bbox[3]],
    opacity,
    visible,
    pickable: false,
    // Smooth IR imagery — bilinear sampling looks better than nearest here.
    textureParameters: {
      minFilter: 'linear',
      magFilter: 'linear',
    },
  });
}

export interface ErrorLayerOptions {
  frame: ManifestFrame;
  sceneBase: string;
  bbox: BBox;
  opacity: number;
  visible: boolean;
}

/**
 * Error / difference heatmap layer. The mock pipeline writes a synthetic diff
 * image next to the frame ("diff/NNN.webp"); if present we render it as a
 * BitmapLayer. Falls back gracefully (invisible) if no diff asset exists.
 */
export function buildErrorLayer(opts: ErrorLayerOptions): Layer {
  const { frame, sceneBase, bbox, opacity, visible } = opts;
  const idx = frame.index.toString().padStart(3, '0');
  const url = sceneAsset(sceneBase, `diff/${idx}.webp`);
  return new BitmapLayer({
    id: `error-bitmap-${frame.index}`,
    image: url,
    bounds: [bbox[0], bbox[1], bbox[2], bbox[3]],
    opacity,
    visible,
    pickable: false,
    // If the diff asset is missing, deck logs a load error but the app is fine.
    onError: () => true,
    textureParameters: { minFilter: 'linear', magFilter: 'linear' },
  });
}

export interface FlowLayerOptions {
  overlay: FlowOverlay;
  visible: boolean;
  /** Multiplier applied on top of the baked display scale. */
  exaggeration?: number;
}

/**
 * Optical-flow overlay: motion vectors rendered as a LineLayer (the shaft) plus
 * an IconLayer-free arrowhead approximated with a short second segment. Color
 * ramps by speed using the IR colormap for visual coherence.
 */
export function buildFlowLayers(opts: FlowLayerOptions): Layer[] {
  const { overlay, visible, exaggeration = 1 } = opts;
  if (!overlay || !Array.isArray(overlay.vectors) || overlay.vectors.length === 0) {
    return [];
  }
  const maxSpeed =
    overlay.vectors.reduce((mx, v) => Math.max(mx, v.speed), 0.0001) || 1;

  const shaft = new LineLayer<FlowVector>({
    id: `flow-shaft-${overlay.frame_index}`,
    data: overlay.vectors,
    visible,
    getSourcePosition: (d) => d.position,
    getTargetPosition: (d) => [
      d.position[0] + d.vector[0] * exaggeration,
      d.position[1] + d.vector[1] * exaggeration,
    ],
    getColor: (d) => {
      const [r, g, b] = sampleColormap('ir', 0.45 + 0.5 * (d.speed / maxSpeed));
      return [r, g, b, 220];
    },
    getWidth: 1.6,
    widthUnits: 'pixels',
    widthMinPixels: 1,
    pickable: false,
  });

  // Arrowheads as small dot icons at the vector tip (cheap, no external image).
  const heads = new IconLayer<FlowVector>({
    id: `flow-heads-${overlay.frame_index}`,
    data: overlay.vectors,
    visible,
    getPosition: (d) => [
      d.position[0] + d.vector[0] * exaggeration,
      d.position[1] + d.vector[1] * exaggeration,
    ],
    getIcon: () => ({
      url: ARROW_DOT_DATA_URI,
      width: 16,
      height: 16,
      anchorX: 8,
      anchorY: 8,
      mask: true,
    }),
    getColor: (d) => {
      const [r, g, b] = sampleColormap('ir', 0.55 + 0.45 * (d.speed / maxSpeed));
      return [r, g, b, 235];
    },
    getSize: 6,
    sizeUnits: 'pixels',
    pickable: false,
  });

  return [shaft, heads];
}

/** Tiny inline circle used as a mask icon for flow-vector tips. */
const ARROW_DOT_DATA_URI =
  'data:image/svg+xml;base64,' +
  btoa(
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><circle cx="8" cy="8" r="5" fill="#fff"/></svg>',
  );

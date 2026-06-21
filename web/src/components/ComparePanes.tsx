import { useEffect, useMemo, useRef, useState } from 'react';
import Compare from '@maplibre/maplibre-gl-compare';
import '@maplibre/maplibre-gl-compare/dist/maplibre-gl-compare.css';
import type { Map as MapLibreMap } from 'maplibre-gl';
import type { Layer } from '@deck.gl/core';
import { MapPane, type MapPaneHandle } from '@/components/MapPane';
import {
  buildRasterLayer,
  buildErrorLayer,
  buildFlowLayers,
} from '@/lib/deckLayers';
import type { Manifest, ManifestFrame } from '@/types/manifest';
import { sceneAsset } from '@/lib/utils';
import { useStore } from '@/hooks/useStore';
import { PaneLabel } from '@/components/PaneLabel';

interface ComparePanesProps {
  manifest: Manifest;
  sceneBase: string;
  /** Active frame (already resolved to the displayed frame). */
  frame: ManifestFrame;
  /**
   * The nearest observed frame to use as "ground truth" in the left pane. For
   * observed frames this equals `frame`; for interpolated frames it is the
   * lower bracket observed frame (so the left pane always shows real data).
   */
  gtFrame: ManifestFrame;
}

/**
 * Side-by-side compare of Ground Truth (left) vs Interpolated (right), driven by
 * the SAME current frame index, with synced pan/zoom via maplibre-gl-compare.
 *
 * Per-frame source switching (Codex P2): `buildRasterLayer` is recomputed for
 * the active frame on every change; its deck.gl layer id encodes the frame
 * index, so the active raster source swaps as the timeline scrubs.
 */
export function ComparePanes({ manifest, sceneBase, frame, gtFrame }: ComparePanesProps) {
  const leftRef = useRef<MapPaneHandle>(null);
  const rightRef = useRef<MapPaneHandle>(null);
  const compareRef = useRef<Compare | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [leftMap, setLeftMap] = useState<MapLibreMap | null>(null);
  const [rightMap, setRightMap] = useState<MapLibreMap | null>(null);

  const compareMode = useStore((s) => s.compareMode);
  const showGt = useStore((s) => s.showGt);
  const showInterp = useStore((s) => s.showInterp);
  const showFlow = useStore((s) => s.showFlow);
  const showError = useStore((s) => s.showError);
  const errorOpacity = useStore((s) => s.errorOpacity);

  const { bbox, min_zoom, max_zoom, tile_size } = manifest;

  // --- Load the flow overlay JSON for the active frame (if any). ---
  const [flow, setFlow] = useState<FlowOverlay | null>(null);
  useEffect(() => {
    let cancelled = false;
    if (!frame.flow_overlay) {
      setFlow(null);
      return;
    }
    fetch(sceneAsset(sceneBase, frame.flow_overlay))
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (!cancelled) setFlow(j as FlowOverlay | null);
      })
      .catch(() => {
        if (!cancelled) setFlow(null);
      });
    return () => {
      cancelled = true;
    };
  }, [frame.flow_overlay, sceneBase]);

  // --- Left pane (Ground Truth): always the observed frame. ---
  const leftLayers = useMemo<Layer[]>(() => {
    return [
      buildRasterLayer({
        role: 'gt',
        frame: gtFrame,
        sceneBase,
        bbox,
        minZoom: min_zoom,
        maxZoom: max_zoom,
        tileSize: tile_size,
        opacity: 1,
        visible: showGt,
      }),
    ];
  }, [gtFrame, sceneBase, bbox, min_zoom, max_zoom, tile_size, showGt]);

  // --- Right pane (Interpolated): the active frame + optional overlays. ---
  const rightLayers = useMemo<Layer[]>(() => {
    const layers: Layer[] = [
      buildRasterLayer({
        role: 'interp',
        frame,
        sceneBase,
        bbox,
        minZoom: min_zoom,
        maxZoom: max_zoom,
        tileSize: tile_size,
        opacity: 1,
        visible: showInterp,
      }),
    ];
    if (showError) {
      layers.push(
        buildErrorLayer({
          frame,
          sceneBase,
          bbox,
          opacity: errorOpacity,
          visible: true,
        }),
      );
    }
    if (showFlow && flow) {
      layers.push(...buildFlowLayers({ overlay: flow, visible: true }));
    }
    return layers;
  }, [
    frame,
    sceneBase,
    bbox,
    min_zoom,
    max_zoom,
    tile_size,
    showInterp,
    showError,
    errorOpacity,
    showFlow,
    flow,
  ]);

  // --- Wire up / tear down maplibre-gl-compare when both maps are ready. ---
  useEffect(() => {
    if (!leftMap || !rightMap || !containerRef.current) return;

    if (compareMode === 'swipe') {
      const cmp = new Compare(leftMap, rightMap, containerRef.current, {
        mousemove: false,
        orientation: 'vertical',
      });
      compareRef.current = cmp;
      return () => {
        cmp.remove();
        compareRef.current = null;
      };
    }

    // Two-pane mode: still keep pan/zoom synced manually.
    const sync = (from: MapLibreMap, to: MapLibreMap) => () => {
      if ((to as unknown as { _ffSyncing?: boolean })._ffSyncing) return;
      (from as unknown as { _ffSyncing?: boolean })._ffSyncing = true;
      to.jumpTo({
        center: from.getCenter(),
        zoom: from.getZoom(),
        bearing: from.getBearing(),
        pitch: from.getPitch(),
      });
      (from as unknown as { _ffSyncing?: boolean })._ffSyncing = false;
    };
    const a = sync(leftMap, rightMap);
    const b = sync(rightMap, leftMap);
    leftMap.on('move', a);
    rightMap.on('move', b);
    return () => {
      leftMap.off('move', a);
      rightMap.off('move', b);
    };
  }, [leftMap, rightMap, compareMode]);

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden">
      <MapPane
        id="ff-gt-pane"
        ref={leftRef}
        layers={leftLayers}
        bounds={bbox}
        showControls={false}
        onReady={setLeftMap}
        className="absolute inset-0 h-full w-full"
      />
      <MapPane
        id="ff-interp-pane"
        ref={rightRef}
        layers={rightLayers}
        bounds={bbox}
        showControls
        onReady={setRightMap}
        className="absolute inset-0 h-full w-full"
      />

      {/* Pane labels */}
      <PaneLabel
        side="left"
        title="Ground Truth"
        sub={gtFrame.kind === 'observed' ? 'observed' : 'reference'}
        tone="cyan"
      />
      <PaneLabel
        side="right"
        title="AI Interpolated"
        sub={frame.kind === 'interpolated' ? `t = ${frame.t?.toFixed(2)}` : 'observed'}
        tone={frame.kind === 'interpolated' ? 'violet' : 'cyan'}
      />
    </div>
  );
}

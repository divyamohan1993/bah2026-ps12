import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import type { Map as MapLibreMap } from 'maplibre-gl';
import { MapboxOverlay } from '@deck.gl/mapbox';
import type { Layer } from '@deck.gl/core';
import 'maplibre-gl/dist/maplibre-gl.css';
import { getBasemapStyle } from '@/lib/basemap';
import type { BBox } from '@/types/manifest';

export interface MapPaneHandle {
  getMap: () => MapLibreMap | null;
}

export interface MapPaneProps {
  id: string;
  /** deck.gl layers to render via the MapboxOverlay (interleaved). */
  layers: Layer[];
  /** Initial fit bounds [w,s,e,n]. */
  bounds: BBox;
  /** Show MapLibre nav controls (only the left/primary pane should). */
  showControls?: boolean;
  className?: string;
  onReady?: (map: MapLibreMap) => void;
}

/**
 * A single MapLibre map pane with a deck.gl MapboxOverlay on top. We use the
 * overlay (rather than @deck.gl/react DeckGL) so two panes can be driven by
 * @maplibre/maplibre-gl-compare with synced pan/zoom for free.
 */
export const MapPane = forwardRef<MapPaneHandle, MapPaneProps>(function MapPane(
  { id, layers, bounds, showControls = false, className, onReady },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  useImperativeHandle(ref, () => ({ getMap: () => mapRef.current }), []);

  // Create the map once.
  useEffect(() => {
    if (!containerRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: getBasemapStyle(false),
      bounds: [
        [bounds[0], bounds[1]],
        [bounds[2], bounds[3]],
      ],
      fitBoundsOptions: { padding: 24 },
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      maxZoom: 10,
      minZoom: 1,
    });
    mapRef.current = map;

    map.touchZoomRotate.disableRotation();
    if (showControls) {
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
      map.addControl(
        new maplibregl.AttributionControl({ compact: true }),
        'bottom-right',
      );
    }

    const overlay = new MapboxOverlay({ interleaved: true, layers: [] });
    overlayRef.current = overlay;
    map.addControl(overlay as unknown as maplibregl.IControl);

    // Handle CARTO raster CDN being unavailable -> swap to inline offline style.
    let swappedOffline = false;
    const onError = (e: { error?: { status?: number } }) => {
      const status = e?.error?.status;
      if (!swappedOffline && (status === 0 || status === 403 || status === 404 || status === 429)) {
        swappedOffline = true;
        try {
          map.setStyle(getBasemapStyle(true));
        } catch {
          /* ignore */
        }
      }
    };
    map.on('error', onError as never);

    map.once('load', () => {
      onReady?.(map);
    });

    return () => {
      map.off('error', onError as never);
      try {
        overlay.finalize();
      } catch {
        /* ignore */
      }
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push new layers to the overlay whenever they change.
  useEffect(() => {
    overlayRef.current?.setProps({ layers });
  }, [layers]);

  return <div id={id} ref={containerRef} className={className} aria-hidden="true" />;
});

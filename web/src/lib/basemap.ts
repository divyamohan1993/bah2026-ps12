import type { StyleSpecification } from 'maplibre-gl';

/**
 * Dark "mission-control" basemap styles. No Mapbox token required.
 *
 * Primary: CARTO dark-matter raster tiles (free, no API key). These are public
 * basemap tiles served from the CARTO CDN.
 * Fallback: a fully inline, network-free style (solid deep-space background with
 * a subtle graticule feel via background only) so the app still renders if the
 * raster CDN is blocked.
 */

const DEEP_SPACE = '#070b12';

/** CARTO dark raster basemap — used when network is available. */
export const cartoDarkStyle: StyleSpecification = {
  version: 8,
  // glyphs are not strictly needed (no symbol layers) but kept for completeness.
  glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
      ],
      tileSize: 256,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
      maxzoom: 19,
    },
  },
  layers: [
    {
      id: 'bg',
      type: 'background',
      paint: { 'background-color': DEEP_SPACE },
    },
    {
      id: 'carto-dark',
      type: 'raster',
      source: 'carto-dark',
      paint: {
        // Dim + slightly desaturate so the IR overlay pops.
        'raster-opacity': 0.85,
        'raster-brightness-max': 0.7,
        'raster-saturation': -0.2,
        'raster-contrast': 0.05,
      },
    },
  ],
};

/** Fully offline inline style (no tile requests). */
export const inlineDarkStyle: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [
    {
      id: 'bg',
      type: 'background',
      paint: { 'background-color': DEEP_SPACE },
    },
  ],
};

/**
 * Pick the basemap style. We default to CARTO dark; callers may pass
 * `offline` to force the inline style (e.g. for air-gapped demos).
 */
export function getBasemapStyle(offline = false): StyleSpecification {
  return offline ? inlineDarkStyle : cartoDarkStyle;
}

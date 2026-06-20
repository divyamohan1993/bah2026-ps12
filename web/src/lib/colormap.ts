/**
 * Scientific colormaps for Thermal-IR brightness temperature.
 *
 * Geostationary IR convention: cold cloud tops (low BT, deep convection) are
 * highlighted in bright colors, warm surface (high BT) fades to dark. We expose
 * a perceptually-ordered "IR" ramp plus an "inferno"-style sequential ramp, and
 * helpers to build CSS gradients for the legend.
 */

export type RGB = [number, number, number];

interface ColorStop {
  /** Normalized position in [0,1] (0 = value_range_k[0], 1 = value_range_k[1]). */
  at: number;
  rgb: RGB;
}

/**
 * Enhanced-IR style ramp (loosely the classic NOAA "IR" enhancement).
 * Low normalized value = warm scene (dark), high = cold cloud tops (bright/white).
 */
const IR_STOPS: ColorStop[] = [
  { at: 0.0, rgb: [8, 12, 20] },
  { at: 0.12, rgb: [20, 30, 55] },
  { at: 0.28, rgb: [20, 70, 120] },
  { at: 0.42, rgb: [16, 130, 150] },
  { at: 0.55, rgb: [30, 175, 120] },
  { at: 0.66, rgb: [150, 200, 70] },
  { at: 0.76, rgb: [240, 200, 40] },
  { at: 0.85, rgb: [240, 130, 30] },
  { at: 0.92, rgb: [225, 60, 50] },
  { at: 0.97, rgb: [200, 60, 160] },
  { at: 1.0, rgb: [245, 240, 255] },
];

/** Sequential "inferno"-like ramp for error / difference heatmaps. */
const INFERNO_STOPS: ColorStop[] = [
  { at: 0.0, rgb: [4, 6, 18] },
  { at: 0.2, rgb: [40, 11, 84] },
  { at: 0.4, rgb: [101, 21, 110] },
  { at: 0.6, rgb: [159, 42, 99] },
  { at: 0.75, rgb: [212, 72, 66] },
  { at: 0.88, rgb: [245, 125, 21] },
  { at: 1.0, rgb: [252, 255, 164] },
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function sampleStops(stops: ColorStop[], tRaw: number): RGB {
  const t = Math.min(1, Math.max(0, tRaw));
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (t >= a.at && t <= b.at) {
      const local = (t - a.at) / (b.at - a.at || 1);
      return [
        Math.round(lerp(a.rgb[0], b.rgb[0], local)),
        Math.round(lerp(a.rgb[1], b.rgb[1], local)),
        Math.round(lerp(a.rgb[2], b.rgb[2], local)),
      ];
    }
  }
  return stops[stops.length - 1].rgb;
}

export type ColormapName = 'ir' | 'inferno';

const RAMPS: Record<ColormapName, ColorStop[]> = {
  ir: IR_STOPS,
  inferno: INFERNO_STOPS,
};

/** Sample a colormap at normalized position t in [0,1]. */
export function sampleColormap(name: ColormapName, t: number): RGB {
  return sampleStops(RAMPS[name] ?? IR_STOPS, t);
}

/** Build a CSS `linear-gradient(...)` string for a legend bar. */
export function gradientCss(name: ColormapName, steps = 24): string {
  const ramp = RAMPS[name] ?? IR_STOPS;
  const parts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const [r, g, b] = sampleStops(ramp, t);
    parts.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

/**
 * Build evenly-spaced Kelvin tick labels for a value range.
 * Note: the IR convention maps the *low* normalized end to the *high* BT (warm)
 * and the high normalized end to the *low* BT (cold), so the legend reads
 * cold -> warm left to right by reversing.
 */
export function legendTicks(rangeK: [number, number], count = 5): number[] {
  const [lo, hi] = rangeK;
  const ticks: number[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    ticks.push(Math.round(lerp(lo, hi, t)));
  }
  return ticks;
}

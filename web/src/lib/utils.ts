import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Tailwind-aware className combiner (shadcn convention). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Clamp a number to [min, max]. */
export function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

/** Format an ISO-8601 timestamp as a compact UTC readout, e.g. "25 Sep · 12:15Z". */
export function formatUTC(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const day = d.getUTCDate().toString().padStart(2, '0');
  const month = d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' });
  const hh = d.getUTCHours().toString().padStart(2, '0');
  const mm = d.getUTCMinutes().toString().padStart(2, '0');
  return `${day} ${month} · ${hh}:${mm}Z`;
}

/** Format just the clock portion, e.g. "12:15:00Z". */
export function formatClock(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const hh = d.getUTCHours().toString().padStart(2, '0');
  const mm = d.getUTCMinutes().toString().padStart(2, '0');
  const ss = d.getUTCSeconds().toString().padStart(2, '0');
  return `${hh}:${mm}:${ss}Z`;
}

/** Minutes elapsed between two ISO timestamps. */
export function minutesBetween(a: string, b: string): number {
  return (new Date(b).getTime() - new Date(a).getTime()) / 60000;
}

/** Round to n decimal places and return a number. */
export function round(v: number, n = 2): number {
  const f = 10 ** n;
  return Math.round(v * f) / f;
}

/** Format a number with fixed decimals, falling back to "—" for nullish/NaN. */
export function fmt(v: number | null | undefined, decimals = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(decimals);
}

/** Resolve a frame-relative asset path against a scene base URL. */
export function sceneAsset(base: string, rel: string): string {
  return `${base.replace(/\/$/, '')}/${rel.replace(/^\//, '')}`;
}

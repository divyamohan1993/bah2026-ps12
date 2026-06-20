import type { Manifest, SceneIndex } from '@/types/manifest';

/** Base path where scene data lives (relative to the deployed app root). */
export const DATA_ROOT = 'data';

export class ManifestError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = 'ManifestError';
  }
}

/** Build the public URL for a scene's directory (honors Vite BASE_URL). */
export function sceneBaseUrl(sceneId: string): string {
  const base = import.meta.env.BASE_URL || '/';
  return `${base.replace(/\/$/, '')}/${DATA_ROOT}/${sceneId}`;
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/**
 * Validate that a parsed JSON blob conforms to the manifest contract.
 * Throws ManifestError with a precise message on the first violation.
 */
export function validateManifest(data: unknown): asserts data is Manifest {
  if (typeof data !== 'object' || data === null) {
    throw new ManifestError('Manifest is not a JSON object.');
  }
  const m = data as Record<string, unknown>;

  const requireStr = (k: string) => {
    if (typeof m[k] !== 'string' || (m[k] as string).length === 0) {
      throw new ManifestError(`Manifest field "${k}" must be a non-empty string.`);
    }
  };
  const requireNum = (k: string) => {
    if (!isFiniteNumber(m[k])) {
      throw new ManifestError(`Manifest field "${k}" must be a finite number.`);
    }
  };

  requireStr('scene_id');
  requireStr('title');
  requireStr('satellite');
  requireStr('channel');
  requireNum('wavelength_um');
  requireNum('tile_size');

  if (!Array.isArray(m.bbox) || m.bbox.length !== 4 || !m.bbox.every(isFiniteNumber)) {
    throw new ManifestError('Manifest "bbox" must be [west, south, east, north].');
  }
  if (
    !Array.isArray(m.value_range_k) ||
    m.value_range_k.length !== 2 ||
    !m.value_range_k.every(isFiniteNumber)
  ) {
    throw new ManifestError('Manifest "value_range_k" must be [min, max].');
  }

  if (!Array.isArray(m.frames) || m.frames.length === 0) {
    throw new ManifestError('Manifest "frames" must be a non-empty array.');
  }
  m.frames.forEach((f, i) => {
    const fr = f as Record<string, unknown>;
    if (!isFiniteNumber(fr.index)) {
      throw new ManifestError(`frames[${i}].index must be a number.`);
    }
    if (typeof fr.time !== 'string') {
      throw new ManifestError(`frames[${i}].time must be an ISO-8601 string.`);
    }
    if (fr.kind !== 'observed' && fr.kind !== 'interpolated') {
      throw new ManifestError(`frames[${i}].kind must be "observed" or "interpolated".`);
    }
    if (typeof fr.image !== 'string') {
      throw new ManifestError(`frames[${i}].image must be a string path.`);
    }
  });

  if (typeof m.metrics !== 'object' || m.metrics === null) {
    throw new ManifestError('Manifest "metrics" object is missing.');
  }
  const metrics = m.metrics as Record<string, unknown>;
  if (!Array.isArray(metrics.per_frame)) {
    throw new ManifestError('Manifest "metrics.per_frame" must be an array.');
  }
}

/** Fetch and validate a scene manifest. */
export async function loadManifest(
  sceneId: string,
  signal?: AbortSignal,
): Promise<Manifest> {
  const url = `${sceneBaseUrl(sceneId)}/manifest.json`;
  let res: Response;
  try {
    res = await fetch(url, { signal, cache: 'no-cache' });
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    throw new ManifestError(`Network error loading manifest for "${sceneId}".`, err);
  }
  if (!res.ok) {
    throw new ManifestError(
      `Failed to load manifest for "${sceneId}" (HTTP ${res.status}). Did you run \`npm run mock\`?`,
    );
  }
  let json: unknown;
  try {
    json = await res.json();
  } catch (err) {
    throw new ManifestError(`Manifest for "${sceneId}" is not valid JSON.`, err);
  }
  validateManifest(json);
  return json;
}

/** Load the scene index (list of available scenes) with a graceful fallback. */
export async function loadSceneIndex(signal?: AbortSignal): Promise<SceneIndex> {
  const base = import.meta.env.BASE_URL || '/';
  const url = `${base.replace(/\/$/, '')}/${DATA_ROOT}/scenes.json`;
  try {
    const res = await fetch(url, { signal, cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = (await res.json()) as SceneIndex;
    if (!json || !Array.isArray(json.scenes) || json.scenes.length === 0) {
      throw new Error('empty scene index');
    }
    return json;
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    // Fallback: assume the canonical demo scenes exist.
    return {
      scenes: [
        { scene_id: 'cyclone-atlantic', title: 'GOES-19 · Atlantic Cyclone', satellite: 'GOES-19' },
        { scene_id: 'synthetic-demo', title: 'Synthetic · Advecting Blob', satellite: 'FrameFlow-Synthetic' },
      ],
    };
  }
}

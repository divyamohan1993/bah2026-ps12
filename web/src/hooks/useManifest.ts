import { useEffect, useState } from 'react';
import type { Manifest } from '@/types/manifest';
import { loadManifest, ManifestError, sceneBaseUrl } from '@/lib/manifest';

export interface ManifestState {
  manifest: Manifest | null;
  loading: boolean;
  error: string | null;
  /** Public base URL for the scene's assets. */
  sceneBase: string;
}

/** Load (and reload on scene change) a scene manifest with abort handling. */
export function useManifest(sceneId: string): ManifestState {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    setManifest(null);

    loadManifest(sceneId, ac.signal)
      .then((m) => {
        setManifest(m);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if ((err as Error)?.name === 'AbortError') return;
        const msg =
          err instanceof ManifestError
            ? err.message
            : `Unexpected error loading scene "${sceneId}".`;
        setError(msg);
        setLoading(false);
      });

    return () => ac.abort();
  }, [sceneId]);

  return { manifest, loading, error, sceneBase: sceneBaseUrl(sceneId) };
}

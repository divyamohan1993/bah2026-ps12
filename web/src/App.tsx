import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import type { ManifestFrame, SceneIndexEntry } from '@/types/manifest';
import { loadSceneIndex } from '@/lib/manifest';
import { useManifest } from '@/hooks/useManifest';
import { usePlayback } from '@/hooks/usePlayback';
import { actions, useStore } from '@/hooks/useStore';
import { Header } from '@/components/Header';
import { ComparePanes } from '@/components/ComparePanes';
import { Timeline } from '@/components/Timeline';
import { MetricStrip } from '@/components/MetricStrip';
import { LayerToggles } from '@/components/LayerToggles';
import { ColorbarLegend } from '@/components/ColorbarLegend';
import { SummaryPanel } from '@/components/SummaryPanel';
import { VideoModal } from '@/components/VideoModal';
import { AboutPanel } from '@/components/AboutPanel';
import { LoadingState, ErrorState } from '@/components/States';
import { Panel } from '@/components/ui/Panel';

export default function App() {
  const sceneId = useStore((s) => s.sceneId);
  const frameIndex = useStore((s) => s.frameIndex);

  const [scenes, setScenes] = useState<SceneIndexEntry[]>([]);
  const { manifest, loading, error } = useManifest(sceneId);

  // Load scene index once.
  useEffect(() => {
    const ac = new AbortController();
    loadSceneIndex(ac.signal)
      .then((idx) => {
        setScenes(idx.scenes);
        // If the default scene isn't in the index, switch to the first one.
        if (idx.scenes.length && !idx.scenes.some((s) => s.scene_id === sceneId)) {
          actions.setScene(idx.scenes[0].scene_id);
        }
      })
      .catch(() => void 0);
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const frameCount = manifest?.frames.length ?? 0;
  usePlayback(frameCount);

  // Resolve the active frame + the "ground truth" frame for the left pane.
  const { activeFrame, gtFrame } = useMemo(() => {
    if (!manifest) return { activeFrame: null, gtFrame: null };
    const idx = Math.min(frameIndex, manifest.frames.length - 1);
    const f = manifest.frames[idx];
    // Left pane shows real data: if f is observed, use it; otherwise the lower
    // bracket observed frame (fallback to nearest observed by index).
    let gt = f;
    if (f.kind === 'interpolated') {
      const lowerIdx = f.bracket?.[0];
      const found =
        lowerIdx != null ? manifest.frames.find((fr) => fr.index === lowerIdx) : undefined;
      gt = found ?? findNearestObserved(manifest.frames, idx) ?? f;
    }
    return { activeFrame: f, gtFrame: gt };
  }, [manifest, frameIndex]);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-space-900">
      {/* subtle grid + radial glow backdrop */}
      <div className="pointer-events-none fixed inset-0 bg-grid bg-grid opacity-40" />
      <div className="pointer-events-none fixed inset-0 bg-radial-fade" />

      <Header
        scenes={scenes}
        sceneId={sceneId}
        onSceneChange={(id) => actions.setScene(id)}
        manifest={manifest}
      />

      <main className="relative z-10 flex min-h-0 flex-1 flex-col gap-3 p-3">
        {error ? (
          <ErrorState message={error} onRetry={() => actions.setScene(sceneId)} />
        ) : loading || !manifest || !activeFrame || !gtFrame ? (
          <LoadingState />
        ) : (
          <>
            {/* Upper region: map compare + right rail */}
            <div className="flex min-h-0 flex-1 gap-3">
              <motion.div
                className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-line shadow-panel"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                <ComparePanes
                  key={manifest.scene_id}
                  manifest={manifest}
                  sceneBase={`${import.meta.env.BASE_URL.replace(/\/$/, '')}/data/${manifest.scene_id}`}
                  frame={activeFrame}
                  gtFrame={gtFrame}
                />
              </motion.div>

              {/* Right rail */}
              <motion.aside
                className="hidden w-[300px] shrink-0 flex-col gap-3 overflow-y-auto lg:flex"
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.05 }}
              >
                <LayerToggles flowAvailable={!!activeFrame.flow_overlay} />
                <ColorbarLegend manifest={manifest} />
                <SummaryPanel manifest={manifest} />
              </motion.aside>
            </div>

            {/* Bottom dock: timeline + metric strip */}
            <motion.div
              className="grid shrink-0 grid-cols-1 gap-3 xl:grid-cols-[1fr_minmax(360px,460px)]"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.08 }}
            >
              <Panel className="px-4 py-3">
                <Timeline manifest={manifest} />
              </Panel>
              <div className="hidden xl:block">
                <MetricStrip manifest={manifest} />
              </div>
            </motion.div>

            {/* Modals / overlays */}
            <VideoModal
              manifest={manifest}
              sceneBase={`${import.meta.env.BASE_URL.replace(/\/$/, '')}/data/${manifest.scene_id}`}
            />
            <AboutPanel manifest={manifest} />
          </>
        )}
      </main>
    </div>
  );
}

/** Find the nearest observed frame at or before `idx`, else after. */
function findNearestObserved(
  frames: ManifestFrame[],
  idx: number,
): ManifestFrame | undefined {
  for (let i = idx; i >= 0; i--) if (frames[i].kind === 'observed') return frames[i];
  for (let i = idx + 1; i < frames.length; i++) if (frames[i].kind === 'observed') return frames[i];
  return undefined;
}

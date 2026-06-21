import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { X, Film, AlertTriangle } from 'lucide-react';
import type { Manifest } from '@/types/manifest';
import { actions, useStore } from '@/hooks/useStore';
import { sceneAsset } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';

interface VideoModalProps {
  manifest: Manifest;
  sceneBase: string;
}

type Which = 'side_by_side' | 'interpolated' | 'observed';

/**
 * Hero "play" mode — plays the precomputed MP4 via an HTML5 <video>.
 *
 * ACCESSIBILITY (AccessLint WCAG 1.2.2): the <video> ALWAYS includes a
 * <track kind="captions" ... default> pointing at captions.vtt. Degrades
 * gracefully with an inline notice if the MP4 is absent (mock data ships no
 * binary video).
 */
export function VideoModal({ manifest, sceneBase }: VideoModalProps) {
  const open = useStore((s) => s.videoOpen);
  const [which, setWhich] = useState<Which>('side_by_side');
  const [errored, setErrored] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Reset error state when switching source or reopening.
  useEffect(() => {
    setErrored(false);
  }, [which, open]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') actions.setVideoOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const src = sceneAsset(sceneBase, manifest.videos[which]);
  const captionsSrc = sceneAsset(sceneBase, 'captions.vtt');

  const tabs: { key: Which; label: string }[] = [
    { key: 'side_by_side', label: 'Side-by-side' },
    { key: 'interpolated', label: 'Interpolated' },
    { key: 'observed', label: 'Ground truth' },
  ];

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          role="dialog"
          aria-modal="true"
          aria-label="Hero playback"
        >
          <div
            className="absolute inset-0 bg-space-950/80 backdrop-blur-sm"
            onClick={() => actions.setVideoOpen(false)}
          />
          <motion.div
            className="relative z-10 w-full max-w-4xl overflow-hidden rounded-2xl border border-line bg-space-850 shadow-panel"
            initial={{ scale: 0.96, y: 10 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.97, y: 8 }}
            transition={{ type: 'spring', stiffness: 320, damping: 28 }}
          >
            <div className="flex items-center justify-between border-b border-line px-4 py-3">
              <div className="flex items-center gap-2">
                <Film className="h-4 w-4 text-cyan" />
                <h2 className="text-sm font-semibold text-ink">{manifest.title} · hero loop</h2>
                <Badge tone="amber">all-intra MP4 · O(1) seek</Badge>
              </div>
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label="Close video"
                onClick={() => actions.setVideoOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Source tabs */}
            <div className="flex items-center gap-1.5 px-4 pt-3">
              {tabs.map((t) => (
                <Button
                  key={t.key}
                  size="sm"
                  variant="outline"
                  active={which === t.key}
                  aria-pressed={which === t.key}
                  onClick={() => setWhich(t.key)}
                >
                  {t.label}
                </Button>
              ))}
            </div>

            {/* Video / fallback */}
            <div className="p-4">
              <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-line bg-space-950">
                {!errored ? (
                  <video
                    key={src}
                    ref={videoRef}
                    className="h-full w-full"
                    src={src}
                    controls
                    autoPlay
                    loop
                    muted
                    playsInline
                    aria-label={`${manifest.title} ${which.replace('_', ' ')} thermal-IR animation`}
                    poster={sceneAsset(sceneBase, manifest.frames[0]?.image ?? '')}
                    onError={() => setErrored(true)}
                  >
                    {/* WCAG 1.2.2 captions track — always present, default on. */}
                    <track
                      kind="captions"
                      srcLang="en"
                      label="English"
                      src={captionsSrc}
                      default
                    />
                    Your browser does not support the video tag.
                  </video>
                ) : (
                  <FallbackNotice posterFrames={manifest} sceneBase={sceneBase} />
                )}
              </div>
              <p className="mt-3 text-xs leading-relaxed text-ink-faint">
                Cinematic playback uses the precomputed all-intra MP4 (every frame
                a keyframe → exact O(1) seek). The scrub/compare view above uses
                per-frame rasters for precision. Captions are provided for
                accessibility.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function FallbackNotice({
  posterFrames,
  sceneBase,
}: {
  posterFrames: Manifest;
  sceneBase: string;
}) {
  const poster = sceneAsset(sceneBase, posterFrames.frames[0]?.image ?? '');
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
      {poster && (
        <img
          src={poster}
          alt="First frame of the thermal-IR animation"
          className="absolute inset-0 h-full w-full object-cover opacity-25"
        />
      )}
      <div className="relative flex flex-col items-center gap-2">
        <AlertTriangle className="h-7 w-7 text-amber" />
        <div className="text-sm font-semibold text-ink">Hero MP4 not bundled</div>
        <p className="max-w-md text-xs leading-relaxed text-ink-dim">
          The mock dataset ships frames + metrics but no binary video. The real
          precompute pipeline (<span className="font-mono">frameflow.serve.video</span>)
          writes all-intra MP4/WebM here. Use the scrubbable compare view for the
          full animation.
        </p>
      </div>
    </div>
  );
}

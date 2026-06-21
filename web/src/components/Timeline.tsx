import { useCallback, useEffect, useMemo, useRef } from 'react';
import { Pause, Play, SkipBack, SkipForward, Repeat, Gauge } from 'lucide-react';
import type { Manifest } from '@/types/manifest';
import { actions, useStore } from '@/hooks/useStore';
import { stepFrame } from '@/hooks/usePlayback';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { cn, formatClock, formatUTC } from '@/lib/utils';

interface TimelineProps {
  manifest: Manifest;
}

const SPEEDS = [0.5, 1, 2, 4];

/**
 * Unified scrubbable timeline. Drives the current frame index for both map
 * panes and the metric cursor. Frame ticks mark observed (solid) vs
 * interpolated (hollow) frames. Fully keyboard operable (Arrow keys / Space).
 */
export function Timeline({ manifest }: TimelineProps) {
  const frames = manifest.frames;
  const count = frames.length;

  const frameIndex = useStore((s) => s.frameIndex);
  const isPlaying = useStore((s) => s.isPlaying);
  const speed = useStore((s) => s.speed);
  const loop = useStore((s) => s.loop);

  const current = frames[frameIndex] ?? frames[0];
  const trackRef = useRef<HTMLDivElement>(null);

  // ----- Keyboard: arrows step, space toggles play -----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      // Ignore when typing in inputs / selects.
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;

      switch (e.key) {
        case 'ArrowRight':
          e.preventDefault();
          stepFrame(e.shiftKey ? 5 : 1, count);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          stepFrame(e.shiftKey ? -5 : -1, count);
          break;
        case 'Home':
          e.preventDefault();
          actions.setFrame(0);
          break;
        case 'End':
          e.preventDefault();
          actions.setFrame(count - 1);
          break;
        case ' ':
        case 'k':
          e.preventDefault();
          actions.togglePlay();
          break;
        case 'l':
          actions.toggleLoop();
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [count]);

  // ----- Pointer scrubbing on the track -----
  const scrubToClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el || count <= 1) return;
      const rect = el.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      const idx = Math.round(ratio * (count - 1));
      actions.setFrame(idx);
    },
    [count],
  );

  const onTrackPointerDown = useCallback(
    (e: React.PointerEvent) => {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      actions.pause();
      scrubToClientX(e.clientX);
    },
    [scrubToClientX],
  );

  const onTrackPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (e.buttons === 1) scrubToClientX(e.clientX);
    },
    [scrubToClientX],
  );

  const progressPct = count > 1 ? (frameIndex / (count - 1)) * 100 : 0;

  const ticks = useMemo(
    () =>
      frames.map((f, i) => ({
        i,
        left: count > 1 ? (i / (count - 1)) * 100 : 0,
        observed: f.kind === 'observed',
      })),
    [frames, count],
  );

  const timeStart = frames[0]?.time;
  const timeEnd = frames[count - 1]?.time;

  return (
    <div className="flex flex-col gap-3">
      {/* Top row: transport controls + readout */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5">
          <Button
            size="icon"
            variant="subtle"
            aria-label="Jump to first frame"
            onClick={() => actions.setFrame(0)}
          >
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="subtle"
            aria-label="Step back one frame"
            onClick={() => stepFrame(-1, count)}
          >
            <span className="text-base leading-none">‹</span>
          </Button>
          <Button
            size="lg"
            variant="accent"
            className="w-[88px]"
            aria-label={isPlaying ? 'Pause animation' : 'Play animation'}
            aria-pressed={isPlaying}
            onClick={() => actions.togglePlay()}
          >
            {isPlaying ? (
              <>
                <Pause className="h-4 w-4" /> Pause
              </>
            ) : (
              <>
                <Play className="h-4 w-4" /> Play
              </>
            )}
          </Button>
          <Button
            size="icon"
            variant="subtle"
            aria-label="Step forward one frame"
            onClick={() => stepFrame(1, count)}
          >
            <span className="text-base leading-none">›</span>
          </Button>
          <Button
            size="icon"
            variant="subtle"
            aria-label="Jump to last frame"
            onClick={() => actions.setFrame(count - 1)}
          >
            <SkipForward className="h-4 w-4" />
          </Button>

          <div className="mx-1 h-6 w-px bg-line" />

          <Button
            size="icon"
            variant="subtle"
            active={loop}
            aria-label="Toggle loop"
            aria-pressed={loop}
            onClick={() => actions.toggleLoop()}
          >
            <Repeat className="h-4 w-4" />
          </Button>

          {/* Speed control */}
          <div
            className="ml-1 flex items-center gap-1 rounded-lg border border-line bg-space-800/60 p-0.5"
            role="group"
            aria-label="Playback speed"
          >
            <Gauge className="ml-1 mr-0.5 h-3.5 w-3.5 text-ink-faint" />
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => actions.setSpeed(s)}
                aria-label={`Set speed ${s}×`}
                aria-pressed={speed === s}
                className={cn(
                  'rounded-md px-2 py-1 text-xs font-semibold tnum transition-colors',
                  speed === s
                    ? 'bg-cyan-deep/25 text-cyan-soft'
                    : 'text-ink-dim hover:text-ink',
                )}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>

        {/* Readout */}
        <div className="flex items-center gap-3">
          {current.kind === 'interpolated' ? (
            <Badge tone="violet" dot>
              INTERPOLATED · t={current.t?.toFixed(2)}
            </Badge>
          ) : (
            <Badge tone="cyan" dot>
              OBSERVED
            </Badge>
          )}
          <div className="text-right">
            <div className="font-mono text-lg font-semibold leading-none text-ink tnum">
              {formatClock(current.time)}
            </div>
            <div className="mt-0.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint tnum">
              {formatUTC(current.time)} · frame {frameIndex + 1}/{count}
            </div>
          </div>
        </div>
      </div>

      {/* Scrubber track */}
      <div className="px-1">
        <div
          ref={trackRef}
          role="slider"
          tabIndex={0}
          aria-label="Timeline scrubber"
          aria-valuemin={0}
          aria-valuemax={count - 1}
          aria-valuenow={frameIndex}
          aria-valuetext={`Frame ${frameIndex + 1} of ${count}, ${formatUTC(current.time)}`}
          onPointerDown={onTrackPointerDown}
          onPointerMove={onTrackPointerMove}
          onKeyDown={(e) => {
            if (e.key === 'ArrowRight') {
              e.preventDefault();
              stepFrame(1, count);
            } else if (e.key === 'ArrowLeft') {
              e.preventDefault();
              stepFrame(-1, count);
            }
          }}
          className={cn(
            'group relative h-10 cursor-pointer touch-none select-none rounded-lg',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan/50',
          )}
        >
          {/* base rail */}
          <div className="absolute left-0 right-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-space-700" />
          {/* progress fill */}
          <div
            className="absolute left-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-gradient-to-r from-cyan-deep to-cyan"
            style={{ width: `${progressPct}%` }}
          />
          {/* frame ticks */}
          {ticks.map((t) => (
            <span
              key={t.i}
              className={cn(
                'absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border transition-colors',
                t.observed
                  ? 'h-2.5 w-2.5 border-cyan bg-cyan'
                  : 'h-2 w-2 border-violet bg-space-900',
                t.i === frameIndex && 'ring-2 ring-white/70',
              )}
              style={{ left: `${t.left}%` }}
              title={`${t.observed ? 'Observed' : 'Interpolated'} frame ${t.i + 1}`}
            />
          ))}
          {/* playhead */}
          <div
            className="pointer-events-none absolute top-1/2 z-10 h-7 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.6)]"
            style={{ left: `${progressPct}%` }}
          />
        </div>

        {/* axis labels + legend */}
        <div className="mt-1.5 flex items-center justify-between text-[10px] font-medium uppercase tracking-wider text-ink-faint">
          <span className="tnum">{timeStart ? formatUTC(timeStart) : ''}</span>
          <div className="flex items-center gap-3 normal-case tracking-normal">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan" /> observed
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full border border-violet bg-space-900" />{' '}
              interpolated
            </span>
          </div>
          <span className="tnum">{timeEnd ? formatUTC(timeEnd) : ''}</span>
        </div>
      </div>
    </div>
  );
}

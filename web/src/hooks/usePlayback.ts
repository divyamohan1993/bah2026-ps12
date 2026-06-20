import { useEffect, useRef } from 'react';
import { getState, setState, useStore } from '@/hooks/useStore';

/** Base interval between frames at 1× speed (ms). */
const BASE_MS = 650;

/**
 * Drives frame advancement when `isPlaying`. Uses a rAF loop with accumulated
 * elapsed time so that speed changes are smooth and we never advance faster than
 * the display refresh. Wraps or stops at the end depending on `loop`.
 */
export function usePlayback(frameCount: number): void {
  const isPlaying = useStore((s) => s.isPlaying);
  const speed = useStore((s) => s.speed);
  const loop = useStore((s) => s.loop);

  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number>(0);
  const accRef = useRef<number>(0);

  useEffect(() => {
    if (!isPlaying || frameCount <= 1) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }

    const interval = BASE_MS / Math.max(0.1, speed);
    lastTsRef.current = performance.now();
    accRef.current = 0;

    const tick = (ts: number) => {
      const dt = ts - lastTsRef.current;
      lastTsRef.current = ts;
      accRef.current += dt;

      if (accRef.current >= interval) {
        const steps = Math.floor(accRef.current / interval);
        accRef.current -= steps * interval;
        advanceFrames(steps, frameCount, loop);
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [isPlaying, speed, loop, frameCount]);
}

/** Advance the current frame by `steps`, honoring loop/stop semantics. */
function advanceFrames(steps: number, frameCount: number, loop: boolean): void {
  const cur = getState().frameIndex;
  let next = cur + steps;
  if (next >= frameCount) {
    if (loop) {
      next = next % frameCount;
    } else {
      setState({ frameIndex: frameCount - 1, isPlaying: false });
      return;
    }
  }
  setState({ frameIndex: next });
}

/** Step the frame by `delta`, clamped, and pause playback. */
export function stepFrame(delta: number, frameCount: number): void {
  const cur = getState().frameIndex;
  const next = Math.min(frameCount - 1, Math.max(0, cur + delta));
  setState({ frameIndex: next, isPlaying: false });
}

import { useSyncExternalStore } from 'react';

/**
 * Minimal external store (Zustand-flavored API) built on useSyncExternalStore.
 * Avoids an extra dependency while giving us a single source of truth for the
 * dashboard's UI state: current frame, playback, active layers, etc.
 */

export type CompareMode = 'swipe' | 'side-by-side';
export type MetricKey = 'psnr' | 'ssim' | 'ms_ssim';

export interface AppState {
  /** Selected scene id. */
  sceneId: string;
  /** Active frame index into manifest.frames (0-based array position). */
  frameIndex: number;
  /** Playback. */
  isPlaying: boolean;
  /** Playback speed multiplier (frames advance every BASE_MS / speed). */
  speed: number;
  /** Loop playback. */
  loop: boolean;
  /** Swipe vs two-pane. */
  compareMode: CompareMode;
  /** Layer toggles. */
  showInterp: boolean;
  showGt: boolean;
  showFlow: boolean;
  showError: boolean;
  /** Error/diff layer opacity [0,1]. */
  errorOpacity: number;
  /** Which metric series the uPlot strip emphasizes / which numbers show. */
  activeMetrics: Record<MetricKey, boolean>;
  /** Hero video modal open. */
  videoOpen: boolean;
  /** About / methodology panel open. */
  aboutOpen: boolean;
}

const initialState: AppState = {
  // Default to the REAL precomputed scene embedded under public/data/demo-0001
  // (genuine IFNet output from `make demo` + scripts/embed_demo_scene.py). The mock
  // scenes (cyclone-atlantic, synthetic-demo) are only present after `npm run mock`.
  sceneId: 'demo-0001',
  frameIndex: 0,
  isPlaying: false,
  speed: 1,
  loop: true,
  compareMode: 'swipe',
  showInterp: true,
  showGt: true,
  showFlow: false,
  showError: false,
  errorOpacity: 0.65,
  activeMetrics: { psnr: true, ssim: true, ms_ssim: false },
  videoOpen: false,
  aboutOpen: false,
};

type Listener = () => void;

let state: AppState = initialState;
const listeners = new Set<Listener>();

function emit() {
  for (const l of listeners) l();
}

export function setState(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)) {
  const next = typeof patch === 'function' ? patch(state) : patch;
  state = { ...state, ...next };
  emit();
}

export function getState(): AppState {
  return state;
}

function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Subscribe to a derived slice of state. */
export function useStore<T>(selector: (s: AppState) => T): T {
  return useSyncExternalStore(
    subscribe,
    () => selector(state),
    () => selector(initialState),
  );
}

// ---- Action helpers (kept colocated for discoverability) ----

export const actions = {
  setScene(sceneId: string) {
    setState({ sceneId, frameIndex: 0, isPlaying: false });
  },
  setFrame(frameIndex: number) {
    setState({ frameIndex });
  },
  play() {
    setState({ isPlaying: true });
  },
  pause() {
    setState({ isPlaying: false });
  },
  togglePlay() {
    setState((s) => ({ isPlaying: !s.isPlaying }));
  },
  setSpeed(speed: number) {
    setState({ speed });
  },
  toggleLoop() {
    setState((s) => ({ loop: !s.loop }));
  },
  setCompareMode(compareMode: CompareMode) {
    setState({ compareMode });
  },
  toggleLayer(key: 'showInterp' | 'showGt' | 'showFlow' | 'showError') {
    setState((s) => ({ [key]: !s[key] }) as Partial<AppState>);
  },
  setErrorOpacity(errorOpacity: number) {
    setState({ errorOpacity });
  },
  toggleMetric(key: MetricKey) {
    setState((s) => ({
      activeMetrics: { ...s.activeMetrics, [key]: !s.activeMetrics[key] },
    }));
  },
  setVideoOpen(videoOpen: boolean) {
    setState({ videoOpen });
  },
  setAboutOpen(aboutOpen: boolean) {
    setState({ aboutOpen });
  },
};

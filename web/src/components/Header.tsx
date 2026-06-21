import { Clapperboard, Info, Radio, Waypoints } from 'lucide-react';
import type { Manifest, SceneIndexEntry } from '@/types/manifest';
import { actions } from '@/hooks/useStore';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { SceneSelector } from '@/components/SceneSelector';

interface HeaderProps {
  scenes: SceneIndexEntry[];
  sceneId: string;
  onSceneChange: (id: string) => void;
  manifest: Manifest | null;
}

export function Header({ scenes, sceneId, onSceneChange, manifest }: HeaderProps) {
  const factor = manifest?.interpolation_factor ?? 2;
  const cin = manifest?.cadence_minutes_input ?? 30;
  const cout = manifest?.cadence_minutes_output ?? 15;

  return (
    <header className="relative z-20 flex flex-wrap items-center justify-between gap-3 border-b border-line bg-space-900/80 px-4 py-2.5 backdrop-blur-md">
      {/* Left: brand + scene */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2.5">
          <Logo />
          <div className="leading-none">
            <div className="flex items-baseline gap-1.5">
              <span className="text-[15px] font-bold tracking-tight text-ink">FrameFlow</span>
              <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan/70">
                mission control
              </span>
            </div>
            <div className="mt-0.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
              ISRO BAH 2026 · PS-12 · temporal interpolation
            </div>
          </div>
        </div>

        <div className="hidden h-8 w-px bg-line md:block" />

        <SceneSelector scenes={scenes} value={sceneId} onChange={onSceneChange} />
      </div>

      {/* Center-right: telemetry badges */}
      <div className="flex flex-wrap items-center gap-2">
        {manifest && (
          <>
            <Badge tone="cyan" dot>
              <Radio className="h-3 w-3" />
              {manifest.satellite}
            </Badge>
            <Badge tone="neutral">
              {manifest.channel} · {manifest.wavelength_um.toFixed(1)} µm
            </Badge>
            <Badge tone="amber">
              <Waypoints className="h-3 w-3" />
              {cin} min → {cout} min · {factor}× via optical-flow AI
            </Badge>
          </>
        )}

        <div className="mx-0.5 hidden h-6 w-px bg-line sm:block" />

        <Button
          size="md"
          variant="default"
          onClick={() => actions.setVideoOpen(true)}
          aria-label="Open hero playback video"
        >
          <Clapperboard className="h-4 w-4" />
          <span className="hidden sm:inline">Hero loop</span>
        </Button>
        <Button
          size="icon"
          variant="subtle"
          onClick={() => actions.setAboutOpen(true)}
          aria-label="About and metrics methodology"
        >
          <Info className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}

function Logo() {
  return (
    <div className="relative flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-deep/40 bg-space-800">
      <div className="absolute inset-0 rounded-lg bg-radial-fade" />
      <svg viewBox="0 0 32 32" className="relative h-6 w-6" aria-hidden="true">
        <circle cx="16" cy="16" r="10" fill="none" stroke="#22d3ee" strokeWidth="1.4" opacity="0.5" />
        <circle cx="16" cy="16" r="6" fill="none" stroke="#22d3ee" strokeWidth="1.4" />
        <circle cx="16" cy="16" r="1.8" fill="#f59e0b" />
        <path
          d="M16 3v4M16 25v4M3 16h4M25 16h4"
          stroke="#22d3ee"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

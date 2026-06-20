import { Layers, Wind, Grid3x3, Satellite, Sparkles, SplitSquareHorizontal, Columns2 } from 'lucide-react';
import { actions, useStore } from '@/hooks/useStore';
import { Panel, PanelHeader } from '@/components/ui/Panel';
import { Switch } from '@/components/ui/Switch';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';

interface LayerTogglesProps {
  /** Whether the active frame has a flow overlay available. */
  flowAvailable: boolean;
}

export function LayerToggles({ flowAvailable }: LayerTogglesProps) {
  const showGt = useStore((s) => s.showGt);
  const showInterp = useStore((s) => s.showInterp);
  const showFlow = useStore((s) => s.showFlow);
  const showError = useStore((s) => s.showError);
  const errorOpacity = useStore((s) => s.errorOpacity);
  const compareMode = useStore((s) => s.compareMode);

  return (
    <Panel className="flex flex-col">
      <PanelHeader title="Layers & overlays" icon={<Layers className="h-3.5 w-3.5" />} />

      <div className="flex flex-col gap-0.5 p-2">
        <ToggleRow
          icon={<Satellite className="h-4 w-4 text-cyan/80" />}
          label="Ground-truth raster"
          checked={showGt}
          onChange={() => actions.toggleLayer('showGt')}
          tone="cyan"
        />
        <ToggleRow
          icon={<Sparkles className="h-4 w-4 text-violet/80" />}
          label="Interpolated raster"
          checked={showInterp}
          onChange={() => actions.toggleLayer('showInterp')}
          tone="violet"
        />
        <ToggleRow
          icon={<Grid3x3 className="h-4 w-4 text-amber/80" />}
          label="Error heatmap"
          sub="|GT − interp|"
          checked={showError}
          onChange={() => actions.toggleLayer('showError')}
          tone="amber"
        />
        {showError && (
          <div className="px-2 pb-1.5 pl-10 pt-0.5">
            <label className="flex items-center gap-2 text-[10px] text-ink-faint">
              <span className="w-10 uppercase tracking-wider">Opacity</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={errorOpacity}
                aria-label="Error heatmap opacity"
                onChange={(e) => actions.setErrorOpacity(Number(e.target.value))}
                className="ff-range flex-1"
              />
              <span className="w-8 text-right font-mono tnum text-ink-dim">
                {Math.round(errorOpacity * 100)}%
              </span>
            </label>
          </div>
        )}
        <ToggleRow
          icon={<Wind className="h-4 w-4 text-cyan/80" />}
          label="Optical-flow vectors"
          sub={flowAvailable ? 'available this frame' : 'not on this frame'}
          checked={showFlow}
          onChange={() => actions.toggleLayer('showFlow')}
          tone="cyan"
          disabled={!flowAvailable}
        />
      </div>

      {/* Compare mode switch */}
      <div className="border-t border-line/70 p-2">
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
          Compare mode
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          <Button
            size="sm"
            variant="outline"
            active={compareMode === 'swipe'}
            aria-pressed={compareMode === 'swipe'}
            onClick={() => actions.setCompareMode('swipe')}
            className="justify-center"
          >
            <SplitSquareHorizontal className="h-3.5 w-3.5" /> Swipe
          </Button>
          <Button
            size="sm"
            variant="outline"
            active={compareMode === 'side-by-side'}
            aria-pressed={compareMode === 'side-by-side'}
            onClick={() => actions.setCompareMode('side-by-side')}
            className="justify-center"
          >
            <Columns2 className="h-3.5 w-3.5" /> Split
          </Button>
        </div>
      </div>
    </Panel>
  );
}

interface ToggleRowProps {
  icon: React.ReactNode;
  label: string;
  sub?: string;
  checked: boolean;
  onChange: () => void;
  tone: 'cyan' | 'amber' | 'violet';
  disabled?: boolean;
}

function ToggleRow({ icon, label, sub, checked, onChange, tone, disabled }: ToggleRowProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 transition-colors',
        disabled ? 'opacity-50' : 'hover:bg-space-800/60',
      )}
    >
      <div className="flex items-center gap-2.5">
        {icon}
        <div className="leading-tight">
          <div className="text-sm text-ink">{label}</div>
          {sub && <div className="text-[10px] text-ink-faint">{sub}</div>}
        </div>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} label={label} tone={tone} disabled={disabled} />
    </div>
  );
}

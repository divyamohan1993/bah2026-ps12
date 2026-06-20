import type { Manifest } from '@/types/manifest';
import { gradientCss, legendTicks, type ColormapName } from '@/lib/colormap';
import { Panel, PanelHeader } from '@/components/ui/Panel';
import { Thermometer } from 'lucide-react';

interface ColorbarLegendProps {
  manifest: Manifest;
}

/**
 * Brightness-temperature colorbar legend. The IR convention shows cold cloud
 * tops bright and warm surface dark; we annotate the Kelvin range from the
 * manifest. Reads cold (left) -> warm (right).
 */
export function ColorbarLegend({ manifest }: ColorbarLegendProps) {
  const name = (manifest.colormap as ColormapName) === 'inferno' ? 'inferno' : 'ir';
  const [lo, hi] = manifest.value_range_k;
  const ticks = legendTicks([lo, hi], 5);

  return (
    <Panel className="flex flex-col">
      <PanelHeader
        title="BT colormap"
        icon={<Thermometer className="h-3.5 w-3.5" />}
        right={
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            {manifest.colormap}
          </span>
        }
      />
      <div className="p-3">
        {/* Reverse the gradient so the bar reads cold (left) to warm (right). */}
        <div
          className="h-3 w-full rounded-full border border-line"
          style={{
            background: gradientCss(name, 32),
            transform: 'scaleX(-1)',
          }}
          role="img"
          aria-label={`Brightness temperature color scale from ${lo} to ${hi} Kelvin`}
        />
        <div className="mt-1.5 flex justify-between font-mono text-[10px] text-ink-dim tnum">
          {ticks.map((t) => (
            <span key={t}>{t}K</span>
          ))}
        </div>
        <div className="mt-1 flex justify-between text-[9px] font-medium uppercase tracking-wider text-ink-faint">
          <span>cold · cloud tops</span>
          <span>warm · surface</span>
        </div>
      </div>
    </Panel>
  );
}

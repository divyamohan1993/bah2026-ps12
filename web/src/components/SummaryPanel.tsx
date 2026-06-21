import { Award, TrendingUp, Minus, TrendingDown } from 'lucide-react';
import type { Manifest } from '@/types/manifest';
import { Panel, PanelHeader } from '@/components/ui/Panel';
import { Badge } from '@/components/ui/Badge';
import { cn, fmt } from '@/lib/utils';

interface SummaryPanelProps {
  manifest: Manifest;
}

/**
 * Aggregate metric summary + baseline comparison. Shows the model's mean PSNR /
 * SSIM / MS-SSIM / BT-RMSE and a small table of baselines (linear blend, TV-L1)
 * with delta badges vs the model.
 */
export function SummaryPanel({ manifest }: SummaryPanelProps) {
  const s = manifest.metrics.summary;
  const baselines = Object.values(manifest.metrics.baselines);
  const model = manifest.model;

  return (
    <Panel className="flex flex-col">
      <PanelHeader
        title="Model vs baselines"
        icon={<Award className="h-3.5 w-3.5" />}
        right={<Badge tone="violet">{model.name}</Badge>}
      />

      {/* Headline numbers */}
      <div className="grid grid-cols-2 gap-px border-b border-line/70 bg-line/40">
        <Stat label="Mean PSNR" value={`${fmt(s.psnr_mean, 2)} dB`} accent="#f59e0b" />
        <Stat label="Mean SSIM" value={fmt(s.ssim_mean, 3)} accent="#22d3ee" />
        <Stat label="MS-SSIM" value={fmt(s.ms_ssim_mean, 3)} accent="#a78bfa" />
        <Stat label="BT-RMSE" value={`${fmt(s.bt_rmse_k_mean, 2)} K`} accent="#34d399" />
      </div>

      {/* Baseline comparison table */}
      <div className="p-2">
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
          PSNR vs classical baselines
        </div>
        <div className="flex flex-col gap-1">
          <BaselineRow
            name={`${model.name} (ours)`}
            psnr={s.psnr_mean}
            ssim={s.ssim_mean}
            highlight
          />
          {baselines.map((b) => (
            <BaselineRow
              key={b.name}
              name={b.name}
              psnr={b.psnr}
              ssim={b.ssim}
              deltaPsnr={s.psnr_mean - b.psnr}
            />
          ))}
        </div>
        <p className="mt-2 px-1 text-[10px] leading-relaxed text-ink-faint">
          Higher PSNR / SSIM and lower BT-RMSE are better. Bar shows the prior-art
          target (Vandal &amp; Nemani 2021: 45.4 dB / 0.933 on GOES Band&nbsp;13).
        </p>
      </div>
    </Panel>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="flex flex-col gap-1 bg-space-850 px-3 py-2.5">
      <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        {label}
      </span>
      <span className="font-mono text-base font-semibold tnum" style={{ color: accent }}>
        {value}
      </span>
    </div>
  );
}

interface BaselineRowProps {
  name: string;
  psnr: number;
  ssim: number;
  deltaPsnr?: number;
  highlight?: boolean;
}

function BaselineRow({ name, psnr, ssim, deltaPsnr, highlight }: BaselineRowProps) {
  // Relative bar width vs a ~50 dB ceiling for visual scale.
  const pct = Math.min(100, (psnr / 50) * 100);
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-lg border px-2.5 py-1.5',
        highlight ? 'border-violet/40 bg-violet/5' : 'border-line bg-space-800/40',
      )}
    >
      <div
        className={cn(
          'absolute inset-y-0 left-0 -z-0 opacity-20',
          highlight ? 'bg-violet' : 'bg-cyan-deep',
        )}
        style={{ width: `${pct}%` }}
      />
      <div className="relative flex items-center justify-between gap-2">
        <span className={cn('truncate text-xs', highlight ? 'font-semibold text-ink' : 'text-ink-dim')}>
          {name}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          <span className="font-mono text-xs tnum text-ink">{fmt(psnr, 1)}</span>
          <span className="font-mono text-[10px] tnum text-ink-faint">{fmt(ssim, 2)}</span>
          {deltaPsnr !== undefined && <DeltaBadge value={deltaPsnr} />}
        </div>
      </div>
    </div>
  );
}

function DeltaBadge({ value }: { value: number }) {
  // value is (ours - baseline); positive means our model is better.
  const positive = value > 0.2;
  const negative = value < -0.2;
  const Icon = positive ? TrendingUp : negative ? TrendingDown : Minus;
  const tone = positive ? 'ok' : negative ? 'bad' : 'neutral';
  return (
    <Badge tone={tone} className="px-1.5">
      <Icon className="h-3 w-3" />
      <span className="tnum">
        {value >= 0 ? '+' : ''}
        {fmt(value, 1)}
      </span>
    </Badge>
  );
}

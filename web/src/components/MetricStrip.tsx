import { useEffect, useMemo, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { Activity } from 'lucide-react';
import type { Manifest, PerFrameMetric } from '@/types/manifest';
import { actions, useStore, type MetricKey } from '@/hooks/useStore';
import { Panel, PanelHeader } from '@/components/ui/Panel';
import { cn, fmt } from '@/lib/utils';

interface MetricStripProps {
  manifest: Manifest;
}

interface SeriesDef {
  key: MetricKey;
  label: string;
  stroke: string;
  /** Which y-axis scale this series binds to. */
  scale: 'snr' | 'unit';
}

const SERIES: SeriesDef[] = [
  { key: 'psnr', label: 'PSNR (dB)', stroke: '#f59e0b', scale: 'snr' },
  { key: 'ssim', label: 'SSIM', stroke: '#22d3ee', scale: 'unit' },
  { key: 'ms_ssim', label: 'MS-SSIM', stroke: '#a78bfa', scale: 'unit' },
];

/**
 * uPlot metric strip plotting PSNR / SSIM / MS-SSIM across interpolated frames,
 * with a vertical cursor synced to the current frame. Series visibility is
 * toggled from the store's `activeMetrics`.
 */
export function MetricStrip({ manifest }: MetricStripProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  const frameIndex = useStore((s) => s.frameIndex);
  const activeMetrics = useStore((s) => s.activeMetrics);

  const frames = manifest.frames;
  const perFrame = manifest.metrics.per_frame;

  // Map global frame index -> the metric record (interpolated frames only).
  const byIndex = useMemo(() => {
    const m = new Map<number, PerFrameMetric>();
    for (const p of perFrame) m.set(p.index, p);
    return m;
  }, [perFrame]);

  // Build uPlot data aligned to ALL frames on the x-axis (observed frames get
  // null for metric values so the line only spans interpolated frames).
  const data = useMemo<uPlot.AlignedData>(() => {
    const xs: number[] = [];
    const psnr: (number | null)[] = [];
    const ssim: (number | null)[] = [];
    const msssim: (number | null)[] = [];
    frames.forEach((f, i) => {
      xs.push(i);
      const p = byIndex.get(f.index);
      psnr.push(p ? p.psnr : null);
      ssim.push(p ? p.ssim : null);
      msssim.push(p ? p.ms_ssim : null);
    });
    return [xs, psnr, ssim, msssim];
  }, [frames, byIndex]);

  // Create the plot once.
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;

    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height: 150,
      padding: [8, 8, 0, 0],
      cursor: {
        x: true,
        y: false,
        points: { size: 6 },
        drag: { x: false, y: false },
      },
      legend: { show: false },
      scales: {
        x: { time: false },
        snr: { auto: true },
        unit: { range: [0.8, 1.0] },
      },
      axes: [
        {
          stroke: '#5b6779',
          grid: { stroke: 'rgba(30,42,61,0.7)', width: 1 },
          ticks: { stroke: 'rgba(30,42,61,0.7)' },
          font: '10px JetBrains Mono, monospace',
          values: (_u, vals) => vals.map((v) => `#${v + 1}`),
        },
        {
          scale: 'snr',
          stroke: '#f59e0b',
          grid: { show: false },
          ticks: { stroke: 'rgba(30,42,61,0.7)' },
          font: '10px JetBrains Mono, monospace',
          size: 42,
        },
        {
          scale: 'unit',
          side: 1,
          stroke: '#22d3ee',
          grid: { show: false },
          ticks: { stroke: 'rgba(30,42,61,0.7)' },
          font: '10px JetBrains Mono, monospace',
          size: 40,
        },
      ],
      series: [
        {},
        {
          label: 'PSNR',
          scale: 'snr',
          stroke: '#f59e0b',
          width: 2,
          points: { show: false },
          spanGaps: false,
        },
        {
          label: 'SSIM',
          scale: 'unit',
          stroke: '#22d3ee',
          width: 2,
          points: { show: false },
          spanGaps: false,
        },
        {
          label: 'MS-SSIM',
          scale: 'unit',
          stroke: '#a78bfa',
          width: 2,
          dash: [4, 3],
          points: { show: false },
          spanGaps: false,
        },
      ],
      hooks: {
        // Clicking the chart seeks the timeline to that frame.
        setSelect: [],
      },
    };

    const u = new uPlot(opts, data, el);
    plotRef.current = u;

    // Click-to-seek
    const onClick = () => {
      const idx = u.cursor.idx;
      if (idx != null) actions.setFrame(idx);
    };
    u.over.addEventListener('click', onClick);

    const ro = new ResizeObserver(() => {
      u.setSize({ width: el.clientWidth || 600, height: 150 });
    });
    ro.observe(el);

    return () => {
      u.over.removeEventListener('click', onClick);
      ro.disconnect();
      u.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push new data when scene changes.
  useEffect(() => {
    plotRef.current?.setData(data);
  }, [data]);

  // Toggle series visibility from store.
  useEffect(() => {
    const u = plotRef.current;
    if (!u) return;
    SERIES.forEach((s, i) => {
      u.setSeries(i + 1, { show: activeMetrics[s.key] });
    });
  }, [activeMetrics]);

  // Sync the cursor to the current frame.
  useEffect(() => {
    const u = plotRef.current;
    if (!u) return;
    const left = u.valToPos(frameIndex, 'x');
    u.setCursor({ left, top: u.bbox.height / window.devicePixelRatio / 2 });
  }, [frameIndex]);

  // Live numeric readout for the current frame.
  const currentFrame = frames[frameIndex];
  const currentMetric = currentFrame ? byIndex.get(currentFrame.index) : undefined;

  return (
    <Panel className="flex h-full flex-col">
      <PanelHeader
        title="Per-frame quality"
        icon={<Activity className="h-3.5 w-3.5" />}
        right={
          <div className="flex items-center gap-1">
            {SERIES.map((s) => (
              <button
                key={s.key}
                onClick={() => actions.toggleMetric(s.key)}
                aria-pressed={activeMetrics[s.key]}
                aria-label={`Toggle ${s.label} series`}
                className={cn(
                  'flex items-center gap-1.5 rounded-md px-1.5 py-1 text-[10px] font-semibold transition-opacity',
                  activeMetrics[s.key] ? 'opacity-100' : 'opacity-35 hover:opacity-70',
                )}
              >
                <span
                  className="h-2 w-2.5 rounded-sm"
                  style={{ background: s.stroke }}
                />
                {s.label}
              </button>
            ))}
          </div>
        }
      />

      <div className="flex-1 px-2 pb-1 pt-2">
        <div ref={containerRef} className="h-[150px] w-full" />
      </div>

      {/* Live readout for the current frame */}
      <div className="grid grid-cols-3 gap-px border-t border-line/70 bg-line/40">
        <Readout
          label="PSNR"
          value={currentMetric ? `${fmt(currentMetric.psnr, 1)} dB` : '—'}
          tone="#f59e0b"
        />
        <Readout
          label="SSIM"
          value={currentMetric ? fmt(currentMetric.ssim, 3) : '—'}
          tone="#22d3ee"
        />
        <Readout
          label="BT-RMSE"
          value={currentMetric ? `${fmt(currentMetric.bt_rmse_k, 2)} K` : '—'}
          tone="#a78bfa"
        />
      </div>
    </Panel>
  );
}

function Readout({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 bg-space-850 px-2 py-2">
      <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        {label}
      </span>
      <span className="font-mono text-sm font-semibold tnum" style={{ color: tone }}>
        {value}
      </span>
    </div>
  );
}

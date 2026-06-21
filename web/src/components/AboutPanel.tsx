import { AnimatePresence, motion } from 'framer-motion';
import { X, BookOpen, CheckCircle2, AlertCircle, CircleDashed } from 'lucide-react';
import type { Manifest } from '@/types/manifest';
import { actions, useStore } from '@/hooks/useStore';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { cn, fmt } from '@/lib/utils';

interface AboutPanelProps {
  manifest: Manifest;
}

/** Slide-over "about / metrics methodology" panel. */
export function AboutPanel({ manifest }: AboutPanelProps) {
  const open = useStore((s) => s.aboutOpen);
  const methods = manifest.crossval.methods_run;

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-space-950/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => actions.setAboutOpen(false)}
          />
          <motion.aside
            className="fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l border-line bg-space-850 shadow-panel"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            role="dialog"
            aria-modal="true"
            aria-label="About and metrics methodology"
          >
            <div className="flex items-center justify-between border-b border-line px-4 py-3">
              <div className="flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-cyan" />
                <h2 className="text-sm font-semibold text-ink">Methodology &amp; provenance</h2>
              </div>
              <Button size="icon-sm" variant="ghost" aria-label="Close panel" onClick={() => actions.setAboutOpen(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex-1 space-y-5 overflow-y-auto p-4">
              <Section title="What you're seeing">
                <p>
                  FrameFlow boosts the temporal resolution of geostationary
                  Thermal-IR imagery using an optical-flow AI frame-interpolation
                  model. The left pane is real{' '}
                  <strong className="text-ink">ground truth</strong>; the right pane
                  is{' '}
                  <strong className="text-violet-soft">AI-interpolated</strong>{' '}
                  intermediate frames synthesized between observed acquisitions.
                </p>
              </Section>

              <Section title="Model">
                <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 font-mono text-xs">
                  <Row k="Name" v={manifest.model.name} />
                  <Row k="Version" v={manifest.model.version} />
                  <Row k="Params" v={`${fmt(manifest.model.params_m, 1)} M`} />
                  <Row k="Factor" v={`${manifest.interpolation_factor}×`} />
                  <Row k="Input cadence" v={`${manifest.cadence_minutes_input} min`} />
                  <Row k="Output cadence" v={`${manifest.cadence_minutes_output} min`} />
                </dl>
              </Section>

              <Section title="Metrics glossary">
                <ul className="space-y-1.5 text-xs">
                  <Gloss term="PSNR" desc="Peak signal-to-noise ratio (dB). Higher = closer to truth." />
                  <Gloss term="SSIM / MS-SSIM" desc="Structural similarity [0,1]. Captures perceived structure." />
                  <Gloss term="FSIM" desc="Feature similarity using phase congruency + gradient." />
                  <Gloss term="GMSD" desc="Gradient magnitude similarity deviation. Lower = better." />
                  <Gloss term="BT-RMSE / bias" desc="Brightness-temperature error in Kelvin (physical units)." />
                  <Gloss term="CSI@235K" desc="Critical success index for cold-cloud (deep convection) detection." />
                  <Gloss term="FSS" desc="Fractions skill score — neighborhood nowcasting skill." />
                  <Gloss term="EPE" desc="Optical-flow end-point error (px) vs reference motion." />
                </ul>
              </Section>

              <Section title="Cross-validation">
                <p className="mb-2">
                  Quality is triangulated across multiple methods, never a single
                  metric or sensor.
                </p>
                <div className="space-y-1.5">
                  {methods.map((m) => (
                    <div
                      key={m.id}
                      className="flex items-center justify-between gap-2 rounded-lg border border-line bg-space-800/50 px-2.5 py-1.5"
                    >
                      <div className="flex items-center gap-2">
                        <StatusIcon status={m.status} />
                        <span className="text-xs text-ink">{m.name}</span>
                      </div>
                      <StatusBadge status={m.status} />
                    </div>
                  ))}
                </div>
              </Section>

              <Section title="The O(1) pitch">
                <p>
                  Interpolation is precomputed and content-addressed; the dashboard
                  serves each frame from a CDN as a constant-time HTTP range request
                  — no model runs while you watch, and every result is cross-checked
                  against multiple satellites.
                </p>
              </Section>

              <div className="border-t border-line pt-3 text-[10px] font-mono text-ink-faint">
                generated {manifest.generated} · manifest v{manifest.version} · scene{' '}
                {manifest.scene_id}
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-cyan/80">
        {title}
      </h3>
      <div className="text-xs leading-relaxed text-ink-dim">{children}</div>
    </section>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-ink-faint">{k}</dt>
      <dd className="text-right text-ink">{v}</dd>
    </>
  );
}

function Gloss({ term, desc }: { term: string; desc: string }) {
  return (
    <li className="flex gap-2">
      <span className="shrink-0 font-mono font-semibold text-ink">{term}</span>
      <span className="text-ink-faint">— {desc}</span>
    </li>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'pass') return <CheckCircle2 className="h-3.5 w-3.5 text-ok" />;
  if (status === 'warn') return <AlertCircle className="h-3.5 w-3.5 text-amber" />;
  if (status === 'fail') return <AlertCircle className="h-3.5 w-3.5 text-bad" />;
  return <CircleDashed className="h-3.5 w-3.5 text-ink-faint" />;
}

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === 'pass' ? 'ok' : status === 'warn' ? 'amber' : status === 'fail' ? 'bad' : 'neutral';
  return (
    <Badge tone={tone as 'ok' | 'amber' | 'bad' | 'neutral'} className={cn('uppercase')}>
      {status}
    </Badge>
  );
}

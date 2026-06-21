import { motion } from 'framer-motion';
import { SatelliteDish, ServerCrash, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/Button';

/** Full-bleed loading state with a scanning animation. */
export function LoadingState({ label = 'Acquiring scene…' }: { label?: string }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-5 bg-space-900">
      <div className="relative h-20 w-20">
        <div className="absolute inset-0 animate-ping rounded-full border border-cyan/30" />
        <div className="absolute inset-2 rounded-full border border-cyan/40" />
        <div className="absolute inset-0 flex items-center justify-center">
          <SatelliteDish className="h-8 w-8 animate-pulseSoft text-cyan" />
        </div>
      </div>
      <div className="text-center">
        <div className="text-sm font-medium text-ink">{label}</div>
        <div className="mt-1 font-mono text-[11px] uppercase tracking-wider text-ink-faint">
          establishing downlink
        </div>
      </div>
      <div className="relative h-1 w-48 overflow-hidden rounded-full bg-space-700">
        <div className="absolute inset-y-0 w-1/3 animate-sweep rounded-full bg-gradient-to-r from-transparent via-cyan to-transparent" />
      </div>
    </div>
  );
}

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

/** Full-bleed error/empty state shown when the manifest fails to load. */
export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex h-full w-full items-center justify-center bg-space-900 p-6">
      <motion.div
        className="max-w-md rounded-2xl border border-bad/30 bg-space-850 p-6 text-center shadow-panel"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full border border-bad/40 bg-bad/10">
          <ServerCrash className="h-7 w-7 text-bad" />
        </div>
        <h2 className="text-base font-semibold text-ink">Telemetry unavailable</h2>
        <p className="mt-2 text-sm leading-relaxed text-ink-dim">{message}</p>
        <div className="mt-4 rounded-lg border border-line bg-space-800/60 p-3 text-left">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-ink-faint">
            Generate the demo data
          </div>
          <code className="block font-mono text-xs text-cyan-soft">npm run mock</code>
        </div>
        {onRetry && (
          <Button variant="accent" className="mt-4" onClick={onRetry}>
            <RefreshCw className="h-4 w-4" /> Retry
          </Button>
        )}
      </motion.div>
    </div>
  );
}

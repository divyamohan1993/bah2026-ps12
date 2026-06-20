import { cn } from '@/lib/utils';

interface PaneLabelProps {
  side: 'left' | 'right';
  title: string;
  sub: string;
  tone: 'cyan' | 'violet';
}

/** Floating corner label identifying each compare pane. */
export function PaneLabel({ side, title, sub, tone }: PaneLabelProps) {
  return (
    <div
      className={cn(
        'pointer-events-none absolute top-3 z-10 select-none',
        side === 'left' ? 'left-3' : 'right-14',
      )}
    >
      <div className="flex items-center gap-2 rounded-lg border border-line bg-space-900/75 px-2.5 py-1.5 backdrop-blur-md">
        <span
          className={cn(
            'h-2 w-2 rounded-full',
            tone === 'cyan' ? 'bg-cyan shadow-[0_0_8px_#22d3ee]' : 'bg-violet shadow-[0_0_8px_#a78bfa]',
          )}
        />
        <div className="leading-tight">
          <div className="text-xs font-semibold text-ink">{title}</div>
          <div className="text-[10px] font-medium uppercase tracking-wider text-ink-faint tnum">
            {sub}
          </div>
        </div>
      </div>
    </div>
  );
}

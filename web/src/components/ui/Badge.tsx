import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

type Tone = 'cyan' | 'amber' | 'violet' | 'neutral' | 'ok' | 'bad';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
  dot?: boolean;
}

const toneClasses: Record<Tone, string> = {
  cyan: 'bg-cyan-deep/15 text-cyan-soft border-cyan-deep/40',
  amber: 'bg-amber/10 text-amber-soft border-amber/40',
  violet: 'bg-violet/10 text-violet-soft border-violet/40',
  neutral: 'bg-space-700 text-ink-dim border-line',
  ok: 'bg-ok/10 text-ok border-ok/40',
  bad: 'bg-bad/10 text-bad border-bad/40',
};

const dotColors: Record<Tone, string> = {
  cyan: 'bg-cyan',
  amber: 'bg-amber',
  violet: 'bg-violet',
  neutral: 'bg-ink-faint',
  ok: 'bg-ok',
  bad: 'bg-bad',
};

export function Badge({ className, tone = 'neutral', dot, children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-none tracking-wide',
        toneClasses[tone],
        className,
      )}
      {...props}
    >
      {dot && <span className={cn('h-1.5 w-1.5 rounded-full', dotColors[tone])} />}
      {children}
    </span>
  );
}

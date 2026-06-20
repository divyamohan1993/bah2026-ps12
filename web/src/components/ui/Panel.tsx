import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface PanelProps extends HTMLAttributes<HTMLDivElement> {
  /** Glassy translucent style for overlays on the map. */
  glass?: boolean;
}

/** A bordered "instrument" panel with the mission-control look. */
export function Panel({ className, glass, children, ...props }: PanelProps) {
  return (
    <div
      className={cn(
        'rounded-xl border border-line shadow-panel shadow-inset',
        glass
          ? 'bg-space-900/70 backdrop-blur-md supports-[backdrop-filter]:bg-space-900/55'
          : 'bg-space-850',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export interface PanelHeaderProps {
  title: ReactNode;
  icon?: ReactNode;
  right?: ReactNode;
  className?: string;
}

export function PanelHeader({ title, icon, right, className }: PanelHeaderProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-between gap-2 border-b border-line/70 px-3 py-2',
        className,
      )}
    >
      <div className="flex items-center gap-2">
        {icon && <span className="text-cyan/80">{icon}</span>}
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-dim">
          {title}
        </h2>
      </div>
      {right}
    </div>
  );
}

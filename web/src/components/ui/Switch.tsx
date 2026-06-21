import { cn } from '@/lib/utils';

export interface SwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: string;
  /** Accent color when on. */
  tone?: 'cyan' | 'amber' | 'violet';
  id?: string;
  disabled?: boolean;
}

const onTrack: Record<NonNullable<SwitchProps['tone']>, string> = {
  cyan: 'bg-cyan-deep/70',
  amber: 'bg-amber/70',
  violet: 'bg-violet/70',
};

const onRing: Record<NonNullable<SwitchProps['tone']>, string> = {
  cyan: 'shadow-[0_0_10px_rgba(34,211,238,0.5)]',
  amber: 'shadow-[0_0_10px_rgba(245,158,11,0.5)]',
  violet: 'shadow-[0_0_10px_rgba(167,139,250,0.5)]',
};

/** Accessible toggle switch (role=switch, keyboard + aria). */
export function Switch({
  checked,
  onCheckedChange,
  label,
  tone = 'cyan',
  id,
  disabled,
}: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      id={id}
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onCheckedChange(!checked)}
      className={cn(
        'relative inline-flex h-[18px] w-8 shrink-0 items-center rounded-full border border-line transition-colors duration-200',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan/60 focus-visible:ring-offset-1 focus-visible:ring-offset-space-900',
        'disabled:opacity-40',
        checked ? onTrack[tone] : 'bg-space-700',
      )}
    >
      <span
        className={cn(
          'inline-block h-3 w-3 transform rounded-full bg-ink transition-transform duration-200',
          checked ? 'translate-x-[15px]' : 'translate-x-[3px]',
          checked && onRing[tone],
        )}
      />
    </button>
  );
}

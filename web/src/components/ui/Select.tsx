import { ChevronDown } from 'lucide-react';
import type { SelectHTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  options: SelectOption[];
  onValueChange: (value: string) => void;
  'aria-label': string;
}

/** Styled native <select> (keeps native a11y + keyboard behavior). */
export function Select({ options, onValueChange, className, value, ...props }: SelectProps) {
  return (
    <div className="relative inline-flex items-center">
      <select
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        className={cn(
          'h-9 appearance-none rounded-lg border border-line bg-space-800 pl-3 pr-8 text-sm text-ink',
          'cursor-pointer transition-colors hover:border-cyan-deep/60',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan/60 focus-visible:ring-offset-1 focus-visible:ring-offset-space-900',
          className,
        )}
        {...props}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} className="bg-space-800 text-ink">
            {o.label}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2.5 h-4 w-4 text-ink-dim" />
    </div>
  );
}

import { forwardRef } from 'react';
import type { ButtonHTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

type Variant = 'default' | 'ghost' | 'outline' | 'accent' | 'subtle';
type Size = 'sm' | 'md' | 'lg' | 'icon' | 'icon-sm';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  active?: boolean;
}

const variantClasses: Record<Variant, string> = {
  default:
    'bg-space-700 text-ink hover:bg-space-600 border border-line hover:border-cyan-deep/60',
  subtle: 'bg-space-800/60 text-ink-dim hover:text-ink hover:bg-space-700 border border-transparent',
  ghost: 'bg-transparent text-ink-dim hover:text-ink hover:bg-space-700/70',
  outline: 'bg-transparent text-ink border border-line hover:border-cyan-deep/60 hover:text-cyan-soft',
  accent:
    'bg-cyan-deep/20 text-cyan-soft border border-cyan-deep/50 hover:bg-cyan-deep/30 shadow-glow',
};

const sizeClasses: Record<Size, string> = {
  sm: 'h-7 px-2.5 text-xs gap-1.5 rounded-md',
  md: 'h-9 px-3.5 text-sm gap-2 rounded-lg',
  lg: 'h-11 px-5 text-sm gap-2 rounded-lg',
  icon: 'h-9 w-9 rounded-lg',
  'icon-sm': 'h-7 w-7 rounded-md',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'md', active, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex select-none items-center justify-center font-medium transition-all duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan/60 focus-visible:ring-offset-1 focus-visible:ring-offset-space-900',
          'disabled:cursor-not-allowed disabled:opacity-40',
          variantClasses[variant],
          sizeClasses[size],
          active && 'ring-1 ring-cyan/60 text-cyan-soft border-cyan-deep/60 bg-cyan-deep/15',
          className,
        )}
        {...props}
      />
    );
  },
);
Button.displayName = 'Button';

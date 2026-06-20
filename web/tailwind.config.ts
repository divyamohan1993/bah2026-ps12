import type { Config } from 'tailwindcss';

/**
 * FrameFlow "mission-control" dark theme.
 * Palette: deep space-blue backgrounds, cyan primary, amber telemetry accent.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Background ladder (darkest -> lightest panel)
        space: {
          950: '#05070d',
          900: '#0a0e14',
          850: '#0d121b',
          800: '#111826',
          700: '#18202f',
          600: '#1f2937',
          500: '#283447',
        },
        // Primary accent — instrument cyan
        cyan: {
          DEFAULT: '#22d3ee',
          soft: '#67e8f9',
          deep: '#0891b2',
        },
        // Secondary accent — telemetry amber
        amber: {
          DEFAULT: '#f59e0b',
          soft: '#fbbf24',
        },
        // Tertiary — interpolated/synthetic violet
        violet: {
          DEFAULT: '#a78bfa',
          soft: '#c4b5fd',
        },
        ok: '#34d399',
        warn: '#f59e0b',
        bad: '#f87171',
        ink: {
          DEFAULT: '#e6edf6',
          dim: '#9aa7ba',
          faint: '#5b6779',
        },
        line: '#1e2a3d',
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          'JetBrains Mono',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(34,211,238,0.18), 0 8px 30px -8px rgba(34,211,238,0.25)',
        panel: '0 10px 40px -12px rgba(0,0,0,0.7)',
        inset: 'inset 0 1px 0 0 rgba(255,255,255,0.04)',
      },
      backgroundImage: {
        grid: 'linear-gradient(rgba(34,211,238,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.05) 1px, transparent 1px)',
        'radial-fade':
          'radial-gradient(ellipse 80% 60% at 50% -10%, rgba(34,211,238,0.10), transparent 60%)',
      },
      backgroundSize: {
        grid: '40px 40px',
      },
      keyframes: {
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
        sweep: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        pulseSoft: 'pulseSoft 2s ease-in-out infinite',
        sweep: 'sweep 1.6s ease-in-out infinite',
        fadeUp: 'fadeUp 0.35s ease-out both',
      },
    },
  },
  plugins: [],
} satisfies Config;

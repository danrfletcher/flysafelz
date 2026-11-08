import { animate, type Transition } from 'motion/react'

/**
 * The motion layer's one config (spec §1/§2). Every enter animation in the app takes its values and timing from `ENTER`; nothing else sets a
 * duration, a spring or a distance. Each kind has a full-motion variant and a reduced one (E12): reduced is opacity only, with no transform key
 * at all, and lasts no longer than a short fade. Nothing here reads the system setting: `enterFor` is told it (src/hooks/useEnter.ts does).
 */
type Values = Record<string, number | number[]>

/** What one element does as it appears. `initial` is where it starts; `animate` is where it ends, or its keyframes when there is no `initial`. */
export interface Enter {
  initial?: Values
  animate: Values
  transition: Transition
}

export type EnterKind = 'pin' | 'card' | 'sheet' | 'drawer' | 'progress' | 'badge'

// A short spring, 0.15 to 0.25 s to its visible end, with a small bounce.
const SPRING = { type: 'spring', visualDuration: 0.2, bounce: 0.15 } as const
// Opacity follows its own fade inside a spring, so a bounce never pushes it past 1.
const FADE = { duration: 0.15, ease: 'easeOut' } as const
const SPRING_AND_FADE = { ...SPRING, opacity: FADE } satisfies Transition

// The pin's drop: it falls 18 px, touches down, rises 4 px and settles, in 0.25 s, then stays still.
const PIN_BOUNCE: Transition = { duration: 0.25, times: [0, 0.5, 0.75, 1], ease: 'easeOut' }

const FADE_IN: Enter = { initial: { opacity: 0 }, animate: { opacity: 1 }, transition: FADE }

export const ENTER: Record<EnterKind, { full: Enter; reduced: Enter }> = {
  pin: {
    full: { animate: { y: [-18, 0, -4, 0] }, transition: PIN_BOUNCE },
    reduced: { animate: { opacity: [0, 1] }, transition: FADE },
  },
  card: {
    full: { initial: { opacity: 0, y: 8, scale: 0.96 }, animate: { opacity: 1, y: 0, scale: 1 }, transition: SPRING_AND_FADE },
    reduced: FADE_IN,
  },
  sheet: {
    full: { initial: { opacity: 0, x: 32 }, animate: { opacity: 1, x: 0 }, transition: SPRING_AND_FADE },
    reduced: FADE_IN,
  },
  drawer: {
    full: { initial: { opacity: 0, y: 40 }, animate: { opacity: 1, y: 0 }, transition: SPRING_AND_FADE },
    reduced: FADE_IN,
  },
  // A progress line only fades in, with or without reduced motion.
  progress: { full: FADE_IN, reduced: FADE_IN },
  badge: {
    full: { initial: { opacity: 0, scale: 0.92 }, animate: { opacity: 1, scale: 1 }, transition: SPRING_AND_FADE },
    reduced: FADE_IN,
  },
}

/** The variant of one kind for the visitor's setting. */
export function enterFor(kind: EnterKind, reduced: boolean): Enter {
  return ENTER[kind][reduced ? 'reduced' : 'full']
}

/**
 * Plays one enter animation on an element Leaflet or Radix made (a motion component is given `initial`, `animate` and `transition` instead).
 * Each value goes from `initial` to `animate`, or follows its keyframes. Returns the controls, or undefined when there is no element.
 */
export function playEnter(element: Element | null | undefined, enter: Enter) {
  if (!element) return undefined
  const target = Object.fromEntries(Object.entries(enter.animate).map(([key, end]) => [key, enter.initial && typeof end === 'number' ? [enter.initial[key] as number, end] : end]))
  return animate(element, target, enter.transition)
}

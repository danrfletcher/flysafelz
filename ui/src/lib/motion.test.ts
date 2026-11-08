import { describe, expect, it } from 'vitest'

import { ENTER, enterFor, type Enter, type EnterKind } from '@/lib/motion'

const KINDS = Object.keys(ENTER) as EnterKind[]
const TRANSFORM_KEYS = ['scale', 'x', 'y', 'rotate']

/** Every number or keyframe list in a transition that is a duration or spring setting, wherever it sits (top level or per property). */
const transitions = (enter: Enter): Record<string, unknown>[] => {
  const top = enter.transition as Record<string, unknown>
  const nested = Object.values(top).filter((value): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value))
  return [top, ...nested]
}
const keysOf = (enter: Enter) => [...Object.keys(enter.initial ?? {}), ...Object.keys(enter.animate)]

describe('the shared motion config', () => {
  it('has an entry for the pin, card, sheet, drawer, progress line and badge', () => {
    expect(KINDS.sort()).toEqual(['badge', 'card', 'drawer', 'pin', 'progress', 'sheet'])
  })

  it('makes every spring a short one: visualDuration 0.15 to 0.25 s and a bounce of at most 0.25', () => {
    const springs = KINDS.flatMap((kind) => transitions(ENTER[kind].full)).filter((transition) => transition.type === 'spring')
    expect(springs.length).toBeGreaterThan(0)
    for (const spring of springs) {
      expect(spring.visualDuration as number).toBeGreaterThanOrEqual(0.15)
      expect(spring.visualDuration as number).toBeLessThanOrEqual(0.25)
      expect(spring.bounce as number).toBeLessThanOrEqual(0.25)
    }
  })

  it('has the card, sheet, drawer and badge spring, and the pin and progress line do not', () => {
    const isSpring = (kind: EnterKind) => (ENTER[kind].full.transition as { type?: string }).type === 'spring'
    expect(KINDS.filter(isSpring).sort()).toEqual(['badge', 'card', 'drawer', 'sheet'])
  })

  it('keeps every fade and the pin drop to 0.25 s or less', () => {
    for (const kind of KINDS) {
      for (const enter of [ENTER[kind].full, ENTER[kind].reduced]) {
        for (const transition of transitions(enter)) {
          if (typeof transition.duration === 'number') expect(transition.duration, kind).toBeLessThanOrEqual(0.25)
        }
      }
    }
  })

  it('has a reduced variant for each animation with no scale, x, y or rotate key, and a fade of 0.15 s or less', () => {
    for (const kind of KINDS) {
      const { reduced } = ENTER[kind]
      for (const key of TRANSFORM_KEYS) expect(keysOf(reduced), `${kind}: ${key}`).not.toContain(key)
      expect(reduced.transition, kind).toEqual(expect.objectContaining({ duration: expect.any(Number) }))
      expect((reduced.transition as { duration: number }).duration, kind).toBeLessThanOrEqual(0.15)
      expect((reduced.transition as { type?: string }).type, kind).toBeUndefined()
    }
  })

  it('has a full variant that moves something, for every kind but the progress line (opacity only in both)', () => {
    for (const kind of KINDS.filter((name) => name !== 'progress')) {
      expect(keysOf(ENTER[kind].full).some((key) => TRANSFORM_KEYS.includes(key)), kind).toBe(true)
    }
    expect(keysOf(ENTER.progress.full)).toEqual(['opacity', 'opacity'])
  })

  it('starts the pin on the way down and ends it where it started: one drop, back at rest', () => {
    const keyframes = ENTER.pin.full.animate.y as number[]
    expect(keyframes[0]).toBeLessThan(0)
    expect(keyframes.at(-1)).toBe(0)
  })

  it('chooses the variant by the setting', () => {
    for (const kind of KINDS) {
      expect(enterFor(kind, true)).toBe(ENTER[kind].reduced)
      expect(enterFor(kind, false)).toBe(ENTER[kind].full)
    }
  })
})

/**
 * The one place a verdict badge's colour is decided (Q2, F7). It reads the segment's machine values and nothing else: never the `display`
 * text, the answer's words or a tier. The result names a theme colour; the classes that paint it are in Badges.tsx.
 */
export type BadgeColour = 'no-go' | 'unverified' | 'amber' | 'conditions' | 'discretion' | 'none'

export interface BadgeColourInput {
  /** The server's verdict value (NO_GO, UNVERIFIED, PILOT_DISCRETION), or null when the segment has none. */
  verdict: string | null
  approaching_limits: boolean
  conditions: readonly string[]
}

/**
 * Most cautious first: NO-GO red, then UNVERIFIED grey, then approaching limits amber (with or without conditions), then conditions
 * blue, then pilot discretion green. No verdict (or a verdict this code does not know) is slate, "none".
 */
export function badgeColour({ verdict, approaching_limits, conditions }: BadgeColourInput): BadgeColour {
  if (verdict === 'NO_GO') return 'no-go'
  if (verdict === 'UNVERIFIED') return 'unverified'
  if (verdict !== 'PILOT_DISCRETION') return 'none'
  if (approaching_limits) return 'amber'
  if (conditions.length > 0) return 'conditions'
  return 'discretion'
}

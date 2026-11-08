import type { AirspaceProps, NotamProps } from '@/map/types'

// The What's-here card's floor order (C2). A floor is a number of feet; the card sorts by it, lowest first. Which rows go in the
// collapsed "Above FL100" group is not decided here: it is the map layers' own rule (belowFl100.ts), read into `aboveFl100` by the
// card's rows, so the card and the "below FL100" and "all heights" layers always agree for the same feature.

const FLIGHT_LEVEL = /^FL\s*(\d+)\b/i
const GROUND = /^(SFC|GND)\b/i

/**
 * An airspace's floor in feet: GND, SFC or a missing lower limit is 0; otherwise the route's `lower_ft` (a flight level is level x 100,
 * an altitude or a height above ground is the feet stated). When the route sends no feet, a flight level in the text is read as
 * level x 100, and anything else counts as 0 so that nothing low is hidden under the group.
 */
export function airspaceFloorFt(props: Pick<AirspaceProps, 'lower' | 'lower_ft'>): number {
  const text = typeof props.lower === 'string' ? props.lower.trim() : ''
  if (text === '' || GROUND.test(text)) return 0
  if (typeof props.lower_ft === 'number' && Number.isFinite(props.lower_ft)) return Math.max(0, props.lower_ft)
  const level = FLIGHT_LEVEL.exec(text)
  return level ? Number(level[1]) * 100 : 0
}

/** A NOTAM's floor in feet: its Q-line lower flight level x 100, or 0 when the Q-line has none. */
export function notamFloorFt(props: Pick<NotamProps, 'lower_fl'>): number {
  if (typeof props.lower_fl !== 'number' || !Number.isFinite(props.lower_fl)) return 0
  return Math.max(0, props.lower_fl * 100)
}

export interface Sortable {
  floorFt: number
  /** Whether the row sits in the "Above FL100" group: `!airspaceBelowFl100(...)` or `!notamBelowFl100(...)` of belowFl100.ts, the map layers' rule. */
  aboveFl100: boolean
  /** The name or NOTAM id: the tie-break. */
  name: string
  /** The last tie-break, so that two items with one floor and one name keep a fixed order. */
  key: string
}

const compareText = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

/** A new list sorted by floor, lowest first; equal floors by name or id, then by key. */
export function sortByFloor<T extends Sortable>(rows: readonly T[]): T[] {
  return [...rows].sort((a, b) => a.floorFt - b.floorFt || compareText(a.name, b.name) || compareText(a.key, b.key))
}

/** Sorted rows split into those below FL100 (shown) and those at or above it (the collapsed group, by `aboveFl100`), each keeping its order. */
export function splitAtFl100<T extends Sortable>(rows: readonly T[]): { below: T[]; above: T[] } {
  const sorted = sortByFloor(rows)
  return { below: sorted.filter((row) => !row.aboveFl100), above: sorted.filter((row) => row.aboveFl100) }
}

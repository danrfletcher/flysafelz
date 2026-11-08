import { FL100_FEET } from '@/map/belowFl100'
import type { AirspaceProps, NotamProps } from '@/map/types'

// The What's-here card's floor rule (C2). A floor is a number of feet; the card sorts by it, lowest first, and puts a floor at or
// above FL100 (10,000 ft) in a collapsed group. This is its own rule, not the base layers' "below FL100" filter (belowFl100.ts):
// that filter keeps every above-ground limit on the map, while here an above-ground limit counts for the feet it states.

export { FL100_FEET }

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

/** A floor of exactly FL100 is in the group above; 9,999 ft is not. */
export function isAboveFl100(floorFt: number): boolean {
  return floorFt >= FL100_FEET
}

export interface Sortable {
  floorFt: number
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

/** Sorted rows split into those below FL100 (shown) and those at or above it (the collapsed group), each keeping its order. */
export function splitAtFl100<T extends Sortable>(rows: readonly T[]): { below: T[]; above: T[] } {
  const sorted = sortByFloor(rows)
  return { below: sorted.filter((row) => !isAboveFl100(row.floorFt)), above: sorted.filter((row) => isAboveFl100(row.floorFt)) }
}

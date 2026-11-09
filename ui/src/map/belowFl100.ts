import type { AirspaceProps, NotamProps } from '@/map/types'

// The one "below FL100" rule. The "below FL100" and "all heights" map layers (layers.ts) and the What's-here card's "Above FL100" group
// (pick/whatsHere.ts) all ask these two functions, so for one feature the map and the card always agree.

/** FL100 in feet. "Below FL100" means the lower limit is under this; a floor exactly at it is not below. */
export const FL100_FEET = 10_000
export const FL100 = 100

const FLIGHT_LEVEL = /^FL\s*(\d+)\b/i
const GROUND = /^(SFC|GND)\b/i
const ABOVE_GROUND = /\bAGL\b/i

/**
 * Whether an airspace's lower limit is below FL100, from openAIP's limit as the route serves it.
 *
 *   missing lower limit, SFC, GND, or any height above ground (AGL, ft or m)   below: an above-ground floor is
 *       measured from the terrain and cannot be compared with a flight level, so it is kept on the map
 *   a flight level or ft AMSL      lower_ft (FL x 100, feet as given) under 10,000
 *
 * `lower_ft` is the number to compare. When the route sends none but the text is a flight level ("FL65"), that level is used.
 */
export function airspaceBelowFl100(props: Pick<AirspaceProps, 'lower' | 'lower_ft'>): boolean {
  const text = typeof props.lower === 'string' ? props.lower.trim() : ''
  if (text === '' || GROUND.test(text) || ABOVE_GROUND.test(text)) return true
  if (typeof props.lower_ft === 'number' && Number.isFinite(props.lower_ft)) return props.lower_ft < FL100_FEET
  const level = FLIGHT_LEVEL.exec(text)
  if (level) return Number(level[1]) < FL100
  return true // a limit the UI cannot read counts as below, so nothing low is hidden
}

/** Whether a NOTAM's Q-line lower flight level is below FL100 (`lower_fl < 100`); a missing one counts as below. */
export function notamBelowFl100(props: Pick<NotamProps, 'lower_fl'>): boolean {
  if (typeof props.lower_fl !== 'number' || !Number.isFinite(props.lower_fl)) return true
  return props.lower_fl < FL100
}

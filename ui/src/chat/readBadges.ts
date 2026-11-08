import { badgeColour, type BadgeColour } from '@/chat/badgeColour'
import type { Payload } from '@/chat/turns'

/** The words for a segment that has no verdict, by the server's `tier`. */
export const NO_VERDICT_LABEL = 'No verdict'
const TIER_TEXT: Record<string, string> = { OUTLOOK: 'Outlook', TOO_FAR: 'Too far ahead' }

/** One badge, ready to draw: a time span, the labels as text, the colour name and the conditions beneath. */
export interface Badge {
  /** "10:00–11:00 GMT", with the day in front when the assessment covers more than one day. */
  span: string
  colour: BadgeColour
  /** The server's `display` exactly as given (every label that applies), or "No verdict" when the segment has none. */
  label: string
  /** "Outlook" or "Too far ahead" for a segment past the verdict range; null otherwise. */
  tier: string | null
  conditions: string[]
}

const isRecord = (value: unknown): value is Payload => typeof value === 'object' && value !== null && !Array.isArray(value)
const LOCAL_TIME = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2})/
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

interface Moment {
  date: string
  day: string
  clock: string
}

/** A local ISO time ("2026-10-31T10:00:00+00:00") is already in the request's zone, so its date and clock are read off the text. */
function moment(iso: unknown): Moment | null {
  const found = typeof iso === 'string' ? LOCAL_TIME.exec(iso) : null
  if (!found) return null
  const [, year, month, day, clock] = found
  const weekday = WEEKDAYS[new Date(Date.UTC(Number(year), Number(month) - 1, Number(day))).getUTCDay()]
  return { date: `${year}-${month}-${day}`, day: `${weekday} ${Number(day)} ${MONTHS[Number(month) - 1]}`, clock }
}

interface Parsed {
  start: Moment
  end: Moment
  tzAbbr: string
  badge: Omit<Badge, 'span'>
}

function parse(segment: unknown): Parsed | null {
  if (!isRecord(segment)) return null
  const start = moment(segment.start)
  const end = moment(segment.end)
  if (!start || !end || typeof segment.tz_abbr !== 'string') return null
  const conditions = Array.isArray(segment.conditions) ? segment.conditions.filter((item): item is string => typeof item === 'string') : []
  const verdict = segment.tier === 'VERDICT' && typeof segment.verdict === 'string' ? segment.verdict : null
  const display = typeof segment.display === 'string' && verdict !== null ? segment.display : null
  const colour = badgeColour({ verdict, approaching_limits: segment.approaching_limits === true, conditions })
  const tier = typeof segment.tier === 'string' ? (TIER_TEXT[segment.tier] ?? null) : null
  return { start, end, tzAbbr: segment.tz_abbr, badge: { colour, label: display ?? NO_VERDICT_LABEL, tier, conditions } }
}

/**
 * The badges of one turn's `assessment`, one per segment, in the order sent. The values come from the segment's machine fields: the
 * colour from `badgeColour`, the labels from `display` as given. A segment that is not an object, or has no readable time, is left out.
 * Returns null when there is nothing to show (no assessment, or no segment), so the caller draws no row at all.
 */
export function readBadges(assessment: Payload | null): Badge[] | null {
  const segments = assessment && Array.isArray(assessment.segments) ? assessment.segments : []
  const parsed = segments.map(parse).filter((item): item is Parsed => item !== null)
  if (parsed.length === 0) return null
  const dates = new Set(parsed.flatMap((item) => [item.start.date, item.end.date]))
  const withDay = dates.size > 1
  return parsed.map(({ start, end, tzAbbr, badge }) => {
    const sameDay = start.date === end.date
    const from = withDay ? `${start.day} ${start.clock}` : start.clock
    const to = withDay && !sameDay ? `${end.day} ${end.clock}` : end.clock
    return { ...badge, span: `${from}–${to} ${tzAbbr}` }
  })
}

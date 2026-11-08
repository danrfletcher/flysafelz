import type { LayerStyle, ThemeToken } from '@/map/airspaceStyle'

// How an answer's highlighted shape is drawn: by the action word the server put on that overlay feature (a NOTAM's action from the
// action table, an airspace area's outcome). The base layers never use these; they are coloured by type alone (Q10).

interface ActionPaint {
  token: ThemeToken
  /** 0 is outline only. */
  fillOpacity: number
}

const NO_FLY: ActionPaint = { token: '--action-no-fly', fillOpacity: 0.2 }
const CONTACT_FIRST: ActionPaint = { token: '--action-contact-first', fillOpacity: 0 }
const AWARENESS: ActionPaint = { token: '--action-awareness', fillOpacity: 0 }

// The legend's five action colours (src/components/Legend.tsx). The same meaning under another server name shares a colour:
// NO_GO (an airspace area from the surface up) is no-fly, AWARENESS_MAP (a NOTAM floored above 10,000 ft) is awareness, PERMISSION and
// RADIO (airspace you must contact first) are contact-first, and an unread NOTAM (UNREAD_OVER_LZ, UNVERIFIED) is the grey of awareness.
const BY_ACTION: Record<string, ActionPaint> = {
  NO_FLY,
  NO_GO: NO_FLY,
  CONTACT_FIRST,
  PERMISSION: CONTACT_FIRST,
  RADIO: CONTACT_FIRST,
  CHECK_SUP: { token: '--action-check-sup', fillOpacity: 0 },
  AVOID_SECTOR: { token: '--action-avoid-sector', fillOpacity: 0 },
  AWARENESS,
  AWARENESS_MAP: AWARENESS,
  UNVERIFIED: AWARENESS,
  UNREAD_OVER_LZ: AWARENESS,
}

const UNKNOWN: ActionPaint = { token: '--type-other', fillOpacity: 0 }
const OUTLINE = 3
const FOCUSED_OUTLINE = 5
const FOCUSED_FILL = 0.3
const APPROX_DASH = '3 7'
const MAX_WARNED_LENGTH = 40

const warned = new Set<string>()

/** Forget the actions already warned about. For tests only. */
export function resetWarnedActions(): void {
  warned.clear()
}

/**
 * The outline of one highlighted shape. A known action gets its legend colour; a feature with no action, or one this code does not know,
 * is slate, and an unknown value is named once in `console.warn` (the value only, cut short; never the feature's text or id).
 * An approximate NOTAM shape is dashed. The shape a NOTAM-id click picked out is drawn heavier.
 */
export function actionStyle(action: string | null, approx: boolean, focused: boolean): LayerStyle {
  let paint = UNKNOWN
  if (action !== null && Object.hasOwn(BY_ACTION, action)) {
    paint = BY_ACTION[action]
  } else if (action !== null && !warned.has(action)) {
    warned.add(action)
    console.warn(`Unknown overlay action: ${JSON.stringify(action.slice(0, MAX_WARNED_LENGTH))}`)
  }
  return {
    token: paint.token,
    weight: focused ? FOCUSED_OUTLINE : OUTLINE,
    fillOpacity: focused ? Math.max(paint.fillOpacity, FOCUSED_FILL) : paint.fillOpacity,
    dashArray: approx ? APPROX_DASH : undefined,
  }
}

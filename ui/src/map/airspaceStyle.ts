import type { PathOptions } from 'leaflet'

/** The theme variables a map outline can use (src/styles/theme.css). Colour lives there and nowhere else. */
export type ThemeToken =
  | '--type-restricted'
  | '--type-danger'
  | '--type-controlled'
  | '--type-other'
  | '--type-notam'
  | '--action-no-fly'
  | '--action-contact-first'
  | '--action-check-sup'
  | '--action-avoid-sector'
  | '--action-awareness'
  | '--primary'
  | '--wind-light'
  | '--wind-moderate'
  | '--wind-strong'
  | '--wind-severe'
  | '--rain'

/** Every name in openAIP's CODES["type"] list (src/flysafelz/clients/openaip.py), in code order. tests/unit/test_ui_map_layers.py keeps the two equal. */
export const AIRSPACE_TYPES = [
  'other', 'restricted', 'danger', 'prohibited', 'ctr', 'tmz', 'rmz', 'tma', 'tra', 'tsa',
  'fir', 'uir', 'adiz', 'atz', 'matz', 'airway', 'mtr', 'alert', 'warning', 'protected',
  'htz', 'gliding', 'trp', 'tiz', 'tia', 'mta', 'cta', 'acc', 'sporting', 'overflight_restriction',
  'mrt', 'tfr', 'vfr_sector', 'fis_sector', 'lta', 'uta', 'mctr',
] as const

export interface LayerStyle {
  /** The theme variable for the outline and the fill. */
  token: ThemeToken
  weight: number
  /** 0 is outline only. */
  fillOpacity: number
  /** SVG dash pattern; undefined is a solid line. */
  dashArray?: string
}

const OUTLINE = 2
const LIGHT_FILL = 0.15

const RED: LayerStyle = { token: '--type-restricted', weight: OUTLINE, fillOpacity: LIGHT_FILL }
const ORANGE: LayerStyle = { token: '--type-danger', weight: OUTLINE, fillOpacity: 0 }
const SKY: LayerStyle = { token: '--type-controlled', weight: OUTLINE, fillOpacity: 0 }
const SLATE: LayerStyle = { token: '--type-other', weight: OUTLINE, fillOpacity: 0 }

// The whole rule (Q10), keyed by the exact `type` string and nothing else: no `action`, no name, no limit.
const BY_TYPE: Record<string, LayerStyle> = {
  restricted: RED,
  prohibited: RED,
  danger: ORANGE,
  ctr: SKY,
  cta: SKY,
  atz: SKY,
}

const warned = new Set<string>()

/** Forget the types already warned about. For tests only. */
export function resetWarnedTypes(): void {
  warned.clear()
}

/**
 * The outline and fill of one airspace type. Restricted and prohibited are red with a light fill, danger is orange, CTR, CTA and
 * ATZ are sky blue, and every other name in openAIP's list (other, tma, matz, rmz, tmz and the rest) is slate.
 * A string that is not in the list is slate too, and `console.warn` names it once (the value only, never the area's name).
 */
export function airspaceStyle(type: string): LayerStyle {
  if (Object.hasOwn(BY_TYPE, type)) return BY_TYPE[type]
  if (!(AIRSPACE_TYPES as readonly string[]).includes(type) && !warned.has(type)) {
    warned.add(type)
    console.warn(`Unknown airspace type: ${JSON.stringify(type)}`)
  }
  return SLATE
}

/** A NOTAM: violet and dashed; an approximate shape (the Q-circle) gets a finer dash pattern. Type only, no fill. */
export function notamStyle(approx: boolean): LayerStyle {
  return { token: '--type-notam', weight: OUTLINE, fillOpacity: 0, dashArray: approx ? '3 7' : '8 6' }
}

/** The colour a theme variable holds on the page (the :root value). Leaflet draws SVG with a colour string, so the variable is read here. */
export function themeColor(token: ThemeToken): string {
  return getComputedStyle(document.documentElement).getPropertyValue(token).trim()
}

/** Leaflet path options for a style. An outline-only shape keeps `fill` on at zero opacity so a hover anywhere inside it still counts. */
export function pathOptions(style: LayerStyle): PathOptions {
  const color = themeColor(style.token)
  return {
    color,
    weight: style.weight,
    opacity: 1,
    fill: true,
    fillColor: color,
    fillOpacity: style.fillOpacity,
    dashArray: style.dashArray,
  }
}

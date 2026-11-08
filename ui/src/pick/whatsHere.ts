import { booleanPointInPolygon } from '@turf/boolean-point-in-polygon'

import { airspaceStyle, type ThemeToken } from '@/map/airspaceStyle'
import { notamLimits, notamTimes } from '@/map/tooltips'
import type { AirspaceProps, MapGeometry, NotamProps } from '@/map/types'
import { isGeometry } from '@/map/useMapLayers'
import { airspaceFloorFt, notamFloorFt, sortByFloor, type Sortable } from '@/pick/floor'
import { warnSkippedOnce } from '@/pick/skipped'

export interface LatLon {
  lat: number
  lon: number
}

/** A loaded feature as the card reads it. The geometry is `unknown` on purpose: a payload can hold a bad one, and it is skipped, not trusted. */
export interface LoadedFeature<P> {
  geometry: unknown
  properties: P
}

/** Everything loaded from the two routes, whether or not its layer is switched on. */
export interface LoadedLayers {
  airspace: readonly LoadedFeature<AirspaceProps>[]
  notams: readonly LoadedFeature<NotamProps>[]
}

/** One line of the card. */
export interface HereRow extends Sortable {
  kind: 'airspace' | 'notam'
  /** The chip's text label: the airspace type in capitals, or NOTAM. */
  chip: string
  /** The theme variable of the chip's colour: the same one the map draws the shape with. */
  token: ThemeToken
  /** Lower to upper limit as published. */
  limits: string
  /** Active times as published. */
  times: string
  /** A NOTAM's raw E text, as published; null for airspace and for a NOTAM with none. Shown as text only, never logged. */
  text: string | null
  /** True for a NOTAM drawn as its Q-line circle. */
  approx: boolean
}

const NOTAM_TOKEN: ThemeToken = '--type-notam'

/** Whether the point is inside the shape or on its edge, and not in a hole. */
function contains(point: LatLon, geometry: MapGeometry): boolean {
  return booleanPointInPolygon([point.lon, point.lat], geometry) // ignoreBoundary stays false: the edge counts as inside
}

function airspaceRow(props: AirspaceProps, position: number): HereRow {
  return {
    kind: 'airspace',
    key: `airspace:${props.id}:${position}`,
    name: props.name,
    floorFt: airspaceFloorFt(props),
    chip: props.type.replace(/_/g, ' ').toUpperCase(),
    token: airspaceStyle(props.type).token,
    limits: `${props.lower} to ${props.upper}`,
    times: props.hours ?? 'no times published',
    text: null,
    approx: false,
  }
}

function notamRow(props: NotamProps, position: number): HereRow {
  return {
    kind: 'notam',
    key: `notam:${props.id}:${position}`,
    name: props.id,
    floorFt: notamFloorFt(props),
    chip: 'NOTAM',
    token: NOTAM_TOKEN,
    limits: notamLimits(props),
    times: notamTimes(props),
    text: typeof props.text === 'string' && props.text !== '' ? props.text : null,
    approx: props.approx,
  }
}

/**
 * Every loaded airspace and NOTAM whose drawn shape contains the point, sorted by floor (lowest first, ties by name or id).
 * It reads the loaded payloads, never what is visible on the map, so a layer switched off is still listed. A feature with no usable
 * shape (null or empty geometry, or a type other than Polygon or MultiPolygon) is left out and named once in `console.warn`.
 */
export function whatsHere(point: LatLon, layers: LoadedLayers): HereRow[] {
  const rows: HereRow[] = []
  layers.airspace.forEach((feature, position) => {
    if (!feature.properties || !isGeometry(feature.geometry)) return warnSkippedOnce('airspace', feature.properties?.id, position)
    if (contains(point, feature.geometry)) rows.push(airspaceRow(feature.properties, position))
  })
  layers.notams.forEach((feature, position) => {
    if (!feature.properties || !isGeometry(feature.geometry)) return warnSkippedOnce('notams', feature.properties?.id, position)
    if (contains(point, feature.geometry)) rows.push(notamRow(feature.properties, position))
  })
  return sortByFloor(rows)
}

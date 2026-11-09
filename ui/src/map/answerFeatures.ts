import type { Payload } from '@/chat/turns'
import type { MapGeometry } from '@/map/types'

// The overlay features of one turn, read from the stream's `overlays` (src/flysafelz/agent/state.py OverlayFeature): GeoJSON Features whose
// properties say what they are. Nothing is fetched; the map draws what the answer carried.

export type AnswerGeometry = MapGeometry | { type: 'Point'; coordinates: [number, number] }

/** One NOTAM or airspace area the answer cited. The LZ circle is not one: the map draws that itself from the pin. */
export interface AnswerFeature {
  /** The NOTAM number or the airspace's name. */
  id: string
  kind: 'notam' | 'airspace'
  /** The server's action word, or null when it sent none. */
  action: string | null
  /** A NOTAM's raw E text as published; null for airspace. */
  text: string | null
  approx: boolean
  /** Null when the geometry is empty or malformed: the feature is not drawn, but a NOTAM's id and text are still usable. */
  geometry: AnswerGeometry | null
}

const isRecord = (value: unknown): value is Payload => typeof value === 'object' && value !== null && !Array.isArray(value)
const isPosition = (value: unknown): value is [number, number] =>
  Array.isArray(value) &&
  value.length >= 2 &&
  typeof value[0] === 'number' &&
  typeof value[1] === 'number' &&
  Number.isFinite(value[0]) &&
  Number.isFinite(value[1]) &&
  Math.abs(value[0]) <= 180 &&
  Math.abs(value[1]) <= 90
const isRing = (value: unknown): boolean => Array.isArray(value) && value.length >= 4 && value.every(isPosition)
const isPolygon = (value: unknown): boolean => Array.isArray(value) && value.length >= 1 && value.every(isRing)

/** The geometry when it can be drawn (a Point, a Polygon or a MultiPolygon with real positions), else null. */
export function readGeometry(value: unknown): AnswerGeometry | null {
  if (!isRecord(value)) return null
  const { type, coordinates } = value
  if (type === 'Point' && isPosition(coordinates)) return { type, coordinates: [coordinates[0], coordinates[1]] }
  if (type === 'Polygon' && isPolygon(coordinates)) return { type, coordinates } as AnswerGeometry
  if (type === 'MultiPolygon' && Array.isArray(coordinates) && coordinates.length >= 1 && coordinates.every(isPolygon)) return { type, coordinates } as AnswerGeometry
  return null
}

/** The NOTAM and airspace features of an `overlays` list, in order. Anything else (the LZ circle, a malformed entry) is left out. */
export function readAnswerFeatures(overlays: readonly Payload[]): AnswerFeature[] {
  const found: AnswerFeature[] = []
  for (const entry of overlays) {
    const props = isRecord(entry.properties) ? entry.properties : null
    if (!props || typeof props.id !== 'string' || props.id === '') continue
    if (props.kind !== 'notam' && props.kind !== 'airspace') continue
    found.push({
      id: props.id,
      kind: props.kind,
      action: typeof props.action === 'string' ? props.action : null,
      text: props.kind === 'notam' && typeof props.text === 'string' ? props.text : null,
      approx: props.kind === 'notam' && props.approx === true,
      geometry: readGeometry(entry.geometry),
    })
  }
  return found
}

/** The NOTAMs of a turn by id: the ids that may be linked in its answer. The first feature with an id wins. */
export function notamsById(overlays: readonly Payload[]): ReadonlyMap<string, AnswerFeature> {
  const index = new Map<string, AnswerFeature>()
  for (const feature of readAnswerFeatures(overlays)) {
    if (feature.kind === 'notam' && !index.has(feature.id)) index.set(feature.id, feature)
  }
  return index
}

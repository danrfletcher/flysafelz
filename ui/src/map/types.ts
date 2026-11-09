// The shapes of GET /map/airspace and GET /map/notams, read from the routes' own code (src/flysafelz/mapdata/airspace.py and
// notams.py) and made-up payloads shaped like their output (src/test/fixtures/map-airspace.json, map-notams.json; tests/unit/test_ui_map_layers.py checks them against the route code). Both routes answer with one
// envelope: a GeoJSON FeatureCollection plus `updated_at` (ISO UTC) and `stale`.

export type Position = [number, number] // [lon, lat], GeoJSON order
export type Ring = Position[]

export type MapGeometry =
  | { type: 'Polygon'; coordinates: Ring[] }
  | { type: 'MultiPolygon'; coordinates: Ring[][] }

/** One airspace. `lower_ft` is the lower limit in feet as stated (SFC is 0, FL65 is 6500), null when there is none. */
export interface AirspaceProps {
  id: string
  name: string
  /** A value of openAIP's CODES["type"] list, or (never from today's server) any other string. */
  type: string
  class: string | null
  /** "SFC", "FL65", "1500 ft AGL", "4500 ft AMSL" or "UNL". */
  lower: string
  upper: string
  lower_ft: number | null
  upper_ft: number | null
  /** openAIP's schedule text, "by NOTAM", or null when none was published. */
  hours: string | null
  /** Always null on this layer: the base layers carry the type only. Never read for colour. */
  action: string | null
}

/** One NOTAM in the PIB's wording. `text` (item E, as published) is shown as plain text in the What's-here card when a row is opened (src/pick), and is never logged. */
export interface NotamProps {
  id: string
  q_code: string
  lower_fl: number | null
  upper_fl: number | null
  item_f: string | null
  item_g: string | null
  valid_from: string
  valid_to: string | null
  permanent: boolean
  est: boolean
  item_d: string | null
  text: string
  approx: boolean
}

export interface MapFeature<P> {
  type: 'Feature'
  geometry: MapGeometry
  properties: P
}

export interface LayerPayload<P> {
  features: MapFeature<P>[]
  updated_at: string
  stale: boolean
}

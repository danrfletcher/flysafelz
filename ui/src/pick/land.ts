import { booleanPointInPolygon } from '@turf/boolean-point-in-polygon'

import ukLandText from '@/data/uk-land.geojson?raw'

// The land polygon is bundled with the app (ui/src/data/uk-land.geojson, rebuilt by ui/scripts/build-uk-land.py) and is never fetched.
// Vite does not treat .geojson as JSON, so the file comes in as text and is parsed once.
// It only decides whether a click drops a pin: the server's UK check (geo.check_uk_bounds) stays the authority.
const LAND = (JSON.parse(ukLandText) as GeoJSON.FeatureCollection<GeoJSON.MultiPolygon>).features[0]

/** The size cap on the bundled polygon, in bytes. land.test.ts holds the file to it. */
export const LAND_MAX_BYTES = 300 * 1024

/**
 * Whether a point is on UK land: Great Britain, Northern Ireland, the Scottish islands and the Isle of Man, by the bundled
 * Natural Earth 1:10m outline simplified to about 0.005 degrees. A point exactly on the outline counts as land.
 * NaN, Infinity and values outside -90..90 / -180..180 are not land.
 */
export function isUkLand(lat: number, lon: number): boolean {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return false
  return booleanPointInPolygon([lon, lat], LAND)
}

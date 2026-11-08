// The basemap: CARTO Dark Matter raster tiles, with an optional public key, and the rule for when the map says the tiles are unavailable.

/** CARTO Dark Matter without a key. {r} asks for retina tiles on high-density screens; Leaflet fills {s}, {z}, {x}, {y} and {r}. */
export const CARTO_TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'

/** The query parameter CARTO's raster-tile docs name for the key (https://github.com/CartoDB/basemap-styles: `...png?key=YOUR_KEY`). */
export const CARTO_KEY_PARAM = 'key'

export const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a> | airspace data <a href="https://www.openaip.net">openAIP</a> (CC BY-NC 4.0) | wind and model rain data <a href="https://open-meteo.com">Open-Meteo</a> (CC BY 4.0) | rain radar <a href="https://www.rainviewer.com">RainViewer</a>'

/**
 * The tile URL template. No key, an empty key or a blank one gives CARTO_TILE_URL unchanged; otherwise the trimmed key is
 * URL-encoded into the `key` parameter, so a key holding `&`, `?`, `#`, `{` or a space cannot break the query or the {z}/{x}/{y}{r} template.
 */
export function cartoTileUrl(key?: string): string {
  const trimmed = key?.trim()
  return trimmed ? `${CARTO_TILE_URL}?${CARTO_KEY_PARAM}=${encodeURIComponent(trimmed)}` : CARTO_TILE_URL
}

/** The tile URL for this build: the one place VITE_CARTO_KEY is read. The key is public by design and goes only into the CARTO tile URL. Read per call so a test can stub it. */
export function tileUrl(): string {
  return cartoTileUrl(import.meta.env.VITE_CARTO_KEY)
}

export type BasemapStatus = 'loading' | 'ok' | 'unavailable'

/** The status plus the tiles of the current batch (since Leaflet's last `load`) that loaded and that failed. */
export interface BasemapState {
  status: BasemapStatus
  loaded: number
  failed: number
}

/** What the tile layer reports: one tile loaded, one tile failed, or a batch finished (every tile asked for has loaded or failed). */
export type BasemapEvent = 'tileload' | 'tileerror' | 'load'

export const INITIAL_BASEMAP: BasemapState = { status: 'loading', loaded: 0, failed: 0 }

/**
 * The status rule. Any tile load makes it `ok`. A finished batch with at least one failed tile and no loaded tile makes it
 * `unavailable`; a later tile load (pan, zoom, network back) makes it `ok` again. A failed tile among loaded ones leaves it `ok`,
 * so it does not flap. A batch is judged once and its counts start again.
 */
export function basemapStatus(prev: BasemapState, event: BasemapEvent): BasemapState {
  switch (event) {
    case 'tileload':
      return { status: 'ok', loaded: prev.loaded + 1, failed: prev.failed }
    case 'tileerror':
      return { ...prev, failed: prev.failed + 1 }
    case 'load':
      return { status: prev.failed > 0 && prev.loaded === 0 ? 'unavailable' : prev.status, loaded: 0, failed: 0 }
  }
}

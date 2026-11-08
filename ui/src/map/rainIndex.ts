// RainViewer's weather-maps index and its tile URLs (G5, E3).
//
// The index, https://api.rainviewer.com/public/weather-maps.json, lists the past 2 hours of radar at 10-minute steps (about 13 frames):
//   { version, generated, host: "https://tilecache.rainviewer.com", radar: { past: [{ time: <unix s>, path: "/v2/radar/<id>" }, ...] } }
// A tile is `host + path + /256/{z}/{x}/{y}/{color}/{smooth}_{snow}.png` (checked against RainViewer's Weather Maps API page, Nov 2026; the
// maximum zoom there is 7). The colour scheme is 2, "Universal Blue", and the options are `1_0`: smoothed, snow not in its own colours. The index
// is the only RainViewer call that is not a tile, and the browser never sends a key or any other data to it.

export const RAINVIEWER_INDEX_URL = 'https://api.rainviewer.com/public/weather-maps.json'

/** Radar tiles exist up to this zoom. The layer is upscaled above it and no tile above it is ever asked for. */
export const MAX_NATIVE_ZOOM = 7

export const TILE_SIZE = 256
const COLOUR_SCHEME = 2
const TILE_OPTIONS = '1_0'

/** One radar frame: when it is for (unix seconds) and where its tiles are (a path under the index's host). */
export interface RainFrame {
  time: number
  path: string
}

/** What the index gave: its tile host and the frames (oldest first, one per time), or `failed`. */
export type RainIndex = { status: 'ok'; host: string; frames: RainFrame[] } | { status: 'failed' }

const FAILED: RainIndex = { status: 'failed' }

// The host must be a RainViewer one over https, and a path must be a plain RainViewer radar path: a changed or hostile index cannot send the browser elsewhere.
const HOST = /^https:\/\/([a-z0-9-]+\.)*rainviewer\.com$/
const PATH = /^\/v2\/radar\/[\w-]+$/

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * The index in a response body. A body that is not an index, a host that is not RainViewer's, or no usable frame in `radar.past` (empty, or every
 * entry malformed) is `failed`. A malformed frame among good ones is left out; the rest stay. One frame is a valid index.
 */
export function parseRainIndex(body: unknown): RainIndex {
  if (!isObject(body) || typeof body.host !== 'string' || !HOST.test(body.host)) return FAILED
  const radar = body.radar
  if (!isObject(radar) || !Array.isArray(radar.past)) return FAILED
  const byTime = new Map<number, RainFrame>()
  for (const entry of radar.past) {
    if (!isObject(entry)) continue
    const { time, path } = entry
    if (typeof time !== 'number' || !Number.isInteger(time) || time <= 0) continue
    if (typeof path !== 'string' || !PATH.test(path)) continue
    byTime.set(time, { time, path })
  }
  if (byTime.size === 0) return FAILED
  return { status: 'ok', host: body.host, frames: [...byTime.values()].sort((a, b) => a.time - b.time) }
}

/** One fetch of the index: `failed` for a network error, any status but 200, or a body that is not JSON or not an index. Never rejects. */
export async function fetchRainIndex(signal: AbortSignal): Promise<RainIndex> {
  try {
    const response = await fetch(RAINVIEWER_INDEX_URL, { signal })
    if (response.status !== 200) return FAILED
    return parseRainIndex(await response.json())
  } catch {
    return FAILED
  }
}

/** The Leaflet tile URL template of one frame; Leaflet fills {z}, {x} and {y}. */
export function tileTemplate(host: string, path: string): string {
  return `${host}${path}/${TILE_SIZE}/{z}/{x}/{y}/${COLOUR_SCHEME}/${TILE_OPTIONS}.png`
}

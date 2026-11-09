import { fetchRoute, useRoutePolling, type LayerStatus } from '@/map/useMapLayers'
import { readWindPayload, WIND_PATH, type WindGrid, type WindPayload } from '@/map/wind'

/** The wind route's state: the panel's status, and the grid on the map (null until one has arrived, and while the route is unavailable). */
export interface WindState extends LayerStatus {
  grid: WindGrid | null
}

const LOADING: WindState = { status: 'loading', grid: null, updatedAt: null, stale: false }
const UNAVAILABLE: WindState = { status: 'unavailable', grid: null, updatedAt: null, stale: false }

/**
 * The state after a fetch. New data replaces the old; a failure keeps the grid and marks it stale, or says unavailable when there is none.
 * An answer with the same `updated_at` as the one on the map is the same cached grid, so the grid object stays: the particle layer is not rebuilt for it.
 */
export function nextWindState(previous: WindState, payload: WindPayload | null): WindState {
  if (payload) {
    const grid = previous.grid && previous.updatedAt === payload.updated_at ? previous.grid : payload.grid
    return { status: 'ok', grid, updatedAt: payload.updated_at, stale: payload.stale }
  }
  if (previous.status === 'ok') return { ...previous, stale: true }
  return UNAVAILABLE
}

const fetchWind = (signal: AbortSignal) => fetchRoute(WIND_PATH, readWindPayload, signal)

/**
 * GET /map/wind on load and every 15 minutes while the page is open. The browser reaches wind only through this route, never Open-Meteo.
 * The fetch runs whether or not the Wind switch is on, so switching it on again draws the grid already held, with no new request.
 */
export function useWind(): WindState {
  return useRoutePolling(fetchWind, nextWindState, LOADING)
}

import type { LayerState } from '@/map/useMapLayers'

/** A layer that has loaded: its features, the route's time, not stale unless asked. */
export function loaded<P>(features: LayerState<P>['features'], over: Partial<LayerState<P>> = {}): LayerState<P> {
  return { status: 'ok', features, updatedAt: '2026-11-08T12:00:00Z', stale: false, ...over }
}

/** A layer whose route has not answered yet. */
export function loading<P>(): LayerState<P> {
  return { status: 'loading', features: [], updatedAt: null, stale: false }
}

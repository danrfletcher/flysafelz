// The Rain layer's zoom fade (G5, Q4): 60% up to zoom 7, falling in a straight line to nothing at zoom 9. Radar tiles stop at zoom 7 (RainViewer's
// maximum), so above it the layer is only upscaled, and by zoom 9 it is too coarse to show.

/** The layer's opacity at and below FADE_START_ZOOM. */
export const RAIN_OPACITY = 0.6
/** The last zoom at the full opacity, and the zoom from which nothing is shown. Zoom 8 is halfway: 30%. */
export const FADE_START_ZOOM = 7
export const HIDDEN_ZOOM = 9

/** The pane opacity at a zoom, fractional zooms included (the map snaps to quarters). A zoom that is not a number gives the full opacity. */
export function rainOpacity(zoom: number): number {
  if (!Number.isFinite(zoom) || zoom <= FADE_START_ZOOM) return RAIN_OPACITY
  if (zoom >= HIDDEN_ZOOM) return 0
  return (RAIN_OPACITY * (HIDDEN_ZOOM - zoom)) / (HIDDEN_ZOOM - FADE_START_ZOOM)
}

/** Whether the layer is faded out completely: the layer panel then says the view is regional only. */
export function rainHidden(zoom: number): boolean {
  return rainOpacity(zoom) === 0
}

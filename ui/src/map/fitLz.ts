import L from 'leaflet'

import type { LatLon } from '@/pick/whatsHere'

/** The radius the chat checks round the LZ, in nautical miles (the What's-here card says "within 2 NM"). */
export const LZ_RADIUS_NM = 2
export const METRES_PER_NM = 1852

/** The width of the chat sheet on a desktop screen, in pixels (26rem at the default font size). The map is padded by this much on the right. */
export const SHEET_WIDTH_PX = 416
/** How much of the map's height the mobile drawer covers. */
export const DRAWER_HEIGHT_FRACTION = 0.55
/** Space kept between the circle and the edge of the part of the map that is still visible. */
const MARGIN_PX = 16

export type ChatLayout = 'sheet' | 'drawer'

/** The box that holds the circle of `LZ_RADIUS_NM` round the point. */
export function lzBounds(point: LatLon): L.LatLngBounds {
  return L.latLng(point.lat, point.lon).toBounds(2 * LZ_RADIUS_NM * METRES_PER_NM)
}

/** Padding for `fitBounds` so the circle sits in the part of the map the chat leaves visible: the sheet covers the right, the drawer the bottom. */
export function fitPadding(layout: ChatLayout, mapSize: { x: number; y: number }): { paddingTopLeft: L.PointTuple; paddingBottomRight: L.PointTuple } {
  const right = layout === 'sheet' ? SHEET_WIDTH_PX : 0
  const bottom = layout === 'drawer' ? Math.round(DRAWER_HEIGHT_FRACTION * mapSize.y) : 0
  return { paddingTopLeft: [MARGIN_PX, MARGIN_PX], paddingBottomRight: [right + MARGIN_PX, bottom + MARGIN_PX] }
}

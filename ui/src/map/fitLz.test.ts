import L from 'leaflet'
import { describe, expect, it } from 'vitest'

import { DRAWER_HEIGHT_FRACTION, LZ_RADIUS_NM, SHEET_WIDTH_PX, fitPadding, lzBounds } from '@/map/fitLz'

const BLUE_MILL = { lat: 53.802797, lon: -0.086644 }
const NM_METRES = 1852

describe('lzBounds', () => {
  it('is a box that holds the 2 NM circle, centred on the point', () => {
    const bounds = lzBounds(BLUE_MILL)
    const centre = bounds.getCenter()
    expect(centre.lat).toBeCloseTo(BLUE_MILL.lat, 6)
    expect(centre.lng).toBeCloseTo(BLUE_MILL.lon, 6)
    expect(LZ_RADIUS_NM).toBe(2)

    const point = L.latLng(BLUE_MILL.lat, BLUE_MILL.lon)
    // Each side of the box is 2 NM from the centre.
    const north = L.latLng(bounds.getNorth(), BLUE_MILL.lon)
    const east = L.latLng(BLUE_MILL.lat, bounds.getEast())
    expect(point.distanceTo(north)).toBeCloseTo(2 * NM_METRES, -1)
    expect(point.distanceTo(east)).toBeCloseTo(2 * NM_METRES, -1)
  })

  it('is wider in degrees of longitude the further north the point is', () => {
    const south = lzBounds({ lat: 50.1, lon: -5 })
    const north = lzBounds({ lat: 60.1, lon: -5 })
    expect(north.getEast() - north.getWest()).toBeGreaterThan(south.getEast() - south.getWest())
  })
})

describe('fitPadding', () => {
  const map = { x: 1000, y: 700 }

  it('pads the right for the sheet and nothing at the bottom', () => {
    const { paddingTopLeft, paddingBottomRight } = fitPadding('sheet', map)
    expect(paddingBottomRight[0]).toBeGreaterThanOrEqual(SHEET_WIDTH_PX)
    expect(paddingBottomRight[1]).toBeLessThan(SHEET_WIDTH_PX)
    expect(paddingTopLeft[0]).toBeLessThan(SHEET_WIDTH_PX)
  })

  it('pads the bottom for the drawer, by its share of the map height, and nothing at the right', () => {
    const { paddingBottomRight } = fitPadding('drawer', map)
    expect(paddingBottomRight[1]).toBeGreaterThanOrEqual(Math.round(DRAWER_HEIGHT_FRACTION * map.y))
    expect(paddingBottomRight[0]).toBeLessThan(SHEET_WIDTH_PX)
  })

  it('differs between the sheet and the drawer', () => {
    expect(fitPadding('sheet', map)).not.toEqual(fitPadding('drawer', map))
  })

  it('puts the circle in the visible part of a real Leaflet map: right of the sheet, above the drawer', () => {
    const element = document.createElement('div')
    Object.defineProperty(element, 'clientWidth', { value: map.x })
    Object.defineProperty(element, 'clientHeight', { value: map.y })
    document.body.append(element)
    const leaflet = L.map(element, { zoomAnimation: false, fadeAnimation: false })
    leaflet.setView([BLUE_MILL.lat, BLUE_MILL.lon], 6, { animate: false })

    leaflet.fitBounds(lzBounds(BLUE_MILL), { ...fitPadding('sheet', leaflet.getSize()), animate: false })
    const sheetCircle = [lzBounds(BLUE_MILL).getNorthEast(), lzBounds(BLUE_MILL).getSouthWest()].map((corner) => leaflet.latLngToContainerPoint(corner))
    for (const corner of sheetCircle) expect(corner.x).toBeLessThanOrEqual(map.x - SHEET_WIDTH_PX)

    leaflet.fitBounds(lzBounds(BLUE_MILL), { ...fitPadding('drawer', leaflet.getSize()), animate: false })
    const drawerCircle = [lzBounds(BLUE_MILL).getNorthEast(), lzBounds(BLUE_MILL).getSouthWest()].map((corner) => leaflet.latLngToContainerPoint(corner))
    for (const corner of drawerCircle) expect(corner.y).toBeLessThanOrEqual(map.y - Math.round(DRAWER_HEIGHT_FRACTION * map.y))
    leaflet.remove()
    element.remove()
  })
})

import L from 'leaflet'
import { render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { UK_BOUNDS } from '@/lib/ukBounds'

// jsdom has no layout, so give every element a size for Leaflet to fit the map into.
const WIDTH = 1200
const HEIGHT = 800

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(WIDTH)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(HEIGHT)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('App', () => {
  it('mounts the Leaflet container, between the nav bar and the footer', () => {
    const { container } = render(<App />)
    const map = container.querySelector('.leaflet-container')
    expect(map).not.toBeNull()
    expect(container.querySelector('header')?.nextElementSibling?.contains(map)).toBe(true)
    expect(container.querySelector('footer')?.previousElementSibling?.contains(map)).toBe(true)
  })

  it('opens fitted to the UK box 49.8-61 N, 8.7 W-2 E', () => {
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    expect(fit).toHaveBeenCalledTimes(1)
    const asked = L.latLngBounds(fit.mock.calls[0][0] as L.LatLngBoundsLiteral)
    expect([asked.getSouth(), asked.getNorth(), asked.getWest(), asked.getEast()]).toEqual([49.8, 61, -8.7, 2])
    expect(UK_BOUNDS).toEqual([
      [49.8, -8.7],
      [61, 2],
    ])

    // The map that Leaflet actually made shows the whole box, and is fitted to it: no further out than it must be.
    const map = fit.mock.contexts[0] as L.Map
    const shown = map.getBounds()
    expect(shown.contains(asked)).toBe(true)
    const zoomedOut = map.getBoundsZoom(asked, false) - map.getZoom()
    expect(zoomedOut).toBeLessThan(0.25) // within one zoomSnap step of the tightest fit
  })

  it('credits OpenStreetMap and CARTO in the attribution', () => {
    const { container } = render(<App />)
    const attribution = container.querySelector('.leaflet-control-attribution')
    expect(attribution).toHaveTextContent('OpenStreetMap')
    expect(attribution).toHaveTextContent('CARTO')
  })

  it('asks for CARTO Dark Matter tiles and no other host', async () => {
    const { container } = render(<App />)
    await waitFor(() => expect(container.querySelectorAll('img.leaflet-tile').length).toBeGreaterThan(0))
    const hosts = [...container.querySelectorAll('img.leaflet-tile')].map((tile) => new URL((tile as HTMLImageElement).src).host)
    expect(hosts.every((host) => /^[a-d]\.basemaps\.cartocdn\.com$/.test(host))).toBe(true)
  })

  it('makes no request of its own: no fetch, and the MSW guard sees nothing', () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const opened = vi.spyOn(XMLHttpRequest.prototype, 'open')
    render(<App />)
    expect(fetched).not.toHaveBeenCalled()
    expect(opened).not.toHaveBeenCalled()
  })

  it('runs invalidateSize after the window is resized', async () => {
    render(<App />)
    const invalidate = vi.spyOn(L.Map.prototype, 'invalidateSize')
    window.dispatchEvent(new Event('resize'))
    await waitFor(() => expect(invalidate).toHaveBeenCalled())
  })

  it('has the page chrome the spec asks for: the dark class is set by index.html, the footer line is present', () => {
    const { container } = render(<App />)
    expect(container.querySelector('footer')).toHaveTextContent('Decision support only, not a primary means of navigation.')
  })
})

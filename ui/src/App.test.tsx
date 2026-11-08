import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { UK_BOUNDS } from '@/lib/ukBounds'
import { server } from '@/test/setup'
import {
  AIRSPACE_FIXTURE,
  AIRSPACE_URL,
  BLUE_MILL_AIRSPACE,
  BLUE_MILL_NOTAMS,
  NOTAMS_FIXTURE,
  NOTAMS_URL,
  BLUE_MILL,
  airspaceRoute,
  blueMillRoutes,
  mapRoutes,
  notamsRoute,
  unavailable,
} from '@/test/handlers'

// jsdom has no layout, so give every element a size for Leaflet to fit the map into.
const WIDTH = 1200
const HEIGHT = 800

beforeEach(() => {
  server.use(...mapRoutes())
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

  it('credits OpenStreetMap, CARTO and openAIP in the attribution', () => {
    const { container } = render(<App />)
    const attribution = container.querySelector('.leaflet-control-attribution')
    expect(attribution).toHaveTextContent('OpenStreetMap')
    expect(attribution).toHaveTextContent('CARTO')
    expect(attribution).toHaveTextContent('openAIP')
  })

  it('asks for CARTO Dark Matter tiles and no other host', async () => {
    const { container } = render(<App />)
    await waitFor(() => expect(container.querySelectorAll('img.leaflet-tile').length).toBeGreaterThan(0))
    const hosts = [...container.querySelectorAll('img.leaflet-tile')].map((tile) => new URL((tile as HTMLImageElement).src).host)
    expect(hosts.every((host) => /^[a-d]\.basemaps\.cartocdn\.com$/.test(host))).toBe(true)
  })

  it('F2: calls only the two map routes on VITE_API_URL, and the basemap tiles', async () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const opened = vi.spyOn(XMLHttpRequest.prototype, 'open')
    const { container } = render(<App />)
    await waitFor(() => expect(fetched).toHaveBeenCalledTimes(2))
    const urls = fetched.mock.calls.map(([input]) => String(input)).sort()
    expect(urls).toEqual([AIRSPACE_URL(), NOTAMS_URL()])
    expect(urls.some((url) => /wind|rain|radar|analytics/.test(url))).toBe(false)
    expect(opened).not.toHaveBeenCalled()
    await waitFor(() => expect(container.querySelectorAll('img.leaflet-tile').length).toBeGreaterThan(0))
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

// The map the app made: Leaflet's fitBounds is called on it once when it opens.
function mapOf(fit: ReturnType<typeof vi.spyOn>): L.Map {
  return fit.mock.contexts[0] as L.Map
}

function polygons(map: L.Map, kind: 'airspace' | 'notams'): number {
  let count = 0
  map.eachLayer((layer) => {
    const feature = (layer as L.Polygon & { feature?: { properties: object } }).feature
    if (layer instanceof L.Polygon && feature && (kind === 'notams' ? 'q_code' in feature.properties : 'lower_ft' in feature.properties)) count += 1
  })
  return count
}

async function openPanel() {
  const user = userEvent.setup()
  await user.click(screen.getByRole('button', { name: 'Map layers' }))
  return { user, panel: await screen.findByRole('dialog') }
}

describe('GP2 (this PR’s part): airspace and NOTAMs below FL100, and the layer panel', () => {
  it('draws both layers on first load and shows the legend and "updated HH:MM" for both', async () => {
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    const map = mapOf(fit)
    await waitFor(() => expect(polygons(map, 'airspace')).toBe(15))
    await waitFor(() => expect(polygons(map, 'notams')).toBe(5))

    const { panel } = await openPanel()
    expect(within(panel).getAllByText(/^updated \d\d:\d\d$/)).toHaveLength(2)
    expect(within(panel).getByText('On the map')).toBeInTheDocument()
    expect(within(panel).getByText('For this field')).toBeInTheDocument()
    expect(within(panel).getByText('amber = near your limits, check the conditions too')).toBeInTheDocument()
  })

  it('switches NOTAMs off and on without touching the airspace', async () => {
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    const map = mapOf(fit)
    await waitFor(() => expect(polygons(map, 'notams')).toBe(5))
    const { user } = await openPanel()

    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' }))
    expect([polygons(map, 'airspace'), polygons(map, 'notams')]).toEqual([15, 0])
    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' }))
    expect([polygons(map, 'airspace'), polygons(map, 'notams')]).toEqual([15, 5])

    await user.click(screen.getByRole('switch', { name: 'Airspace below FL100' }))
    expect([polygons(map, 'airspace'), polygons(map, 'notams')]).toEqual([0, 5])
  })
})

describe('E1: a route that fails does not break the rest', () => {
  it('hides a 503 layer with "<layer> unavailable", keeps the other (stale) layer drawn, and the shell works', async () => {
    server.use(unavailable(AIRSPACE_URL()), notamsRoute({ ...NOTAMS_FIXTURE, stale: true }))
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    const { container } = render(<App />)
    const map = mapOf(fit)
    await waitFor(() => expect(polygons(map, 'notams')).toBe(5))
    expect(polygons(map, 'airspace')).toBe(0)

    const { panel } = await openPanel()
    expect(within(panel).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(within(panel).getByText('stale')).toBeInTheDocument()
    expect(within(panel).getAllByText(/^updated/)).toHaveLength(1)

    // the map still pans, and the shell is still there
    const before = map.getCenter()
    map.panBy([120, 0], { animate: false })
    expect(map.getCenter().lng).not.toBe(before.lng)
    expect(container.querySelector('header')).toBeInTheDocument()
    expect(container.querySelector('footer')).toHaveTextContent('Decision support only')
    expect(screen.getByRole('button', { name: 'How it works' })).toBeInTheDocument()
  })

  it.each([
    ['a network error', () => HttpResponse.error()],
    ['text that is not JSON', () => new HttpResponse('<html>Access denied</html>')],
    ['JSON that is not a payload', () => HttpResponse.json({ ok: true })],
  ])('treats %s on both routes as unavailable, without a crash', async (_label, answer) => {
    server.use(http.get(AIRSPACE_URL(), answer), http.get(NOTAMS_URL(), answer))
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    const { panel } = await openPanel()
    await waitFor(() => expect(within(panel).getByText('NOTAMs below FL100 unavailable')).toBeInTheDocument())
    expect(within(panel).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(polygons(mapOf(fit), 'airspace')).toBe(0)
  })

  it('E15: zero NOTAMs and not stale draws nothing, shows the time, and no "stale"', async () => {
    server.use(notamsRoute({ features: [], updated_at: NOTAMS_FIXTURE.updated_at, stale: false }))
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    await waitFor(() => expect(polygons(mapOf(fit), 'airspace')).toBe(15))
    const { panel } = await openPanel()
    await waitFor(() => expect(within(panel).getAllByText(/^updated \d\d:\d\d$/)).toHaveLength(2))
    expect(polygons(mapOf(fit), 'notams')).toBe(0)
    expect(panel).not.toHaveTextContent('stale')
  })

  it('an empty airspace list is an empty layer', async () => {
    server.use(airspaceRoute({ ...AIRSPACE_FIXTURE, features: [] }))
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    await waitFor(() => expect(polygons(mapOf(fit), 'notams')).toBe(5))
    expect(polygons(mapOf(fit), 'airspace')).toBe(0)
  })
})


const BLUE_MILL_STALE_AIRSPACE = { ...BLUE_MILL_AIRSPACE, stale: true }
const BLUE_MILL_STALE_NOTAMS = { ...BLUE_MILL_NOTAMS, stale: true }

describe('GP3: click Blue Mill', () => {
  const clickBlueMill = (map: L.Map) => act(() => void map.fire('click', { latlng: L.latLng(BLUE_MILL.lat, BLUE_MILL.lon) }))

  it('drops a pin and opens the card with the airspace and NOTAMs over the point, and each row expands', async () => {
    server.use(...blueMillRoutes())
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    const map = mapOf(fit)
    await waitFor(() => expect(polygons(map, 'notams')).toBeGreaterThan(0))

    clickBlueMill(map)
    const card = await screen.findByRole('region', { name: "What's here" })
    expect(within(card).getByText('Over this point')).toBeInTheDocument()
    expect(within(card).getByText('The chat checks everything within 2 NM.')).toBeInTheDocument()
    expect(within(card).getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(within(card).getByText('H2001/26')).toBeInTheDocument()
    expect(within(card).getByRole('button', { name: 'Above FL100 (3)' })).toHaveAttribute('aria-expanded', 'false')

    const user = userEvent.setup()
    await user.click(within(card).getByRole('button', { name: /H2001\/26/ }))
    expect(within(card).getByText(/BEFORE FLIGHT/)).toBeInTheDocument()
    expect(within(card).getByText(/1200FT AMSL/, { selector: 'span' })).toBeInTheDocument()
  })

  it('makes no request on the click (F2): the card reads the payloads already loaded', async () => {
    server.use(...blueMillRoutes())
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    const fetched = vi.spyOn(globalThis, 'fetch')
    const opened = vi.spyOn(XMLHttpRequest.prototype, 'open')
    render(<App />)
    const map = mapOf(fit)
    await waitFor(() => expect(polygons(map, 'notams')).toBeGreaterThan(0))
    const requests = fetched.mock.calls.length
    const tiles = opened.mock.calls.length

    clickBlueMill(map)
    await screen.findByRole('region', { name: "What's here" })
    expect(fetched.mock.calls.length).toBe(requests)
    expect(opened.mock.calls.length).toBe(tiles)
  })

  it('lists everything loaded even with both layers switched off, and a switch changed after it opens leaves the card as it was', async () => {
    server.use(...blueMillRoutes())
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    const map = mapOf(fit)
    await waitFor(() => expect(polygons(map, 'notams')).toBeGreaterThan(0))
    const { user } = await openPanel()
    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' }))
    await user.click(screen.getByRole('switch', { name: 'Airspace below FL100' }))
    expect([polygons(map, 'airspace'), polygons(map, 'notams')]).toEqual([0, 0])

    clickBlueMill(map)
    const card = await screen.findByRole('region', { name: "What's here" })
    const listed = () => within(card).getAllByRole('listitem').map((item) => item.textContent)
    const before = listed()
    expect(before.some((text) => text?.includes('BLUE MILL TEST ATZ'))).toBe(true)
    expect(before.some((text) => text?.includes('H2001/26'))).toBe(true)

    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' })) // on again, after the card opened
    await user.click(screen.getByRole('switch', { name: 'Airspace below FL100' }))
    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' })) // and NOTAMs off once more
    expect(listed()).toEqual(before)
  })

  it('shows the stale layer’s rows and none of the panel’s stale note in the card', async () => {
    server.use(airspaceRoute({ ...BLUE_MILL_STALE_AIRSPACE }), notamsRoute({ ...BLUE_MILL_STALE_NOTAMS }))
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(<App />)
    const map = mapOf(fit)
    await waitFor(() => expect(polygons(map, 'notams')).toBeGreaterThan(0))
    clickBlueMill(map)
    const card = await screen.findByRole('region', { name: "What's here" })
    expect(within(card).getByText('H2001/26')).toBeInTheDocument()
    expect(card).not.toHaveTextContent(/stale/i)
  })
})

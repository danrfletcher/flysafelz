import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { STARTER_CHIPS } from '@/chat/copy'
import { ANSWER_PANE } from '@/map/answerOverlays'
import { HIGHLIGHTS_RUN, chatServer, scriptOf } from '@/test/chatServer'
import { BLUE_MILL, blueMillRoutes } from '@/test/handlers'
import { server } from '@/test/setup'

// E2 through the whole app: the basemap tiles fail, the status rule says `unavailable`, and everything else carries on.
// The tile layer is the real one on the real map; the failure is Leaflet's own `tileerror` and `load` events fired on it.
// Phones in the recorded run are fiction-range.

const NORTH_SEA = { lat: 54.0, lon: 1.5 }
const SLOW = { timeout: 5000 }

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})
afterEach(() => vi.restoreAllMocks())

function open() {
  const fake = chatServer(scriptOf(HIGHLIGHTS_RUN))
  server.use(...blueMillRoutes(), ...fake.handlers)
  const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  const view = render(<App />)
  const map = fit.mock.contexts[0] as L.Map
  let tiles!: L.TileLayer
  map.eachLayer((layer) => {
    if (layer instanceof L.TileLayer) tiles = layer
  })
  const fire = (name: 'tileload' | 'tileerror' | 'load') => act(() => void tiles.fire(name, { tile: document.createElement('img'), coords: { x: 1, y: 1, z: 5 } }))
  const click = (point: { lat: number; lon: number }) => act(() => void map.fire('click', { latlng: L.latLng(point.lat, point.lon) }))
  return { map, tiles, fire, click, view, user: userEvent.setup() }
}

const inPane = (layer: L.Layer) => (layer as L.Path).options?.pane === ANSWER_PANE
function count(map: L.Map, test: (layer: L.Layer) => boolean): number {
  let found = 0
  map.eachLayer((layer) => {
    if (test(layer)) found += 1
  })
  return found
}
const basePolygons = (map: L.Map) => count(map, (layer) => layer instanceof L.Polygon && !(layer instanceof L.Circle) && !inPane(layer))

describe('E2: the basemap is unavailable', () => {
  it('shows "basemap unavailable" in the layer panel, hides the tiles, and leaves the map container on the theme colour', async () => {
    const { fire, tiles, view, user } = open()
    await user.click(screen.getByRole('button', { name: 'Map layers' }))
    const panel = await screen.findByRole('dialog', { name: /layers/i })
    expect(within(panel).queryByText('basemap unavailable')).toBeNull()

    fire('tileerror')
    fire('tileerror')
    fire('load')

    expect(within(panel).getByText('basemap unavailable')).toBeInTheDocument()
    expect(tiles.options.opacity).toBe(0)
    expect(view.container.querySelector('.leaflet-container')).toHaveClass('basemap-unavailable')
    await waitFor(() => expect(within(panel).getAllByText(/^updated \d\d:\d\d$/)).toHaveLength(2)) // the two route layers only; the basemap has no time
  })

  it('draws airspace and NOTAMs, drops the pin on land and opens What’s here, and shows the sea toast', async () => {
    const { map, fire, click, view } = open()
    fire('tileerror')
    fire('load')
    await waitFor(() => expect(basePolygons(map)).toBeGreaterThan(0))

    click(BLUE_MILL)
    expect(await screen.findByText('Over this point')).toBeInTheDocument()
    expect(view.container.querySelector('.pick-pin')).not.toBeNull()

    click(NORTH_SEA)
    expect(screen.getByRole('status')).toHaveTextContent('Pick a field on land in the UK.')
    expect(view.container.querySelectorAll('.pick-pin')).toHaveLength(1)
  })

  it('opens the chat and gets an answer with its badges and highlights', async () => {
    const { map, fire, click, user } = open()
    fire('tileerror')
    fire('load')
    click(BLUE_MILL)
    act(() => (screen.getByRole('button', { name: 'Chat about this LZ' }) as HTMLButtonElement).click())
    const sheet = await screen.findByRole('dialog', { name: /\d/ })
    await user.click(within(sheet).getByRole('button', { name: STARTER_CHIPS[0] }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)

    const badges = within(within(sheet).getByRole('list', { name: 'Verdict by time' })).getAllByRole('button')
    expect(badges).toHaveLength(2)
    expect(badges[0]).toHaveTextContent('APPROACHING LIMITS')
    expect(count(map, (layer) => layer instanceof L.Circle && inPane(layer))).toBe(1)
    expect(count(map, (layer) => layer instanceof L.Polygon && !(layer instanceof L.Circle) && inPane(layer))).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Map layers' })).toBeEnabled()
  })

  it('recovers: a tile that loads later brings the tiles back and removes the note, and no other tile layer is added', async () => {
    const { map, fire, tiles, view, user } = open()
    await user.click(screen.getByRole('button', { name: 'Map layers' }))
    const panel = await screen.findByRole('dialog', { name: /layers/i })
    fire('tileerror')
    fire('load')
    expect(within(panel).getByText('basemap unavailable')).toBeInTheDocument()

    fire('tileload')
    expect(within(panel).queryByText('basemap unavailable')).toBeNull()
    expect(tiles.options.opacity).toBe(1)
    expect(view.container.querySelector('.leaflet-container')).not.toHaveClass('basemap-unavailable')

    expect(count(map, (layer) => layer instanceof L.TileLayer)).toBe(1) // the one CARTO layer, hidden and shown again: no second provider
  })
})

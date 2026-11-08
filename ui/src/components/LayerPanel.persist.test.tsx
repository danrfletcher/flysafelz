import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { LAYER_PREFS_KEY, resetLayerPrefsWarning } from '@/lib/layerPrefs'
import { BLUE_MILL, blueMillRoutes, mapRoutes } from '@/test/handlers'
import { server } from '@/test/setup'

// The whole path for saved layer choices: the real App reads jsdom's storage at load and writes it on every switch, the recorded
// /map/airspace and /map/notams payloads are served by MSW (any other request fails the test), and "a reload" is unmount + render.

const AIRSPACE = 15 // drawn polygons in the recorded payloads (src/test/fixtures/map-*.json)
const NOTAMS = 5

beforeEach(() => {
  resetLayerPrefsWarning()
  vi.spyOn(console, 'warn').mockImplementation(() => {})
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})
afterEach(() => vi.restoreAllMocks())

/** Renders the app (a page load) and returns the map it made. */
function load(routes = mapRoutes()) {
  server.use(...routes)
  const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  fit.mockClear()
  const view = render(<App />)
  const map = fit.mock.contexts.at(-1) as L.Map
  return { view, map, user: userEvent.setup() }
}

/** How many of a base layer's shapes the map draws. */
function drawn(map: L.Map, kind: 'airspace' | 'notams'): number {
  let count = 0
  map.eachLayer((layer) => {
    const feature = (layer as L.Polygon & { feature?: { properties: object } }).feature
    if (layer instanceof L.Polygon && feature && (kind === 'notams' ? 'q_code' in feature.properties : 'lower_ft' in feature.properties)) count += 1
  })
  return count
}

const NAMES = { wind: 'Wind', rain: 'Rain', airspace: 'Airspace below FL100', notams: 'NOTAMs below FL100', answer: 'Answer overlays' } as const

async function openPanel(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'Map layers' }))
  return screen.findByRole('dialog', { name: /layers/i })
}

const checked = (id: keyof typeof NAMES) => screen.getByRole('switch', { name: NAMES[id] }).getAttribute('aria-checked') === 'true'
const stored = () => JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) as string) as Record<string, unknown>

describe('G1: choices survive a reload', () => {
  it('has all four layers on with empty storage, and writes nothing until the first switch', async () => {
    const set = vi.spyOn(Storage.prototype, 'setItem')
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(NOTAMS))
    await openPanel(user)
    expect([checked('wind'), checked('rain'), checked('airspace'), checked('notams'), checked('answer')]).toEqual([true, true, true, true, true])
    expect(drawn(map, 'airspace')).toBe(AIRSPACE)
    expect(set).not.toHaveBeenCalled()
    expect(window.localStorage.getItem(LAYER_PREFS_KEY)).toBeNull()
  })

  it('stores each switch and keeps the others as they were, then a remount shows the same switches and the same drawn layers', async () => {
    const first = load()
    await waitFor(() => expect(drawn(first.map, 'notams')).toBe(NOTAMS))
    await openPanel(first.user)

    await first.user.click(screen.getByRole('switch', { name: NAMES.airspace }))
    expect(stored()).toEqual({ wind: true, rain: true, airspace: false, notams: true, answer: true })
    expect([drawn(first.map, 'airspace'), drawn(first.map, 'notams')]).toEqual([0, NOTAMS])

    await first.user.click(screen.getByRole('switch', { name: NAMES.answer }))
    expect(stored()).toEqual({ wind: true, rain: true, airspace: false, notams: true, answer: false })

    await first.user.click(screen.getByRole('switch', { name: NAMES.notams }))
    expect(stored()).toEqual({ wind: true, rain: true, airspace: false, notams: false, answer: false })
    await first.user.click(screen.getByRole('switch', { name: NAMES.notams }))
    expect(stored()).toEqual({ wind: true, rain: true, airspace: false, notams: true, answer: false })

    first.view.unmount() // the reload
    const again = load()
    await waitFor(() => expect(drawn(again.map, 'notams')).toBe(NOTAMS))
    expect(drawn(again.map, 'airspace')).toBe(0)
    await openPanel(again.user)
    expect([checked('airspace'), checked('notams'), checked('answer')]).toEqual([false, true, false])
  })

  it('ends with a stored value equal to the final switch state after ten clicks on one switch', async () => {
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(NOTAMS))
    await openPanel(user)
    for (let click = 0; click < 10; click++) await user.click(screen.getByRole('switch', { name: NAMES.notams }))
    expect(checked('notams')).toBe(true)
    expect(stored().notams).toBe(true)
    await user.click(screen.getByRole('switch', { name: NAMES.notams }))
    expect(checked('notams')).toBe(false)
    expect(stored().notams).toBe(false)
    expect(drawn(map, 'notams')).toBe(0)
  })

  it('applies a stored Wind and Rain choice and keeps it on the next write (Rain joins the one layers.v1 key)', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, '{"wind":false,"rain":false,"airspace":false}')
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(NOTAMS))
    const panel = await openPanel(user)
    expect(within(panel).getAllByRole('switch')).toHaveLength(5)
    expect([checked('wind'), checked('rain'), checked('airspace'), checked('notams'), checked('answer')]).toEqual([false, false, false, true, true])

    await user.click(screen.getByRole('switch', { name: NAMES.notams }))
    expect(stored()).toEqual({ wind: false, rain: false, airspace: false, notams: false, answer: true })
  })

  it('never reads a later layers.v2 key', async () => {
    window.localStorage.setItem('flysafelz.layers.v2', '{"airspace":false,"notams":false,"answer":false}')
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(NOTAMS))
    await openPanel(user)
    expect([checked('wind'), checked('rain'), checked('airspace'), checked('notams'), checked('answer')]).toEqual([true, true, true, true, true])
  })

  it('G10/C2: What’s-here still lists switched-off layers after a reload with them off', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, '{"airspace":false,"notams":false}')
    const { map, user } = load(blueMillRoutes())
    await waitFor(() => expect(screen.getByRole('button', { name: 'Map layers' })).toBeInTheDocument())
    await waitFor(() => expect(drawn(map, 'airspace')).toBe(0))
    // Wait for both routes: the card says which are still loading until they have arrived.
    await openPanel(user)
    await waitFor(() => expect(screen.getAllByText(/^updated \d\d:\d\d$/)).toHaveLength(3))
    await user.keyboard('{Escape}')

    act(() => void map.fire('click', { latlng: L.latLng(BLUE_MILL.lat, BLUE_MILL.lon) }))
    const card = await screen.findByRole('region', { name: "What's here" })
    expect(within(card).getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(within(card).getByText('H2001/26')).toBeInTheDocument()
    expect([drawn(map, 'airspace'), drawn(map, 'notams')]).toEqual([0, 0])
  })
})

describe('E13: storage that is missing, corrupt or failing', () => {
  it('shows the defaults for a corrupt value, renders with no error, and the next switch overwrites it with valid JSON', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, '{not json')
    const errors: unknown[] = []
    const onError = (event: ErrorEvent) => errors.push(event.error)
    window.addEventListener('error', onError)
    const { map, user, view } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(NOTAMS))
    expect(drawn(map, 'airspace')).toBe(AIRSPACE)
    await openPanel(user)
    expect([checked('wind'), checked('rain'), checked('airspace'), checked('notams'), checked('answer')]).toEqual([true, true, true, true, true])
    expect(view.container.querySelector('.leaflet-container')).not.toBeNull()
    expect(screen.queryByText(/something went wrong/i)).toBeNull()

    await user.click(screen.getByRole('switch', { name: NAMES.airspace }))
    expect(stored()).toEqual({ wind: true, rain: true, airspace: false, notams: true, answer: true })
    window.removeEventListener('error', onError)
    expect(errors).toEqual([])
  })

  it.each([['null'], ['[]'], ['"true"'], ['42'], ['{"airspace":"yes"}'], ['{"airspace":"yes","notams":false}']])(
    'shows the defaults for the bad entries in %s and applies the valid ones',
    async (value) => {
      window.localStorage.setItem(LAYER_PREFS_KEY, value)
      const { map, user } = load()
      await waitFor(() => expect(drawn(map, 'airspace')).toBe(AIRSPACE))
      await openPanel(user)
      expect([checked('airspace'), checked('answer')]).toEqual([true, true])
      expect(checked('notams')).toBe(!value.includes('"notams":false'))
    },
  )

  it('works for the session when window.localStorage throws a SecurityError: defaults, switches change the map, nothing shown', async () => {
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    })
    const { map, user, view } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(NOTAMS))
    expect(drawn(map, 'airspace')).toBe(AIRSPACE)
    await openPanel(user)
    await user.click(screen.getByRole('switch', { name: NAMES.airspace }))
    expect(checked('airspace')).toBe(false)
    expect(drawn(map, 'airspace')).toBe(0)
    expect(drawn(map, 'notams')).toBe(NOTAMS)
    await user.click(screen.getByRole('switch', { name: NAMES.airspace }))
    expect(drawn(map, 'airspace')).toBe(AIRSPACE)
    expect(screen.queryByRole('alert')).toBeNull()
    expect(view.container).not.toHaveTextContent(/storage|quota|error/i)
  })

  it('keeps the map and a switch working when setItem throws QuotaExceededError, and shows nothing about it', async () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
    })
    const { map, user, view } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(NOTAMS))
    await openPanel(user)
    await user.click(screen.getByRole('switch', { name: NAMES.notams }))
    expect(drawn(map, 'notams')).toBe(0)
    expect(checked('notams')).toBe(false)
    expect(screen.queryByRole('alert')).toBeNull()
    expect(view.container).not.toHaveTextContent(/storage|quota|error/i)
  })

  it('still runs click → What’s-here → Chat about this LZ with storage blocked', async () => {
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    })
    const { map } = load(blueMillRoutes())
    await waitFor(() => expect(drawn(map, 'notams')).toBeGreaterThan(0))
    act(() => void map.fire('click', { latlng: L.latLng(BLUE_MILL.lat, BLUE_MILL.lon) }))
    const card = await screen.findByRole('region', { name: "What's here" })
    expect(within(card).getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    act(() => screen.getByRole('button', { name: 'Chat about this LZ' }).click())
    const sheet = await screen.findByRole('dialog', { name: /\d/ })
    expect(within(sheet).getByRole('textbox', { name: 'Ask about this LZ' })).toBeInTheDocument()
  })
})

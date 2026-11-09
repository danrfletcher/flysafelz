import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { LAYER_PREFS_KEY } from '@/lib/layerPrefs'
import { RAINVIEWER_INDEX_URL } from '@/map/rainIndex'
import { RAIN_PANE } from '@/map/rainPane'
import { RAIN_GRID_OK, RAIN_GRID_URL, frameTimes, rainGridRoute, rainIndexRoute, type RainIndexAnswer } from '@/mocks/rain'
import { mapRoutes } from '@/test/handlers'
import { server } from '@/test/setup'

// The Rain layer through the real App (G5, E3, E12): RainViewer's index and /map/rain-grid come from MSW; the radar is the real Leaflet tile layers on the real map.
// jsdom loads no images, so a test fires `load` on the radar's tile images when it wants them to have arrived.

/**
 * The radar's tile throttle (85 a minute, see rainScheduler.ts) would make these tests wait minutes for a screenful of tiles, so they run it on a fast clock: every
 * read of Date.now moves 100 ms on. The throttle itself is tested with fake timers in rainScheduler.test.ts.
 */
function fastClock() {
  let at = Date.now()
  vi.spyOn(Date, 'now').mockImplementation(() => (at += 100))
}

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

// --- helpers -----------------------------------------------------------------------------------------------------------

/** Mounts the app with the map routes, RainViewer's index answering as told, and (when given) the rain-grid route. */
function load(index: RainIndexAnswer | null = { frames: 3 }, grid: 'ok' | 'stale' | 'unavailable' | null = null, fast = true) {
  server.use(...mapRoutes())
  if (index) server.use(rainIndexRoute(index))
  if (grid) server.use(rainGridRoute(grid))
  // The wind particles redraw a moment after every zoom, on a timer that can fire after the test's map is gone, so these tests run with Wind off (windLayer.test.tsx has it on).
  window.localStorage.setItem(LAYER_PREFS_KEY, JSON.stringify({ wind: false, ...(JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) ?? '{}') as object) }))
  const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  if (fast) fastClock()
  const view = render(<App />)
  return { view, container: view.container, map: fit.mock.contexts[0] as L.Map, user: userEvent.setup() }
}

const pane = (container: HTMLElement) => container.querySelector('.leaflet-rain-pane') as HTMLElement
/** The radar's tile images that have a URL (the throttle holds the rest back with none yet). The pane exists only while a rain layer is drawn. */
const tiles = (container: HTMLElement) => [...(pane(container)?.querySelectorAll('img') ?? [])].filter((img) => img.getAttribute('src'))
const squares = (container: HTMLElement) => pane(container)?.querySelectorAll('svg path') ?? []
const rainLayers = (map: L.Map) => {
  const found: L.TileLayer[] = []
  map.eachLayer((layer) => {
    if (layer instanceof L.TileLayer && layer.options.pane === RAIN_PANE) found.push(layer)
  })
  return found
}
const arrive = (container: HTMLElement) => act(() => tiles(container).forEach((img) => img.dispatchEvent(new Event('load'))))
/** Pane z-indexes as Leaflet's stylesheet sets them (jsdom does not load it). */
const LEAFLET_CSS = readFileSync(resolve(__dirname, '../../node_modules/leaflet/dist/leaflet.css'), 'utf8')
const leafletZ = (pane: string) => Number(new RegExp(`\\.leaflet-${pane}-pane\\s*\\{\\s*z-index:\\s*(\\d+)`).exec(LEAFLET_CSS)?.[1])
/** Fires `load` on the radar's tiles again and again until `count` frame layers exist, and (when `all`) every one of them has its tiles: the throttle hands tiles out over time. */
const arriveUntil = (container: HTMLElement, map: L.Map, count: number) =>
  waitFor(
    () => {
      arrive(container)
      expect(rainLayers(map)).toHaveLength(count)
    },
    { timeout: 5000 },
  )
const stamp = () => screen.queryByLabelText('Radar frame time')
const rainSwitch = () => screen.getByRole('switch', { name: 'Rain' })
const stored = () => JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) as string) as Record<string, unknown>

async function openPanel(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'Map layers' }))
  return screen.findByRole('dialog', { name: /layers/i })
}
const rainRow = (panel: HTMLElement) => within(panel).getAllByRole('listitem')[1]

/** "HH:MM" as the app shows it: the browser's time zone, 24 hour clock. */
const hhmm = (when: Date) => new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(when)

function mockReducedMotion(initial: boolean) {
  let reduced = initial
  const listeners = new Set<(event: MediaQueryListEvent) => void>()
  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return query.includes('prefers-reduced-motion') ? reduced : false
    },
    media: query,
    addEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
    onchange: null,
  }))
  return (value: boolean) => {
    reduced = value
    act(() => listeners.forEach((listener) => listener({ matches: value } as MediaQueryListEvent)))
  }
}

// --- the radar path ----------------------------------------------------------------------------------------------------

describe('G5: the radar', () => {
  it('draws the latest frame first from RainViewer’s tile host, shows its time on the map and in the panel row, and Rain is on by default', async () => {
    const times = frameTimes(Date.now(), 3)
    const { container, user } = load({ times })
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    const latest = `/v2/radar/${times[2].toString(16)}/256/`
    expect(tiles(container).length).toBeGreaterThan(0)
    expect(tiles(container).every((img) => img.src.startsWith(`https://tilecache.rainviewer.com${latest}`))).toBe(true)
    const expected = hhmm(new Date(times[2] * 1000))
    await waitFor(() => expect(stamp()).toHaveTextContent(`Radar ${expected}`))
    const panel = await openPanel(user)
    expect(rainSwitch()).toBeChecked()
    expect(rainRow(panel)).toHaveTextContent(new RegExp(`^Rain\\s*frame ${expected}$`))
    expect(rainRow(panel).textContent).not.toMatch(/updated/)
  })

  it('asks for the other frames once the latest has its tiles, and loops through them with the time following the frame on screen', async () => {
    const times = frameTimes(Date.now(), 3)
    const { container, map, user } = load({ times })
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    expect(rainLayers(map)).toHaveLength(1)
    await arriveUntil(container, map, 3)
    const panel = await openPanel(user)
    // the loop starts once every frame has its tiles, holds the latest for 1.5 s, then goes oldest first
    await waitFor(
      () => {
        arrive(container)
        expect(stamp()).toHaveTextContent(`Radar ${hhmm(new Date(times[0] * 1000))}`)
      },
      { timeout: 6000 },
    )
    expect(rainRow(panel)).toHaveTextContent(`frame ${hhmm(new Date(times[0] * 1000))}`)
    await waitFor(() => expect(stamp()).toHaveTextContent(`Radar ${hhmm(new Date(times[1] * 1000))}`), { timeout: 4000 })
  })

  it('never asks for a tile above zoom 7, and fades: 60% to zoom 7, 30% at zoom 8, nothing from zoom 9, with the panel saying "radar: regional view only"', async () => {
    const { container, map, user } = load()
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    const panel = await openPanel(user)
    const zooms = () => tiles(container).map((img) => Number(/\/256\/(\d+)\//.exec(img.src)?.[1] ?? 0))
    for (const [zoom, opacity, regional] of [
      [6, '0.6', false],
      [7, '0.6', false],
      [8, '0.3', false],
      [9, '0', true],
      [12, '0', true],
      [6, '0.6', false],
    ] as const) {
      act(() => void map.setZoom(zoom, { animate: false }))
      await waitFor(() => expect(Number(pane(container).style.opacity)).toBeCloseTo(Number(opacity), 6))
      expect(Math.max(0, ...zooms())).toBeLessThanOrEqual(7)
      expect(within(panel).queryByText('radar: regional view only') !== null).toBe(regional)
      expect(stamp() === null).toBe(regional)
    }
  })

  it('puts the rain between the basemap and every outline, takes no pointer events, and calls only the index (no rain-grid, no Open-Meteo)', async () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const { container } = load()
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    expect(pane(container).style.zIndex).toBe('250')
    expect(pane(container).style.pointerEvents).toBe('none')
    expect(leafletZ('tile')).toBeLessThan(250) // the basemap
    expect(leafletZ('overlay')).toBeGreaterThan(250) // airspace, NOTAM and answer shapes
    const urls = fetched.mock.calls.map(([input]) => String(input))
    expect(urls.filter((url) => url === RAINVIEWER_INDEX_URL)).toHaveLength(1)
    expect(urls.some((url) => url === RAIN_GRID_URL() || /open-meteo/.test(url))).toBe(false)
  })

  it('shows the legend line "Rain radar, past 2 h"', async () => {
    const { user } = load()
    const panel = await openPanel(user)
    await waitFor(() => expect(within(panel).getByText('Rain radar, past 2 h')).toBeInTheDocument())
    expect(within(panel).queryByText('rain (model, not radar)')).toBeNull()
  })

  it('credits RainViewer, with a link, and Open-Meteo in the map’s attribution', () => {
    const { container } = load()
    const credits = container.querySelector('.leaflet-control-attribution') as HTMLElement
    expect(credits).toHaveTextContent('RainViewer')
    expect(credits).toHaveTextContent('Open-Meteo')
    const links = [...credits.querySelectorAll('a')].map((a) => a.getAttribute('href'))
    expect(links).toEqual(expect.arrayContaining(['https://www.rainviewer.com', 'https://open-meteo.com']))
  })

  it('shows a single frame still, with no error', async () => {
    const times = frameTimes(Date.now(), 1)
    const { container, map } = load({ times })
    await waitFor(() => expect(stamp()).toHaveTextContent(`Radar ${hhmm(new Date(times[0] * 1000))}`))
    arrive(container)
    await act(() => new Promise<void>((done) => setTimeout(done, 100)))
    expect(rainLayers(map)).toHaveLength(1)
    expect(stamp()).toHaveTextContent(`Radar ${hhmm(new Date(times[0] * 1000))}`)
  })

  it('leaves a gap for a tile that fails, and keeps going', async () => {
    const { container, map } = load()
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    // every tile of every frame fails, as they are handed out
    await waitFor(
      () => {
        act(() => tiles(container).forEach((img) => img.dispatchEvent(new Event('error'))))
        expect(rainLayers(map)).toHaveLength(3) // a frame whose tiles all failed still counts as in
      },
      { timeout: 5000 },
    )
    expect(tiles(container).length).toBeGreaterThan(0)
    expect(tiles(container).some((img) => img.src.startsWith('data:image/gif'))).toBe(true) // a transparent gap, not a broken image
    expect(screen.queryByText('Rain unavailable')).toBeNull()
  })

  it('drops frames older than 2 hours from the index', async () => {
    const now = Date.now()
    const ok = frameTimes(now, 2)
    const old = Math.floor(now / 1000) - 3 * 3600
    const { container, map } = load({ times: [old, ...ok] })
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    await arriveUntil(container, map, 2)
  })
})

// --- the switch --------------------------------------------------------------------------------------------------------

describe('G1, G5: the Rain switch', () => {
  it('takes the radar off and the timestamp with it, and on again without asking the index again', async () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const { container, map, user } = load()
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    await arriveUntil(container, map, 3)
    await openPanel(user)

    await user.click(rainSwitch())
    expect(rainSwitch()).not.toBeChecked()
    expect(rainLayers(map)).toHaveLength(0)
    expect(tiles(container)).toHaveLength(0)
    expect(stamp()).toBeNull()
    expect(stored().rain).toBe(false)

    await user.click(rainSwitch())
    expect(rainSwitch()).toBeChecked()
    await waitFor(() => expect(rainLayers(map)).toHaveLength(1))
    expect(stamp()).not.toBeNull()
    expect(stored().rain).toBe(true)
    expect(fetched.mock.calls.filter(([input]) => String(input) === RAINVIEWER_INDEX_URL)).toHaveLength(1)
  })

  it('off and on again quickly leaves one layer set and one loop', async () => {
    const times = frameTimes(Date.now(), 3)
    const { container, map, user } = load({ times })
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    await openPanel(user)
    for (let i = 0; i < 5; i += 1) {
      await user.click(rainSwitch())
      await user.click(rainSwitch())
    }
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    expect(rainLayers(map)).toHaveLength(1)
    await arriveUntil(container, map, 3)
    expect(pane(container)).not.toBeNull()
    expect(container.querySelectorAll('.leaflet-rain-pane')).toHaveLength(1)
  })

  it('fast zooming 7 to 9 to 7 leaves one layer set and the opacity at the rule’s value', async () => {
    const { container, map } = load()
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    await arriveUntil(container, map, 3)
    for (let i = 0; i < 4; i += 1) {
      act(() => void map.setZoom(9, { animate: false }))
      act(() => void map.setZoom(7, { animate: false }))
    }
    await waitFor(() => expect(Number(pane(container).style.opacity)).toBeCloseTo(0.6, 6))
    expect(rainLayers(map)).toHaveLength(3)
    expect(container.querySelectorAll('.leaflet-rain-pane')).toHaveLength(1)
  })

  it('a stored "off" choice draws no radar at all', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, JSON.stringify({ rain: false }))
    const { container, map, user } = load()
    const panel = await openPanel(user)
    await waitFor(() => expect(rainRow(panel)).toHaveTextContent(/frame \d\d:\d\d/))
    expect(rainSwitch()).not.toBeChecked()
    expect(rainLayers(map)).toHaveLength(0)
    expect(container.querySelectorAll('.leaflet-rain-pane img')).toHaveLength(0)
    expect(stamp()).toBeNull()
  })
})

// --- E12 ---------------------------------------------------------------------------------------------------------------

describe('E12: reduced motion', () => {
  it('makes the latest frame only: no other frame’s tiles, no loop, a time that does not change', async () => {
    mockReducedMotion(true)
    const times = frameTimes(Date.now(), 3)
    const { container, map } = load({ times })
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    arrive(container)
    await act(() => new Promise<void>((done) => setTimeout(done, 2500)))
    expect(rainLayers(map)).toHaveLength(1)
    expect(tiles(container).every((img) => img.src.includes(`/v2/radar/${times[2].toString(16)}/`))).toBe(true)
    expect(stamp()).toHaveTextContent(`Radar ${hhmm(new Date(times[2] * 1000))}`)
  }, 10_000)

  it('follows the setting live: motion back on brings the other frames', async () => {
    const set = mockReducedMotion(true)
    const { container, map } = load()
    await waitFor(() => expect(tiles(container).length).toBeGreaterThan(0))
    act(() => tiles(container).forEach((img) => img.dispatchEvent(new Event('load'))))
    expect(rainLayers(map)).toHaveLength(1)
    set(false)
    await arriveUntil(container, map, 3)
  })
})

// --- the fallback ------------------------------------------------------------------------------------------------------

describe('E3: RainViewer’s index fails', () => {
  it.each<[string, RainIndexAnswer]>([
    ['a network error', 'network-error'],
    ['a 500', 'server-error'],
    ['a body that is not JSON', 'not-json'],
    ['a body that is not an index', 'not-an-index'],
    ['an empty radar.past', 'empty'],
  ])('with %s, draws /map/rain-grid as soft cells, says "rain (model, not radar)", and has no radar tiles', async (_name, index) => {
    const { container, user } = load(index, 'ok')
    await waitFor(() => expect(squares(container).length).toBeGreaterThan(0))
    const wet = RAIN_GRID_OK.grid.filter((cell) => cell.precipitation >= 0.05).length
    expect(squares(container)).toHaveLength(wet)
    expect(tiles(container)).toHaveLength(0)
    expect(stamp()).toBeNull()
    expect(pane(container)).toHaveClass('rain-pane--model')
    const panel = await openPanel(user)
    expect(within(panel).getByText('rain (model, not radar)')).toBeInTheDocument()
    expect(within(panel).queryByText('Rain radar, past 2 h')).toBeNull()
    expect(rainRow(panel)).toHaveTextContent(`updated ${hhmm(new Date(RAIN_GRID_OK.updated_at))}`)
    expect(rainRow(panel).textContent).not.toMatch(/stale|frame/)
  })

  it('shows "stale" in the row when the route says so', async () => {
    const { user } = load('server-error', 'stale')
    const panel = await openPanel(user)
    await waitFor(() => expect(rainRow(panel)).toHaveTextContent(/updated \d\d:\d\d\s*stale/))
  })

  it('draws cells display-only: no pointer events, no tooltip and no value in the text', async () => {
    const { container } = load('server-error', 'ok')
    await waitFor(() => expect(squares(container).length).toBeGreaterThan(0))
    expect(pane(container).style.pointerEvents).toBe('none')
    expect(container.querySelectorAll('.leaflet-tooltip, .leaflet-popup')).toHaveLength(0)
    expect(pane(container).textContent).toBe('')
    expect([...squares(container)].some((path) => path.classList.contains('leaflet-interactive'))).toBe(false)
  })

  it('has the same zoom fade as the radar, and the same panel note', async () => {
    const { container, map, user } = load('server-error', 'ok')
    await waitFor(() => expect(squares(container).length).toBeGreaterThan(0))
    const panel = await openPanel(user)
    act(() => void map.setZoom(8, { animate: false }))
    await waitFor(() => expect(Number(pane(container).style.opacity)).toBeCloseTo(0.3, 6))
    act(() => void map.setZoom(9, { animate: false }))
    await waitFor(() => expect(Number(pane(container).style.opacity)).toBe(0))
    expect(within(panel).getByText('rain (model): regional view only')).toBeInTheDocument()
  })

  it('goes through /map/rain-grid: the browser never calls Open-Meteo', async () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const { container } = load('server-error', 'ok')
    await waitFor(() => expect(squares(container).length).toBeGreaterThan(0))
    const urls = fetched.mock.calls.map(([input]) => String(input))
    expect(urls.filter((url) => url === RAIN_GRID_URL())).toHaveLength(1)
    expect(urls.some((url) => /open-meteo/.test(url))).toBe(false)
  })

  it('goes away with the switch', async () => {
    const { container, user } = load('server-error', 'ok')
    await waitFor(() => expect(squares(container).length).toBeGreaterThan(0))
    await openPanel(user)
    await user.click(rainSwitch())
    expect(squares(container)).toHaveLength(0)
  })
})

describe('E3: both fail', () => {
  it.each<[string, 'unavailable' | null]>([
    ['the rain-grid route answers 503', 'unavailable'],
    ['the rain-grid route is unreachable', null],
  ])('hides the layer and says "Rain unavailable" when %s, and the other layers carry on', async (_name, grid) => {
    if (grid === null) server.use(http.get(RAIN_GRID_URL(), () => HttpResponse.error()))
    const { container, user } = load('server-error', grid)
    const panel = await openPanel(user)
    await waitFor(() => expect(within(panel).getByText('Rain unavailable')).toBeInTheDocument())
    expect(tiles(container)).toHaveLength(0)
    expect(squares(container)).toHaveLength(0)
    expect(stamp()).toBeNull()
    expect(within(panel).queryByText('rain (model, not radar)')).toBeNull()
    expect(within(panel).queryByText('Rain radar, past 2 h')).toBeNull()
    await waitFor(() => expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(/^Wind\s*updated \d\d:\d\d$/))
    await waitFor(() => expect(within(panel).getAllByRole('listitem')[2]).toHaveTextContent(/^Airspace below FL100\s*updated \d\d:\d\d/))
    expect(screen.getByRole('button', { name: 'Map layers' })).toBeEnabled()
  })
})

describe('the row while loading', () => {
  it('says Loading… until RainViewer answers', async () => {
    server.use(...mapRoutes())
    let release!: () => void
    const gate = new Promise<void>((done) => (release = done))
    server.use(
      http.get(RAINVIEWER_INDEX_URL, async () => {
        await gate
        return HttpResponse.json({ host: 'https://tilecache.rainviewer.com', radar: { past: [{ time: Math.floor(Date.now() / 1000) - 300, path: '/v2/radar/a' }] } })
      }),
    )
    const user = userEvent.setup()
    render(<App />)
    const panel = await openPanel(user)
    expect(rainRow(panel)).toHaveTextContent(/^Rain\s*Loading…$/)
    release()
    await waitFor(() => expect(rainRow(panel)).toHaveTextContent(/frame \d\d:\d\d/))
  })
})

import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { LAYER_PREFS_KEY } from '@/lib/layerPrefs'
import { WIND_OK, WIND_UNAVAILABLE, WIND_URL, windRoute } from '@/mocks/wind'
import { BLUE_MILL, blueMillRoutes, mapRoutes } from '@/test/handlers'
import { server } from '@/test/setup'

// The Wind layer through the real App: the recorded /map/wind answers come from MSW (ok, stale, 503), the particle layer is the real leaflet-velocity
// drawing on a canvas (jsdom's canvas context is a stub, src/test/setup.ts), and animation frames are counted by wrapping requestAnimationFrame.

let fit: ReturnType<typeof vi.spyOn>

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
  server.use(...mapRoutes())
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

// --- helpers -----------------------------------------------------------------------------------------------------------

/** Frames asked for and not yet run or cancelled, and how many have been asked for in all. */
function watchFrames() {
  const pending = new Set<number>()
  let requested = 0
  const request = window.requestAnimationFrame.bind(window)
  const cancel = window.cancelAnimationFrame.bind(window)
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
    requested += 1
    const id = request((time) => {
      pending.delete(id)
      callback(time)
    })
    pending.add(id)
    return id
  })
  vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((id) => {
    pending.delete(id)
    cancel(id)
  })
  return { pending, requested: () => requested }
}

/** A matchMedia that answers `prefers-reduced-motion: reduce` as told and fires `change` when told to. */
function mockReducedMotion(initial: boolean) {
  const listeners = new Set<(event: MediaQueryListEvent) => void>()
  let reduced = initial
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
  return {
    set(value: boolean) {
      reduced = value
      act(() => listeners.forEach((listener) => listener({ matches: value } as MediaQueryListEvent)))
    },
    listeners: () => listeners.size,
  }
}

const canvases = (container: HTMLElement) => container.querySelectorAll('canvas.velocity-overlay').length
const arrows = (container: HTMLElement) => container.querySelectorAll('.wind-arrow').length

function load() {
  fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  const view = render(<App />)
  return { view, container: view.container, map: fit.mock.contexts[0] as L.Map, user: userEvent.setup() }
}

async function openPanel(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'Map layers' }))
  return screen.findByRole('dialog', { name: /layers/i })
}

const windSwitch = () => screen.getByRole('switch', { name: 'Wind' })
const stored = () => JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) as string) as Record<string, unknown>
const flush = (ms = 60) => act(() => new Promise<void>((done) => setTimeout(done, ms)))

// --- on by default, particles ----------------------------------------------------------------------------------------------

describe('G4: the particle layer', () => {
  it('draws one particle canvas once /map/wind has arrived, with Wind on and its time in the panel', async () => {
    const { container, user } = load()
    expect(canvases(container)).toBe(0) // nothing before the grid is held
    await waitFor(() => expect(canvases(container)).toBe(1))
    expect(arrows(container)).toBe(0)
    const panel = await openPanel(user)
    expect(windSwitch()).toBeChecked()
    expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(/^Wind\s*updated \d\d:\d\d$/)
  })

  it('shows the time in the browser’s time zone, from the route’s updated_at', async () => {
    const { user } = load()
    const panel = await openPanel(user)
    const expected = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(WIND_OK.updated_at))
    await waitFor(() => expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(`updated ${expected}`))
  })

  it('makes the one request to the app’s own /map/wind route and none to Open-Meteo', async () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const { container } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    const urls = fetched.mock.calls.map(([input]) => String(input))
    expect(urls.filter((url) => url === WIND_URL())).toHaveLength(1)
    expect(urls.some((url) => /open-meteo/.test(url))).toBe(false)
  })

  it('animates with the library’s own loop while on', async () => {
    const frames = watchFrames()
    const { container } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    await waitFor(() => expect(frames.requested()).toBeGreaterThan(0))
    await flush(100)
    expect(frames.pending.size).toBeGreaterThan(0)
  })

  it('gives the canvas no pointer events, so a click on the map goes through to it (E14)', async () => {
    const css = (await import('node:fs')).readFileSync((await import('node:path')).resolve(__dirname, '../styles/index.css'), 'utf8')
    expect(css).toMatch(/\.velocity-overlay\s*\{[^}]*pointer-events:\s*none/)
    const { container, map } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    const click = vi.fn()
    map.on('click', click)
    act(() => void map.fire('click', { latlng: L.latLng(BLUE_MILL.lat, BLUE_MILL.lon) }))
    expect(click).toHaveBeenCalledTimes(1)
  })
})

// --- toggling -----------------------------------------------------------------------------------------------------------------

describe('G1: the Wind switch', () => {
  it('turning it off removes the canvas and stops its animation loop, and turning it on again uses the grid already fetched', async () => {
    const frames = watchFrames()
    const requests: string[] = []
    server.use(
      http.get(WIND_URL(), () => {
        requests.push('wind')
        return HttpResponse.json(WIND_OK)
      }),
    )
    const { container, user } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    await flush(100)
    await openPanel(user)

    await user.click(windSwitch())
    expect(windSwitch()).not.toBeChecked()
    expect(canvases(container)).toBe(0)
    const askedBefore = frames.requested()
    await flush(150) // a running loop would ask for more frames in this time
    expect(frames.requested()).toBe(askedBefore)
    expect(frames.pending.size).toBe(0)

    await user.click(windSwitch())
    await waitFor(() => expect(canvases(container)).toBe(1))
    expect(requests).toEqual(['wind']) // no new request
  })

  it('leaves at most one velocity canvas after ten quick toggles, and none after an even number', async () => {
    const frames = watchFrames()
    const { container, user } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    await openPanel(user)
    for (let click = 0; click < 10; click += 1) {
      await user.click(windSwitch())
      expect(canvases(container)).toBeLessThanOrEqual(1)
    }
    expect(windSwitch()).toBeChecked() // ten clicks end where they began
    await waitFor(() => expect(canvases(container)).toBe(1))
    await user.click(windSwitch())
    expect(canvases(container)).toBe(0)
    await flush(150)
    expect(frames.pending.size).toBe(0)
  })

  it('does not throw from a frame or a timer the layer left behind after a quick off', async () => {
    const errors: unknown[] = []
    const onError = (event: ErrorEvent) => errors.push(String(event.error?.stack))
    window.addEventListener('error', onError)
    const { container, user } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    await openPanel(user)
    for (let click = 0; click < 6; click += 1) await user.click(windSwitch())
    await flush(200)
    window.removeEventListener('error', onError)
    expect(errors).toEqual([])
  })

  it('leaves none of its map listeners behind when it goes off', async () => {
    const { container, map, user } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    const events = map as unknown as { _events: Record<string, unknown[]> }
    const count = () => ['dragstart', 'dragend', 'zoomstart', 'zoomend', 'resize', 'moveend'].map((name) => events._events[name]?.length ?? 0)
    const withWind = count()
    await openPanel(user)
    await user.click(windSwitch())
    expect(canvases(container)).toBe(0)
    const without = count()
    expect(without.reduce((a, b) => a + b, 0)).toBeLessThan(withWind.reduce((a, b) => a + b, 0))
    await user.click(windSwitch())
    await waitFor(() => expect(canvases(container)).toBe(1))
    await user.click(windSwitch())
    expect(count()).toEqual(without) // on and off again adds nothing that stays
  })

  it('stays off and draws nothing when it was saved off, and the saved value is under flysafelz.layers.v1', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, '{"wind":false}')
    const { container, user } = load()
    const panel = await openPanel(user)
    await waitFor(() => expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(/updated/)) // the grid is held
    expect(windSwitch()).not.toBeChecked()
    expect(canvases(container)).toBe(0)
    expect(arrows(container)).toBe(0)
  })

  it('saves its choice, and a reload keeps it', async () => {
    const first = load()
    await waitFor(() => expect(canvases(first.container)).toBe(1))
    await openPanel(first.user)
    await first.user.click(windSwitch())
    expect(stored()).toEqual({ wind: false, rain: true, airspace: true, notams: true, answer: true })
    first.view.unmount() // the reload
    const again = load()
    await openPanel(again.user)
    expect(windSwitch()).not.toBeChecked()
    expect(canvases(again.container)).toBe(0)
  })

  it('E13: a stored choice from before this PR, with no wind entry, leaves Wind on', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, '{"airspace":false,"notams":true,"answer":true}')
    const { container, user } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    await openPanel(user)
    expect(windSwitch()).toBeChecked()
  })

  it('E13: a corrupt value, a blocked storage and a wind entry that is not a boolean all leave Wind on and crash nothing', async () => {
    for (const bad of ['{not json', '{"wind":"off"}', 'null']) {
      window.localStorage.setItem(LAYER_PREFS_KEY, bad)
      const { container, view } = load()
      await waitFor(() => expect(canvases(container)).toBe(1))
      view.unmount()
    }
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    })
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { container } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
  })
})

// --- unmount -----------------------------------------------------------------------------------------------------------------

describe('cleanup', () => {
  it('removes the canvas, the loop and the timer when the app unmounts', async () => {
    const frames = watchFrames()
    const { container, view } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    await flush(100)
    view.unmount()
    expect(document.querySelectorAll('canvas.velocity-overlay')).toHaveLength(0)
    const askedBefore = frames.requested()
    await flush(150)
    expect(frames.requested()).toBe(askedBefore)
    expect(frames.pending.size).toBe(0)
  })
})

// --- reduced motion ------------------------------------------------------------------------------------------------------------

describe('E12: reduced motion', () => {
  it('draws one static arrow per grid point and no particle canvas, and never starts an animation loop', async () => {
    mockReducedMotion(true)
    const frames = watchFrames()
    const { container } = load()
    await waitFor(() => expect(arrows(container)).toBe(143))
    expect(canvases(container)).toBe(0)
    await flush(150)
    expect(frames.requested()).toBe(0)
    expect(frames.pending.size).toBe(0)
  })

  it('points each arrow downwind from the same u/v payload, with its length and speed band', async () => {
    mockReducedMotion(true)
    const { container } = load()
    await waitFor(() => expect(arrows(container)).toBe(143))
    const first = container.querySelector('.wind-arrow') as HTMLElement
    const svg = first.querySelector('svg') as SVGElement | null
    const calm = [...container.querySelectorAll('.wind-arrow')].filter((el) => el.querySelector('circle'))
    expect(calm).toHaveLength(1) // the one calm cell of the recorded grid is a dot
    expect(svg?.getAttribute('style')).toMatch(/rotate\(\d+deg\)/)
    const bands = new Set([...container.querySelectorAll('.wind-arrow')].flatMap((el) => [...el.classList].filter((name) => name.startsWith('wind-arrow--'))))
    expect(bands.size).toBeGreaterThan(1)
    for (const el of container.querySelectorAll('.wind-arrow')) expect(el.classList.length).toBeGreaterThanOrEqual(2)
  })

  it('takes no clicks: an arrow is not interactive', async () => {
    mockReducedMotion(true)
    const { container } = load()
    await waitFor(() => expect(arrows(container)).toBe(143))
    expect(container.querySelectorAll('.wind-arrow.leaflet-interactive')).toHaveLength(0)
  })

  it('swaps both ways with a live change of the setting, with no reload and no extra request', async () => {
    const media = mockReducedMotion(false)
    const requests: string[] = []
    server.use(
      http.get(WIND_URL(), () => {
        requests.push('wind')
        return HttpResponse.json(WIND_OK)
      }),
    )
    const frames = watchFrames()
    const { container, view } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    expect(arrows(container)).toBe(0)

    media.set(true)
    await waitFor(() => expect(arrows(container)).toBe(143))
    expect(canvases(container)).toBe(0)
    await flush(50)
    const askedBefore = frames.requested()
    await flush(150)
    expect(frames.requested()).toBe(askedBefore) // the particle loop stopped with the swap
    expect(frames.pending.size).toBe(0)

    media.set(false)
    await waitFor(() => expect(canvases(container)).toBe(1))
    expect(arrows(container)).toBe(0)
    expect(requests).toEqual(['wind'])

    view.unmount()
    expect(media.listeners()).toBe(0) // the listener is taken off
  })

  it('keeps the Wind switch and the time working in arrows mode, and switching off removes the arrows', async () => {
    mockReducedMotion(true)
    const { container, user } = load()
    await waitFor(() => expect(arrows(container)).toBe(143))
    const panel = await openPanel(user)
    expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(/updated \d\d:\d\d/)
    await user.click(windSwitch())
    expect(arrows(container)).toBe(0)
    await user.click(windSwitch())
    await waitFor(() => expect(arrows(container)).toBe(143))
  })
})

// --- stale, 503, a slow route ------------------------------------------------------------------------------------------------

describe('G6 and E1: stale and unavailable', () => {
  it('shows "stale" next to the time for the stale answer and still draws the grid', async () => {
    server.use(windRoute('stale'))
    const { container, user } = load()
    await waitFor(() => expect(canvases(container)).toBe(1))
    const panel = await openPanel(user)
    expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(/^Wind\s*updated \d\d:\d\d\s*stale$/)
    expect(within(panel).getAllByText('stale')).toHaveLength(1)
  })

  it('shows "Wind unavailable" for a 503, draws nothing, and the map, the other layers and What’s-here still work', async () => {
    server.use(windRoute('unavailable'), ...blueMillRoutes()) // the first handler for a URL wins
    const { container, map, user } = load()
    const panel = await openPanel(user)
    await waitFor(() => expect(within(panel).getByText('Wind unavailable')).toBeInTheDocument())
    expect(within(panel).queryByText(/Airspace below FL100 unavailable|NOTAMs below FL100 unavailable/)).toBeNull()
    expect(canvases(container)).toBe(0)
    expect(arrows(container)).toBe(0)
    expect(WIND_UNAVAILABLE.reason).not.toMatch(/\n/)
    await user.keyboard('{Escape}')

    act(() => void map.fire('click', { latlng: L.latLng(BLUE_MILL.lat, BLUE_MILL.lon) }))
    const card = await screen.findByRole('region', { name: "What's here" })
    expect(within(card).getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
  })

  it.each([
    ['a network error', () => HttpResponse.error()],
    ['text that is not JSON', () => new HttpResponse('<html>Access denied</html>')],
    ['a grid with a null in it', () => HttpResponse.json({ ...WIND_OK, grid: [{ ...WIND_OK.grid[0], data: WIND_OK.grid[0].data.map((x, i) => (i === 3 ? null : x)) }, WIND_OK.grid[1]] })],
  ])('treats %s as unavailable, draws no part of a grid', async (_label, answer) => {
    server.use(http.get(WIND_URL(), answer))
    const { container, user } = load()
    const panel = await openPanel(user)
    await waitFor(() => expect(within(panel).getByText('Wind unavailable')).toBeInTheDocument())
    expect(canvases(container)).toBe(0)
  })

  it('keeps the last grid and marks it stale when a refetch fails, and recovers on the next good answer', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['setInterval', 'clearInterval'] })
    try {
      const { container, user } = load()
      await waitFor(() => expect(canvases(container)).toBe(1))
      const panel = await openPanel(user)
      server.use(http.get(WIND_URL(), () => HttpResponse.error()))
      await act(async () => void vi.advanceTimersByTime(15 * 60 * 1000))
      await waitFor(() => expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(/updated \d\d:\d\d\s*stale/))
      expect(canvases(container)).toBe(1) // the old grid stays on the map
      server.use(windRoute('ok'))
      await act(async () => void vi.advanceTimersByTime(15 * 60 * 1000))
      await waitFor(() => expect(within(panel).getAllByRole('listitem')[0]).not.toHaveTextContent('stale'))
    } finally {
      vi.useRealTimers()
    }
  })

  it('refetches /map/wind every 15 minutes, and not before', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['setInterval', 'clearInterval'] })
    try {
      const requests: number[] = []
      server.use(
        http.get(WIND_URL(), () => {
          requests.push(requests.length)
          return HttpResponse.json(WIND_OK)
        }),
      )
      const { container } = load()
      await waitFor(() => expect(requests).toHaveLength(1))
      await act(async () => void vi.advanceTimersByTime(14 * 60 * 1000))
      expect(requests).toHaveLength(1)
      await act(async () => void vi.advanceTimersByTime(60 * 1000 + 50))
      await waitFor(() => expect(requests).toHaveLength(2))
      expect(canvases(container)).toBe(1) // the same cached grid: still one layer
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not wait on a slow wind route to show the map and the other layers', async () => {
    let release: () => void = () => {}
    const gate = new Promise<void>((done) => (release = done))
    server.use(
      http.get(WIND_URL(), async () => {
        await gate
        return HttpResponse.json(WIND_OK)
      }),
    )
    const { container, map, user } = load()
    const panel = await openPanel(user)
    await waitFor(() => expect(within(panel).getAllByText(/^updated \d\d:\d\d$/)).toHaveLength(2)) // airspace and NOTAMs
    expect(within(panel).getAllByRole('listitem')[0]).toHaveTextContent(/Loading/)
    expect(map.getZoom()).toBeGreaterThan(0)
    expect(canvases(container)).toBe(0)
    release()
    await waitFor(() => expect(canvases(container)).toBe(1))
  })
})

import { booleanPointInPolygon } from '@turf/boolean-point-in-polygon'
import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { LAYER_PREFS_KEY, resetLayerPrefsWarning } from '@/lib/layerPrefs'
import { airspaceBelowFl100, notamBelowFl100 } from '@/map/belowFl100'
import { RAINVIEWER_INDEX_URL } from '@/map/rainIndex'
import { WIND_URL, windRoute } from '@/mocks/wind'
import {
  ALL_HEIGHTS_AIRSPACE,
  ALL_HEIGHTS_NOTAMS,
  AIRSPACE_URL,
  NOTAMS_URL,
  airspaceRoute,
  allHeightsNotams503Routes,
  allHeightsRoutes,
  allHeightsStaleRoutes,
  notamsRoute,
} from '@/test/handlers'
import { rainIndexRoute } from '@/mocks/rain'
import { server } from '@/test/setup'

// The whole path for the "all heights" layers (PR-11.F1): the real App, the synthetic stacked payloads served by MSW (any other request fails the
// test), jsdom's storage, and "a reload" is unmount + render.

const LOW_AIRSPACE = ALL_HEIGHTS_AIRSPACE.features.filter((f) => airspaceBelowFl100(f.properties)).length
const LOW_NOTAMS = ALL_HEIGHTS_NOTAMS.features.filter((f) => notamBelowFl100(f.properties)).length
const ALL_AIRSPACE = ALL_HEIGHTS_AIRSPACE.features.length
const ALL_NOTAMS = ALL_HEIGHTS_NOTAMS.features.length

beforeEach(() => {
  resetLayerPrefsWarning()
  vi.spyOn(console, 'warn').mockImplementation(() => {})
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})
afterEach(() => vi.restoreAllMocks())

function load(routes = allHeightsRoutes()) {
  server.use(...routes)
  const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  fit.mockClear()
  const view = render(<App />)
  const map = fit.mock.contexts.at(-1) as L.Map
  return { view, map, user: userEvent.setup() }
}

function drawn(map: L.Map, kind: 'airspace' | 'notams'): number {
  let count = 0
  map.eachLayer((layer) => {
    const feature = (layer as L.Polygon & { feature?: { properties: object } }).feature
    if (layer instanceof L.Polygon && feature && (kind === 'notams' ? 'q_code' in feature.properties : 'lower_ft' in feature.properties)) count += 1
  })
  return count
}

async function openPanel(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'Map layers' }))
  return screen.findByRole('dialog', { name: /layers/i })
}

const NAMES = { airspaceAll: 'Airspace all heights', notamsAll: 'NOTAMs all heights', airspace: 'Airspace below FL100', notams: 'NOTAMs below FL100' } as const
const sw = (id: keyof typeof NAMES) => screen.getByRole('switch', { name: NAMES[id] })
const stored = () => JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) as string) as Record<string, unknown>

describe('G1: the all-heights layers on the real page', () => {
  it('starts with both off and only the below-FL100 shapes drawn; Wind, Rain and both below layers are on', async () => {
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(LOW_NOTAMS))
    expect(drawn(map, 'airspace')).toBe(LOW_AIRSPACE)
    expect(LOW_AIRSPACE).toBeLessThan(ALL_AIRSPACE)
    await openPanel(user)
    expect(sw('airspaceAll')).not.toBeChecked()
    expect(sw('notamsAll')).not.toBeChecked()
    for (const id of ['airspace', 'notams'] as const) expect(sw(id)).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Wind' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Rain' })).toBeChecked()
  })

  it('draws every feature with all heights on, each once whether or not the below switch is on, and the set the other switch implies when one goes off', async () => {
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(LOW_NOTAMS))
    await openPanel(user)

    await user.click(sw('airspaceAll'))
    await user.click(sw('notamsAll'))
    expect([drawn(map, 'airspace'), drawn(map, 'notams')]).toEqual([ALL_AIRSPACE, ALL_NOTAMS]) // both on: once each, not below + all
    await user.click(sw('airspace'))
    await user.click(sw('notams'))
    expect([drawn(map, 'airspace'), drawn(map, 'notams')]).toEqual([ALL_AIRSPACE, ALL_NOTAMS]) // all alone
    await user.click(sw('airspaceAll'))
    await user.click(sw('notamsAll'))
    expect([drawn(map, 'airspace'), drawn(map, 'notams')]).toEqual([0, 0])
    await user.click(sw('airspace'))
    expect([drawn(map, 'airspace'), drawn(map, 'notams')]).toEqual([LOW_AIRSPACE, 0]) // each route on its own
    expect(stored()).toMatchObject({ airspace: true, airspaceAll: false, notams: false, notamsAll: false })
  })

  it('F2: with every layer on, fetches /map/airspace and /map/notams once each, plus wind and RainViewer’s index, and nothing else; a switch fetches nothing', async () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const { map, user } = load()
    await waitFor(() => expect(fetched).toHaveBeenCalledTimes(4))
    await waitFor(() => expect(drawn(map, 'notams')).toBe(LOW_NOTAMS))
    await openPanel(user)
    await user.click(sw('airspaceAll'))
    await user.click(sw('notamsAll'))
    expect(drawn(map, 'airspace')).toBe(ALL_AIRSPACE)
    const urls = fetched.mock.calls.map(([input]) => String(input)).sort()
    expect(urls).toEqual([AIRSPACE_URL(), NOTAMS_URL(), WIND_URL(), RAINVIEWER_INDEX_URL].sort())
  })

  it('F2: logs no feature or NOTAM text with every layer on (console.warn carries ids and types only)', async () => {
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(LOW_NOTAMS))
    await openPanel(user)
    await user.click(sw('airspaceAll'))
    await user.click(sw('notamsAll'))
    const logged = JSON.stringify(vi.mocked(console.warn).mock.calls)
    expect(logged).not.toContain('EXAMPLE AIR DISPLAY')
    expect(logged).not.toContain('7946')
    expect(logged).not.toMatch(/SAMPLE [A-Z ]+ \d{3}/) // no airspace name
  })

  it('saves both choices under flysafelz.layers.v1 and a reload keeps them, with the shapes drawn', async () => {
    const first = load()
    await waitFor(() => expect(drawn(first.map, 'notams')).toBe(LOW_NOTAMS))
    await openPanel(first.user)
    await first.user.click(sw('airspaceAll'))
    await first.user.click(sw('notamsAll'))
    expect(stored()).toEqual({ wind: true, rain: true, airspace: true, airspaceAll: true, notams: true, notamsAll: true, answer: true })
    first.view.unmount()

    const second = load()
    await waitFor(() => expect(drawn(second.map, 'airspace')).toBe(ALL_AIRSPACE))
    expect(drawn(second.map, 'notams')).toBe(ALL_NOTAMS)
    await openPanel(second.user)
    expect(sw('airspaceAll')).toBeChecked()
    expect(sw('notamsAll')).toBeChecked()
  })

  it('E13: a choice stored before this PR (no new keys) keeps its values and leaves the new layers off', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, '{"wind":true,"rain":false,"airspace":false,"notams":true,"answer":true}')
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(LOW_NOTAMS))
    expect(drawn(map, 'airspace')).toBe(0)
    await openPanel(user)
    expect(sw('airspace')).not.toBeChecked()
    expect(sw('airspaceAll')).not.toBeChecked()
    expect(sw('notamsAll')).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'Rain' })).not.toBeChecked()
  })

  it('E13: corrupt storage gives every default and a working page', async () => {
    window.localStorage.setItem(LAYER_PREFS_KEY, '{not json')
    const { map, user } = load()
    await waitFor(() => expect(drawn(map, 'notams')).toBe(LOW_NOTAMS))
    await openPanel(user)
    expect(sw('airspaceAll')).not.toBeChecked()
    expect(sw('airspace')).toBeChecked()
  })
})

describe('E1/E15: stale, 503 and empty routes', () => {
  it('E1: a stale route draws its cached data in both layers and the panel says "stale" for both', async () => {
    const { map, user } = load(allHeightsStaleRoutes())
    await waitFor(() => expect(drawn(map, 'notams')).toBe(LOW_NOTAMS))
    const dialog = await openPanel(user)
    await user.click(sw('airspaceAll'))
    expect(drawn(map, 'airspace')).toBe(ALL_AIRSPACE)
    expect(within(dialog).getAllByText('stale')).toHaveLength(4) // each route's two layers
  })

  it('E1/G6: /map/notams answering 503 hides both NOTAM layers with one note; airspace, both of its layers, and the map still work', async () => {
    const { map, user } = load(allHeightsNotams503Routes())
    await waitFor(() => expect(drawn(map, 'airspace')).toBe(LOW_AIRSPACE))
    const dialog = await openPanel(user)
    await user.click(sw('notamsAll'))
    await user.click(sw('airspaceAll'))
    expect(drawn(map, 'notams')).toBe(0)
    expect(drawn(map, 'airspace')).toBe(ALL_AIRSPACE)
    expect(within(dialog).getAllByText(/unavailable/)).toHaveLength(1)
    expect(within(dialog).getByText('NOTAMs below FL100 unavailable')).toBeInTheDocument()
    const before = map.getCenter()
    map.panBy([100, 0], { animate: false })
    expect(map.getCenter().lng).not.toBe(before.lng)
  })

  it('E15: zero NOTAMs gives two empty NOTAM layers with "updated HH:MM" and no "stale"', async () => {
    const { map, user } = load([airspaceRoute(ALL_HEIGHTS_AIRSPACE), notamsRoute({ features: [], updated_at: ALL_HEIGHTS_NOTAMS.updated_at, stale: false }), windRoute(), rainIndexRoute()])
    await waitFor(() => expect(drawn(map, 'airspace')).toBe(LOW_AIRSPACE))
    const dialog = await openPanel(user)
    await user.click(sw('notamsAll'))
    expect(drawn(map, 'notams')).toBe(0)
    await waitFor(() => expect(within(dialog).getAllByText(/^updated \d\d:\d\d$/)).toHaveLength(5))
    expect(dialog).not.toHaveTextContent('stale')
  })
})

describe('G10: the What’s-here card does not depend on the all-heights layers', () => {
  // A payload with only high shapes, so that the point is covered by nothing below FL100.
  const high = { ...ALL_HEIGHTS_AIRSPACE, features: ALL_HEIGHTS_AIRSPACE.features.filter((f) => !airspaceBelowFl100(f.properties)) }
  const POINT = L.latLng(52.5, -1.5) // on land, under several stacked high shapes

  const click = (map: L.Map) => act(() => void map.fire('click', { latlng: POINT }))

  it.each([['off', false], ['on', true]])('lists a point covered only by above-FL100 features, the same with the all-heights layer %s', async (_label, allOn) => {
    const { map, user } = load([airspaceRoute(high), notamsRoute({ ...ALL_HEIGHTS_NOTAMS, features: [] }), windRoute(), rainIndexRoute()])
    await waitFor(() => expect(drawn(map, 'airspace')).toBe(0)) // the below layer has nothing here to draw
    await openPanel(user)
    if (allOn) {
      await user.click(sw('airspaceAll'))
      expect(drawn(map, 'airspace')).toBe(high.features.length)
    }
    await user.keyboard('{Escape}')
    click(map)
    const card = await screen.findByRole('region', { name: "What's here" })
    const covering = high.features.filter((f) => booleanPointInPolygon([POINT.lng, POINT.lat], f.geometry)).length
    expect(covering).toBeGreaterThan(1)
    const group = within(card).getByRole('button', { name: `Above FL100 (${covering})` })
    expect(group).toHaveAttribute('aria-expanded', 'false')
    await user.click(group)
    expect(within(card).getAllByRole('listitem')).toHaveLength(covering)
    expect(card).not.toHaveTextContent('Nothing in the loaded layers')
  })
})

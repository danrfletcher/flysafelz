import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import L from 'leaflet'
import { render } from '@testing-library/react'
import { useEffect } from 'react'
import { MapContainer, useMap } from 'react-leaflet'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { UK_BOUNDS } from '@/lib/ukBounds'
import { ANSWER_PANE, AnswerOverlays } from '@/map/answerOverlays'
import { APPROX_LABEL, BaseLayers } from '@/map/BaseLayers'
import { airspaceBelowFl100, notamBelowFl100 } from '@/map/belowFl100'
import { resetWarnedTypes, themeColor } from '@/map/airspaceStyle'
import type { AirspaceProps, MapFeature, NotamProps } from '@/map/types'
import { HIGHLIGHTS_RUN } from '@/test/chatServer'
import { ALL_HEIGHTS_AIRSPACE, ALL_HEIGHTS_NOTAMS } from '@/test/handlers'

// The "all heights" layers drawn on a real Leaflet map in jsdom, from the recorded payloads (240 airspaces, 60 NOTAMs).

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
  resetWarnedTypes()
  vi.spyOn(console, 'warn').mockImplementation(() => {})
})
afterEach(() => {
  vi.restoreAllMocks()
})

const SLATE = themeColor('--type-other')
const VIOLET = themeColor('--type-notam')
const RED = themeColor('--type-restricted')

const holder: { map: L.Map | null } = { map: null }
const mapNow = () => holder.map as L.Map

function Probe() {
  const found = useMap()
  useEffect(() => {
    holder.map = found
  }, [found])
  return null
}

interface Switches {
  showAirspace: boolean
  showNotams: boolean
  showAirspaceAll: boolean
  showNotamsAll: boolean
}

const element = (switches: Switches) => (
  <MapContainer bounds={UK_BOUNDS}>
    <Probe />
    <BaseLayers airspace={ALL_HEIGHTS_AIRSPACE.features} notams={ALL_HEIGHTS_NOTAMS.features} {...switches} />
  </MapContainer>
)

function draw(switches: Switches) {
  let current = switches
  const view = render(element(current))
  return {
    ...view,
    set: (next: Partial<Switches>) => {
      current = { ...current, ...next }
      view.rerender(element(current))
    },
  }
}

const OFF: Switches = { showAirspace: false, showNotams: false, showAirspaceAll: false, showNotamsAll: false }

function shapes(): L.Polygon[] {
  const found: L.Polygon[] = []
  mapNow().eachLayer((layer) => {
    if (layer instanceof L.Polygon && (layer as L.Polygon & { feature?: unknown }).feature) found.push(layer)
  })
  return found
}
const featureOf = <P,>(layer: L.Polygon) => (layer as L.Polygon & { feature: MapFeature<P> }).feature
const airspaceShapes = () => shapes().filter((layer) => 'lower_ft' in featureOf<AirspaceProps>(layer).properties)
const notamShapes = () => shapes().filter((layer) => 'q_code' in featureOf<NotamProps>(layer).properties)
/** Every layer on the map but Leaflet's own SVG renderer, which it adds on the first path drawn and keeps. */
const everyLayer = () => {
  let count = 0
  mapNow().eachLayer((layer) => {
    if (!(layer instanceof L.Renderer)) count += 1
  })
  return count
}

const LOW_AIRSPACE = ALL_HEIGHTS_AIRSPACE.features.filter((f) => airspaceBelowFl100(f.properties)).length
const LOW_NOTAMS = ALL_HEIGHTS_NOTAMS.features.filter((f) => notamBelowFl100(f.properties)).length
const ALL_AIRSPACE = ALL_HEIGHTS_AIRSPACE.features.length
const ALL_NOTAMS = ALL_HEIGHTS_NOTAMS.features.length
const APPROX_DRAWN = ALL_HEIGHTS_NOTAMS.features.filter((f) => f.properties.approx).length

describe('G1: what each combination draws on the map', () => {
  it('draws nothing for a route with both switches off, and mounts no group', () => {
    draw(OFF)
    expect(shapes()).toEqual([])
    expect(everyLayer()).toBe(0)
  })

  it.each([
    ['below on only', true, false, LOW_AIRSPACE, LOW_NOTAMS],
    ['all heights on only', false, true, ALL_AIRSPACE, ALL_NOTAMS],
    ['both on', true, true, ALL_AIRSPACE, ALL_NOTAMS],
  ])('draws the right set with %s, for each route separately', (_name, below, every, airspace, notams) => {
    const first = draw({ showAirspace: below, showAirspaceAll: every, showNotams: false, showNotamsAll: false })
    expect([airspaceShapes().length, notamShapes().length]).toEqual([airspace, 0])
    first.unmount()
    draw({ showAirspace: false, showAirspaceAll: false, showNotams: below, showNotamsAll: every })
    expect([airspaceShapes().length, notamShapes().length]).toEqual([0, notams])
  })

  it('with both on, draws each feature id once: no doubled outline or fill', () => {
    draw({ showAirspace: true, showAirspaceAll: true, showNotams: true, showNotamsAll: true })
    const airspace = airspaceShapes().map((layer) => featureOf<AirspaceProps>(layer).properties.id)
    const notams = notamShapes().map((layer) => featureOf<NotamProps>(layer).properties.id)
    expect(airspace).toHaveLength(ALL_AIRSPACE)
    expect(new Set(airspace).size).toBe(ALL_AIRSPACE)
    expect(notams).toHaveLength(ALL_NOTAMS)
    expect(new Set(notams).size).toBe(ALL_NOTAMS)
  })

  it('turning either switch off leaves the set the other implies (all on, then below off; below on, then all off)', () => {
    const { set } = draw({ ...OFF, showAirspace: true, showAirspaceAll: true })
    set({ showAirspace: false })
    expect(airspaceShapes()).toHaveLength(ALL_AIRSPACE)
    set({ showAirspace: true, showAirspaceAll: false })
    expect(airspaceShapes()).toHaveLength(LOW_AIRSPACE)
    set({ showAirspace: false })
    expect(airspaceShapes()).toHaveLength(0)
  })

  it('draws the high features of the all-heights layer too: none has a floor below FL100', () => {
    const { set } = draw({ ...OFF, showAirspace: true })
    const lowIds = new Set(airspaceShapes().map((layer) => featureOf<AirspaceProps>(layer).properties.id))
    set({ showAirspaceAll: true })
    const high = airspaceShapes().filter((layer) => !lowIds.has(featureOf<AirspaceProps>(layer).properties.id))
    expect(high.length).toBe(ALL_AIRSPACE - LOW_AIRSPACE)
    for (const layer of high) expect(airspaceBelowFl100(featureOf<AirspaceProps>(layer).properties)).toBe(false)
  })
})

describe('G1/G2/G3: the same colours and labels as the below-FL100 layers', () => {
  it('draws an all-heights airspace by its type: restricted red with a light fill, an unknown type slate, nothing logged beyond the type', () => {
    draw({ ...OFF, showAirspaceAll: true })
    const high = airspaceShapes().filter((layer) => !airspaceBelowFl100(featureOf<AirspaceProps>(layer).properties))
    const restricted = high.find((layer) => featureOf<AirspaceProps>(layer).properties.type === 'restricted')!
    expect(restricted.options).toMatchObject({ color: RED, fillOpacity: 0.15 })
    const unknown = high.find((layer) => featureOf<AirspaceProps>(layer).properties.type === 'balloon_zone')!
    expect(unknown.options).toMatchObject({ color: SLATE })
    const airway = high.find((layer) => featureOf<AirspaceProps>(layer).properties.type === 'airway')!
    expect(airway.options).toMatchObject({ color: SLATE })
    expect(vi.mocked(console.warn).mock.calls).toEqual([['Unknown airspace type: "balloon_zone"']]) // E11: the type value, once, no text
  })

  it('draws every all-heights NOTAM violet and dashed, approx ones with the shorter dash and the word "approx"', () => {
    const { container } = draw({ ...OFF, showNotamsAll: true })
    const drawn = notamShapes()
    expect(drawn).toHaveLength(ALL_NOTAMS)
    for (const layer of drawn) {
      const approx = featureOf<NotamProps>(layer).properties.approx
      expect(layer.options).toMatchObject({ color: VIOLET, dashArray: approx ? '3 7' : '8 6', fillOpacity: 0 })
    }
    expect(container.querySelectorAll('.approx-label')).toHaveLength(APPROX_DRAWN)
    expect(container.querySelector('.approx-label')).toHaveTextContent(APPROX_LABEL)
    const high = drawn.filter((layer) => !notamBelowFl100(featureOf<NotamProps>(layer).properties))
    expect(high.length).toBeGreaterThan(0)
  })

  it('keeps the tooltip of a high feature as it is on the low ones: name, limits and times, never NOTAM text', () => {
    draw({ ...OFF, showAirspaceAll: true, showNotamsAll: true })
    const text = (layer: L.Polygon) => (layer.getTooltip()?.getContent() as HTMLElement).textContent ?? ''
    const tma = airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.lower === 'FL195')!
    expect(text(tma)).toContain('Limits FL195 to UNL')
    const notam = notamShapes().find((layer) => featureOf<NotamProps>(layer).properties.lower_fl === 245)!
    expect(text(notam)).toContain(featureOf<NotamProps>(notam).properties.id)
    expect(text(notam)).not.toContain('EXAMPLE AIR DISPLAY')
  })
})

describe('the layers are the base layers’ own: same pane, so the answer pane stays above them (G16)', () => {
  const POINT = { lat: 53.802797, lon: -0.086644 }
  const HIGHLIGHTS = (HIGHLIGHTS_RUN.events.at(-1)!.data as { overlays: Record<string, unknown>[] }).overlays
  const leafletCss = readFileSync(resolve(__dirname, '../../node_modules/leaflet/dist/leaflet.css'), 'utf8')
  // jsdom does not apply a stylesheet's z-index, so read the overlay pane's from Leaflet's own stylesheet; the answer pane's is set inline by AnswerOverlays
  const OVERLAY_PANE_Z = Number(/\.leaflet-overlay-pane\s*\{\s*z-index:\s*(\d+)/.exec(leafletCss)?.[1])

  const withAnswer = (switches: Switches, shown: boolean) => (
    <MapContainer bounds={UK_BOUNDS}>
      <Probe />
      <BaseLayers airspace={ALL_HEIGHTS_AIRSPACE.features} notams={ALL_HEIGHTS_NOTAMS.features} {...switches} />
      <AnswerOverlays point={POINT} highlight={shown ? { turnId: 't1', overlays: HIGHLIGHTS, focusId: null } : null} visible layout="sheet" />
    </MapContainer>
  )
  const answerZ = () => Number(mapNow().getPane(ANSWER_PANE)!.style.zIndex)
  const paneOf = (layer: L.Layer) => ((layer as L.Path).getElement() as Element).closest('.leaflet-pane') as HTMLElement

  it.each([
    ['both all-heights on', { ...OFF, showAirspaceAll: true, showNotamsAll: true }],
    ['every switch on', { showAirspace: true, showNotams: true, showAirspaceAll: true, showNotamsAll: true }],
    ['below only', { ...OFF, showAirspace: true, showNotams: true }],
  ])('draws a turn’s highlights in a pane above the base shapes with %s', (_name, switches) => {
    render(withAnswer(switches, true))
    const base = shapes().filter((layer) => layer.options.pane !== ANSWER_PANE) // the answer shapes are GeoJSON polygons too
    const answer: L.Path[] = []
    mapNow().eachLayer((layer) => {
      if (layer instanceof L.Path && layer.options.pane === ANSWER_PANE) answer.push(layer)
    })
    expect(base.length).toBeGreaterThan(0)
    expect(answer.length).toBeGreaterThan(1) // the 2 NM circle and the turn's shapes
    for (const layer of base) expect(paneOf(layer)).toBe(mapNow().getPane('overlayPane'))
    for (const layer of answer) expect(paneOf(layer)).toBe(mapNow().getPane(ANSWER_PANE))
    expect(OVERLAY_PANE_Z).toBe(400)
    expect(answerZ()).toBeGreaterThan(OVERLAY_PANE_Z)
  })

  it('keeps the highlights drawn, above the shapes, as the all-heights switches go on and off under them', () => {
    const view = render(withAnswer(OFF, true))
    const inAnswerPane = () => mapNow().getPane(ANSWER_PANE)!.querySelectorAll('path').length
    const highlights = inAnswerPane()
    expect(highlights).toBeGreaterThan(1)
    for (const change of [{ showAirspaceAll: true }, { showNotamsAll: true }, { showAirspace: true }, { showAirspaceAll: false, showNotamsAll: false }]) {
      view.rerender(withAnswer({ ...OFF, ...change }, true))
      expect(inAnswerPane()).toBe(highlights)
      expect(answerZ()).toBeGreaterThan(OVERLAY_PANE_Z)
    }
  })
})

describe('many overlapping high polygons: 240 airspaces and 60 NOTAMs', () => {
  it('toggles every all-heights layer on and off in well under 500 ms each', () => {
    const { set } = draw(OFF)
    const timed = (change: Partial<Switches>) => {
      const start = performance.now()
      set(change)
      return performance.now() - start
    }
    const on = timed({ showAirspaceAll: true, showNotamsAll: true })
    expect(airspaceShapes()).toHaveLength(ALL_AIRSPACE)
    const off = timed({ showAirspaceAll: false, showNotamsAll: false })
    expect(shapes()).toHaveLength(0)
    expect(on).toBeLessThan(500)
    expect(off).toBeLessThan(500)
  })

  it('ends in the state the last of ten quick toggles implies, with no Leaflet layer left behind', () => {
    const { set } = draw(OFF)
    for (let toggle = 1; toggle <= 10; toggle += 1) set({ showAirspaceAll: toggle % 2 === 1, showNotamsAll: toggle % 4 === 1 })
    // the 10th toggle: airspace all off, NOTAMs all off
    expect(shapes()).toHaveLength(0)
    mapNow().eachLayer((l) => process.stdout.write('LEFT ' + JSON.stringify([l.constructor.name, Object.keys(l.options ?? {}), (l as L.Layer & {_url?:string})._url]) + '\n'))
    expect(everyLayer()).toBe(0)
    for (let toggle = 1; toggle <= 9; toggle += 1) set({ showAirspaceAll: toggle % 2 === 1, showAirspace: toggle % 2 === 0 })
    // the 9th: all on, below off
    expect(airspaceShapes()).toHaveLength(ALL_AIRSPACE)
    const withNotams = everyLayer()
    set({ showAirspaceAll: false })
    expect(everyLayer()).toBeLessThan(withNotams)
    set({ showAirspaceAll: true })
    expect(everyLayer()).toBe(withNotams) // the same count after going round: nothing leaked
  })
})

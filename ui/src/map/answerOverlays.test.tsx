import L from 'leaflet'
import { act, render } from '@testing-library/react'
import { useEffect } from 'react'
import { MapContainer, useMap } from 'react-leaflet'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { actionStyle, resetWarnedActions } from '@/map/actionStyle'
import { themeColor } from '@/map/airspaceStyle'
import { readAnswerFeatures, readGeometry } from '@/map/answerFeatures'
import { ANSWER_PANE, AnswerOverlays, buildAnswerGroup, type AnswerHighlight } from '@/map/answerOverlays'
import { DRAWER_HEIGHT_FRACTION, LZ_RADIUS_NM, METRES_PER_NM, SHEET_WIDTH_PX } from '@/map/fitLz'
import { UK_BOUNDS } from '@/lib/ukBounds'
import { BLUE_MILL_RUN, HIGHLIGHTS_RUN, type Recording } from '@/test/chatServer'

const POINT = { lat: 53.802797, lon: -0.086644 }
const overlaysOf = (run: Recording) => (run.events.at(-1)!.data as { overlays: Record<string, unknown>[] }).overlays
const HIGHLIGHTS = overlaysOf(HIGHLIGHTS_RUN)
const BLUE_MILL = overlaysOf(BLUE_MILL_RUN)

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
  resetWarnedActions()
  vi.spyOn(console, 'warn').mockImplementation(() => {})
})
afterEach(() => vi.restoreAllMocks())

const holder: { map: L.Map | null } = { map: null }
function Probe() {
  const found = useMap()
  useEffect(() => {
    holder.map = found
  }, [found])
  return null
}

type Props = Parameters<typeof AnswerOverlays>[0]
const tree = (props: Partial<Props>) => (
  <MapContainer bounds={UK_BOUNDS}>
    <Probe />
    <AnswerOverlays point={POINT} highlight={null} visible layout="sheet" {...props} />
  </MapContainer>
)

const highlight = (overlays: Record<string, unknown>[], turnId = 't1', focusId: string | null = null): AnswerHighlight => ({ turnId, overlays, focusId })

/** Everything drawn in the answer pane: the circle and the shapes. */
function drawn(): L.Path[] {
  const found: L.Path[] = []
  holder.map!.eachLayer((layer) => {
    if (layer instanceof L.Path && (layer.options as { pane?: string }).pane === ANSWER_PANE) found.push(layer)
  })
  return found
}
const shapesDrawn = () => drawn().filter((layer) => !(layer instanceof L.Circle))
const circleDrawn = () => drawn().filter((layer): layer is L.Circle => layer instanceof L.Circle)
const labels = () => holder.map!.getPane(ANSWER_PANE)?.querySelectorAll('.approx-label') ?? []

describe('actionStyle', () => {
  it('gives each of the legend’s five actions its colour, and no-fly a light fill', () => {
    const colour = (action: string) => actionStyle(action, false, false)
    expect(colour('NO_FLY')).toMatchObject({ token: '--action-no-fly', fillOpacity: 0.2 })
    expect(colour('CONTACT_FIRST')).toMatchObject({ token: '--action-contact-first', fillOpacity: 0 })
    expect(colour('CHECK_SUP')).toMatchObject({ token: '--action-check-sup', fillOpacity: 0 })
    expect(colour('AVOID_SECTOR')).toMatchObject({ token: '--action-avoid-sector', fillOpacity: 0 })
    expect(colour('AWARENESS')).toMatchObject({ token: '--action-awareness', fillOpacity: 0 })
  })

  it('gives NO_GO and AWARENESS_MAP the colour of the action they mean', () => {
    expect(actionStyle('NO_GO', false, false).token).toBe('--action-no-fly')
    expect(actionStyle('AWARENESS_MAP', false, false).token).toBe('--action-awareness')
  })

  it('dashes an approximate shape, draws a picked one heavier and with a fill', () => {
    expect(actionStyle('CHECK_SUP', true, false).dashArray).toBeDefined()
    expect(actionStyle('CHECK_SUP', false, false).dashArray).toBeUndefined()
    const picked = actionStyle('CHECK_SUP', false, true)
    expect(picked.weight).toBeGreaterThan(actionStyle('CHECK_SUP', false, false).weight)
    expect(picked.fillOpacity).toBeGreaterThan(0)
  })

  it('draws a missing or unknown action in slate, and warns once about an unknown one with the value only (E11)', () => {
    const warn = vi.mocked(console.warn)
    expect(actionStyle(null, false, false).token).toBe('--type-other')
    expect(warn).not.toHaveBeenCalled()
    expect(actionStyle('HOLD_FIRE', false, false).token).toBe('--type-other')
    expect(actionStyle('HOLD_FIRE', true, false).token).toBe('--type-other')
    expect(warn).toHaveBeenCalledOnce()
    expect(warn.mock.calls[0][0]).toContain('HOLD_FIRE')
    actionStyle('__proto__', false, false)
    expect(actionStyle('toString', false, false).token).toBe('--type-other')
  })
})

describe('readAnswerFeatures', () => {
  it('reads the NOTAMs and airspace of an answer in order, and leaves out the LZ circle', () => {
    const features = readAnswerFeatures(HIGHLIGHTS)
    expect(features.map((f) => [f.kind, f.id])).toEqual([['notam', 'H7171/26'], ['notam', 'H7172/26'], ['notam', 'V32/26'], ['airspace', 'Blue Mill ATZ']])
    expect(features[1]).toMatchObject({ approx: true, action: 'AVOID_SECTOR' })
    expect(features[3].text).toBeNull()
  })

  it('keeps a NOTAM whose geometry is empty or malformed, with its text, and gives it no geometry (E7)', () => {
    const notam = (geometry: unknown) => ({ geometry, properties: { id: 'H1/26', kind: 'notam', action: 'NO_FLY', text: 'RAW TEXT', approx: false } })
    for (const geometry of [null, {}, { type: 'Polygon', coordinates: [] }, { type: 'Polygon', coordinates: [[[1, 2], [3, 4]]] }, { type: 'Polygon', coordinates: [[[200, 99], [1, 1], [2, 2], [200, 99]]] }, { type: 'MultiPolygon', coordinates: [] }, 'x']) {
      const [feature] = readAnswerFeatures([notam(geometry)])
      expect(feature, JSON.stringify(geometry)).toMatchObject({ id: 'H1/26', text: 'RAW TEXT', geometry: null })
    }
  })

  it('skips entries that are not features of a NOTAM or an area', () => {
    expect(readAnswerFeatures([{}, { properties: null }, { properties: { id: '', kind: 'notam' } }, { properties: { id: 'x', kind: 'lz' } }, { properties: { id: 5, kind: 'notam' } }])).toEqual([])
  })

  it('reads a Point and a MultiPolygon', () => {
    expect(readGeometry({ type: 'Point', coordinates: [-0.1, 53.8] })).toEqual({ type: 'Point', coordinates: [-0.1, 53.8] })
    const square = [[[-0.1, 53.8], [-0.09, 53.8], [-0.09, 53.81], [-0.1, 53.8]]]
    expect(readGeometry({ type: 'MultiPolygon', coordinates: [square] })).not.toBeNull()
  })
})

describe('AnswerOverlays on the map', () => {
  it('draws the 2 NM circle round the LZ and each NOTAM and airspace shape in the colour of its action, in the answer pane', () => {
    render(tree({ highlight: highlight(HIGHLIGHTS) }))
    const [circle] = circleDrawn()
    expect(circleDrawn()).toHaveLength(1)
    expect(circle.getLatLng().lat).toBeCloseTo(POINT.lat, 6)
    expect(circle.getRadius()).toBe(LZ_RADIUS_NM * METRES_PER_NM)
    expect(circle.options.interactive).toBe(false)

    const colours = shapesDrawn().map((layer) => layer.options.color)
    expect(shapesDrawn()).toHaveLength(4)
    expect(colours).toEqual([
      themeColor('--action-contact-first'), // H7171/26
      themeColor('--action-avoid-sector'), // H7172/26
      themeColor('--type-other'), // V32/26, an action this code does not know
      themeColor('--action-no-fly'), // Blue Mill ATZ
    ])
    for (const layer of shapesDrawn()) expect(layer.options.interactive).toBe(false)
  })

  it('dashes and labels the approximate shape, and only that one', () => {
    render(tree({ highlight: highlight(HIGHLIGHTS) }))
    const dashed = shapesDrawn().filter((layer) => layer.options.dashArray)
    expect(dashed).toHaveLength(1)
    expect(dashed[0].options.color).toBe(themeColor('--action-avoid-sector'))
    expect(labels()).toHaveLength(1)
    expect(labels()[0]).toHaveTextContent('approx')
  })

  it('draws the shape a NOTAM-id click picked out heavier and last, and does not move the map', () => {
    const view = render(tree({ highlight: highlight(HIGHLIGHTS) }))
    const fit = vi.spyOn(holder.map!, 'fitBounds')
    const plainWeight = shapesDrawn()[2].options.weight
    view.rerender(tree({ highlight: highlight(HIGHLIGHTS, 't1', 'V32/26') }))
    const picked = shapesDrawn().filter((layer) => layer.options.weight! > plainWeight!)
    expect(picked).toHaveLength(1)
    expect(picked[0].options.color).toBe(themeColor('--type-other'))
    expect(shapesDrawn().at(-1)).toBe(picked[0])
    expect(fit).not.toHaveBeenCalled()
  })

  it('draws a feature without geometry nowhere and still draws the others (E7)', () => {
    const broken = HIGHLIGHTS.map((entry, index) => (index === 1 ? { ...entry, geometry: { type: 'Polygon', coordinates: [] } } : entry))
    render(tree({ highlight: highlight(broken) }))
    expect(shapesDrawn()).toHaveLength(3)
    expect(circleDrawn()).toHaveLength(1)
  })

  it('fits the map to the circle and the shapes, padded for the layout', () => {
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    render(tree({ highlight: highlight(HIGHLIGHTS), layout: 'sheet' }))
    expect(fit).toHaveBeenCalled()
    const [bounds, options] = fit.mock.calls.at(-1) as [L.LatLngBounds, L.FitBoundsOptions]
    const circle = circleDrawn()[0].getBounds()
    expect(bounds.contains(circle)).toBe(true)
    expect((options.paddingBottomRight as L.PointTuple)[0]).toBeGreaterThan(0)
  })

  it('pads the fit for the sheet on desktop and for the drawer on mobile', () => {
    const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
    const view = render(tree({ highlight: highlight(HIGHLIGHTS), layout: 'sheet' }))
    const [, sheet] = fit.mock.calls.at(-1) as [L.LatLngBounds, L.FitBoundsOptions]
    view.rerender(tree({ highlight: highlight(HIGHLIGHTS), layout: 'drawer' }))
    const [, drawer] = fit.mock.calls.at(-1) as [L.LatLngBounds, L.FitBoundsOptions]
    expect((sheet.paddingBottomRight as L.PointTuple)[0]).toBeGreaterThanOrEqual(SHEET_WIDTH_PX)
    expect((sheet.paddingBottomRight as L.PointTuple)[1]).toBeLessThan((drawer.paddingBottomRight as L.PointTuple)[1])
    expect((drawer.paddingBottomRight as L.PointTuple)[1]).toBeGreaterThanOrEqual(Math.round(DRAWER_HEIGHT_FRACTION * 800))
    expect((drawer.paddingBottomRight as L.PointTuple)[0]).toBeLessThan(SHEET_WIDTH_PX)
  })

  it('replaces the whole set when another turn is shown, and clears it when no turn is', () => {
    const view = render(tree({ highlight: highlight(HIGHLIGHTS, 't1') }))
    expect(shapesDrawn()).toHaveLength(4)
    view.rerender(tree({ highlight: highlight(BLUE_MILL, 't2') }))
    expect(shapesDrawn()).toHaveLength(2)
    expect(circleDrawn()).toHaveLength(1)
    expect(labels()).toHaveLength(0)
    view.rerender(tree({ highlight: null }))
    expect(drawn()).toHaveLength(0)
  })

  it('draws nothing while the Answer overlays switch is off, and everything again when it goes back on', () => {
    const view = render(tree({ highlight: highlight(HIGHLIGHTS), visible: false }))
    expect(drawn()).toHaveLength(0)
    view.rerender(tree({ highlight: highlight(HIGHLIGHTS), visible: true }))
    expect(drawn()).toHaveLength(5)
    view.rerender(tree({ highlight: highlight(HIGHLIGHTS), visible: false }))
    expect(drawn()).toHaveLength(0)
  })

  it('draws nothing without an LZ, and removes everything on unmount', () => {
    const none = render(tree({ highlight: highlight(HIGHLIGHTS), point: null }))
    expect(drawn()).toHaveLength(0)
    none.unmount()
    const view = render(tree({ highlight: highlight(HIGHLIGHTS) }))
    const map = holder.map!
    expect(drawn()).toHaveLength(5)
    view.unmount()
    const left: L.Layer[] = []
    map.eachLayer((layer) => left.push(layer))
    expect(left.filter((layer) => layer instanceof L.Path)).toHaveLength(0)
  })

  it('puts the answer pane above the base layers’ pane and does not touch the base layer groups', () => {
    render(tree({ highlight: highlight(HIGHLIGHTS) }))
    const pane = holder.map!.getPane(ANSWER_PANE)!
    expect(Number(pane.style.zIndex)).toBeGreaterThan(Number(holder.map!.getPane('overlayPane')!.style.zIndex || 400) - 1)
    expect(Number(pane.style.zIndex)).toBeLessThan(Number(holder.map!.getPane('markerPane')!.style.zIndex || 600))
  })

  it('builds a group of circle plus shapes that a picked NOTAM id reorders to the top (buildAnswerGroup)', () => {
    const group = buildAnswerGroup(POINT, readAnswerFeatures(HIGHLIGHTS), 'H7171/26')
    const weights = group
      .getLayers()
      .filter((layer): layer is L.GeoJSON => layer instanceof L.GeoJSON)
      .map((shape) => (shape.getLayers()[0] as L.Path).options.weight!)
    expect(weights).toHaveLength(4)
    expect(weights.slice(0, 3).every((weight) => weight < weights[3])).toBe(true)
    expect(group.getLayers().filter((layer) => layer instanceof L.GeoJSON).length).toBe(4)
  })

  it('does not throw on a render with no overlays at all', () => {
    act(() => void render(tree({ highlight: highlight([]) })))
    expect(circleDrawn()).toHaveLength(1)
  })
})

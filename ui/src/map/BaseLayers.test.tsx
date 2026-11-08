import L from 'leaflet'
import { act, render } from '@testing-library/react'
import { useEffect } from 'react'
import { MapContainer, useMap } from 'react-leaflet'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { APPROX_LABEL, BaseLayers, buildAirspaceGroup } from '@/map/BaseLayers'
import { resetWarnedTypes, themeColor } from '@/map/airspaceStyle'
import { UK_BOUNDS } from '@/lib/ukBounds'
import type { AirspaceProps, MapFeature, NotamProps } from '@/map/types'
import { AIRSPACE_FIXTURE, NOTAMS_FIXTURE } from '@/test/handlers'

// jsdom has no layout, so give every element a size for Leaflet to fit the map into.
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
  resetWarnedTypes()
  vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  vi.restoreAllMocks()
})

// The colours are read from the theme variables, as the map reads them; airspaceStyle.test.ts holds the values.
const RED = themeColor('--type-restricted')
const ORANGE = themeColor('--type-danger')
const SKY = themeColor('--type-controlled')
const SLATE = themeColor('--type-other')
const VIOLET = themeColor('--type-notam')

// The map the layers are on, set by Probe once it is mounted (an effect: render itself stays free of side effects).
const holder: { map: L.Map | null } = { map: null }
const mapNow = () => holder.map as L.Map

function Probe() {
  const found = useMap()
  useEffect(() => {
    holder.map = found
  }, [found])
  return null
}

function renderLayers(props: Partial<Parameters<typeof BaseLayers>[0]> = {}) {
  const all = {
    airspace: AIRSPACE_FIXTURE.features,
    notams: NOTAMS_FIXTURE.features,
    showAirspace: true,
    showNotams: true,
    showAirspaceAll: false,
    showNotamsAll: false,
    ...props,
  }
  const view = render(
    <MapContainer bounds={UK_BOUNDS}>
      <Probe />
      <BaseLayers {...all} />
    </MapContainer>,
  )
  return {
    ...view,
    rerenderWith: (next: Partial<typeof all>) =>
      view.rerender(
        <MapContainer bounds={UK_BOUNDS}>
          <Probe />
          <BaseLayers {...{ ...all, ...next }} />
        </MapContainer>,
      ),
  }
}

/** The drawn shapes: every polygon on the map that carries a GeoJSON feature. */
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
const groups = () => {
  let count = 0
  mapNow().eachLayer((layer) => {
    if (layer instanceof L.FeatureGroup && !(layer instanceof L.GeoJSON)) count += 1
  })
  return count
}
const tooltipText = (layer: L.Polygon) => (layer.getTooltip()?.getContent() as HTMLElement).textContent ?? ''

describe('theme colours', () => {
  it('are five different, non-empty values', () => {
    const all = [RED, ORANGE, SKY, SLATE, VIOLET]
    expect(all.every((colour) => /^#[0-9a-f]{6}$/.test(colour))).toBe(true)
    expect(new Set(all).size).toBe(5)
  })
})

describe('which features are drawn', () => {
  it('draws the features whose lower limit is below FL100, and no others', () => {
    renderLayers()
    const names = airspaceShapes().map((layer) => featureOf<AirspaceProps>(layer).properties.name)
    expect(names).toHaveLength(15)
    expect(names).not.toContain('SAMPLE CTA FL100 FLOOR') // exactly FL100: not below
    expect(names).not.toContain('SAMPLE TMA ABOVE FL100')
    expect(names).toContain('SAMPLE TMA 300 M AGL') // an above-ground floor counts as below
    const ids = notamShapes().map((layer) => featureOf<NotamProps>(layer).properties.id)
    expect(ids.sort()).toEqual(['H1001/26', 'H1002/26', 'H1003/26', 'H1004/26', 'H1007/26'])
  })

  it('matches a count made from the fixture itself', () => {
    renderLayers()
    const low = AIRSPACE_FIXTURE.features.filter((f) => (f.properties.lower_ft ?? 0) < 10000)
    expect(airspaceShapes()).toHaveLength(low.length)
    expect(notamShapes()).toHaveLength(NOTAMS_FIXTURE.features.filter((f) => (f.properties.lower_fl ?? 0) < 100).length)
  })

  it('renders an empty airspace list and an empty NOTAM list as empty layers', () => {
    renderLayers({ airspace: [], notams: [] })
    expect(shapes()).toEqual([])
    expect(groups()).toBe(2)
  })

  it('renders a MultiPolygon and a polygon with a hole without throwing', () => {
    renderLayers()
    const two = airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.name === 'SAMPLE DANGER TWO PARTS')
    expect(featureOf<AirspaceProps>(two!).geometry.type).toBe('MultiPolygon')
    expect((two!.getLatLngs() as L.LatLng[][][]).length).toBe(2)
    const holed = airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.name === 'SAMPLE CTR WITH HOLE')
    expect((holed!.getLatLngs() as L.LatLng[][]).length).toBe(2) // outer ring and one hole
  })
})

describe('colour comes from the type alone (E11)', () => {
  it('colours each airspace by type, and ignores an action on the feature', () => {
    renderLayers()
    const byName = Object.fromEntries(airspaceShapes().map((layer) => [featureOf<AirspaceProps>(layer).properties.name, layer.options]))
    expect(byName['TESTVILLE RESTRICTED AREA']).toMatchObject({ color: RED, fillOpacity: 0.15 })
    expect(byName['EXAMPLEBY PROHIBITED AREA']).toMatchObject({ color: RED, fillOpacity: 0.15 })
    expect(byName['SAMPLE MOOR DANGER AREA']).toMatchObject({ color: ORANGE, fillOpacity: 0 })
    expect(byName['EXAMPLEBY ATZ']).toMatchObject({ color: SKY, fillOpacity: 0 })
    expect(byName['SAMPLE TMA 1']).toMatchObject({ color: SLATE })
    expect(byName['SAMPLE MATZ']).toMatchObject({ color: SLATE })
    expect(byName['SAMPLE OTHER AREA']).toMatchObject({ color: SLATE })
    // TESTVILLE CTR carries action NO_FLY in the fixture; it is still sky blue, the colour of a CTR.
    const ctr = airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.name === 'TESTVILLE CTR')!
    expect(featureOf<AirspaceProps>(ctr).properties.action).toBe('NO_FLY')
    expect(ctr.options).toMatchObject({ color: SKY, fillOpacity: 0 })
  })

  it('draws an unknown type slate and warns once with the value only', () => {
    renderLayers()
    expect(byName('SAMPLE UNKNOWN KIND')!.options).toMatchObject({ color: SLATE })
    const warnings = vi.mocked(console.warn).mock.calls
    expect(warnings).toEqual([['Unknown airspace type: "balloon_zone"']])
    function byName(name: string) {
      return airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.name === name)
    }
  })

  it('draws NOTAMs violet and dashed, with no fill', () => {
    renderLayers()
    for (const layer of notamShapes()) {
      const approx = featureOf<NotamProps>(layer).properties.approx
      expect(layer.options).toMatchObject({ color: VIOLET, dashArray: approx ? '3 7' : '8 6', fillOpacity: 0 })
    }
  })
})

describe('NOTAM geometry exactly as served', () => {
  it('passes every NOTAM’s fixture geometry through, coordinate for coordinate', () => {
    renderLayers()
    const drawn = notamShapes()
    expect(drawn.length).toBe(5)
    for (const layer of drawn) {
      const served = NOTAMS_FIXTURE.features.find((f) => f.properties.id === featureOf<NotamProps>(layer).properties.id)!
      expect(featureOf<NotamProps>(layer).geometry).toBe(served.geometry) // the same object: nothing copied or changed
      const ring = (layer.getLatLngs() as L.LatLng[][])[0]
      // Leaflet keeps a ring open (it drops the repeat of the first point at the end); every other coordinate is as served.
      expect(ring.map((point) => [point.lng, point.lat])).toEqual(served.geometry.coordinates[0].slice(0, -1))
    }
  })
})

describe('approx label', () => {
  it('puts the word on each drawn approx NOTAM and on no other', () => {
    const { container } = renderLayers()
    const labels = [...container.querySelectorAll('.approx-label')]
    expect(labels).toHaveLength(1) // H1003 is approx and drawn; H1006 is approx but above FL100
    expect(labels[0]).toHaveTextContent(APPROX_LABEL)
    expect(container.querySelectorAll('.approx-label__text')).toHaveLength(1)
  })

  it('is at the centre of the shape and does not take the mouse', () => {
    renderLayers()
    const approx = notamShapes().find((layer) => featureOf<NotamProps>(layer).properties.approx)!
    let marker: L.Marker | undefined
    mapNow().eachLayer((layer) => {
      if (layer instanceof L.Marker) marker = layer
    })
    expect(marker!.getLatLng()).toEqual(approx.getBounds().getCenter())
    expect(marker!.options.interactive).toBe(false)
  })

  it('is gone with the layer', () => {
    const { container, rerenderWith } = renderLayers()
    rerenderWith({ showNotams: false })
    expect(container.querySelectorAll('.approx-label')).toHaveLength(0)
  })
})

describe('tooltips', () => {
  it('shows an airspace’s name, limits and active times', () => {
    renderLayers()
    const holed = airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.name === 'SAMPLE MOOR DANGER AREA')!
    expect(tooltipText(holed)).toBe('SAMPLE MOOR DANGER AREALimits SFC to 12000 ft AMSLActive by NOTAM')
    const ctr = airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.name === 'TESTVILLE CTA 1')!
    expect(tooltipText(ctr)).toBe('TESTVILLE CTA 1Limits 1500 ft AMSL to 4500 ft AMSLActive no times published')
  })

  it('shows a NOTAM’s id, limits and times, and never its text', () => {
    renderLayers()
    const first = notamShapes().find((layer) => featureOf<NotamProps>(layer).properties.id === 'H1001/26')!
    const text = tooltipText(first)
    expect(text).toContain('H1001/26')
    expect(text).toContain('Limits SFC to 800FT AMSL')
    expect(text).toMatch(/Times \d+ \w+, \d\d:\d\d \w+ to \d+ \w+, \d\d:\d\d \w+/)
    expect(text).not.toContain('approx')
    for (const layer of notamShapes()) {
      const raw = featureOf<NotamProps>(layer).properties.text
      expect(raw.length).toBeGreaterThan(10)
      expect(tooltipText(layer)).not.toContain(raw.slice(0, 20))
      expect(tooltipText(layer)).not.toMatch(/\d{5}\s?\d{6}|CTC/) // no phone number, no contact line
    }
  })

  it('marks an approximate shape in its tooltip', () => {
    renderLayers()
    const approx = notamShapes().find((layer) => featureOf<NotamProps>(layer).properties.id === 'H1003/26')!
    expect(tooltipText(approx)).toContain('approx')
  })

  it('opens on hover and shows the same text in the page', () => {
    const { container } = renderLayers()
    const restricted = airspaceShapes().find((layer) => featureOf<AirspaceProps>(layer).properties.name === 'TESTVILLE RESTRICTED AREA')!
    act(() => {
      restricted.fire('mouseover', { latlng: restricted.getBounds().getCenter() })
    })
    const tip = container.querySelector('.leaflet-tooltip')
    expect(tip).not.toBeNull()
    expect(tip).toHaveTextContent('TESTVILLE RESTRICTED AREA')
    expect(tip).toHaveTextContent('SFC to 5000 ft AMSL')
  })

  it('builds the text with text nodes, so a name with markup is shown as text', () => {
    const feature: MapFeature<AirspaceProps> = {
      ...AIRSPACE_FIXTURE.features[0],
      properties: { ...AIRSPACE_FIXTURE.features[0].properties, name: '<img src=x onerror=alert(1)>' },
    }
    const group = buildAirspaceGroup([feature])
    const layer = (group.getLayers()[0] as L.GeoJSON).getLayers()[0] as L.Polygon
    const content = layer.getTooltip()!.getContent() as HTMLElement
    expect(content.querySelector('img')).toBeNull()
    expect(content.textContent).toContain('<img src=x onerror=alert(1)>')
  })
})

describe('switching layers', () => {
  it('draws neither, one, or both layers by their switches, each on its own', () => {
    const { rerenderWith } = renderLayers()
    expect([airspaceShapes().length, notamShapes().length]).toEqual([15, 5])
    rerenderWith({ showNotams: false })
    expect([airspaceShapes().length, notamShapes().length]).toEqual([15, 0])
    rerenderWith({ showNotams: true, showAirspace: false })
    expect([airspaceShapes().length, notamShapes().length]).toEqual([0, 5])
    rerenderWith({ showNotams: false, showAirspace: false })
    expect(shapes()).toEqual([])
    expect(groups()).toBe(0)
  })

  it('ends in the state of the last of ten quick toggles, with no layer drawn twice', () => {
    const { rerenderWith } = renderLayers()
    for (let toggle = 1; toggle <= 10; toggle += 1) rerenderWith({ showAirspace: toggle % 2 === 0, showNotams: toggle % 2 === 1 })
    // the 10th toggle: airspace on, NOTAMs off
    expect([airspaceShapes().length, notamShapes().length]).toEqual([15, 0])
    expect(groups()).toBe(1)
    for (let toggle = 1; toggle <= 9; toggle += 1) rerenderWith({ showAirspace: toggle % 2 === 0, showNotams: toggle % 2 === 1 })
    expect([airspaceShapes().length, notamShapes().length]).toEqual([0, 5]) // ended on an odd toggle
    expect(groups()).toBe(1)
  })

  it('swaps the drawing, not adds to it, when the features change', () => {
    const { rerenderWith } = renderLayers()
    rerenderWith({ airspace: AIRSPACE_FIXTURE.features.slice(0, 3) })
    expect(airspaceShapes()).toHaveLength(3)
    expect(groups()).toBe(2)
  })
})

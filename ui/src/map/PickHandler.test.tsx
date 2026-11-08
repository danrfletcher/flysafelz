import L from 'leaflet'
import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useEffect } from 'react'
import { MapContainer, useMap } from 'react-leaflet'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PickHandler } from '@/map/PickHandler'
import type { AirspaceProps, NotamProps } from '@/map/types'
import type { LayerState } from '@/map/useMapLayers'
import { BLUE_MILL, BLUE_MILL_AIRSPACE, BLUE_MILL_NOTAMS } from '@/test/handlers'
import { loaded, loading } from '@/test/layers'

// jsdom has no layout, so give every element a size for Leaflet to fit the map into.
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

const holder: { map: L.Map | null } = { map: null }
const mapNow = () => holder.map as L.Map

function Probe() {
  const found = useMap()
  useEffect(() => {
    holder.map = found
  }, [found])
  return null
}

const loadedAirspace = loaded<AirspaceProps>(BLUE_MILL_AIRSPACE.features)
const loadedNotams = loaded<NotamProps>(BLUE_MILL_NOTAMS.features)

const EDINBURGH = { lat: 55.9533, lon: -3.1883 }
const NORTH_SEA = { lat: 54.0, lon: 1.5 }
const PARIS = { lat: 48.8566, lon: 2.3522 }

function renderMap(airspace: LayerState<AirspaceProps> = loadedAirspace, notams: LayerState<NotamProps> = loadedNotams) {
  const tree = (a: LayerState<AirspaceProps>, n: LayerState<NotamProps>) => (
    <MapContainer center={[BLUE_MILL.lat, BLUE_MILL.lon]} zoom={10}>
      <Probe />
      <PickHandler airspace={a} notams={n} />
    </MapContainer>
  )
  const view = render(tree(airspace, notams))
  return { ...view, rerenderWith: (a: LayerState<AirspaceProps>, n: LayerState<NotamProps>) => view.rerender(tree(a, n)) }
}

/** A click on the map, as Leaflet reports it. */
function clickAt(point: { lat: number; lon: number }) {
  act(() => {
    mapNow().fire('click', { latlng: L.latLng(point.lat, point.lon) })
  })
}

function pins(): L.Marker[] {
  const found: L.Marker[] = []
  mapNow().eachLayer((layer) => {
    if (layer instanceof L.Marker) found.push(layer)
  })
  return found
}

const cards = () => screen.queryAllByRole('region', { name: "What's here" })
const popups = (container: HTMLElement) => container.querySelectorAll('.leaflet-popup')

describe('G8: a click on land', () => {
  it('places one pin on the point and opens the card at it', () => {
    const { container } = renderMap()
    expect(pins()).toHaveLength(0)
    expect(cards()).toHaveLength(0)

    clickAt(BLUE_MILL)
    expect(pins()).toHaveLength(1)
    expect(pins()[0].getLatLng()).toMatchObject({ lat: BLUE_MILL.lat, lng: BLUE_MILL.lon })
    expect(cards()).toHaveLength(1)
    expect(popups(container)).toHaveLength(1)
    expect(screen.getByText('Over this point')).toBeInTheDocument()
    expect(screen.getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(pins()[0].isPopupOpen()).toBe(true)
    expect(screen.queryByText('Pick a field on land in the UK.')).toBeNull()
  })

  it('moves the pin and replaces the card on a second land click: never two pins or two cards', async () => {
    const user = userEvent.setup()
    const { container } = renderMap()
    clickAt(BLUE_MILL)
    await user.click(screen.getByRole('button', { name: /BLUE MILL TEST ATZ/ })) // an open row on the first card
    expect(screen.getByText('MON-FRI 0800-1700')).toBeInTheDocument()

    clickAt(EDINBURGH)
    expect(pins()).toHaveLength(1)
    expect(pins()[0].getLatLng()).toMatchObject({ lat: EDINBURGH.lat, lng: EDINBURGH.lon })
    expect(cards()).toHaveLength(1)
    expect(popups(container)).toHaveLength(1)
    expect(screen.getByText('Nothing in the loaded layers over this point.')).toBeInTheDocument() // the new point's card
    expect(screen.queryByText('BLUE MILL TEST ATZ')).toBeNull()
    expect(screen.queryByText('MON-FRI 0800-1700')).toBeNull()

    clickAt(BLUE_MILL)
    expect(pins()).toHaveLength(1)
    expect(screen.getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(screen.queryByText('MON-FRI 0800-1700')).toBeNull() // a fresh card, rows closed
  })

  it('reads the wrapped longitude when the map has been panned round the world', () => {
    renderMap()
    clickAt({ lat: BLUE_MILL.lat, lon: BLUE_MILL.lon + 360 })
    expect(pins()).toHaveLength(1)
    expect(pins()[0].getLatLng().lng).toBeCloseTo(BLUE_MILL.lon, 6)
    expect(screen.getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
  })

  it('works from a real DOM click on the map, through Leaflet’s own mouse handling', () => {
    const { container } = renderMap()
    act(() => mapNow().setView([BLUE_MILL.lat, BLUE_MILL.lon], 10, { animate: false }))
    const mapElement = container.querySelector('.leaflet-container') as HTMLElement
    fireEvent.click(mapElement, { clientX: 600, clientY: 400 }) // the centre of the 1200 x 800 map
    expect(pins()).toHaveLength(1)
    expect(pins()[0].getLatLng().lat).toBeCloseTo(BLUE_MILL.lat, 2)
    expect(cards()).toHaveLength(1)
  })

  it('opens the pin’s card again from the pin itself, and the card has no close button or chat button', () => {
    renderMap()
    clickAt(BLUE_MILL)
    expect(screen.queryByRole('button', { name: /close|chat/i })).toBeNull()
    // fireEvent, not userEvent: two quick userEvent clicks read as a double-tap to Leaflet, which jsdom cannot place
    fireEvent.click(screen.getByRole('button', { name: 'Picked field' }))
    expect(cards()).toHaveLength(0) // Leaflet's own toggle: the pin closes its card
    fireEvent.click(screen.getByRole('button', { name: 'Picked field' }))
    expect(cards()).toHaveLength(1)
    expect(pins()).toHaveLength(1)
  })
})

describe('G9: a click on sea or outside the UK', () => {
  it('shows the toast and drops no pin', () => {
    renderMap()
    clickAt(NORTH_SEA)
    expect(screen.getByRole('status')).toHaveTextContent('Pick a field on land in the UK.')
    expect(pins()).toHaveLength(0)
    expect(cards()).toHaveLength(0)
  })

  it.each([
    ['Paris', PARIS],
    ['the Irish Sea', { lat: 53.7, lon: -4.5 }],
    ['a point east of the antimeridian', { lat: 0, lon: 179.9 }],
  ])('treats %s as not land', (_name, point) => {
    renderMap()
    clickAt(point)
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(pins()).toHaveLength(0)
  })

  it('leaves an existing pin and card exactly as they were', async () => {
    const user = userEvent.setup()
    const { container } = renderMap()
    clickAt(BLUE_MILL)
    await user.click(screen.getByRole('button', { name: /BLUE MILL TEST ATZ/ }))
    const pin = pins()[0]
    const card = cards()[0]

    clickAt(NORTH_SEA)
    expect(screen.getByRole('status')).toHaveTextContent('Pick a field on land in the UK.')
    expect(pins()).toEqual([pin])
    expect(pin.getLatLng()).toMatchObject({ lat: BLUE_MILL.lat, lng: BLUE_MILL.lon })
    expect(cards()).toEqual([card])
    expect(popups(container)).toHaveLength(1)
    expect(screen.getByText('MON-FRI 0800-1700')).toBeInTheDocument() // the open row is still open
  })

  it('shows one toast for several sea clicks and removes it after a few seconds', () => {
    vi.useFakeTimers()
    renderMap()
    clickAt(NORTH_SEA)
    clickAt(PARIS)
    expect(screen.getAllByRole('status')).toHaveLength(1)
    act(() => void vi.advanceTimersByTime(3000))
    clickAt(NORTH_SEA) // a new toast restarts the time
    act(() => void vi.advanceTimersByTime(3000))
    expect(screen.getAllByRole('status')).toHaveLength(1)
    act(() => void vi.advanceTimersByTime(1500))
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('does not hide the card of a pin that is already placed, and a land click after a toast removes nothing but the pick', () => {
    renderMap()
    clickAt(NORTH_SEA)
    clickAt(BLUE_MILL)
    expect(screen.getByRole('status')).toBeInTheDocument() // the toast runs its own time
    expect(pins()).toHaveLength(1)
  })
})

describe('rapid clicks', () => {
  it('shows only the last point’s card, with one pin, and fills it in when the layers arrive', () => {
    const { rerenderWith, container } = renderMap(loading(), loading())
    clickAt(BLUE_MILL)
    clickAt(EDINBURGH)
    clickAt(NORTH_SEA)
    clickAt(BLUE_MILL)
    clickAt(EDINBURGH) // the last land click
    expect(pins()).toHaveLength(1)
    expect(pins()[0].getLatLng()).toMatchObject({ lat: EDINBURGH.lat, lng: EDINBURGH.lon })
    expect(cards()).toHaveLength(1)
    expect(popups(container)).toHaveLength(1)
    expect(screen.getByText('Still loading: Airspace, NOTAMs')).toBeInTheDocument()

    rerenderWith(loadedAirspace, loadedNotams)
    expect(screen.queryByText(/Still loading/)).toBeNull()
    expect(screen.getByText('Nothing in the loaded layers over this point.')).toBeInTheDocument()
    expect(screen.queryByText('BLUE MILL TEST ATZ')).toBeNull()
    expect(pins()).toHaveLength(1)
    expect(cards()).toHaveLength(1)
  })

  it('ends on the Blue Mill card when Blue Mill was the last click', () => {
    const { rerenderWith } = renderMap(loading(), loading())
    clickAt(EDINBURGH)
    clickAt(BLUE_MILL)
    rerenderWith(loadedAirspace, loadedNotams)
    expect(pins()).toHaveLength(1)
    expect(pins()[0].getLatLng()).toMatchObject({ lat: BLUE_MILL.lat, lng: BLUE_MILL.lon })
    expect(screen.getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
  })
})

describe('click propagation', () => {
  /** Counts the click events the map itself receives. */
  function watchMap() {
    const seen = vi.fn()
    mapNow().on('click', seen)
    return seen
  }

  it('control: a DOM click on the map does reach it', () => {
    const { container } = renderMap()
    const seen = watchMap()
    fireEvent.click(container.querySelector('.leaflet-container') as HTMLElement, { clientX: 600, clientY: 400 })
    expect(seen).toHaveBeenCalledTimes(1)
  })

  it('a click on the card, a row toggle and the Above FL100 group never reaches the map, so the pin stays', () => {
    renderMap()
    clickAt(BLUE_MILL)
    const seen = watchMap()
    const pin = pins()[0]
    const before = pin.getLatLng()

    // fireEvent, not userEvent: quick userEvent clicks read as a double-tap to Leaflet, which jsdom cannot place
    fireEvent.click(screen.getByRole('heading', { name: 'Over this point' }))
    fireEvent.click(screen.getByText('The chat checks everything within 2 NM.'))
    fireEvent.click(screen.getByRole('button', { name: /BLUE MILL TEST ATZ/ }))
    fireEvent.click(screen.getByRole('button', { name: /H2001\/26/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Above FL100 (3)' }))
    expect(screen.getByText('MON-FRI 0800-1700')).toBeInTheDocument() // the clicks did reach the rows

    expect(seen).not.toHaveBeenCalled()
    expect(pins()).toEqual([pin])
    expect(pin.getLatLng()).toEqual(before)
    expect(cards()).toHaveLength(1)
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('a click on the toast never reaches the map, so it makes no pin and no second toast', () => {
    renderMap()
    clickAt(NORTH_SEA)
    const seen = watchMap()
    fireEvent.click(screen.getByRole('status'))
    expect(seen).not.toHaveBeenCalled()
    expect(pins()).toHaveLength(0)
    expect(screen.getAllByRole('status')).toHaveLength(1)
  })
})

describe('the card follows the loaded data', () => {
  it('shows the new data in the open card when a layer is replaced, without another click', () => {
    const { rerenderWith } = renderMap(loadedAirspace, loaded<NotamProps>([]))
    clickAt(BLUE_MILL)
    expect(screen.queryByText('H2001/26')).toBeNull()
    rerenderWith(loadedAirspace, loadedNotams)
    expect(screen.getByText('H2001/26')).toBeInTheDocument()
  })

  it('re-measures the card (popup update) when its size changes', () => {
    const observed: Array<() => void> = []
    const original = globalThis.ResizeObserver
    globalThis.ResizeObserver = class {
      constructor(callback: () => void) {
        observed.push(callback)
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver
    try {
      renderMap()
      clickAt(BLUE_MILL)
      const update = vi.spyOn(L.Popup.prototype, 'update')
      act(() => observed.forEach((callback) => callback()))
      expect(update).toHaveBeenCalled()
    } finally {
      globalThis.ResizeObserver = original
    }
  })
})

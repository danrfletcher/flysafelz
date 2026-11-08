import L from 'leaflet'
import { act, render, screen } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { useMap } from 'react-leaflet'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { UkMap } from '@/components/UkMap'
import { cartoTileUrl, type BasemapStatus } from '@/map/basemap'
import { readUi } from '@/test/files'

// jsdom has no layout, so give every element a size for Leaflet to fit the map into.
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

/** Hands the Leaflet map out of the react-leaflet tree. */
function GetMap({ onMap }: { onMap: (map: L.Map) => void }) {
  const map = useMap()
  useEffect(() => onMap(map), [map, onMap])
  return null
}

/** The map as App holds it: the status in state, passed down and reported back. */
function Harness({ onMap }: { onMap: (map: L.Map) => void }) {
  const [status, setStatus] = useState<BasemapStatus>('loading')
  return (
    <>
      <UkMap basemap={status} onBasemapStatus={setStatus}>
        <GetMap onMap={onMap} />
      </UkMap>
      <output aria-label="basemap status">{status}</output>
    </>
  )
}

function mount() {
  let theMap: L.Map | undefined
  const view = render(<Harness onMap={(map) => {
        theMap = map
      }} />)
  let tiles: L.TileLayer | undefined
  theMap!.eachLayer((layer) => {
    if (layer instanceof L.TileLayer) tiles = layer
  })
  const tile = (name: 'tileload' | 'tileerror' | 'load') => act(() => void tiles!.fire(name, { tile: document.createElement('img'), coords: { x: 1, y: 1, z: 5 } }))
  return { map: theMap!, tiles: tiles!, tile, container: view.container.querySelector('.leaflet-container') as HTMLElement }
}

const status = () => screen.getByLabelText('basemap status').textContent

describe('the basemap tile layer', () => {
  it('starts loading, with the tiles shown and no unavailable class on the map', () => {
    const { tiles, container } = mount()
    expect(status()).toBe('loading')
    expect(tiles.options.opacity).toBe(1)
    expect(container).not.toHaveClass('basemap-unavailable')
  })

  it('goes unavailable when a batch ends with failed tiles and none loaded: tiles hidden, the container takes the unavailable class', () => {
    const { tiles, tile, container } = mount()
    tile('tileerror')
    tile('tileerror')
    expect(status()).toBe('loading') // not judged until the batch ends
    tile('load')
    expect(status()).toBe('unavailable')
    expect(tiles.options.opacity).toBe(0)
    expect(container).toHaveClass('basemap-unavailable')
  })

  it('goes back to ok and shows the tiles again when a tile loads', () => {
    const { tiles, tile, container } = mount()
    tile('tileerror')
    tile('load')
    expect(status()).toBe('unavailable')
    tile('tileload')
    expect(status()).toBe('ok')
    expect(tiles.options.opacity).toBe(1)
    expect(container).not.toHaveClass('basemap-unavailable')
  })

  it('stays ok when one tile fails among loaded ones', () => {
    const { tile, container } = mount()
    tile('tileload')
    tile('tileload')
    tile('tileerror')
    tile('load')
    expect(status()).toBe('ok')
    expect(container).not.toHaveClass('basemap-unavailable')
  })

  it('keeps the OpenStreetMap, CARTO, openAIP and Open-Meteo (wind) credits, with links, in both states', () => {
    const { map, tile, container } = mount()
    const credits = () => container.querySelector('.leaflet-control-attribution') as HTMLElement
    const check = () => {
      expect(credits()).toHaveTextContent('OpenStreetMap')
      expect(credits()).toHaveTextContent('CARTO')
      expect(credits()).toHaveTextContent('openAIP')
      expect(credits()).toHaveTextContent('wind data Open-Meteo (CC BY 4.0)')
      const links = [...credits().querySelectorAll('a')].map((a) => a.getAttribute('href'))
      expect(links).toEqual(expect.arrayContaining(['https://www.openstreetmap.org/copyright', 'https://carto.com/attributions', 'https://www.openaip.net', 'https://open-meteo.com']))
    }
    check()
    tile('tileerror')
    tile('load')
    expect(status()).toBe('unavailable')
    let tileLayers = 0
    map.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) tileLayers += 1
    })
    expect(tileLayers).toBe(1) // hidden, not removed, and no second provider added
    check()
  })

  it('draws the map background from the theme variable, and the stylesheet gives the unavailable map that same colour (never Leaflet’s grey)', () => {
    const css = readUi('src/styles/index.css')
    expect(css).toMatch(/\.leaflet-container,\s*\.leaflet-container\.basemap-unavailable\s*\{[^}]*background:\s*var\(--map-background\)/)
    expect(readUi('src/styles/theme.css')).toMatch(/--map-background:\s*var\(--background\)/)
    expect(css).not.toMatch(new RegExp(['#d', 'dd'].join(''), 'i')) // Leaflet's default grey, built from pieces so that the colour-literal rule does not match this file
  })
})

describe('the tile URL the layer is given (env)', () => {
  const templateOf = (layer: L.TileLayer) => (layer as unknown as { _url: string })._url

  it.each([['empty', ''], ['blank', '   ']])('is the keyless builder output with VITE_CARTO_KEY %s', (_label, value) => {
    vi.stubEnv('VITE_CARTO_KEY', value)
    const { tiles } = mount()
    expect(templateOf(tiles)).toBe(cartoTileUrl())
    expect(templateOf(tiles)).not.toContain('key=')
  })

  it('is the builder output for the key when VITE_CARTO_KEY is set, and Leaflet turns it into a tile URL carrying the key once', () => {
    vi.stubEnv('VITE_CARTO_KEY', ' fake&key 1 ')
    const { tiles } = mount()
    expect(templateOf(tiles)).toBe(cartoTileUrl('fake&key 1'))
    const url = new URL(tiles.getTileUrl({ x: 1, y: 2, z: 3 } as L.Coords))
    expect(url.host).toMatch(/^[a-d]\.basemaps\.cartocdn\.com$/)
    expect(url.pathname).toMatch(/^\/dark_all\/\d+\/1\/2\.png$/)
    expect([...url.searchParams.entries()]).toEqual([['key', 'fake&key 1']])
  })
})

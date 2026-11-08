import { afterEach, describe, expect, it, vi } from 'vitest'

import { basemapStatus, cartoTileUrl, CARTO_KEY_PARAM, CARTO_TILE_URL, INITIAL_BASEMAP, tileUrl, type BasemapEvent, type BasemapState } from '@/map/basemap'

afterEach(() => {
  vi.unstubAllEnvs()
})

// The keyless URL the core PR drew the map with, written out here so a change to the constant shows up as a failing test.
const CORE_KEYLESS_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'

describe('cartoTileUrl', () => {
  it('names CARTO’s documented parameter, `key`', () => {
    expect(CARTO_KEY_PARAM).toBe('key')
    expect(CARTO_TILE_URL).toBe(CORE_KEYLESS_URL)
  })

  it.each([
    ['undefined', undefined],
    ['an empty string', ''],
    ['only spaces', '   '],
    ['only tabs and a newline', '\t\n '],
  ])('is the keyless core URL, byte for byte, for %s', (_label, key) => {
    expect(cartoTileUrl(key)).toBe(CORE_KEYLESS_URL)
    expect(cartoTileUrl(key)).not.toContain('?')
  })

  it.each([
    ['a plain key', 'abc123XYZ', 'abc123XYZ'],
    ['a key with reserved characters', 'a&b?c#d e=f/g+h%i', 'a%26b%3Fc%23d%20e%3Df%2Fg%2Bh%25i'],
    ['a padded key', '  abc123  ', 'abc123'],
    ['a key with a placeholder-looking part', 'x{z}y', 'x%7Bz%7Dy'],
    ['a key with non-ASCII text', 'clé', 'cl%C3%A9'],
  ])('puts %s once, encoded, after the template', (_label, key, encoded) => {
    const url = cartoTileUrl(key)
    expect(url).toBe(`${CORE_KEYLESS_URL}?key=${encoded}`)
    expect(url.split('?')).toHaveLength(2)
    expect(url.match(/[?&]key=/g)).toHaveLength(1)
    // the template part is untouched, so Leaflet still finds exactly its five placeholders
    expect([...url.matchAll(/\{([a-z]+)\}/g)].map((match) => match[1])).toEqual(['s', 'z', 'x', 'y', 'r'])
  })

  it('does not let a key end the query early or start a second parameter', () => {
    const url = new URL(cartoTileUrl('k&x=1#frag').replace('{s}', 'a').replace('{z}', '3').replace('{x}', '1').replace('{y}', '2').replace('{r}', ''))
    expect([...url.searchParams.keys()]).toEqual(['key'])
    expect(url.searchParams.get('key')).toBe('k&x=1#frag')
    expect(url.hash).toBe('')
    expect(url.host).toBe('a.basemaps.cartocdn.com')
  })
})

describe('tileUrl (the one place the key is read)', () => {
  it('is the keyless URL with VITE_CARTO_KEY unset, empty or blank', () => {
    for (const value of ['', '   ']) {
      vi.stubEnv('VITE_CARTO_KEY', value)
      expect(tileUrl()).toBe(CORE_KEYLESS_URL)
    }
  })

  it('carries the trimmed, encoded key when VITE_CARTO_KEY is set', () => {
    vi.stubEnv('VITE_CARTO_KEY', '  fake key&1 ')
    expect(tileUrl()).toBe(cartoTileUrl('fake key&1'))
    expect(tileUrl()).toBe(`${CORE_KEYLESS_URL}?key=fake%20key%261`)
  })
})

/** Runs the events through the reducer, as the tile layer would, and returns the state after each. */
function run(events: BasemapEvent[], from: BasemapState = INITIAL_BASEMAP): BasemapState[] {
  const states: BasemapState[] = []
  let state = from
  for (const event of events) {
    state = basemapStatus(state, event)
    states.push(state)
  }
  return states
}
const statuses = (events: BasemapEvent[], from?: BasemapState) => run(events, from).map((state) => state.status)

describe('basemapStatus', () => {
  it('starts as loading, with nothing counted', () => {
    expect(INITIAL_BASEMAP).toEqual({ status: 'loading', loaded: 0, failed: 0 })
  })

  it('loading → ok on the first tile that loads, and stays ok when the batch ends', () => {
    expect(statuses(['tileload'])).toEqual(['ok'])
    expect(statuses(['tileload', 'tileload', 'load'])).toEqual(['ok', 'ok', 'ok'])
  })

  it('loading → unavailable when a batch ends with a failed tile and no loaded tile, not before', () => {
    expect(statuses(['tileerror', 'tileerror', 'load'])).toEqual(['loading', 'loading', 'unavailable'])
  })

  it('stays loading when a batch ends with nothing loaded and nothing failed', () => {
    expect(statuses(['load'])).toEqual(['loading'])
  })

  it('unavailable → ok as soon as a later tile loads (pan, zoom, network back), and stays ok at the batch end', () => {
    const down = run(['tileerror', 'load']).at(-1)!
    expect(down.status).toBe('unavailable')
    expect(statuses(['tileload', 'load'], down)).toEqual(['ok', 'ok'])
  })

  it('stays unavailable when the next batch fails too', () => {
    const down = run(['tileerror', 'load']).at(-1)!
    expect(statuses(['tileerror', 'tileerror', 'load'], down)).toEqual(['unavailable', 'unavailable', 'unavailable'])
  })

  it('stays ok when one tile fails among loaded ones, in either order', () => {
    expect(statuses(['tileload', 'tileerror', 'tileload', 'load'])).toEqual(['ok', 'ok', 'ok', 'ok'])
    expect(statuses(['tileerror', 'tileload', 'tileload', 'load'])).toEqual(['loading', 'ok', 'ok', 'ok'])
  })

  it('does not flap over repeated batches: ok, one failed tile among loaded ones, ok', () => {
    const batch: BasemapEvent[] = ['tileload', 'tileload', 'tileerror', 'load']
    expect(statuses([...batch, ...batch, ...batch]).every((status) => status === 'ok')).toBe(true)
  })

  it('judges each batch on its own counts: loads from an earlier batch do not rescue a later all-failed one', () => {
    expect(statuses(['tileload', 'load', 'tileerror', 'load'])).toEqual(['ok', 'ok', 'ok', 'unavailable'])
  })

  it('does not change the state it is given', () => {
    const before = { ...INITIAL_BASEMAP }
    basemapStatus(INITIAL_BASEMAP, 'tileerror')
    basemapStatus(INITIAL_BASEMAP, 'load')
    expect(INITIAL_BASEMAP).toEqual(before)
  })
})

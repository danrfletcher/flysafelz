import L from 'leaflet'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { RAIN_PANE } from '@/map/rainPane'
import { EMPTY_TILE, FRAME_MS, LAST_FRAME_MS, RainRadar } from '@/map/rainRadar'
import type { RainFrame } from '@/map/rainIndex'
import { TileGate, TokenBucket } from '@/map/rainScheduler'

// The radar on a real Leaflet map in jsdom, with fake timers (setTimeout and clearTimeout only). Tiles never load in jsdom, so a test says when they do by firing
// `load` (or `error`) on the tile images, as the browser would.

const HOST = 'https://tilecache.rainviewer.com'
const frame = (n: number): RainFrame => ({ time: 1_794_140_000 + n * 600, path: `/v2/radar/f${n}` }) // f0 is the oldest
const FRAMES = [frame(0), frame(1), frame(2)]

let maps: L.Map[] = []
function makeMap(zoom = 5) {
  const el = document.createElement('div')
  Object.defineProperty(el, 'clientWidth', { value: 800 })
  Object.defineProperty(el, 'clientHeight', { value: 600 })
  document.body.append(el)
  const map = L.map(el, { zoomSnap: 0.25 }).setView([54, -2], zoom)
  maps.push(map)
  return map
}

const tilesOf = (map: L.Map, path?: string) =>
  [...map.getPane(RAIN_PANE)!.querySelectorAll('img')].filter((img) => !path || img.src.includes(path))
const settle = (map: L.Map, path?: string) => tilesOf(map, path).forEach((img) => img.dispatchEvent(new Event('load')))
const layersIn = (map: L.Map) => {
  const found: L.TileLayer[] = []
  map.eachLayer((layer) => {
    if (layer instanceof L.TileLayer && layer.options.pane === RAIN_PANE) found.push(layer)
  })
  return found
}

/** A gate that never makes a tile wait for a token (the throttle has its own tests in rainScheduler.test.ts). */
const openGate = () => new TileGate(new TokenBucket(100_000, 100_000))

function setup(over: { zoom?: number; gate?: TileGate } = {}) {
  const map = makeMap(over.zoom)
  const gate = over.gate ?? openGate()
  const shown: (number | null)[] = []
  const radar = new RainRadar(map, gate, (time) => shown.push(time))
  return { map, gate, radar, shown }
}

beforeEach(() => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] }))
afterEach(() => {
  vi.useRealTimers()
  maps.forEach((map) => map.remove())
  maps = []
  document.body.innerHTML = ''
})

describe('loading: the latest frame first', () => {
  it('makes only the latest frame’s layer until its tiles are in, and shows it', () => {
    const { map, radar, shown } = setup()
    radar.update(HOST, FRAMES, false)
    expect(layersIn(map)).toHaveLength(1)
    expect(tilesOf(map).length).toBeGreaterThan(0)
    expect(tilesOf(map).every((img) => img.src.startsWith(`${HOST}/v2/radar/f2/256/`))).toBe(true)
    expect(shown.at(-1)).toBe(FRAMES[2].time)
  })

  it('asks for the other frames once the latest has its tiles, newest first, and starts the loop once every one has', () => {
    const { map, radar, shown } = setup()
    radar.update(HOST, FRAMES, false)
    settle(map)
    expect(layersIn(map)).toHaveLength(3)
    expect(tilesOf(map, '/f1/').length).toBeGreaterThan(0)
    // the loop has not started: the other two frames have not got their tiles
    vi.advanceTimersByTime(10_000)
    expect(shown.at(-1)).toBe(FRAMES[2].time)
    settle(map, '/f1/')
    vi.advanceTimersByTime(10_000)
    expect(shown.at(-1)).toBe(FRAMES[2].time)
    settle(map, '/f0/')
    const before = shown.length
    vi.advanceTimersByTime(LAST_FRAME_MS)
    expect(shown.length).toBe(before + 1)
  })

  it('draws tiles at the radar’s own zoom and never above 7, however far in the map is', () => {
    const { map, radar } = setup({ zoom: 11 })
    radar.update(HOST, FRAMES, false)
    const zooms = tilesOf(map).map((img) => Number(/\/256\/(\d+)\//.exec(img.src)![1]))
    expect(zooms.length).toBeGreaterThan(0)
    expect(Math.max(...zooms)).toBe(7)
    map.setZoom(9)
    expect(Math.max(...tilesOf(map).map((img) => Number(/\/256\/(\d+)\//.exec(img.src)![1])))).toBe(7)
  })
})

describe('the loop', () => {
  function looping() {
    const made = setup()
    made.radar.update(HOST, FRAMES, false)
    settle(made.map)
    settle(made.map)
    return made
  }

  it('steps oldest to latest in time order, holds the latest a little longer, and wraps', () => {
    const { shown } = looping()
    shown.length = 0
    vi.advanceTimersByTime(LAST_FRAME_MS)
    expect(shown).toEqual([FRAMES[0].time])
    vi.advanceTimersByTime(FRAME_MS)
    vi.advanceTimersByTime(FRAME_MS)
    expect(shown).toEqual([FRAMES[0].time, FRAMES[1].time, FRAMES[2].time])
    vi.advanceTimersByTime(LAST_FRAME_MS)
    expect(shown.at(-1)).toBe(FRAMES[0].time)
  })

  it('shows one frame at a time: the others’ layers have opacity 0', () => {
    const { map, shown } = looping()
    vi.advanceTimersByTime(LAST_FRAME_MS)
    const opacities = layersIn(map).map((layer) => layer.options.opacity)
    expect(opacities.filter((o) => o === 1)).toHaveLength(1)
    expect(opacities.filter((o) => o === 0)).toHaveLength(2)
    expect(shown.at(-1)).toBe(FRAMES[0].time)
  })

  it('runs one loop however often it is told the same frames', () => {
    const { radar, shown, map } = looping()
    for (let i = 0; i < 5; i += 1) radar.update(HOST, FRAMES, false)
    expect(layersIn(map)).toHaveLength(3)
    shown.length = 0
    vi.advanceTimersByTime(LAST_FRAME_MS + 2 * FRAME_MS + LAST_FRAME_MS + 2 * FRAME_MS)
    expect(shown).toHaveLength(6) // two laps of three steps: a doubled timer would give more
  })

  it('stops while paused (the zoom has faded the layer out) and carries on when it is not', () => {
    const { radar, shown } = looping()
    radar.setPaused(true)
    shown.length = 0
    vi.advanceTimersByTime(60_000)
    expect(shown).toEqual([])
    radar.setPaused(false)
    radar.setPaused(false)
    vi.advanceTimersByTime(LAST_FRAME_MS)
    expect(shown).toHaveLength(1)
  })

  it('pausing and resuming many times leaves one loop', () => {
    const { radar, shown } = looping()
    for (let i = 0; i < 10; i += 1) {
      radar.setPaused(true)
      radar.setPaused(false)
    }
    shown.length = 0
    vi.advanceTimersByTime(LAST_FRAME_MS + FRAME_MS * 2)
    expect(shown).toHaveLength(3)
  })

  it('goes on past a tile that failed: it leaves a transparent gap, throws nothing, and the loop still runs', () => {
    const { map, radar, shown } = setup()
    radar.update(HOST, FRAMES, false)
    settle(map)
    settle(map, '/f1/')
    const failing = tilesOf(map, '/f0/')
    expect(failing.length).toBeGreaterThan(0)
    expect(() => failing.forEach((img) => img.dispatchEvent(new Event('error')))).not.toThrow()
    expect(failing.every((img) => img.src === EMPTY_TILE)).toBe(true)
    shown.length = 0
    vi.advanceTimersByTime(LAST_FRAME_MS + FRAME_MS * 2)
    expect(shown).toHaveLength(3)
  })
})

describe('one frame, no frame, reduced motion', () => {
  it('shows a single frame still, without a loop and without an error', () => {
    const { map, radar, shown } = setup()
    radar.update(HOST, [FRAMES[2]], false)
    settle(map)
    shown.length = 0
    vi.advanceTimersByTime(60_000)
    expect(layersIn(map)).toHaveLength(1)
    expect(shown).toEqual([])
  })

  it('with reduced motion makes the latest frame only: no other frame is fetched and nothing loops', () => {
    const { map, radar, shown, gate } = setup()
    radar.update(HOST, FRAMES, true)
    settle(map)
    expect(layersIn(map)).toHaveLength(1)
    expect(tilesOf(map).every((img) => img.src.includes('/f2/'))).toBe(true)
    expect(shown.at(-1)).toBe(FRAMES[2].time)
    shown.length = 0
    vi.advanceTimersByTime(60_000)
    expect(shown).toEqual([])
    expect(gate.requests).toBe(tilesOf(map).length)
  })

  it('switching to reduced motion drops the other frames and the loop; switching back brings them', () => {
    const { map, radar, shown } = setup()
    radar.update(HOST, FRAMES, false)
    settle(map)
    settle(map)
    radar.update(HOST, FRAMES, true)
    expect(layersIn(map)).toHaveLength(1)
    shown.length = 0
    vi.advanceTimersByTime(60_000)
    expect(shown).toEqual([])
    radar.update(HOST, FRAMES, false)
    settle(map)
    expect(layersIn(map)).toHaveLength(3)
  })

  it('shows nothing for no frames', () => {
    const { map, radar, shown } = setup()
    radar.update(HOST, [], false)
    expect(layersIn(map)).toHaveLength(0)
    expect(shown.at(-1)).toBeNull()
  })
})

describe('refresh', () => {
  it('appends a new frame and drops one that has left the window, keeping the others’ layers', () => {
    const { map, radar } = setup()
    radar.update(HOST, FRAMES, false)
    settle(map)
    settle(map)
    const kept = layersIn(map).find((layer) => layer.getTileUrl({ x: 0, y: 0, z: 3 } as L.Coords).includes('/f1/'))
    const fresh = frame(3)
    radar.update(HOST, [FRAMES[1], FRAMES[2], fresh], false)
    const paths = layersIn(map).map((layer) => /radar\/(f\d)/.exec(layer.getTileUrl({ x: 0, y: 0, z: 3 } as L.Coords))![1])
    expect(paths.sort()).toEqual(['f1', 'f2', 'f3'])
    expect(layersIn(map)).toContain(kept)
  })

  it('does not fetch a tile again that it has: a frame kept across a refresh costs no request', () => {
    const { map, radar, gate } = setup()
    radar.update(HOST, FRAMES, false)
    settle(map)
    settle(map)
    const before = gate.requests
    radar.update(HOST, FRAMES, false)
    expect(gate.requests).toBe(before)
  })

  it('tries the tiles that failed again at the next refresh', () => {
    const { map, radar, gate } = setup()
    radar.update(HOST, [FRAMES[2]], false)
    const tiles = tilesOf(map)
    tiles.forEach((img) => img.dispatchEvent(new Event('error')))
    const before = gate.requests
    radar.update(HOST, [FRAMES[2]], false)
    expect(gate.requests).toBeGreaterThan(before)
  })
})

describe('switching off (destroy)', () => {
  it('removes every layer, stops the loop and the queued tile requests, and tells the page there is no frame', () => {
    const { map, radar, shown, gate } = setup()
    radar.update(HOST, FRAMES, false)
    settle(map)
    settle(map)
    radar.destroy()
    expect(layersIn(map)).toHaveLength(0)
    expect(map.getPane(RAIN_PANE)!.querySelectorAll('img')).toHaveLength(0)
    expect(shown.at(-1)).toBeNull()
    shown.length = 0
    vi.advanceTimersByTime(60_000)
    expect(shown).toEqual([])
    expect(gate.requests).toBeGreaterThan(0)
    radar.update(HOST, FRAMES, false) // a radar that is destroyed ignores later calls
    expect(layersIn(map)).toHaveLength(0)
  })

  it('switching on again with the same gate makes one layer set, one loop, and no request for tiles already loaded', () => {
    const gate = openGate()
    const first = setup({ gate })
    first.radar.update(HOST, FRAMES, false)
    settle(first.map)
    settle(first.map)
    const requests = gate.requests
    first.radar.destroy()

    const { map, radar, shown } = setup({ gate })
    radar.update(HOST, FRAMES, false)
    settle(map)
    settle(map)
    expect(gate.requests).toBe(requests + 0) // the same view: every URL was loaded before, so none is asked for again
    expect(layersIn(map)).toHaveLength(3)
    shown.length = 0
    vi.advanceTimersByTime(LAST_FRAME_MS + 2 * FRAME_MS)
    expect(shown).toHaveLength(3)
  })

  it('off and on again and again leaves exactly one layer set', () => {
    const gate = openGate()
    const map = makeMap()
    for (let i = 0; i < 6; i += 1) {
      const radar = new RainRadar(map, gate, () => undefined)
      radar.update(HOST, FRAMES, false)
      if (i < 5) radar.destroy()
    }
    expect(layersIn(map)).toHaveLength(1) // the latest frame, waiting for its tiles
    expect(map.getPane(RAIN_PANE)!.children.length).toBeGreaterThan(0)
  })
})

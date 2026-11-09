import L from 'leaflet'

import { ensureRainPane, RAIN_PANE } from '@/map/rainPane'
import { MAX_NATIVE_ZOOM, tileTemplate, TILE_SIZE, type RainFrame } from '@/map/rainIndex'
import { planFrames, type TileGate } from '@/map/rainScheduler'

// The radar: one Leaflet tile layer per frame, all in the rain pane, one of them showing. The latest frame loads first and shows alone; the rest load behind it,
// newest first; once every one of them has its tiles (loaded or failed) the loop starts, and steps through them in time order. Every tile request goes
// through the TileGate (throttled, once per URL).

/** How long a frame stays on screen, and the last one (the latest: "now"), in milliseconds. */
export const FRAME_MS = 700
export const LAST_FRAME_MS = 1500

/** A transparent 1x1 GIF: a tile that failed (404, 5xx, network) shows this, which is a gap in the radar and nothing else. */
export const EMPTY_TILE = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=='

/** A radar tile layer whose tiles ask the gate before they go to the network, and which says `settled` when it has no tile still waiting. */
class FrameLayer extends L.TileLayer {
  private readonly gate: TileGate
  private readonly pending = new Set<HTMLElement>()
  private readonly gone = new WeakSet<HTMLElement>()

  constructor(url: string, options: L.TileLayerOptions, gate: TileGate) {
    super(url, options)
    this.gate = gate
    // A tile the map drops (panned away, layer removed) is no longer waited for, and a queued request for it is not made.
    const drop = (event: L.TileEvent) => {
      this.gone.add(event.tile)
      this.pending.delete(event.tile)
      this.checkSettled()
    }
    this.on('tileunload', drop)
    this.on('tileabort', drop)
  }

  onAdd(map: L.Map): this {
    super.onAdd(map)
    this.checkSettled() // the first view's tiles have been asked for by now; none to wait for is settled too
    return this
  }

  private checkSettled() {
    if (this.pending.size === 0) this.fire('settled')
  }

  createTile(coords: L.Coords, done: L.DoneCallback): HTMLElement {
    const tile = document.createElement('img')
    tile.alt = ''
    tile.setAttribute('role', 'presentation')
    const url = this.getTileUrl(coords)
    this.pending.add(tile)
    let finished = false
    const finish = (error?: Error) => {
      if (finished) return
      finished = true
      this.pending.delete(tile)
      done(error, tile)
      this.checkSettled()
    }
    tile.addEventListener('load', () => {
      if (tile.src === EMPTY_TILE) return
      this.gate.loaded(url)
      finish()
    })
    tile.addEventListener('error', () => {
      if (tile.src === EMPTY_TILE) return
      this.gate.failed(url)
      tile.src = EMPTY_TILE // a transparent gap, not a broken-image mark
      finish(new Error('radar tile failed'))
    })
    const gap = (error: Error) => {
      tile.src = EMPTY_TILE
      setTimeout(() => finish(error), 0) // Leaflet wants `done` after createTile has returned
    }
    this.gate.request(url, {
      alive: () => !this.gone.has(tile),
      start: () => {
        tile.src = url
      },
      skip: () => gap(new Error('radar tile skipped')),
    })
    return tile
  }
}

interface Entry {
  frame: RainFrame
  layer: FrameLayer
}

/**
 * The radar on one map. `update` is told the frames, and builds, keeps and removes layers to match; `setPaused` stops and restarts the loop (the layer is
 * faded out); `destroy` takes everything off the map, stops the timer and drops the queued tile requests. There is at most one timer at any time.
 */
export class RainRadar {
  private readonly entries = new Map<number, Entry>()
  private readonly settled = new Set<number>()
  private host = ''
  private reduced = false
  private paused = false
  private started = false
  private current: number | null = null
  private timer: ReturnType<typeof setTimeout> | null = null
  private destroyed = false
  /** The frames to show, latest first (just the latest with reduced motion). */
  private wanted: RainFrame[] = []

  constructor(
    private readonly map: L.Map,
    private readonly gate: TileGate,
    private readonly onFrame: (time: number | null) => void,
  ) {
    ensureRainPane(map)
  }

  update(host: string, frames: readonly RainFrame[], reduced: boolean) {
    if (this.destroyed) return
    if (host !== this.host) this.removeAll() // another tile host: none of the layers' URLs are right
    this.host = host
    if (reduced !== this.reduced) this.started = false
    this.reduced = reduced
    // Tiles that failed at the last refresh are tried again with this one.
    const retry = this.gate.forgetFailed()

    this.wanted = planFrames(frames, reduced)
    const keep = new Set(this.wanted.map((frame) => frame.time))
    for (const time of [...this.entries.keys()]) if (!keep.has(time)) this.remove(time)

    const latest = this.wanted[0]
    if (!latest) {
      this.stop()
      this.show(null)
      return
    }
    if (retry) for (const { layer } of this.entries.values()) layer.redraw()
    this.ensure(latest) // the latest frame, alone, first
    if (this.settled.has(latest.time)) this.ensureAll() // then the others, newest first
    if (this.current === null || !this.entries.has(this.current)) this.show(latest.time)
    this.sync()
  }

  setPaused(paused: boolean) {
    if (this.destroyed || paused === this.paused) return
    this.paused = paused
    this.sync()
  }

  destroy() {
    this.destroyed = true
    this.stop()
    this.removeAll()
    this.gate.clear()
    this.onFrame(null)
  }

  private ensure(frame: RainFrame) {
    if (this.entries.has(frame.time)) return
    const layer = new FrameLayer(
      tileTemplate(this.host, frame.path),
      { pane: RAIN_PANE, tileSize: TILE_SIZE, maxNativeZoom: MAX_NATIVE_ZOOM, maxZoom: 20, opacity: this.current === frame.time ? 1 : 0, keepBuffer: 1 },
      this.gate,
    )
    this.entries.set(frame.time, { frame, layer })
    layer.once('settled', () => this.onSettled(frame.time, layer))
    layer.addTo(this.map)
  }

  private onSettled(time: number, layer: FrameLayer) {
    if (this.destroyed || this.entries.get(time)?.layer !== layer) return
    this.settled.add(time)
    if (time === this.wanted[0]?.time) this.ensureAll() // the latest frame is in: ask for the others, newest first
    this.sync()
  }

  private ensureAll() {
    for (const frame of this.wanted) this.ensure(frame)
  }

  private remove(time: number) {
    const entry = this.entries.get(time)
    if (!entry) return
    entry.layer.remove()
    this.entries.delete(time)
    this.settled.delete(time)
  }

  private removeAll() {
    for (const time of [...this.entries.keys()]) this.remove(time)
    this.current = null
    this.started = false
  }

  private show(time: number | null) {
    this.current = time
    for (const { frame, layer } of this.entries.values()) layer.setOpacity(frame.time === time ? 1 : 0)
    this.onFrame(time)
  }

  private stop() {
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
  }

  /** (Re)starts the loop when it should run, and stops it when it should not. Whatever it does, at most one timer is left. */
  private sync() {
    this.stop()
    if (this.destroyed || this.reduced || this.paused) return
    const ready = this.readyTimes()
    if (ready.length < 2) return // one frame is shown still
    if (!this.started) {
      if (ready.length < this.entries.size) return // the loop starts once every frame has its tiles
      this.started = true
    }
    this.timer = setTimeout(() => this.step(), this.current === ready[ready.length - 1] ? LAST_FRAME_MS : FRAME_MS)
  }

  private readyTimes(): number[] {
    return [...this.settled].filter((time) => this.entries.has(time)).sort((a, b) => a - b)
  }

  private step() {
    this.timer = null
    const ready = this.readyTimes()
    if (ready.length < 2) return
    const after = ready.find((time) => this.current === null || time > this.current)
    this.show(after ?? ready[0])
    this.sync()
  }
}

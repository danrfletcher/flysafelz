// The Rain layer's scheduling (G5): which frames are on the loop, in what order their tiles are asked for, and how fast. Nothing here touches Leaflet or
// React, so the rules can be tested with a fake clock.
//
//   framesInWindow / mergeFrames   the 2 hour window: a refresh appends new frames and drops those older than 2 hours
//   loadOrder / planFrames         the latest frame first; with reduced motion, the latest frame only
//   TokenBucket                    at most 95 tile requests in any minute, under RainViewer's 100 per minute per IP
//   TileGate                       every tile request goes through it: throttled, queued in order, and made once per URL

import type { RainFrame } from '@/map/rainIndex'

/** Radar older than this is not shown. */
export const WINDOW_MS = 2 * 60 * 60 * 1000

/** The frames not older than the window at `nowMs` (a frame exactly 2 hours old stays), oldest first, one per time. */
export function framesInWindow(frames: readonly RainFrame[], nowMs: number): RainFrame[] {
  const byTime = new Map<number, RainFrame>()
  for (const frame of frames) {
    if (nowMs - frame.time * 1000 <= WINDOW_MS) byTime.set(frame.time, frame)
  }
  return [...byTime.values()].sort((a, b) => a.time - b.time)
}

/**
 * The frames after a refresh: those already held and those in the new index, inside the window. A frame held before keeps its object, so
 * the layer sees the same frame, not a new one. `added` are the frames that were not held, `dropped` the held ones that left the window or the index.
 */
export function mergeFrames(
  held: readonly RainFrame[],
  incoming: readonly RainFrame[],
  nowMs: number,
): { frames: RainFrame[]; added: RainFrame[]; dropped: RainFrame[] } {
  const known = new Map(held.map((frame) => [frame.time, frame]))
  const frames = framesInWindow([...held, ...incoming.filter((frame) => !known.has(frame.time))], nowMs)
  const kept = new Set(frames.map((frame) => frame.time))
  return {
    frames,
    added: frames.filter((frame) => !known.has(frame.time)),
    dropped: held.filter((frame) => !kept.has(frame.time)),
  }
}

/** The order frames are loaded in: the latest first, then back in time. */
export function loadOrder(frames: readonly RainFrame[]): RainFrame[] {
  return [...frames].sort((a, b) => b.time - a.time)
}

/** The frames to load at all: every frame, or with reduced motion the latest one and nothing else (E12). */
export function planFrames(frames: readonly RainFrame[], reducedMotion: boolean): RainFrame[] {
  const ordered = loadOrder(frames)
  return reducedMotion ? ordered.slice(0, 1) : ordered
}

/**
 * The most requests the bucket allows in any one minute is its size plus a minute of refill: 95, under RainViewer's 100 per minute per IP (the
 * 5 left over are for the index). Sustained, it lets 85 a minute through, so the 13 frames of a screenful of tiles are in within a few minutes.
 */
export const BURST = 10
export const PER_MINUTE = 85
export const MAX_PER_MINUTE = BURST + PER_MINUTE

/** A token bucket on the clock `now` (milliseconds): `capacity` tokens at the start, `perMinute` more each minute, never above `capacity`. */
export class TokenBucket {
  private tokens: number
  private last: number

  constructor(
    private readonly capacity = BURST,
    private readonly perMinute = PER_MINUTE,
    private readonly now: () => number = Date.now,
  ) {
    this.tokens = capacity
    this.last = now()
  }

  private refill() {
    const at = this.now()
    this.tokens = Math.min(this.capacity, this.tokens + ((at - this.last) * this.perMinute) / 60_000)
    this.last = at
  }

  /** Takes one token if there is one. */
  take(): boolean {
    this.refill()
    if (this.tokens < 1) return false
    this.tokens -= 1
    return true
  }

  /** Milliseconds until a token is there (0 when one is). */
  msUntilToken(): number {
    this.refill()
    return this.tokens >= 1 ? 0 : Math.ceil(((1 - this.tokens) * 60_000) / this.perMinute)
  }
}

/** One tile the layer wants. `alive` says whether it is still wanted (its tile is still on the map), `start` makes the request, `skip` shows a gap instead. */
export interface TileJob {
  alive: () => boolean
  start: () => void
  skip: () => void
}

/**
 * Every radar tile request goes through here. A URL is requested from the network once per page load: after it has loaded, a tile for it starts at once
 * and costs no token (the browser serves it from its cache), and a URL that failed is not asked for again until `forgetFailed`
 * (the 15-minute index refresh), so a tile that fails leaves a gap and never loops. A new URL waits in order for a token. A job whose tile has left
 * the map is dropped without a request. A tile removed in flight is not recorded as loaded, so it is requested (and counted) again if it comes back.
 */
export class TileGate {
  private queue: { url: string; job: TileJob }[] = []
  private readonly done = new Set<string>()
  private readonly broken = new Set<string>()
  private timer: ReturnType<typeof setTimeout> | null = null
  /** How many requests have been let through to the network (each took a token). */
  requests = 0

  constructor(private readonly bucket: TokenBucket = new TokenBucket()) {}

  request(url: string, job: TileJob) {
    if (this.done.has(url)) return job.start()
    if (this.broken.has(url)) return job.skip()
    this.queue.push({ url, job })
    this.drain()
  }

  /** The tile for `url` loaded. */
  loaded(url: string) {
    this.done.add(url)
  }

  /** The tile for `url` failed (404, 5xx, network). */
  failed(url: string) {
    this.broken.add(url)
  }

  /** Lets failed URLs be asked for again; true when there were any. */
  forgetFailed(): boolean {
    const any = this.broken.size > 0
    this.broken.clear()
    return any
  }

  /** Drops every waiting request and the timer that would release them. Loaded and failed URLs are kept. */
  clear() {
    this.queue = []
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
  }

  private drain() {
    if (this.timer !== null) return
    while (this.queue.length > 0) {
      const { url, job } = this.queue[0]
      if (!job.alive()) {
        this.queue.shift()
      } else if (this.done.has(url)) {
        this.queue.shift()
        job.start()
      } else if (this.broken.has(url)) {
        this.queue.shift()
        job.skip()
      } else if (this.bucket.take()) {
        this.queue.shift()
        this.requests += 1
        job.start()
      } else {
        this.timer = setTimeout(() => {
          this.timer = null
          this.drain()
        }, Math.max(1, this.bucket.msUntilToken()))
        return
      }
    }
  }
}

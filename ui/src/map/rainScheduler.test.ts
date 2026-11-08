import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RainFrame } from '@/map/rainIndex'
import {
  BURST,
  framesInWindow,
  loadOrder,
  MAX_PER_MINUTE,
  mergeFrames,
  PER_MINUTE,
  planFrames,
  TileGate,
  TokenBucket,
  WINDOW_MS,
  type TileJob,
} from '@/map/rainScheduler'

const NOW = 1_794_140_000_000
const SECONDS = NOW / 1000
const frame = (minutesAgo: number): RainFrame => ({ time: SECONDS - minutesAgo * 60, path: `/v2/radar/${minutesAgo}` })
const times = (frames: readonly RainFrame[]) => frames.map((f) => (SECONDS - f.time) / 60)

describe('the 2 hour window', () => {
  it('is two hours', () => {
    expect(WINDOW_MS).toBe(7_200_000)
  })

  it('keeps frames up to exactly 2 hours old and drops older ones, oldest first', () => {
    const kept = framesInWindow([frame(0), frame(121), frame(120), frame(60), frame(180)], NOW)
    expect(times(kept)).toEqual([120, 60, 0])
  })

  it('keeps one frame per time', () => {
    expect(framesInWindow([frame(10), { ...frame(10), path: '/other' }], NOW)).toHaveLength(1)
  })

  it('keeps a frame that is in the future (a clock a little behind the server’s)', () => {
    expect(framesInWindow([frame(-3)], NOW)).toHaveLength(1)
  })
})

describe('mergeFrames (the 15 minute refresh)', () => {
  const held = [frame(110), frame(100), frame(90)]

  it('appends new frames and drops those that have left the window', () => {
    const later = NOW + 20 * 60_000 // the frame 110 minutes old at NOW is 130 old now (dropped), the one 100 old is exactly 120 (kept)
    const fresh = frame(-10) // 10 minutes after NOW: 10 minutes old at `later`
    const result = mergeFrames(held, [frame(90), fresh], later)
    expect(times(result.frames)).toEqual([100, 90, -10])
    expect(result.added).toEqual([fresh])
    expect(result.dropped.map((f) => f.path)).toEqual(['/v2/radar/110'])
  })

  it('keeps the frame objects it already held, so a layer sees the same frame', () => {
    const result = mergeFrames(held, [{ ...held[0] }, { ...held[1] }, { ...held[2] }, frame(80)], NOW)
    expect(result.frames[0]).toBe(held[0])
    expect(result.frames[1]).toBe(held[1])
    expect(result.added).toHaveLength(1)
    expect(result.dropped).toEqual([])
  })

  it('holds a frame the new index no longer lists while it is in the window', () => {
    const result = mergeFrames(held, [frame(80)], NOW)
    expect(times(result.frames)).toEqual([110, 100, 90, 80])
    expect(result.dropped).toEqual([])
  })

  it('does nothing when nothing changed', () => {
    const result = mergeFrames(held, held, NOW)
    expect(result.frames).toEqual(held)
    expect(result.added).toEqual([])
    expect(result.dropped).toEqual([])
  })
})

describe('load order and the reduced-motion plan', () => {
  const frames = [frame(30), frame(10), frame(0), frame(20)]

  it('loads the latest frame first, then back in time', () => {
    expect(times(loadOrder(frames))).toEqual([0, 10, 20, 30])
  })

  it('does not change the list it is given', () => {
    const copy = [...frames]
    loadOrder(frames)
    expect(frames).toEqual(copy)
  })

  it('plans every frame, latest first, when motion is fine', () => {
    expect(times(planFrames(frames, false))).toEqual([0, 10, 20, 30])
  })

  it('plans the latest frame only with reduced motion: no other frame is fetched', () => {
    expect(times(planFrames(frames, true))).toEqual([0])
  })

  it('plans nothing for no frames', () => {
    expect(planFrames([], true)).toEqual([])
    expect(planFrames([], false)).toEqual([])
  })
})

describe('TokenBucket (fake timers)', () => {
  beforeEach(() => vi.useFakeTimers({ now: 0 }))
  afterEach(() => vi.useRealTimers())

  it('stays under 100 a minute: its burst plus a minute of refill', () => {
    expect(BURST + PER_MINUTE).toBe(MAX_PER_MINUTE)
    expect(MAX_PER_MINUTE).toBeLessThan(100)
  })

  it('lets the burst through at once, then refuses', () => {
    const bucket = new TokenBucket()
    let taken = 0
    while (bucket.take()) taken += 1
    expect(taken).toBe(BURST)
    expect(bucket.msUntilToken()).toBeGreaterThan(0)
  })

  it('refills at the rate and never above its capacity', () => {
    const bucket = new TokenBucket(5, 60)
    for (let i = 0; i < 5; i += 1) expect(bucket.take()).toBe(true)
    expect(bucket.take()).toBe(false)
    vi.advanceTimersByTime(1000) // 60 a minute is one a second
    expect(bucket.take()).toBe(true)
    expect(bucket.take()).toBe(false)
    vi.advanceTimersByTime(60 * 60_000) // an hour later, still only 5
    let taken = 0
    while (bucket.take()) taken += 1
    expect(taken).toBe(5)
  })

  it('tells how long to wait for the next token', () => {
    const bucket = new TokenBucket(1, 60)
    expect(bucket.msUntilToken()).toBe(0)
    bucket.take()
    expect(bucket.msUntilToken()).toBe(1000)
    vi.advanceTimersByTime(400)
    expect(bucket.msUntilToken()).toBe(600)
  })
})

describe('TileGate (fake timers)', () => {
  beforeEach(() => vi.useFakeTimers({ now: 0 }))
  afterEach(() => vi.useRealTimers())

  /** A job that records when it started, and whether it was skipped. */
  function job(alive = true): TileJob & { started: number[]; skipped: number } {
    const made = { started: [] as number[], skipped: 0, alive: () => alive, start: () => void made.started.push(Date.now()), skip: () => void made.skipped++ }
    return made
  }

  it('never lets more than 100 requests through in any rolling minute, however many tiles are asked for', () => {
    const gate = new TileGate()
    const started: number[] = []
    for (let i = 0; i < 1000; i += 1) {
      gate.request(`https://t/${i}.png`, { alive: () => true, start: () => void started.push(Date.now()), skip: () => undefined })
    }
    vi.advanceTimersByTime(10 * 60_000)
    expect(started.length).toBeGreaterThan(500) // it keeps going: this is a throttle, not a stop
    let worst = 0
    for (let from = 0, to = 0; to < started.length; to += 1) {
      while (started[to] - started[from] >= 60_000) from += 1
      worst = Math.max(worst, to - from + 1)
    }
    expect(worst).toBeLessThanOrEqual(MAX_PER_MINUTE)
    expect(worst).toBeLessThan(100)
    expect(gate.requests).toBe(started.length)
  })

  it('starts the first tiles at once and the rest in the order they were asked for', () => {
    const gate = new TileGate()
    const jobs = Array.from({ length: BURST + 3 }, () => job())
    jobs.forEach((j, i) => gate.request(`u${i}`, j))
    expect(jobs.filter((j) => j.started.length).length).toBe(BURST)
    vi.advanceTimersByTime(60_000)
    const order = jobs.map((j) => j.started[0])
    expect(order.every((at) => at !== undefined)).toBe(true)
    expect([...order].sort((a, b) => a - b)).toEqual(order)
  })

  it('fetches a URL once: a tile that loaded starts again at once, for no token', () => {
    const gate = new TileGate()
    const first = job()
    gate.request('u', first)
    gate.loaded('u')
    const before = gate.requests
    const again = job()
    gate.request('u', again)
    expect(again.started).toHaveLength(1)
    expect(gate.requests).toBe(before)
  })

  it('does not ask again for a URL that failed until the next refresh forgets it, and leaves a gap meanwhile', () => {
    const gate = new TileGate()
    gate.request('u', job())
    gate.failed('u')
    const again = job()
    gate.request('u', again)
    expect(again.skipped).toBe(1)
    expect(again.started).toEqual([])
    expect(gate.forgetFailed()).toBe(true)
    expect(gate.forgetFailed()).toBe(false)
    const retry = job()
    gate.request('u', retry)
    expect(retry.started).toHaveLength(1)
  })

  it('drops a job whose tile has left the map without a request', () => {
    const gate = new TileGate()
    for (let i = 0; i < BURST; i += 1) gate.request(`u${i}`, job())
    const waiting = { ...job(), alive: () => false }
    const live = job()
    gate.request('late', waiting)
    gate.request('later', live)
    const before = gate.requests
    vi.advanceTimersByTime(60_000)
    expect(waiting.started).toEqual([])
    expect(gate.requests).toBe(before + 1)
    expect(live.started).toHaveLength(1)
  })

  it('clear() stops the waiting requests and the timer, and keeps what it learned', () => {
    const gate = new TileGate()
    for (let i = 0; i < BURST; i += 1) gate.request(`u${i}`, job())
    const waiting = job()
    gate.request('w', waiting)
    expect(vi.getTimerCount()).toBe(1)
    gate.clear()
    expect(vi.getTimerCount()).toBe(0)
    vi.advanceTimersByTime(10 * 60_000)
    expect(waiting.started).toEqual([])
    gate.loaded('u0')
    const known = job()
    gate.request('u0', known)
    expect(known.started).toHaveLength(1)
  })

  it('has at most one timer however many requests wait', () => {
    const gate = new TileGate()
    for (let i = 0; i < 300; i += 1) gate.request(`u${i}`, job())
    expect(vi.getTimerCount()).toBe(1)
  })
})

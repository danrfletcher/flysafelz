import { describe, expect, it } from 'vitest'

import { createSessions, pinKey, readResult } from '@/chat/turns'
import { BLUE_MILL_RUN, DECLINE_RUN, FALLBACK_RUN } from '@/test/chatServer'

const finalOf = (run: { events: { data: unknown }[] }) => run.events.at(-1)!.data

describe('readResult', () => {
  it('reads the answer, guard, assessment and overlays of the Blue Mill run exactly as sent', () => {
    const values = finalOf(BLUE_MILL_RUN) as { answer: string; assessment: object; overlays: object[] }
    const result = readResult(values)!
    expect(result.answer).toBe(values.answer)
    expect(result.guard).toBe('passed')
    expect(result.assessment).toEqual(values.assessment)
    expect(result.overlays).toEqual(values.overlays)
  })

  it('reads nothing before the run has an answer', () => {
    expect(readResult({ messages: [], answer: null, guard: null, overlays: [], assessment: null })).toBeNull()
    expect(readResult(undefined)).toBeNull()
    expect(readResult('x')).toBeNull()
  })

  it('keeps a missing or unknown guard as null or the string it is, and bad assessment and overlays as empty', () => {
    expect(readResult({ answer: 'a' })).toEqual({ answer: 'a', guard: null, assessment: null, overlays: [] })
    expect(readResult({ answer: 'a', guard: 7, assessment: 'x', overlays: 'y' })).toEqual({ answer: 'a', guard: null, assessment: null, overlays: [] })
    expect(readResult({ answer: 'a', guard: 'surprise', overlays: [1, { id: 2 }, null] })).toMatchObject({ guard: 'surprise', overlays: [{ id: 2 }] })
  })

  it('reads a fallback and a decline', () => {
    expect(readResult(finalOf(FALLBACK_RUN))).toMatchObject({ guard: 'fallback' })
    expect(readResult(finalOf(DECLINE_RUN))).toMatchObject({ answer: 'FlySafeLZ only covers the UK.', assessment: null, overlays: [] })
  })
})

describe('turn store', () => {
  const key = pinKey({ lat: 53.8, lon: -0.09 })
  const blue = readResult(finalOf(BLUE_MILL_RUN))!
  const decline = readResult(finalOf(DECLINE_RUN))!

  it('keeps each turn assessment and overlays, and a later turn with assessment null does not overwrite an earlier one', () => {
    const sessions = createSessions()
    sessions.ask(key, 't1', 'Can I fly here this afternoon?')
    sessions.complete(key, 't1', blue)
    sessions.ask(key, 't2', 'What is the weather like here?')
    sessions.complete(key, 't2', decline)

    const [first, second] = sessions.get(key).turns
    expect(first.assessment).toEqual(blue.assessment)
    expect(first.overlays).toEqual(blue.overlays)
    expect(first.overlays.length).toBeGreaterThan(0)
    expect(second.assessment).toBeNull()
    expect(second.overlays).toEqual([])
    expect(first.answer).toBe(blue.answer)
    expect(second.answer).toBe(decline.answer)
  })

  it('changes only the turn it is told to', () => {
    const sessions = createSessions()
    sessions.ask(key, 't1', 'one')
    sessions.ask(key, 't2', 'two')
    sessions.fail(key, 't1')
    expect(sessions.get(key).turns.map((turn) => turn.status)).toEqual(['error', 'running'])
    sessions.resume(key, 't1')
    sessions.complete(key, 't2', decline)
    expect(sessions.get(key).turns.map((turn) => turn.status)).toEqual(['running', 'done'])
  })

  it('keeps one session per pin, and the same coordinates are the same pin', () => {
    const sessions = createSessions()
    sessions.setThreadId(key, 'thread-1')
    sessions.ask(key, 't1', 'one')
    const other = pinKey({ lat: 51.5, lon: -0.12 })
    expect(sessions.get(other)).toEqual({ threadId: null, turns: [] })
    expect(sessions.get(pinKey({ lat: 53.8, lon: -0.09 })).threadId).toBe('thread-1')
  })

  it('returns the same object until a pin changes, and tells listeners when it does', () => {
    const sessions = createSessions()
    let calls = 0
    const stop = sessions.subscribe(() => calls++)
    const before = sessions.get(key)
    expect(sessions.get(key)).toBe(before)
    sessions.ask(key, 't1', 'one')
    expect(calls).toBe(1)
    expect(sessions.get(key)).not.toBe(before)
    stop()
    sessions.ask(key, 't2', 'two')
    expect(calls).toBe(1)
  })

  it('holds nothing in a fresh store: threads are lost on reload (Q8)', () => {
    const first = createSessions()
    first.setThreadId(key, 'thread-1')
    first.ask(key, 't1', 'one')
    const afterReload = createSessions()
    expect(afterReload.get(key)).toEqual({ threadId: null, turns: [] })
  })
})

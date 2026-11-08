import { act, renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { createSessions, pinKey, type Sessions } from '@/chat/turns'
import { storedResult, useLzThread } from '@/chat/useLzThread'
import { BLUE_MILL_RUN, DECLINE_RUN, FALLBACK_RUN, NETWORK_ERROR_RUN, chatServer, scriptOf, type ChatServer } from '@/test/chatServer'
import { server } from '@/test/setup'

const BLUE_MILL = { lat: 53.802797, lon: -0.086644 }
const OTHER = { lat: 51.5074, lon: -0.1278 }
const QUESTION = 'Can I fly here this afternoon?'

function setup(...script: Parameters<typeof chatServer>) {
  const fake = chatServer(...script)
  server.use(...fake.handlers)
  return { fake, sessions: createSessions() }
}

const mount = (point: { lat: number; lon: number }, sessions: Sessions) => renderHook(() => useLzThread(point, sessions))
const settled = (hook: ReturnType<typeof mount>, index = 0) =>
  waitFor(() => expect(hook.result.current.turns[index]?.status).toMatch(/done|error/), { timeout: 5000 })

describe('useLzThread', () => {
  it('sends lat, lon and the browser IANA tz with the question, and stores the answer, guard, assessment and overlays that came back', async () => {
    const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN))
    const hook = mount(BLUE_MILL, sessions)
    act(() => hook.result.current.send(QUESTION))
    await settled(hook)

    const input = fake.runs[0].body.input as { messages: { type: string; content: string }[]; lat: number; lon: number; tz: string }
    expect(input.lat).toBe(BLUE_MILL.lat)
    expect(input.lon).toBe(BLUE_MILL.lon)
    expect(input.tz).toBe(Intl.DateTimeFormat().resolvedOptions().timeZone)
    expect(input.messages).toEqual([{ type: 'human', content: QUESTION, id: hook.result.current.turns[0].id }])
    expect(fake.runs[0].body.stream_mode).toEqual(['custom', 'values'])

    const turn = hook.result.current.turns[0]
    const final = BLUE_MILL_RUN.events.at(-1)!.data as { answer: string; assessment: object; overlays: object[] }
    expect(turn).toMatchObject({ status: 'done', question: QUESTION, answer: final.answer, guard: 'passed', assessment: final.assessment, overlays: final.overlays })
    expect(turn.overlays).toHaveLength(3)
  })

  it('sends Unicode and a 2,000-character question unchanged', async () => {
    const { fake, sessions } = setup(scriptOf(DECLINE_RUN))
    const hook = mount(BLUE_MILL, sessions)
    const long = `Ça va? ✈️ 風 ${'x'.repeat(1990)}`.slice(0, 2000)
    act(() => hook.result.current.send(long))
    await settled(hook)
    expect((fake.runs[0].body.input as { messages: { content: string }[] }).messages[0].content).toBe(long)
    expect(hook.result.current.turns[0].question).toBe(long)
  })

  it('makes one thread per pin: the first run makes it, later turns reuse it', async () => {
    const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN))
    const hook = mount(BLUE_MILL, sessions)
    act(() => hook.result.current.send(QUESTION))
    await settled(hook)
    act(() => hook.result.current.send('And tomorrow?'))
    await settled(hook, 1)

    expect(fake.threads).toEqual(['thread-1'])
    expect(fake.runs.map((run) => run.threadId)).toEqual(['thread-1', 'thread-1'])
    expect(sessions.get(pinKey(BLUE_MILL)).threadId).toBe('thread-1')
  })

  it('gives a new pin a new thread, and the same coordinates again reuse the first pin thread and its turns', async () => {
    const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN))
    const first = mount(BLUE_MILL, sessions)
    act(() => first.result.current.send(QUESTION))
    await settled(first)
    first.unmount()

    const second = mount(OTHER, sessions)
    expect(second.result.current.turns).toEqual([])
    act(() => second.result.current.send(QUESTION))
    await settled(second)
    second.unmount()
    expect(fake.threads).toEqual(['thread-1', 'thread-2'])

    const back = mount({ ...BLUE_MILL }, sessions)
    expect(back.result.current.turns.map((turn) => turn.question)).toEqual([QUESTION])
    expect(back.result.current.turns[0].status).toBe('done')
    act(() => back.result.current.send('And tomorrow?'))
    await settled(back, 1)
    expect(fake.threads).toEqual(['thread-1', 'thread-2'])
    expect(fake.runs.at(-1)!.threadId).toBe('thread-1')
  })

  it('starts only one run when the question is sent twice in the same tick', async () => {
    const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN))
    const hook = mount(BLUE_MILL, sessions)
    act(() => {
      hook.result.current.send(QUESTION)
      hook.result.current.send(QUESTION)
    })
    await settled(hook)
    expect(fake.runs).toHaveLength(1)
    expect(hook.result.current.turns).toHaveLength(1)
  })

  it('ignores empty and whitespace-only questions', () => {
    const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN))
    const hook = mount(BLUE_MILL, sessions)
    act(() => {
      hook.result.current.send('')
      hook.result.current.send('  \n\t ')
    })
    expect(hook.result.current.turns).toEqual([])
    expect(fake.runs).toEqual([])
    expect(fake.threads).toEqual([])
  })

  it('shows each progress line in turn, then clears it when the run ends', async () => {
    const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 3 }))
    const hook = mount(BLUE_MILL, sessions)
    act(() => hook.result.current.send(QUESTION))
    expect(hook.result.current.running).toBe(true)
    expect(hook.result.current.line).toBeNull()

    await waitFor(() => expect(hook.result.current.line).toBe('Checking airspace…'))
    expect(hook.result.current.turns[0].answer).toBeNull() // nothing is shown before the run ends
    act(() => fake.runs[0].release())
    await settled(hook)
    expect(hook.result.current.line).toBeNull()
    expect(hook.result.current.running).toBe(false)
  })

  it('stores a guard = fallback answer with its guard and assessment', async () => {
    const { sessions } = setup(scriptOf(FALLBACK_RUN))
    const hook = mount(BLUE_MILL, sessions)
    act(() => hook.result.current.send(QUESTION))
    await settled(hook)
    const turn = hook.result.current.turns[0]
    expect(turn.guard).toBe('fallback')
    expect(turn.assessment).not.toBeNull()
    expect(turn.status).toBe('done')
  })

  it('keeps the UK-only decline as a normal done turn', async () => {
    const { sessions } = setup(scriptOf(DECLINE_RUN))
    const hook = mount({ lat: 49.2, lon: -2.13 }, sessions)
    act(() => hook.result.current.send('What is the weather like here?'))
    await settled(hook)
    expect(hook.result.current.turns[0]).toMatchObject({ status: 'done', answer: 'FlySafeLZ only covers the UK.', guard: 'passed', assessment: null, overlays: [] })
  })

  describe('a pin change while a run is going (E8)', () => {
    it('cancels the old run, never writes its result, and keeps the old pin thread for when the pilot comes back', async () => {
      const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 3 }))
      const old = mount(BLUE_MILL, sessions)
      act(() => old.result.current.send(QUESTION))
      await waitFor(() => expect(old.result.current.line).toBe('Checking airspace…'))
      old.unmount() // the sheet is closed or another pin is picked: Conversation is unmounted

      await waitFor(() => expect(fake.runs[0].aborted).toBe(true))
      fake.runs[0].release()
      await new Promise((resolve) => setTimeout(resolve, 50))

      // Nothing from the old run reached the old pin as an answer, and the new pin has nothing at all.
      expect(sessions.get(pinKey(BLUE_MILL)).turns[0]).toMatchObject({ status: 'error', answer: null, assessment: null, overlays: [] })
      expect(sessions.get(pinKey(OTHER)).turns).toEqual([])
      expect(sessions.get(pinKey(BLUE_MILL)).threadId).toBe('thread-1')

      const fresh = mount(OTHER, sessions)
      expect(fresh.result.current.turns).toEqual([])
      expect(fresh.result.current.line).toBeNull()
      expect(fresh.result.current.running).toBe(false)

      const back = mount(BLUE_MILL, sessions)
      expect(back.result.current.turns).toHaveLength(1)
      expect(back.result.current.turns[0].question).toBe(QUESTION)
    })
  })

  describe('errors and retry (E7)', () => {
    async function dropped(fake: ChatServer, sessions: Sessions) {
      const hook = mount(BLUE_MILL, sessions)
      act(() => hook.result.current.send(QUESTION))
      await settled(hook)
      expect(fake.runs).toHaveLength(1)
      return hook
    }

    it('turns a connection that drops mid-run into an error turn with no partial answer', async () => {
      const { fake, sessions } = setup(scriptOf(NETWORK_ERROR_RUN))
      const hook = await dropped(fake, sessions)
      expect(hook.result.current.turns[0]).toMatchObject({ status: 'error', answer: null, guard: null, assessment: null, overlays: [] })
      expect(hook.result.current.line).toBeNull()
      expect(hook.result.current.running).toBe(false)
    })

    it('turns a server that is down (503) into an error turn', async () => {
      const { sessions } = setup({ events: [], status: 503 })
      const hook = mount(BLUE_MILL, sessions)
      act(() => hook.result.current.send(QUESTION))
      await settled(hook)
      expect(hook.result.current.turns[0]).toMatchObject({ status: 'error', answer: null })
    })

    it('re-sends the question on the same thread when the server has no answer for it, and never makes a second thread', async () => {
      const { fake, sessions } = setup(scriptOf(NETWORK_ERROR_RUN), scriptOf(BLUE_MILL_RUN))
      const hook = await dropped(fake, sessions)
      const id = hook.result.current.turns[0].id

      act(() => hook.result.current.retry())
      expect(hook.result.current.turns[0].status).toBe('running')
      await waitFor(() => expect(hook.result.current.turns[0].status).toBe('done'), { timeout: 5000 })

      expect(fake.threads).toEqual(['thread-1'])
      expect(fake.runs.map((run) => run.threadId)).toEqual(['thread-1', 'thread-1'])
      expect(hook.result.current.turns).toHaveLength(1) // the retry is the same turn, not a second one
      const input = fake.runs[1].body.input as { messages: { id: string; content: string }[] }
      expect(input.messages).toEqual([{ type: 'human', content: QUESTION, id }]) // the same message id, so the thread replaces it
      expect(hook.result.current.turns[0].answer).toContain("Pilot's call")
    })

    it('shows the stored answer, without a new run, when the server already holds one for this turn', async () => {
      const { fake, sessions } = setup(scriptOf(NETWORK_ERROR_RUN))
      const hook = await dropped(fake, sessions)
      const id = hook.result.current.turns[0].id
      const final = BLUE_MILL_RUN.events.at(-1)!.data as Record<string, unknown>
      fake.stored.set('thread-1', { ...final, messages: [{ type: 'human', id, content: QUESTION }, { type: 'ai', id: 'a1', content: 'x' }] })

      act(() => hook.result.current.retry())
      await waitFor(() => expect(hook.result.current.turns[0].status).toBe('done'), { timeout: 5000 })

      expect(fake.runs).toHaveLength(1)
      expect(fake.threads).toEqual(['thread-1'])
      expect(hook.result.current.turns[0]).toMatchObject({ answer: final.answer, guard: 'passed' })
      expect(hook.result.current.turns[0].assessment).toEqual(final.assessment)
    })

    it('does not take an earlier turn answer for the failed turn', async () => {
      const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN), scriptOf(NETWORK_ERROR_RUN), scriptOf(DECLINE_RUN))
      const hook = mount(BLUE_MILL, sessions)
      act(() => hook.result.current.send(QUESTION))
      await settled(hook)
      act(() => hook.result.current.send('And tomorrow?'))
      await settled(hook, 1)
      expect(hook.result.current.turns[1].status).toBe('error')
      // The server still holds the first turn's answer as the latest `answer`, but the latest question is the first one's.
      const final = BLUE_MILL_RUN.events.at(-1)!.data as Record<string, unknown>
      fake.stored.set('thread-1', { ...final, messages: [{ type: 'human', id: hook.result.current.turns[0].id, content: QUESTION }] })

      act(() => hook.result.current.retry())
      await waitFor(() => expect(hook.result.current.turns[1].status).toBe('done'), { timeout: 5000 })
      expect(fake.runs).toHaveLength(3) // the failed turn was sent again; the stored answer was not its answer
      expect(hook.result.current.turns[1].answer).toBe('FlySafeLZ only covers the UK.')
      expect(hook.result.current.turns[0].answer).toContain("Pilot's call")
    })

    it('has nothing to retry when the latest turn did not fail', async () => {
      const { fake, sessions } = setup(scriptOf(BLUE_MILL_RUN))
      const hook = mount(BLUE_MILL, sessions)
      act(() => hook.result.current.send(QUESTION))
      await settled(hook)
      act(() => hook.result.current.retry())
      expect(fake.runs).toHaveLength(1)
    })
  })
})

describe('storedResult', () => {
  const answered = { messages: [{ type: 'human', id: 'q2' }], answer: 'Yes.', guard: 'passed', overlays: [], assessment: null }

  it('reads the answer when the turn is the thread latest question', () => {
    expect(storedResult(answered, 'q2')).toMatchObject({ answer: 'Yes.', guard: 'passed' })
  })

  it('reads nothing for a turn that is not the latest question, or a state with no answer', () => {
    expect(storedResult(answered, 'q1')).toBeNull()
    expect(storedResult({ ...answered, answer: null }, 'q2')).toBeNull()
    expect(storedResult({}, 'q2')).toBeNull()
    expect(storedResult(null, 'q2')).toBeNull()
  })
})

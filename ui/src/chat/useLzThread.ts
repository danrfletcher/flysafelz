import { useStream } from '@langchain/langgraph-sdk/react'
import { useEffect, useReducer, useRef, useState, useSyncExternalStore } from 'react'

import { nextLine } from '@/chat/progress'
import { pinKey, readFinalResult, type Sessions, type Turn, type TurnResult } from '@/chat/turns'
import { apiUrl } from '@/config'
import type { LatLon } from '@/pick/whatsHere'

/** The graph's name in langgraph.json. */
export const ASSISTANT_ID = 'agent'

/** The browser's IANA time zone, sent with every run so the server speaks the pilot's local times. */
export function browserTz(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone
}

/** The slice of the thread's state this PR reads: the messages, and the output fields code writes at the end of a turn. */
interface ChatState {
  messages: { id?: string; type?: string }[]
  answer?: string | null
  guard?: string | null
  overlays?: unknown[]
  assessment?: unknown
}

/** One run in flight (or settling) on this pin's thread. */
interface Run {
  turnId: string
  /** The SDK reported an error for this run. */
  failed: boolean
  /** `submit` has returned: the stream is over, whatever way it ended. */
  settled: boolean
}

/**
 * A retry's first question: does the server already hold the answer to this turn? It does when the turn's message is the thread's latest
 * question, the state carries an `answer`, and the thread ends with the reply to it (readFinalResult). `answer` alone is not enough: code clears
 * it when the turn's input check runs, so before that it is still the previous turn's.
 */
export function storedResult(values: unknown, turnId: string): TurnResult | null {
  const messages = (values as ChatState | null)?.messages
  if (!Array.isArray(messages)) return null
  const lastQuestion = messages.findLast((message) => message?.type === 'human')
  return lastQuestion?.id === turnId ? readFinalResult(values) : null
}

export interface LzThread {
  turns: readonly Turn[]
  /** The live progress line while a run is going, else null. */
  line: string | null
  /** True while the latest turn has no answer yet. */
  running: boolean
  /** Sends a question as a new turn. Does nothing while a run is going or when the text is empty or only whitespace. The text is sent as given. */
  send: (question: string) => void
  /** Retries the latest turn when it failed or was stopped. */
  retry: () => void
}

/**
 * The conversation of one pin. Mount it keyed by the pin: one `useStream` per pin, so a different pin is a different instance. Unmounting
 * cancels a run still going (the SDK aborts the request, and the server cancels the run), marks the turn `stopped` (not `error`: the server did nothing
 * wrong), and nothing from that run is written afterwards.
 *
 * The thread id is kept in `sessions` for the pin, so coming back to the same coordinates reuses the thread. The first `submit` makes the
 * thread; a retry reuses it and never makes another. The run ends on the final `values`, the state that carries `answer` and ends with the reply to this turn's question (readFinalResult): its fields become the
 * turn's answer, guard, assessment and overlays. Custom events only drive the status line; nothing streams into the answer.
 */
export function useLzThread(point: LatLon, sessions: Sessions): LzThread {
  const key = pinKey(point)
  const session = useSyncExternalStore(sessions.subscribe, () => sessions.get(key))
  const [line, dispatchLine] = useReducer(nextLine, null)
  const [, setSettled] = useState(0)
  const run = useRef<Run | null>(null)
  const busy = useRef(false) // set at once when a question is taken, so two taps in the same tick start one run
  const alive = useRef(true)
  const stopRef = useRef<() => Promise<void>>(async () => {})

  const stream = useStream<ChatState>({
    apiUrl: apiUrl(),
    assistantId: ASSISTANT_ID,
    threadId: session.threadId,
    onThreadId: (threadId) => sessions.setThreadId(key, threadId),
    // The SDK would send a run again after a failed request; a second POST would be a second run on the thread.
    callerOptions: { maxRetries: 0 },
    onCustomEvent: (event) => {
      if (alive.current && run.current) dispatchLine({ type: 'event', event })
    },
    onError: () => {
      if (run.current) run.current.failed = true
    },
  })

  useEffect(() => {
    stopRef.current = stream.stop
  })

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
      const current = run.current
      run.current = null
      busy.current = false
      if (current) {
        sessions.stop(key, current.turnId) // the pilot left (another pin, or the sheet closed): stopped, which is not a server failure
        void stopRef.current()
      }
    }
  }, [sessions, key])

  const finish = (current: Run, result: TurnResult | null) => {
    if (run.current !== current) return
    run.current = null
    busy.current = false
    dispatchLine({ type: 'end' })
    if (result) sessions.complete(key, current.turnId, result)
    else sessions.fail(key, current.turnId)
  }

  // Runs after the render that follows `submit` returning, so `stream.values` is the final state of the run.
  useEffect(() => {
    const current = run.current
    if (!current?.settled) return
    finish(current, current.failed ? null : readFinalResult(stream.values))
  })

  const begin = (turnId: string): Run => {
    busy.current = true
    dispatchLine({ type: 'end' })
    const current: Run = { turnId, failed: false, settled: false }
    run.current = current
    return current
  }

  const submitRun = async (current: Run, question: string) => {
    try {
      await stream.submit(
        { messages: [{ type: 'human', content: question, id: current.turnId }], lat: point.lat, lon: point.lon, tz: browserTz() } as never,
        // `onDisconnect: 'cancel'` is the SDK's default for a stream that is not resumable; it is said here so a change of that default cannot
        // leave an old run going on the server after the pilot changed pin or closed the sheet (the server cancels a run whose client went away).
        { streamMode: ['custom', 'values'], onDisconnect: 'cancel' },
      )
    } catch {
      current.failed = true
    }
    if (!alive.current || run.current !== current) return
    current.settled = true
    setSettled((count) => count + 1)
  }

  const send = (question: string) => {
    if (busy.current || question.trim() === '') return
    const turnId = crypto.randomUUID()
    const current = begin(turnId)
    sessions.ask(key, turnId, question)
    void submitRun(current, question)
  }

  const retry = () => {
    const last = session.turns.at(-1)
    if (!last || (last.status !== 'error' && last.status !== 'stopped') || busy.current) return
    const current = begin(last.id)
    sessions.resume(key, last.id)
    void (async () => {
      let stored: TurnResult | null = null
      if (session.threadId) {
        try {
          stored = storedResult((await stream.client.threads.getState(session.threadId)).values, last.id)
        } catch {
          // The server cannot be asked: send the question again.
        }
      }
      if (run.current !== current) return // the pin changed while asking
      if (stored) finish(current, stored)
      else await submitRun(current, last.question)
    })()
  }

  const last = session.turns.at(-1)
  return { turns: session.turns, line, running: last?.status === 'running', send, retry }
}

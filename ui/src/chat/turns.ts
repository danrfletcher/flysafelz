import type { LatLon } from '@/pick/whatsHere'

/** A turn's `assessment` or one overlay feature, exactly as the server sent it. This PR only keeps them; PR-6.F1 reads them. */
export type Payload = Record<string, unknown>

export type TurnStatus = 'running' | 'done' | 'error'

/** One question and what came back for it. `assessment` and `overlays` stay with the turn they arrived with, and no later turn touches them. */
export interface Turn {
  /** The id of the question's message. A retry sends the same id, so the thread's message reducer replaces the message rather than doubling it. */
  id: string
  question: string
  status: TurnStatus
  answer: string | null
  /** The server's `guard` value, or null when it sent none (or something that is not a string). Only the exact string "fallback" means anything to the UI. */
  guard: string | null
  assessment: Payload | null
  overlays: Payload[]
}

/** What one finished run gave back: the final `values` fields, read as they are. */
export interface TurnResult {
  answer: string
  guard: string | null
  assessment: Payload | null
  overlays: Payload[]
}

/** One LZ's conversation: its LangGraph thread id (null until its first run makes the thread) and its turns, oldest first. */
export interface Session {
  threadId: string | null
  turns: readonly Turn[]
}

const EMPTY: Session = { threadId: null, turns: [] }

/** The key of a pin. The same coordinates are the same LZ for the session. */
export function pinKey(point: LatLon): string {
  return `${point.lat},${point.lon}`
}

const isRecord = (value: unknown): value is Payload => typeof value === 'object' && value !== null && !Array.isArray(value)

/**
 * Reads the output fields of a thread's `values`. Returns null when there is no `answer` string yet. A missing or unknown `guard` is
 * kept as null (a normal answer); `assessment` that is not an object becomes null; `overlays` that is not a list becomes [].
 */
export function readResult(values: unknown): TurnResult | null {
  if (!isRecord(values) || typeof values.answer !== 'string') return null
  return {
    answer: values.answer,
    guard: typeof values.guard === 'string' ? values.guard : null,
    assessment: isRecord(values.assessment) ? values.assessment : null,
    overlays: Array.isArray(values.overlays) ? values.overlays.filter(isRecord) : [],
  }
}

/**
 * The conversations of this page load, one per pin, kept in memory only: a reload starts empty (Q8). It works with
 * `useSyncExternalStore`: `get(key)` returns the same object until that pin changes.
 */
export interface Sessions {
  get(key: string): Session
  setThreadId(key: string, threadId: string): void
  /** Adds a question as a running turn. */
  ask(key: string, id: string, question: string): void
  /** Puts a running or failed turn back to running (a retry). */
  resume(key: string, id: string): void
  complete(key: string, id: string, result: TurnResult): void
  fail(key: string, id: string): void
  subscribe(listener: () => void): () => void
}

export function createSessions(): Sessions {
  const sessions = new Map<string, Session>()
  const listeners = new Set<() => void>()

  const update = (key: string, change: (session: Session) => Session) => {
    sessions.set(key, change(sessions.get(key) ?? EMPTY))
    listeners.forEach((listener) => listener())
  }
  const changeTurn = (key: string, id: string, change: (turn: Turn) => Turn) =>
    update(key, (session) => ({ ...session, turns: session.turns.map((turn) => (turn.id === id ? change(turn) : turn)) }))

  return {
    get: (key) => sessions.get(key) ?? EMPTY,
    setThreadId: (key, threadId) => update(key, (session) => ({ ...session, threadId })),
    ask: (key, id, question) =>
      update(key, (session) => ({
        ...session,
        turns: [...session.turns, { id, question, status: 'running', answer: null, guard: null, assessment: null, overlays: [] }],
      })),
    resume: (key, id) => changeTurn(key, id, (turn) => ({ ...turn, status: 'running' })),
    complete: (key, id, result) => changeTurn(key, id, (turn) => ({ ...turn, ...result, status: 'done' })),
    fail: (key, id) => changeTurn(key, id, (turn) => ({ ...turn, status: 'error' })),
    subscribe: (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}

import { http, HttpResponse } from 'msw'

import { apiUrl } from '@/config'
import blueMill from '@/test/fixtures/chat/blue-mill-assess.json'
import followUp from '@/test/fixtures/chat/blue-mill-follow-up.json'
import highlights from '@/test/fixtures/chat/blue-mill-highlights.json'
import fallback from '@/test/fixtures/chat/guard-fallback.json'
import networkError from '@/test/fixtures/chat/network-error.json'
import outlook from '@/test/fixtures/chat/outlook-too-far.json'
import progressSequence from '@/test/fixtures/chat/progress-sequence.json'
import decline from '@/test/fixtures/chat/uk-only-decline.json'

/** One recorded stream event: the SSE `event:` name and its JSON `data:`. */
export interface RecordedEvent {
  event: string
  data: unknown
}

/** A recorded run: the events the server sent, and the question and point it was recorded for. */
export interface Recording {
  lat?: number
  lon?: number
  question?: string
  events: RecordedEvent[]
  then?: string
}

export const BLUE_MILL_RUN = blueMill as Recording
/** The Blue Mill answer with an approximate NOTAM, a NOTAM with an action the UI does not know, and ids in code, in a link and not in the overlays. */
export const HIGHLIGHTS_RUN = highlights as Recording
/** The follow-up on the same thread: `assessment` null and no overlays, its answer naming H7171/26. */
export const FOLLOW_UP_RUN = followUp as Recording
/** An answer whose `assessment` has only OUTLOOK and TOO_FAR segments, and whose overlays are the 2 NM circle alone. */
export const OUTLOOK_RUN = outlook as Recording
export const DECLINE_RUN = decline as Recording
export const FALLBACK_RUN = fallback as Recording
export const NETWORK_ERROR_RUN = networkError as Recording
/** Every custom event of one run, in the order the server sent them, and the status line each one shows. */
export const PROGRESS_SEQUENCE = progressSequence as { events: unknown[]; lines: string[] }

/** What the fake server does with one run request. */
export interface RunScript {
  /** The events to send, in order. Events from `gate` on are held back until the gate opens. */
  events: RecordedEvent[]
  /** Hold back the events from this index until `release()` is called; the stream stays open meanwhile. */
  holdAt?: number
  /** After the events, break the connection (a network drop) rather than end the stream cleanly. */
  drop?: boolean
  /** Answer the request itself with this status instead of a stream (a server that is down). */
  status?: number
}

export interface RunCall {
  threadId: string
  runId: string
  /** The JSON body of the request: `input`, `stream_mode`, `assistant_id`, ... */
  body: { input?: Record<string, unknown>; [key: string]: unknown }
  release: () => void
  aborted: boolean
}

export interface ChatServer {
  /** The thread ids made, in order. */
  threads: string[]
  /** Every run request, in order. */
  runs: RunCall[]
  /** What `GET /threads/:id/state` says for a thread (the retry check). Absent: the thread has no stored answer. */
  stored: Map<string, Record<string, unknown>>
  /** The scripts the next run requests will use, first to last. The last one repeats. */
  script: RunScript[]
  handlers: ReturnType<typeof http.post>[]
}

const encoder = new TextEncoder()
const sse = (event: string, data: unknown) => encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)

/** The script for a recorded run: all of its events, and the connection broken after them when the recording says "network-error". */
export function scriptOf(recording: Recording, options: Partial<RunScript> = {}): RunScript {
  return { events: recording.events, drop: recording.then === 'network-error', ...options }
}

/**
 * A fake LangGraph server on the API URL, for the tests: it makes threads, plays a run's events as a real SSE stream, records every
 * request, and answers the cancel and state calls. The caller registers `handlers` with MSW's `server.use`.
 */
export function chatServer(...script: RunScript[]): ChatServer {
  const base = () => apiUrl()
  const state: ChatServer = { threads: [], runs: [], stored: new Map(), script, handlers: [] }

  state.handlers = [
    http.post(`${base()}/threads`, () => {
      const threadId = `thread-${state.threads.length + 1}`
      state.threads.push(threadId)
      return HttpResponse.json({ thread_id: threadId, created_at: '2026-11-08T12:00:00Z', updated_at: '2026-11-08T12:00:00Z', metadata: {}, status: 'idle', values: {} })
    }),
    http.get(`${base()}/threads/:threadId/state`, ({ params }) => {
      const values = state.stored.get(String(params.threadId))
      return HttpResponse.json({ values: values ?? {}, next: [], tasks: [], checkpoint: { thread_id: String(params.threadId), checkpoint_id: 'c1', checkpoint_ns: '' }, metadata: {}, created_at: null, parent_checkpoint: null })
    }),
    http.post(`${base()}/threads/:threadId/history`, () => HttpResponse.json([])),
    http.get(`${base()}/threads/:threadId/history`, () => HttpResponse.json([])),
    // The browser does not call this one: a run is cancelled by the stream's `on_disconnect: 'cancel'` (the tests read that from each run's body). Answered
    // anyway, so a change in the SDK that did call it would not fail as an unmocked request.
    http.post(`${base()}/threads/:threadId/runs/:runId/cancel`, () => new HttpResponse(null, { status: 202 })),
    http.post(`${base()}/threads/:threadId/runs/stream`, async ({ request, params }) => {
      const body = (await request.json()) as RunCall['body']
      const runId = `run-${state.runs.length + 1}`
      let opened!: () => void
      const gate = new Promise<void>((resolve) => (opened = resolve))
      const call: RunCall = { threadId: String(params.threadId), runId, body, release: () => opened(), aborted: false }
      state.runs.push(call)
      const plan = state.script[Math.min(state.runs.length - 1, state.script.length - 1)]
      if (plan.status) return HttpResponse.json({ detail: 'down' }, { status: plan.status })

      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          request.signal.addEventListener('abort', () => {
            call.aborted = true
            opened()
          })
          controller.enqueue(sse('metadata', { run_id: runId, attempt: 1 }))
          // The recorded metadata event names the recorded run; the fake server sends its own, with this request's run id.
          const playing = plan.events.filter((item) => item.event !== 'metadata')
          for (const [index, item] of playing.entries()) {
            if (index === plan.holdAt) await gate
            if (call.aborted) return
            controller.enqueue(sse(item.event, item.data))
            await new Promise((resolve) => setTimeout(resolve, 0))
          }
          if (plan.holdAt !== undefined && plan.holdAt >= playing.length) await gate
          if (call.aborted) return
          if (plan.drop) controller.error(new TypeError('network error'))
          else controller.close()
        },
      })
      return new HttpResponse(stream, { headers: { 'content-type': 'text/event-stream', 'content-location': `/threads/${params.threadId}/runs/${runId}` } })
    }),
  ] as never

  return state
}

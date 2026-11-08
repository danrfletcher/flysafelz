import { useEffect, useRef, useState, type FormEvent } from 'react'

import { Answer } from '@/chat/Answer'
import { ERROR_TEXT, STARTER_CHIPS } from '@/chat/copy'
import { pinKey, type Sessions } from '@/chat/turns'
import { useLeaveWarning } from '@/chat/useLeaveWarning'
import { useLzThread } from '@/chat/useLzThread'
import { Drawer, DrawerContent } from '@/components/ui/drawer'
import { DialogTitle } from '@/components/ui/dialog'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { useIsDesktop } from '@/lib/useIsDesktop'
import type { LatLon } from '@/pick/whatsHere'

/** The status shown from the moment a question is sent until the first progress line arrives. */
const WORKING_TEXT = 'Working…'

/** The coordinates as the header shows them: four decimal places. */
export function formatPoint(point: LatLon): string {
  return `${point.lat.toFixed(4)}, ${point.lon.toFixed(4)}`
}

interface ConversationProps {
  point: LatLon
  sessions: Sessions
  onClose: () => void
}

/** The chat for one LZ: header, turns, the live status line (or the error with its retry button), starter chips and the question box. */
function Conversation({ point, sessions, onClose }: ConversationProps) {
  const { turns, line, running, send, retry } = useLzThread(point, sessions)
  useLeaveWarning(turns.length > 0)
  const [text, setText] = useState('')
  const end = useRef<HTMLDivElement>(null)
  const last = turns.at(-1)

  useEffect(() => {
    end.current?.scrollIntoView?.({ block: 'end' })
  }, [turns, line])

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (running || text.trim() === '') return
    send(text)
    setText('')
  }

  return (
    <>
      <header className="flex shrink-0 items-center justify-between gap-3 border-b px-4 py-3">
        <DialogTitle className="font-mono text-base">{formatPoint(point)}</DialogTitle>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md border-[length:var(--border-width)] px-3 py-1.5 text-sm font-medium hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          Change LZ
        </button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-3">
        {turns.length === 0 && (
          <ul className="flex flex-wrap gap-2" aria-label="Suggested questions">
            {STARTER_CHIPS.map((chip) => (
              <li key={chip}>
                <button
                  type="button"
                  disabled={running}
                  onClick={() => send(chip)}
                  className="rounded-full border-[length:var(--border-width)] px-3 py-1.5 text-left text-sm hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:opacity-50"
                >
                  {chip}
                </button>
              </li>
            ))}
          </ul>
        )}

        {turns.map((turn) => (
          <section key={turn.id} className="space-y-2" aria-label="Question and answer">
            <p className="ml-auto w-fit max-w-[85%] rounded-lg bg-muted px-3 py-2 text-sm whitespace-pre-wrap [overflow-wrap:anywhere]">{turn.question}</p>
            {turn.status === 'done' && turn.answer !== null && <Answer answer={turn.answer} guard={turn.guard} />}
          </section>
        ))}

        <div role="status" aria-live="polite">
          {last?.status === 'running' && (
            <p key={line ?? WORKING_TEXT} className="text-sm text-muted-foreground transition-opacity duration-300 motion-reduce:transition-none starting:opacity-0">
              {line ?? WORKING_TEXT}
            </p>
          )}
        </div>

        {last?.status === 'error' && (
          <div role="alert" className="flex flex-wrap items-center gap-3 text-sm">
            <p className="text-destructive">{ERROR_TEXT}</p>
            <button
              type="button"
              onClick={retry}
              className="rounded-md border-[length:var(--border-width)] px-3 py-1.5 font-medium hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            >
              Retry
            </button>
          </div>
        )}
        <div ref={end} />
      </div>

      <form onSubmit={submit} className="flex shrink-0 gap-2 border-t p-3">
        <input
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          disabled={running}
          aria-label="Ask about this LZ"
          placeholder="Ask about this LZ"
          autoComplete="off"
          className="min-w-0 flex-1 rounded-md border-[length:var(--border-width)] bg-background px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={running || text.trim() === ''}
          className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </>
  )
}

export interface ChatPanelProps {
  /** The LZ the chat is open for, or null when it is closed. */
  point: LatLon | null
  sessions: Sessions
  /** "Change LZ", Esc: close the chat. The pin and its card stay. */
  onClose: () => void
  /** Where the panel is placed: the map area, so the nav bar and footer stay clear. */
  container?: HTMLElement | null
}

/**
 * The chat: a right-hand Sheet from 768 px up, a bottom Drawer below. Both leave the map visible and clickable. The conversation inside is
 * mounted by the pin, so another pin starts a fresh one and closing the panel ends any run that is still going.
 */
export function ChatPanel({ point, sessions, onClose, container }: ChatPanelProps) {
  const desktop = useIsDesktop()
  const body = point && <Conversation key={pinKey(point)} point={point} sessions={sessions} onClose={onClose} />
  const onOpenChange = (open: boolean) => {
    if (!open) onClose()
  }

  return desktop ? (
    <Sheet open={point !== null} onOpenChange={onOpenChange}>
      <SheetContent container={container} aria-describedby={undefined}>
        {body}
      </SheetContent>
    </Sheet>
  ) : (
    <Drawer open={point !== null} onOpenChange={onOpenChange}>
      <DrawerContent container={container} aria-describedby={undefined}>
        {body}
      </DrawerContent>
    </Drawer>
  )
}

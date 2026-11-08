import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ChatPanel, formatPoint } from '@/chat/ChatPanel'
import { ERROR_TEXT, FALLBACK_HEADER, STARTER_CHIPS } from '@/chat/copy'
import { createSessions, pinKey, type Sessions } from '@/chat/turns'
import { BLUE_MILL_RUN, DECLINE_RUN, FALLBACK_RUN, NETWORK_ERROR_RUN, chatServer, scriptOf } from '@/test/chatServer'
import { server } from '@/test/setup'

const BLUE_MILL = { lat: 53.802797, lon: -0.086644 }
const OTHER = { lat: 51.5074, lon: -0.1278 }
const FIRST_CHIP = STARTER_CHIPS[0]

const setWidth = (width: number) => {
  window.innerWidth = width
  window.dispatchEvent(new Event('resize'))
}

function setup(...script: Parameters<typeof chatServer>) {
  const fake = chatServer(...script)
  server.use(...fake.handlers)
  const sessions = createSessions()
  const onClose = vi.fn()
  const view = (point: typeof BLUE_MILL | null, store: Sessions = sessions) => <ChatPanel point={point} sessions={store} onClose={onClose} />
  return { fake, sessions, onClose, view }
}

beforeEach(() => setWidth(1024))
afterEach(() => setWidth(1024))

describe('ChatPanel layout', () => {
  it('is a Sheet at 1024 px with the coordinates to 4 dp in the mono font and a Change LZ button (G11)', () => {
    const { view } = setup(scriptOf(BLUE_MILL_RUN))
    render(view(BLUE_MILL))
    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveAttribute('data-slot', 'sheet-content')
    const title = within(dialog).getByRole('heading')
    expect(title).toHaveTextContent('53.8028, -0.0866')
    expect(title).toHaveClass('font-mono')
    expect(within(dialog).getByRole('button', { name: 'Change LZ' })).toBeInTheDocument()
  })

  it('is a Drawer at 375 px (E14)', () => {
    setWidth(375)
    const { view } = setup(scriptOf(BLUE_MILL_RUN))
    render(view(BLUE_MILL))
    expect(screen.getByRole('dialog')).toHaveAttribute('data-slot', 'drawer-content')
  })

  it('is a Sheet at exactly 768 px and a Drawer at 767 px', () => {
    const { view } = setup(scriptOf(BLUE_MILL_RUN))
    setWidth(768)
    const { unmount } = render(view(BLUE_MILL))
    expect(screen.getByRole('dialog')).toHaveAttribute('data-slot', 'sheet-content')
    unmount()
    setWidth(767)
    render(view(BLUE_MILL))
    expect(screen.getByRole('dialog')).toHaveAttribute('data-slot', 'drawer-content')
  })

  it('switches between Sheet and Drawer when the window is resized', () => {
    const { view } = setup(scriptOf(BLUE_MILL_RUN))
    render(view(BLUE_MILL))
    expect(screen.getByRole('dialog')).toHaveAttribute('data-slot', 'sheet-content')
    act(() => setWidth(500))
    expect(screen.getByRole('dialog')).toHaveAttribute('data-slot', 'drawer-content')
  })

  it('shows nothing when no point is open', () => {
    const { view } = setup(scriptOf(BLUE_MILL_RUN))
    render(view(null))
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('does not cover the page: the map beside it stays clickable', () => {
    const { view } = setup(scriptOf(BLUE_MILL_RUN))
    render(
      <>
        <button type="button">map</button>
        {view(BLUE_MILL)}
      </>,
    )
    expect(screen.getByRole('button', { name: 'map' })).not.toHaveAttribute('aria-hidden')
    expect(document.body.style.pointerEvents).not.toBe('none')
  })

  it('formats coordinates to four decimal places', () => {
    expect(formatPoint({ lat: 51.5, lon: -0.12345678 })).toBe('51.5000, -0.1235')
  })
})

describe('ChatPanel Change LZ', () => {
  it('asks to close, and Esc does too', async () => {
    const { view, onClose } = setup(scriptOf(BLUE_MILL_RUN))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: 'Change LZ' }))
    expect(onClose).toHaveBeenCalledTimes(1)
    await user.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalledTimes(2)
  })

  it('restores the turns when the same pin is opened again', async () => {
    const { view, sessions } = setup(scriptOf(BLUE_MILL_RUN))
    const user = userEvent.setup()
    const { rerender } = render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    await screen.findByText(/Pilot's call this afternoon/, undefined, { timeout: 5000 })

    rerender(view(null))
    expect(screen.queryByRole('dialog')).toBeNull()
    rerender(view({ ...BLUE_MILL }))
    expect(screen.getByText(FIRST_CHIP)).toBeInTheDocument()
    expect(screen.getByText(/Pilot's call this afternoon/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: FIRST_CHIP })).toBeNull() // the chips are for an empty thread
    expect(sessions.get(pinKey(BLUE_MILL)).turns).toHaveLength(1)
  })
})

describe('ChatPanel conversation', () => {
  it('shows the three starter chips in an empty thread, and tapping one sends its text', async () => {
    const { view, fake } = setup(scriptOf(BLUE_MILL_RUN))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    const chips = within(screen.getByRole('list', { name: 'Suggested questions' })).getAllByRole('button')
    expect(chips.map((chip) => chip.textContent)).toEqual(['Can I fly here this afternoon?', 'What airspace is above me?', 'Any NOTAMs I should know about?'])

    await user.click(chips[1])
    await waitFor(() => expect(fake.runs).toHaveLength(1))
    expect((fake.runs[0].body.input as { messages: { content: string }[] }).messages[0].content).toBe('What airspace is above me?')
    expect(screen.queryByRole('list', { name: 'Suggested questions' })).toBeNull()
  })

  it('shows the progress lines in sequence, then the whole answer, and no line afterwards (GP5, G13)', async () => {
    const { view, fake } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 3 }))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))

    const status = screen.getByRole('status')
    await waitFor(() => expect(status).toHaveTextContent('Checking airspace…'))
    expect(screen.queryByText(/Pilot's call/)).toBeNull() // nothing of the answer before the run ends

    act(() => fake.runs[0].release())
    await screen.findByText(/Pilot's call this afternoon/, undefined, { timeout: 5000 })
    expect(status).toBeEmptyDOMElement()
    expect(screen.getByText(/Decision support only/)).toBeInTheDocument()
    expect(screen.queryByText(FALLBACK_HEADER)).toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('says it is working until the first progress line arrives', async () => {
    const { view, fake } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 0 }))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    expect(screen.getByRole('status')).toHaveTextContent('Working…')
    act(() => fake.runs[0].release())
    await screen.findByText(/Pilot's call/, undefined, { timeout: 5000 })
  })

  it('keeps the last progress line on screen until the run ends', async () => {
    const { view, fake } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 7 }))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    // Held at the last event: all six progress lines have been sent, and the last one stays on screen.
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Checking the answer…'))
    act(() => fake.runs[0].release())
    await screen.findByText(/Pilot's call/, undefined, { timeout: 5000 })
  })

  it('shows a fallback answer as given under the muted header (G15)', async () => {
    const { view } = setup(scriptOf(FALLBACK_RUN))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    await screen.findByText(FALLBACK_HEADER, undefined, { timeout: 5000 })
    expect(screen.getByText(/Safe summary for 10:00-14:00 GMT/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('shows the UK-only decline as a normal answer (E6)', async () => {
    const { view } = setup(scriptOf(DECLINE_RUN))
    const user = userEvent.setup()
    render(view({ lat: 49.2, lon: -2.13 }))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    await screen.findByText('FlySafeLZ only covers the UK.', undefined, { timeout: 5000 })
    expect(screen.queryByText(FALLBACK_HEADER)).toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('turns a dropped connection into the error text and a Retry button, with no partial answer (E7)', async () => {
    const { view, fake } = setup(scriptOf(NETWORK_ERROR_RUN), scriptOf(BLUE_MILL_RUN))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))

    const alert = await screen.findByRole('alert', undefined, { timeout: 5000 })
    expect(alert).toHaveTextContent("Couldn't reach FlySafeLZ. Try again.")
    expect(alert).toHaveTextContent(ERROR_TEXT)
    expect(screen.getByRole('status')).toBeEmptyDOMElement()
    expect(screen.queryByText(/Pilot's call/)).toBeNull()

    await user.click(within(alert).getByRole('button', { name: 'Retry' }))
    await screen.findByText(/Pilot's call this afternoon/, undefined, { timeout: 5000 })
    expect(screen.queryByRole('alert')).toBeNull()
    expect(fake.threads).toEqual(['thread-1'])
    expect(fake.runs.map((run) => run.threadId)).toEqual(['thread-1', 'thread-1'])
  })

  it('blocks a double submit: two taps in one tick start one run, and the input is disabled while it runs', async () => {
    const { view, fake } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 3 }))
    render(view(BLUE_MILL))
    const chip = screen.getByRole('button', { name: FIRST_CHIP })
    const other = screen.getByRole('button', { name: STARTER_CHIPS[1] })
    act(() => {
      chip.click()
      other.click()
      chip.click()
    })
    await waitFor(() => expect(fake.runs).toHaveLength(1))
    const box = screen.getByRole('textbox', { name: 'Ask about this LZ' })
    expect(box).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    act(() => fake.runs[0].release())
    await screen.findByText(/Pilot's call/, undefined, { timeout: 5000 })
    expect(fake.runs).toHaveLength(1)
    expect(box).toBeEnabled()
  })

  it('cannot send empty or whitespace-only input', async () => {
    const { view, fake } = setup(scriptOf(BLUE_MILL_RUN))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    const box = screen.getByRole('textbox', { name: 'Ask about this LZ' })
    const send = screen.getByRole('button', { name: 'Send' })
    expect(send).toBeDisabled()
    await user.type(box, '   ')
    expect(send).toBeDisabled()
    await user.type(box, '{Enter}')
    expect(fake.runs).toEqual([])
    expect(fake.threads).toEqual([])
  })

  it('sends typed input unchanged, Unicode and 2,000 characters included, and clears the box', async () => {
    const { view, fake } = setup(scriptOf(DECLINE_RUN))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    const long = `  ✈️ ça va? ${'é'.repeat(1985)}`.slice(0, 2000)
    const box = screen.getByRole('textbox', { name: 'Ask about this LZ' })
    await user.click(box)
    await user.paste(long)
    await user.keyboard('{Enter}')
    await waitFor(() => expect(fake.runs).toHaveLength(1))
    expect((fake.runs[0].body.input as { messages: { content: string }[] }).messages[0].content).toBe(long)
    expect(box).toHaveValue('')
    await screen.findByText('FlySafeLZ only covers the UK.', undefined, { timeout: 5000 })
  })

  it('does not show the result of a run for the old pin on the new pin (E8)', async () => {
    const { view, fake, sessions } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 3 }))
    const user = userEvent.setup()
    const { rerender } = render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Checking airspace…'))

    rerender(view(OTHER))
    expect(screen.getByRole('heading')).toHaveTextContent('51.5074, -0.1278')
    expect(screen.queryAllByRole('region', { name: 'Question and answer' })).toEqual([]) // no turn of the old pin
    expect(screen.getByRole('status')).toBeEmptyDOMElement()
    expect(screen.getAllByRole('button', { name: /fly here|airspace|NOTAMs/ })).toHaveLength(3) // the new pin starts empty, with chips
    await waitFor(() => expect(fake.runs[0].aborted).toBe(true))

    fake.runs[0].release()
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText(/Pilot's call/)).toBeNull()
    expect(sessions.get(pinKey(OTHER)).turns).toEqual([])

    rerender(view(BLUE_MILL))
    expect(screen.getAllByRole('region', { name: 'Question and answer' })).toHaveLength(1) // the old pin kept its turn
    expect(screen.queryByText(/Pilot's call/)).toBeNull() // cancelled, so it has no answer; the pilot can retry
    expect(screen.getByRole('alert')).toHaveTextContent(ERROR_TEXT)
  })

  it('does not show the result of a run when the sheet is closed mid-run', async () => {
    const { view, fake } = setup(scriptOf(BLUE_MILL_RUN, { holdAt: 3 }))
    const user = userEvent.setup()
    const { rerender } = render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Checking airspace…'))
    rerender(view(null))
    await waitFor(() => expect(fake.runs[0].aborted).toBe(true))
    fake.runs[0].release()
    rerender(view(OTHER))
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText(/Pilot's call/)).toBeNull()
  })

  it('writes nothing about the question or the answer to the console during a full run (F2)', async () => {
    const spies = (['log', 'info', 'warn', 'error', 'debug'] as const).map((name) => vi.spyOn(console, name).mockImplementation(() => {}))
    const { view } = setup(scriptOf(BLUE_MILL_RUN))
    const user = userEvent.setup()
    render(view(BLUE_MILL))
    await user.click(screen.getByRole('button', { name: FIRST_CHIP }))
    await screen.findByText(/Pilot's call this afternoon/, undefined, { timeout: 5000 })
    const logged = spies.flatMap((spy) => spy.mock.calls).flat().map(String).join('\n')
    expect(logged).not.toContain('Can I fly here')
    expect(logged).not.toContain("Pilot's call")
    expect(logged).not.toContain('01632')
    vi.restoreAllMocks()
  })
})

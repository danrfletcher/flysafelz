import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Badges } from '@/chat/Badges'
import { ChatPanel } from '@/chat/ChatPanel'
import { readBadges } from '@/chat/readBadges'
import { createSessions, pinKey, readResult } from '@/chat/turns'
import { FALLBACK_HEADER } from '@/chat/copy'
import { BLUE_MILL_RUN, FALLBACK_RUN, FOLLOW_UP_RUN, OUTLOOK_RUN, type Recording } from '@/test/chatServer'
import { BLUE_MILL } from '@/test/handlers'
import { mockReducedMotion } from '@/test/reducedMotion'

const finalOf = (run: Recording) => run.events.at(-1)!.data as { assessment: Record<string, unknown> | null }
const BLUE = finalOf(BLUE_MILL_RUN).assessment
const OUTLOOK = finalOf(OUTLOOK_RUN).assessment

const segment = (over: Record<string, unknown>) => ({
  start: '2026-10-31T10:00:00+00:00',
  end: '2026-10-31T11:00:00+00:00',
  tz_abbr: 'GMT',
  tier: 'VERDICT',
  verdict: 'PILOT_DISCRETION',
  approaching_limits: false,
  with_conditions: false,
  display: 'PILOT DISCRETION',
  conditions: [],
  ...over,
})

describe('Badges', () => {
  it('draws one badge per segment, with the time span in the zone and its abbreviation, and the display as given (G14, GP6)', () => {
    render(<Badges assessment={BLUE} />)
    const badges = screen.getAllByRole('button')
    expect(badges).toHaveLength(2)
    expect(badges[0]).toHaveTextContent('10:00–11:00 GMT')
    expect(badges[0]).toHaveTextContent('PILOT DISCRETION – APPROACHING LIMITS · WITH CONDITIONS')
    expect(badges[1]).toHaveTextContent('11:00–14:00 GMT')
    expect(badges[1].textContent).toMatch(/PILOT DISCRETION$/)
  })

  it('colours the Blue Mill 10:00-11:00 badge amber and the plain pilot-discretion one green, and shows its label as text too', () => {
    render(<Badges assessment={BLUE} />)
    const [amber, green] = screen.getAllByRole('button')
    expect(amber).toHaveClass('border-verdict-amber')
    expect(green).toHaveClass('border-verdict-discretion')
    for (const badge of [amber, green]) expect(badge.textContent?.trim()).not.toBe('') // a label in words, not colour alone
    expect(within(amber).getByText(/APPROACHING LIMITS/)).toBeInTheDocument()
    expect(within(amber).getByText(/WITH CONDITIONS/)).toBeInTheDocument()
  })

  it('lists the conditions beneath the badge they belong to', () => {
    render(<Badges assessment={BLUE} />)
    const items = screen.getAllByRole('listitem').filter((item) => item.parentElement?.getAttribute('aria-label') === 'Verdict by time')
    expect(items).toHaveLength(2)
    expect(within(items[0]).getByText('Contact 01632 960003 before flying near H7171/26.')).toBeInTheDocument()
    expect(within(items[1]).queryAllByRole('listitem')).toHaveLength(0)
    const text = items[0].textContent ?? ''
    expect(text.indexOf('APPROACHING LIMITS')).toBeLessThan(text.indexOf('Contact 01632'))
  })

  it('shows OUTLOOK and TOO_FAR segments as slate "No verdict" with the tier text and no verdict colour (E9)', () => {
    render(<Badges assessment={OUTLOOK} />)
    const [outlook, tooFar] = screen.getAllByRole('button')
    expect(outlook).toHaveTextContent('Wed 11 Nov 10:00–14:00 GMT')
    expect(outlook).toHaveTextContent('No verdict')
    expect(outlook).toHaveTextContent('Outlook')
    expect(tooFar).toHaveTextContent('No verdict')
    expect(tooFar).toHaveTextContent('Too far ahead')
    for (const badge of [outlook, tooFar]) {
      expect(badge).toHaveClass('border-type-other')
      expect(badge.className).not.toMatch(/verdict-/)
    }
  })

  it('colours from the machine values, not from the display words', () => {
    const lies = { segments: [segment({ verdict: 'NO_GO', display: 'PILOT DISCRETION' }), segment({ verdict: 'PILOT_DISCRETION', display: 'NO-GO', start: '2026-10-31T11:00:00+00:00', end: '2026-10-31T12:00:00+00:00' })], overall: null }
    render(<Badges assessment={lies} />)
    const [first, second] = screen.getAllByRole('button')
    expect(first).toHaveClass('border-verdict-no-go')
    expect(first).toHaveTextContent('PILOT DISCRETION')
    expect(second).toHaveClass('border-verdict-discretion')
    expect(second).toHaveTextContent('NO-GO')
  })

  it('renders nothing at all for a null assessment, with no empty container (E9)', () => {
    const { container } = render(<Badges assessment={null} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing when the assessment has no readable segment', () => {
    for (const assessment of [{ segments: [] }, { segments: 'x' }, {}, { segments: [null, 3, { start: 'soon', end: 'later', tz_abbr: 'GMT' }] }]) {
      const { container } = render(<Badges assessment={assessment} />)
      expect(container).toBeEmptyDOMElement()
    }
  })

  it('leaves out a segment it cannot read and keeps the others', () => {
    render(<Badges assessment={{ segments: [{ nonsense: true }, segment({})] }} />)
    expect(screen.getAllByRole('button')).toHaveLength(1)
  })

  it('shows the day on every badge when the assessment spans more than one day, and the end day when one badge crosses midnight', () => {
    const badges = readBadges({
      segments: [
        segment({ start: '2026-10-31T22:00:00+00:00', end: '2026-11-01T01:00:00+00:00' }),
        segment({ start: '2026-11-01T01:00:00+00:00', end: '2026-11-01T03:00:00+00:00' }),
      ],
    })
    expect(badges?.map((badge) => badge.span)).toEqual(['Sat 31 Oct 22:00–Sun 1 Nov 01:00 GMT', 'Sun 1 Nov 01:00–03:00 GMT'])
  })

  it('reads the clock from the text, so a summer-time offset shows local time with its abbreviation', () => {
    const [badge] = readBadges({ segments: [segment({ start: '2026-07-04T10:00:00+01:00', end: '2026-07-04T12:00:00+01:00', tz_abbr: 'BST' })] })!
    expect(badge.span).toBe('10:00–12:00 BST')
  })

  it('asks for this turn’s highlights when a badge is clicked', async () => {
    const onSelect = vi.fn()
    render(<Badges assessment={BLUE} onSelect={onSelect} />)
    await userEvent.setup().click(screen.getAllByRole('button')[1])
    expect(onSelect).toHaveBeenCalledOnce()
  })
})

const finalValues = (run: Recording) => run.events.at(-1)!.data
const atRest = (value: string) => value === '' || value === 'none'
const turnBadges = (turn: HTMLElement) => within(turn).queryByRole('list', { name: 'Verdict by time' })

describe('Badges across turns (G14)', () => {
  afterEach(() => vi.unstubAllGlobals())

  function open() {
    mockReducedMotion(false)
    const sessions = createSessions()
    const key = pinKey(BLUE_MILL)
    const view = (point: typeof BLUE_MILL | null, onShowTurn: (id: string) => void = () => {}) => (
      <ChatPanel point={point} sessions={sessions} onClose={() => {}} onShowTurn={onShowTurn} />
    )
    const answer = (id: string, run: Recording) => act(() => sessions.complete(key, id, readResult(finalValues(run))!))
    const ask = (id: string) => act(() => sessions.ask(key, id, `question ${id}`))
    return { sessions, key, view, answer, ask }
  }

  /** Records every change to the style of the badges in `list` from now on. */
  function watch(list: HTMLElement) {
    const changes: string[] = []
    new MutationObserver((records) => records.forEach((record) => changes.push(String(record.target.nodeName)))).observe(list, { attributes: true, attributeFilter: ['style'], subtree: true })
    return changes
  }

  it('enters only the badges of the turn that just arrived: the first answer’s badges enter, the badge-less follow-up has none, and a redraw or a re-render never replays the first', async () => {
    const { view, answer, ask } = open()
    ask('t1')
    const { rerender } = render(view(BLUE_MILL))
    answer('t1', BLUE_MILL_RUN)

    const first = screen.getAllByRole('region', { name: 'Question and answer' })[0]
    const items = within(turnBadges(first)!).getAllByRole('listitem').filter((item) => item.querySelector(':scope > button'))
    expect(items).toHaveLength(2)
    for (const item of items) {
      expect(item.style.opacity).toBe('0') // they mount at the start values and enter
      expect(item.style.transform).toBe('scale(0.92)')
    }
    await waitFor(() => expect(items.every((item) => item.style.opacity === '1' && atRest(item.style.transform))).toBe(true))

    const changes = watch(turnBadges(first)!)
    ask('t2')
    answer('t2', FOLLOW_UP_RUN)
    const turns = screen.getAllByRole('region', { name: 'Question and answer' })
    expect(turns).toHaveLength(2)
    expect(turnBadges(turns[1])).toBeNull() // nothing to animate on a badge-less turn

    const onShowTurn = vi.fn()
    rerender(view(BLUE_MILL, onShowTurn)) // the map redrew: the chat re-renders with new callbacks
    await userEvent.setup().click(within(turnBadges(turns[0])!).getAllByRole('button')[0])
    expect(onShowTurn).toHaveBeenCalledWith('t1')
    await new Promise((resolve) => setTimeout(resolve, 300)) // longer than any enter animation
    expect(changes).toEqual([])
    expect(items.every((item) => item.style.opacity === '1' && atRest(item.style.transform))).toBe(true)
  })

  it('draws the badges of earlier turns at rest when the pin’s chat is opened again, and enters only a turn that lands afterwards', async () => {
    const { view, answer, ask } = open()
    ask('t1')
    const { rerender } = render(view(BLUE_MILL))
    answer('t1', BLUE_MILL_RUN)
    await waitFor(() => expect(screen.getAllByRole('listitem').some((item) => item.style.opacity === '1')).toBe(true))

    rerender(view(null))
    expect(screen.queryByRole('dialog')).toBeNull()
    rerender(view({ ...BLUE_MILL }))
    const back = screen.getAllByRole('region', { name: 'Question and answer' })[0]
    for (const item of within(turnBadges(back)!).getAllByRole('listitem').filter((entry) => entry.querySelector(':scope > button'))) {
      expect(item.style.opacity).toBe('1')
      expect(atRest(item.style.transform)).toBe(true)
    }

    ask('t2')
    answer('t2', BLUE_MILL_RUN) // a second turn that carries badges: these are the ones that just arrived
    const latest = screen.getAllByRole('region', { name: 'Question and answer' })[1]
    for (const item of within(turnBadges(latest)!).getAllByRole('listitem').filter((entry) => entry.querySelector(':scope > button'))) {
      expect(item.style.transform).toBe('scale(0.92)')
    }
  })

  it('enters slate "no verdict" badges like any other, and a turn with no assessment or only a null one has nothing to animate (E9)', () => {
    const { view, answer, ask } = open()
    ask('t1')
    render(view(BLUE_MILL))
    answer('t1', OUTLOOK_RUN)
    const outlook = within(turnBadges(screen.getAllByRole('region', { name: 'Question and answer' })[0])!).getAllByRole('listitem').filter((item) => item.querySelector(':scope > button'))
    expect(outlook).toHaveLength(2)
    for (const item of outlook) expect(item.style.transform).toBe('scale(0.92)')
    expect(within(outlook[0]).getByRole('button')).toHaveClass('border-type-other')

    ask('t2')
    answer('t2', FOLLOW_UP_RUN)
    const turns = screen.getAllByRole('region', { name: 'Question and answer' })
    expect(turnBadges(turns[1])).toBeNull()
    expect(within(turns[1]).queryAllByRole('listitem')).toHaveLength(0)
  })

  it('shows a fallback answer’s warning header with nothing animating on it, and enters its badges like any other', async () => {
    const { view, answer, ask } = open()
    ask('t1')
    render(view(BLUE_MILL))
    answer('t1', FALLBACK_RUN)
    const header = screen.getByText(FALLBACK_HEADER)
    expect(header.style.opacity).toBe('') // the header has no animation of its own
    expect(header.style.transform).toBe('')
    const turn = screen.getAllByRole('region', { name: 'Question and answer' })[0]
    const items = within(turnBadges(turn)!).getAllByRole('listitem').filter((entry) => entry.querySelector(':scope > button'))
    expect(items).toHaveLength(2)
    for (const item of items) expect(item.style.transform).toBe('scale(0.92)')
    await waitFor(() => expect(header).toBeVisible()) // once the chat panel has entered, the header is plainly there
  })

  it('never animates the badges of a run that was discarded when another LZ was picked onto the new LZ’s chat (E8)', async () => {
    const { sessions, key, view, answer, ask } = open()
    const other = { lat: 51.5074, lon: -0.1278 }
    ask('t1')
    const { rerender } = render(view(BLUE_MILL))
    rerender(view(other)) // a new pin: the old chat and its run are gone
    answer('t1', BLUE_MILL_RUN) // the discarded run's result still lands in the old pin's session
    expect(screen.queryByRole('list', { name: 'Verdict by time' })).toBeNull()
    expect(screen.queryAllByRole('region', { name: 'Question and answer' })).toHaveLength(0)
    expect(sessions.get(pinKey(other)).turns).toHaveLength(0)
    expect(sessions.get(key).turns[0].status).toBe('done')
  })
})

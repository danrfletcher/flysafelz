import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { Badges } from '@/chat/Badges'
import { readBadges } from '@/chat/readBadges'
import { BLUE_MILL_RUN, OUTLOOK_RUN, type Recording } from '@/test/chatServer'

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

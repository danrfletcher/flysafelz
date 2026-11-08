import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { AirspaceProps, NotamProps } from '@/map/types'
import { CARD_EMPTY, CARD_FOOTNOTE, CARD_TITLE, WhatsHereCard } from '@/pick/WhatsHereCard'
import { BLUE_MILL, BLUE_MILL_AIRSPACE, BLUE_MILL_NOTAMS } from '@/test/handlers'
import { loaded, loading } from '@/test/layers'

const AIRSPACE = loaded<AirspaceProps>(BLUE_MILL_AIRSPACE.features)
const NOTAMS = loaded<NotamProps>(BLUE_MILL_NOTAMS.features)

const rowButtons = () => screen.getAllByRole('button').filter((button) => button.hasAttribute('aria-expanded'))

describe('the fixed copy (G10)', () => {
  it('has the header, the 2 NM footnote and no other button than the rows and the group', () => {
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    expect(screen.getByRole('heading', { name: 'Over this point' })).toBeInTheDocument()
    expect(CARD_TITLE).toBe('Over this point')
    expect(screen.getByText('The chat checks everything within 2 NM.')).toBeInTheDocument()
    expect(CARD_FOOTNOTE).toBe('The chat checks everything within 2 NM.')
    expect(screen.getAllByRole('button').every((button) => button.hasAttribute('aria-expanded'))).toBe(true)
    expect(screen.queryByText(/chat about/i)).toBeNull()
  })

  it('says "Nothing in the loaded layers over this point." for an empty list, and still has the footnote', () => {
    render(<WhatsHereCard point={{ lat: 56, lon: -4 }} airspace={AIRSPACE} notams={NOTAMS} />)
    expect(screen.getByText('Nothing in the loaded layers over this point.')).toBeInTheDocument()
    expect(CARD_EMPTY).toBe('Nothing in the loaded layers over this point.')
    expect(screen.getByText(CARD_FOOTNOTE)).toBeInTheDocument()
    expect(screen.queryAllByRole('listitem')).toHaveLength(0)
    expect(screen.queryByText(/Above FL100/)).toBeNull()
  })

  it('does not say "Nothing…" when no layer has loaded yet', () => {
    render(<WhatsHereCard point={BLUE_MILL} airspace={loading()} notams={loading()} />)
    expect(screen.queryByText(CARD_EMPTY)).toBeNull()
  })
})

describe('rows (G10 rows, C2 order)', () => {
  it('lists rows by floor, lowest first, each with a chip label, its name or id and its limits', () => {
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    const rows = screen.getAllByRole('listitem')
    // five below FL100 shown at once, the three at or above in the collapsed group
    expect(rows.map((row) => row.querySelector('button')?.textContent)).toEqual([
      'ATZBLUE MILL TEST ATZSFC to 2000 ft AMSL',
      'NOTAMH2001/26SFC to 1200FT AMSL',
      'NOTAMapproxH2002/26SFC to 3000FT AMSL',
      'NOTAMH2004/26? to ?',
      'CTABLUE MILL TEST CTA1500 ft AMSL to 4500 ft AMSL',
    ])
  })

  it('labels an approx NOTAM "approx" and no other row', () => {
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    expect(screen.getAllByText('approx')).toHaveLength(1)
    expect(within(screen.getByRole('button', { name: /H2002\/26/ })).getByText('approx')).toBeInTheDocument()
  })

  it('expands a row to its active times (and, for a NOTAM, the raw text), and collapses it again', async () => {
    const user = userEvent.setup()
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    const atz = screen.getByRole('button', { name: /BLUE MILL TEST ATZ/ })
    expect(atz).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('MON-FRI 0800-1700')).toBeNull()

    await user.click(atz)
    expect(atz).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('MON-FRI 0800-1700')).toBeInTheDocument()
    expect(screen.queryByText('NOTAM text')).toBeNull() // airspace has no NOTAM text

    const notam = screen.getByRole('button', { name: /H2001\/26/ })
    await user.click(notam)
    expect(screen.getByText(/CTC 01632 960001 BEFORE FLIGHT/)).toBeInTheDocument()
    expect(screen.getByText('NOTAM text')).toBeInTheDocument()
    expect(screen.getAllByText('Active')).toHaveLength(2)

    await user.click(atz)
    expect(screen.queryByText('MON-FRI 0800-1700')).toBeNull()
    expect(screen.getByText(/CTC 01632 960001/)).toBeInTheDocument() // the other row stays open
  })

  it('says "no times published" for airspace with no schedule', async () => {
    const user = userEvent.setup()
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    await user.click(screen.getByRole('button', { name: /BLUE MILL TEST CTA/ }))
    expect(screen.getByText('no times published')).toBeInTheDocument()
  })

  it('shows the chip colour as the map draws it: the type token, with the label as text', () => {
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    const atz = screen.getByText('ATZ')
    expect(atz.getAttribute('style')).toContain('var(--type-controlled)') // jsdom keeps var() only in the style attribute
    expect(screen.getAllByText('NOTAM')[0].getAttribute('style')).toContain('var(--type-notam)')
  })
})

describe('the "Above FL100 (n)" group', () => {
  it('is collapsed, counts its rows, and opens on click to show them in floor order', async () => {
    const user = userEvent.setup()
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    const group = screen.getByRole('button', { name: 'Above FL100 (3)' })
    expect(group).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('BLUE MILL FL100 FLOOR CTA')).toBeNull()
    expect(screen.queryByText('BLUE MILL TEST TMA ABOVE FL100')).toBeNull()

    await user.click(group)
    expect(group).toHaveAttribute('aria-expanded', 'true')
    const rows = rowButtons().filter((button) => button !== group)
    expect(rows.slice(-3).map((row) => row.textContent)).toEqual([
      'CTABLUE MILL FL100 FLOOR CTAFL100 to FL245',
      'TMABLUE MILL TEST TMA ABOVE FL100FL105 to FL195',
      'NOTAMH2003/26FL105 to FL195',
    ])

    await user.click(group)
    expect(screen.queryByText('BLUE MILL FL100 FLOOR CTA')).toBeNull()
  })

  it('has no group at all when nothing is over the point at or above FL100', () => {
    render(<WhatsHereCard point={BLUE_MILL} airspace={loaded(BLUE_MILL_AIRSPACE.features.slice(0, 2))} notams={loaded([])} />)
    expect(screen.queryByText(/Above FL100/)).toBeNull()
  })

  it('rows inside the group expand like any other', async () => {
    const user = userEvent.setup()
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    await user.click(screen.getByRole('button', { name: 'Above FL100 (3)' }))
    await user.click(screen.getByRole('button', { name: /H2003\/26/ }))
    expect(screen.getByText(/AIR DISPLAY/)).toBeInTheDocument()
  })
})

describe('text is never HTML', () => {
  const hostile = '<script>window.__pwned = 1</script><img src=x onerror="window.__pwned=1"><b>bold</b> CAFÉ ÅNGSTRÖM 飛行 🛩'
  const long = `${'A'.repeat(1000)} ${'B'.repeat(1000)}`

  it('renders a NOTAM’s text, name and limits as plain text, whatever markup they hold', async () => {
    const user = userEvent.setup()
    const base = BLUE_MILL_NOTAMS.features[0]
    const feature = {
      ...base,
      properties: { ...base.properties, id: '<i>H1/26</i>', text: hostile, item_f: '<u>SFC</u>' },
    }
    const { container } = render(<WhatsHereCard point={BLUE_MILL} airspace={loaded([])} notams={loaded([feature])} />)
    await user.click(screen.getByRole('button', { name: /H1\/26/ }))
    expect(screen.getByText(hostile)).toBeInTheDocument()
    expect(container.querySelector('script, img, b, i, u')).toBeNull()
    expect(container).toHaveTextContent('<i>H1/26</i>')
    expect(container).toHaveTextContent('<u>SFC</u>')
    expect((window as unknown as { __pwned?: number }).__pwned).toBeUndefined()
  })

  it('renders an airspace name as plain text', () => {
    const base = BLUE_MILL_AIRSPACE.features[0]
    const feature = { ...base, properties: { ...base.properties, name: '<b>BOLD ATZ</b>' } }
    const { container } = render(<WhatsHereCard point={BLUE_MILL} airspace={loaded([feature])} notams={loaded([])} />)
    expect(container.querySelector('b')).toBeNull()
    expect(container).toHaveTextContent('<b>BOLD ATZ</b>')
  })

  it('keeps a 2,000-character text whole, in a block that wraps, with its line breaks', async () => {
    const user = userEvent.setup()
    const base = BLUE_MILL_NOTAMS.features[0]
    const feature = { ...base, properties: { ...base.properties, text: `LINE ONE\nLINE TWO ${long}` } }
    render(<WhatsHereCard point={BLUE_MILL} airspace={loaded([])} notams={loaded([feature])} />)
    await user.click(screen.getByRole('button', { name: /H2001\/26/ }))
    const text = screen.getByText(/LINE ONE/)
    expect(text.textContent).toBe(`LINE ONE\nLINE TWO ${long}`)
    expect(text.className).toContain('whitespace-pre-wrap')
    expect(text.className).toContain('overflow-wrap:anywhere')
  })

  it('leaves out the text block for a NOTAM whose text is empty', async () => {
    const user = userEvent.setup()
    const base = BLUE_MILL_NOTAMS.features[0]
    const feature = { ...base, properties: { ...base.properties, text: '' } }
    render(<WhatsHereCard point={BLUE_MILL} airspace={loaded([])} notams={loaded([feature])} />)
    await user.click(screen.getByRole('button', { name: /H2001\/26/ }))
    expect(screen.queryByText('NOTAM text')).toBeNull()
    expect(screen.getByText('Active')).toBeInTheDocument()
  })
})

describe('the text of a NOTAM is never logged', () => {
  it('writes nothing to the console while the card opens and rows expand', async () => {
    const spies = (['log', 'info', 'warn', 'error', 'debug'] as const).map((name) => vi.spyOn(console, name).mockImplementation(() => {}))
    const user = userEvent.setup()
    render(<WhatsHereCard point={BLUE_MILL} airspace={AIRSPACE} notams={NOTAMS} />)
    await user.click(screen.getByRole('button', { name: /H2001\/26/ }))
    expect(screen.getByText(/BEFORE FLIGHT/)).toBeInTheDocument()
    expect(spies.map((spy) => spy.mock.calls)).toEqual([[], [], [], [], []])
    vi.restoreAllMocks()
  })
})

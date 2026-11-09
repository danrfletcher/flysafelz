import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Answer } from '@/chat/Answer'
import { FALLBACK_HEADER } from '@/chat/copy'
import { BLUE_MILL_RUN, DECLINE_RUN, FALLBACK_RUN } from '@/test/chatServer'

const finalOf = (run: { events: { data: unknown }[] }) => run.events.at(-1)!.data as { answer: string; guard: string }

describe('Answer', () => {
  it('renders markdown: bold, a list', () => {
    const { container } = render(<Answer answer={finalOf(BLUE_MILL_RUN).answer} guard="passed" />)
    expect(container.querySelector('strong')).toHaveTextContent("Pilot's call this afternoon.")
    expect(container.querySelectorAll('li')).toHaveLength(3)
  })

  it('shows the fallback header above the answer only for guard = fallback (G15)', () => {
    const fallback = finalOf(FALLBACK_RUN)
    const { container, rerender } = render(<Answer answer={fallback.answer} guard={fallback.guard} />)
    const header = screen.getByText(FALLBACK_HEADER)
    expect(header).toHaveClass('text-muted-foreground')
    expect(container.firstElementChild?.firstElementChild).toBe(header)
    expect(container).toHaveTextContent('Safe summary for 10:00-14:00 GMT.') // the answer, as given

    for (const guard of ['passed', null, '', 'FALLBACK', 'unknown', 'fallbacks']) {
      rerender(<Answer answer={fallback.answer} guard={guard} />)
      expect(screen.queryByText(FALLBACK_HEADER), `guard ${JSON.stringify(guard)}`).toBeNull()
    }
  })

  it('shows the UK-only decline as normal text, with no header (E6)', () => {
    const decline = finalOf(DECLINE_RUN)
    const { container } = render(<Answer answer={decline.answer} guard={decline.guard} />)
    expect(container).toHaveTextContent('FlySafeLZ only covers the UK.')
    expect(screen.queryByText(FALLBACK_HEADER)).toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('renders raw HTML as text and runs nothing', () => {
    const ran = vi.fn()
    ;(window as unknown as { ran: () => void }).ran = ran
    const attack = 'Hello <script>window.ran()</script> <img src=x onerror="window.ran()"> <b onclick="window.ran()">bold</b> <iframe src="https://example.com"></iframe>'
    const { container } = render(<Answer answer={attack} guard={null} />)
    expect(container.querySelector('script, img, iframe, b')).toBeNull()
    expect(container.querySelector('[onclick], [onerror]')).toBeNull()
    expect(container).toHaveTextContent('<script>window.ran()</script>')
    expect(container).toHaveTextContent('<img src=x onerror="window.ran()">')
    expect(ran).not.toHaveBeenCalled()
  })

  it('drops markdown images, so the browser never fetches from another server for an answer', () => {
    const { container } = render(<Answer answer="Look: ![x](https://tracker.example/pixel.png) done" guard={null} />)
    expect(container.querySelector('img')).toBeNull()
  })

  it('opens links in a new tab without handing over the opener, and does not run a javascript: link', () => {
    const { container } = render(<Answer answer="[site](https://example.com) [bad](javascript:alert(1))" guard={null} />)
    const links = [...container.querySelectorAll('a')]
    expect(links[0]).toHaveAttribute('href', 'https://example.com')
    expect(links[0]).toHaveAttribute('target', '_blank')
    expect(links[0].getAttribute('rel')).toContain('noopener')
    for (const link of links) expect(link.getAttribute('href') ?? '').not.toMatch(/^javascript:/i)
  })
})

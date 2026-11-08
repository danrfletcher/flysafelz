import L from 'leaflet'
import { act, render, screen, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { MapContainer, useMap } from 'react-leaflet'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { Badges } from '@/chat/Badges'
import { ChatPanel } from '@/chat/ChatPanel'
import { ProgressLine } from '@/chat/ProgressLine'
import { createSessions } from '@/chat/turns'
import { ENTER } from '@/lib/motion'
import { PickHandler } from '@/map/PickHandler'
import type { AirspaceProps, NotamProps } from '@/map/types'
import { BLUE_MILL_RUN } from '@/test/chatServer'
import { BLUE_MILL, BLUE_MILL_AIRSPACE, BLUE_MILL_NOTAMS } from '@/test/handlers'
import { loaded } from '@/test/layers'
import { mockReducedMotion } from '@/test/reducedMotion'

// `animate` is the real one, watched: the pin and the card are Leaflet's elements, so they are animated through it. The sheet, drawer, progress
// line and badges are motion components, which are seen through the styles motion writes (the start values, every value on the way, the end).
vi.mock('motion/react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('motion/react')>()
  return { ...actual, animate: vi.fn(actual.animate) }
})
const { animate } = await import('motion/react')

const BLUE = (BLUE_MILL_RUN.events.at(-1)!.data as { assessment: Record<string, unknown> }).assessment

const holder: { map: L.Map | null } = { map: null }
function Probe() {
  const found = useMap()
  useEffect(() => {
    holder.map = found
  }, [found])
  return null
}

function renderMap() {
  return render(
    <MapContainer center={[BLUE_MILL.lat, BLUE_MILL.lon]} zoom={10}>
      <Probe />
      <PickHandler airspace={loaded<AirspaceProps>(BLUE_MILL_AIRSPACE.features)} notams={loaded<NotamProps>(BLUE_MILL_NOTAMS.features)} />
    </MapContainer>,
  )
}

const clickAt = (lat: number, lon: number) => act(() => void holder.map!.fire('click', { latlng: L.latLng(lat, lon) }))
const atRest = (value: string) => value === '' || value === 'none'

/** Every `transform` an element's style holds from now on, newest last, so "at no point" can be asserted. */
function watchTransforms(element: HTMLElement) {
  const seen = [element.style.transform]
  new MutationObserver(() => seen.push(element.style.transform)).observe(element, { attributes: true, attributeFilter: ['style'] })
  return seen
}

describe.each([
  ['reduce', true, 'reduced'],
  ['no-preference', false, 'full'],
] as const)('enter animations under prefers-reduced-motion: %s', (_name, reduced, variant) => {
  beforeEach(() => {
    mockReducedMotion(reduced)
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
    vi.mocked(animate).mockClear()
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    window.innerWidth = 1024
  })

  const played = (selector: string) => vi.mocked(animate).mock.calls.filter(([element]) => (element as Element).matches(selector))

  it('drops the pin with the shared pin variant, once per click, and leaves one pin', async () => {
    const { container } = renderMap()
    clickAt(BLUE_MILL.lat, BLUE_MILL.lon)
    const drop = container.querySelector<HTMLElement>('.pick-pin__drop')!
    const calls = played('.pick-pin__drop')
    expect(calls).toHaveLength(1)
    expect(calls[0][0]).toBe(drop)
    expect(calls[0][1]).toEqual(ENTER.pin[variant].animate)
    expect(calls[0][2]).toEqual(ENTER.pin[variant].transition)
    if (reduced) expect(Object.keys(calls[0][1] as object)).toEqual(['opacity']) // no y, so no bounce

    clickAt(54.5, -1.5)
    expect(played('.pick-pin__drop')).toHaveLength(2)
    expect(played('.pick-pin__drop')[1][0]).not.toBe(drop) // the new pin's own element
    expect(container.querySelectorAll('.pick-pin')).toHaveLength(1)
    await waitFor(() => expect(atRest(container.querySelector<HTMLElement>('.pick-pin__drop')!.style.transform)).toBe(true))
  })

  it('springs the What’s-here card in with the shared card variant', async () => {
    const { container } = renderMap()
    clickAt(BLUE_MILL.lat, BLUE_MILL.lon)
    const bubble = container.querySelector<HTMLElement>('.leaflet-popup-content-wrapper')!
    const calls = played('.leaflet-popup-content-wrapper')
    expect(calls).toHaveLength(1)
    expect(calls[0][0]).toBe(bubble)
    const { initial, animate: end, transition } = ENTER.card[variant]
    // from the start values to the end values, with the shared timing
    expect(Object.keys(calls[0][1] as object).sort()).toEqual(Object.keys(end).sort())
    for (const [key, to] of Object.entries(end)) expect(calls[0][1]).toHaveProperty(key, [(initial as Record<string, number>)[key], to])
    expect(calls[0][2]).toEqual(transition)
    await waitFor(() => expect(bubble.style.opacity).toBe('1'))
    await waitFor(() => expect(atRest(bubble.style.transform)).toBe(true))
  })

  it.each([
    [1024, 'sheet'],
    [375, 'drawer'],
  ] as const)('enters the chat at %i px as the %s variant, and ends with no transform', async (width, kind) => {
    window.innerWidth = width
    const view = render(<ChatPanel point={BLUE_MILL} sessions={createSessions()} onClose={() => {}} />)
    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveAttribute('data-slot', `${kind}-content`)
    const seen = watchTransforms(dialog)
    const start = ENTER[kind][variant].initial as Record<string, number>
    expect(dialog.style.opacity).toBe(String(start.opacity))
    if (reduced) expect(atRest(dialog.style.transform)).toBe(true)
    else expect(dialog.style.transform).toBe(kind === 'sheet' ? `translateX(${start.x}px)` : `translateY(${start.y}px)`)
    await waitFor(() => expect(dialog.style.opacity).toBe('1'))
    await waitFor(() => expect(atRest(dialog.style.transform)).toBe(true))
    if (reduced) expect(seen.every(atRest)).toBe(true) // no transform at any point
    view.unmount()
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('fades a progress line in, and never moves it', async () => {
    render(<ProgressLine line="Checking weather…" />)
    const line = screen.getByText('Checking weather…')
    const seen = watchTransforms(line)
    expect(line.style.opacity).toBe('0')
    await waitFor(() => expect(line.style.opacity).toBe('1'))
    expect(seen.every(atRest)).toBe(true)
  })

  it('enters the badges of a turn that just arrived, and ends with no transform', async () => {
    render(<Badges assessment={BLUE} enter />)
    const badges = screen.getAllByRole('listitem').filter((item) => item.querySelector(':scope > button'))
    expect(badges).toHaveLength(2)
    const seen = badges.map(watchTransforms)
    const start = ENTER.badge[variant].initial as Record<string, number>
    for (const badge of badges) {
      expect(badge.style.opacity).toBe(String(start.opacity))
      if (reduced) expect(atRest(badge.style.transform)).toBe(true)
      else expect(badge.style.transform).toBe(`scale(${start.scale})`)
    }
    await waitFor(() => expect(badges.every((badge) => badge.style.opacity === '1')).toBe(true))
    await waitFor(() => expect(badges.every((badge) => atRest(badge.style.transform))).toBe(true))
    if (reduced) expect(seen.flat().every(atRest)).toBe(true)
  })

  it('draws badges that did not just arrive at rest, with no start values', () => {
    render(<Badges assessment={BLUE} />)
    for (const badge of screen.getAllByRole('listitem').filter((item) => item.querySelector(':scope > button'))) {
      expect(badge.style.opacity).toBe('1')
      expect(atRest(badge.style.transform)).toBe(true)
    }
  })
})

describe('a change of the system setting while the page is open', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('makes the next pin drop follow the new setting, with no reload, both ways', () => {
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
    const media = mockReducedMotion(false)
    vi.mocked(animate).mockClear()
    const { container } = renderMap()
    const dropped = () => vi.mocked(animate).mock.calls.filter(([element]) => (element as Element).matches('.pick-pin__drop')).map(([, target]) => Object.keys(target as object))

    clickAt(BLUE_MILL.lat, BLUE_MILL.lon)
    media.set(true)
    clickAt(54.5, -1.5)
    media.set(false)
    clickAt(55.5, -2.5)
    expect(dropped()).toEqual([['y'], ['opacity'], ['y']])
    expect(container.querySelectorAll('.pick-pin')).toHaveLength(1)
  })
})

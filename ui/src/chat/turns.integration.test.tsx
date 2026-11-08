import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { FALLBACK_HEADER, STARTER_CHIPS } from '@/chat/copy'
import { themeColor } from '@/map/airspaceStyle'
import { ANSWER_PANE } from '@/map/answerOverlays'
import { FALLBACK_RUN, FOLLOW_UP_RUN, HIGHLIGHTS_RUN, chatServer, scriptOf } from '@/test/chatServer'
import { BLUE_MILL, blueMillRoutes } from '@/test/handlers'
import { server } from '@/test/setup'

// The whole path: the recorded Blue Mill run is replayed over MSW as a real SSE stream, the chat store keeps each turn's `assessment` and
// `overlays`, and the map draws the open pin's turn. Nothing here calls a helper in place of the code under test.

const LONDON = { lat: 51.5074, lon: -0.1278 }
const FIRST_CHIP = STARTER_CHIPS[0]
const SLOW = { timeout: 5000 }

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})
afterEach(() => vi.restoreAllMocks())

function open(...script: Parameters<typeof chatServer>) {
  const fake = chatServer(...script)
  server.use(...blueMillRoutes(), ...fake.handlers)
  const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  const view = render(<App />)
  const map = fit.mock.contexts[0] as L.Map
  const pick = (point: { lat: number; lon: number }) => act(() => void map.fire('click', { latlng: L.latLng(point.lat, point.lon) }))
  return { fake, fit, map, pick, view, user: userEvent.setup() }
}

async function chatAbout() {
  const button = await screen.findByRole('button', { name: 'Chat about this LZ' })
  act(() => button.click())
  return screen.findByRole('dialog', { name: /\d/ })
}

/** What the answer pane holds: every path drawn in it, with the colours as set. */
function answerPaths(map: L.Map): L.Path[] {
  const found: L.Path[] = []
  map.eachLayer((layer) => {
    if (layer instanceof L.Path && (layer.options as { pane?: string }).pane === ANSWER_PANE) found.push(layer)
  })
  return found
}
const circles = (map: L.Map) => answerPaths(map).filter((layer) => layer instanceof L.Circle)
const shapes = (map: L.Map) => answerPaths(map).filter((layer) => !(layer instanceof L.Circle))

/** The base layers' shapes: the map's polygons that are not in the answer pane. */
function basePolygons(map: L.Map): L.Polygon[] {
  const found: L.Polygon[] = []
  map.eachLayer((layer) => {
    if (layer instanceof L.Polygon && !(layer instanceof L.Circle) && (layer.options as { pane?: string }).pane !== ANSWER_PANE) found.push(layer)
  })
  return found
}

async function ask(user: ReturnType<typeof userEvent.setup>, sheet: HTMLElement, text: string) {
  const box = within(sheet).getByRole('textbox', { name: 'Ask about this LZ' })
  await user.type(box, text)
  await user.click(within(sheet).getByRole('button', { name: 'Send' }))
}

describe('answer badges, NOTAM links and highlights through the app', () => {
  it('GP6: the Blue Mill answer lands with one badge per segment above it, the amber one first, and its highlights drawn and fitted', async () => {
    const { pick, user, map, fit, fake } = open(scriptOf(HIGHLIGHTS_RUN))
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)

    const list = within(sheet).getByRole('list', { name: 'Verdict by time' })
    const badges = within(list).getAllByRole('button')
    expect(badges).toHaveLength(2)
    expect(badges[0]).toHaveTextContent('10:00–11:00 GMT')
    expect(badges[0]).toHaveTextContent('APPROACHING LIMITS')
    expect(badges[0]).toHaveTextContent('WITH CONDITIONS')
    expect(badges[0]).toHaveClass('border-verdict-amber')
    expect(list.compareDocumentPosition(screen.getByText(/Pilot's call this afternoon/)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(within(list).getByText('Contact 01632 960003 before flying near H7171/26.')).toBeInTheDocument()

    expect(circles(map)).toHaveLength(1)
    expect(shapes(map)).toHaveLength(4)
    const [bounds] = fit.mock.calls.at(-1) as [L.LatLngBounds]
    expect(bounds.contains(L.latLng(BLUE_MILL.lat, BLUE_MILL.lon))).toBe(true)
    expect(fake.runs).toHaveLength(1)
  })

  it('GP7: clicking a NOTAM id draws its shape heavier, opens its raw text with the phone as delivered, and turns the switch back on', async () => {
    const { pick, user, map } = open(scriptOf(HIGHLIGHTS_RUN))
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)

    // Turn the switch off first: the click on the id must bring the highlights back.
    await user.click(screen.getByRole('button', { name: 'Map layers' }))
    const panel = await screen.findByRole('dialog', { name: /layers/i })
    await user.click(within(panel).getByRole('switch', { name: 'Answer overlays' }))
    expect(answerPaths(map)).toHaveLength(0)
    await user.keyboard('{Escape}')

    const ids = within(sheet).getAllByRole('button', { name: /^[A-Z]\d+\/\d+$/ })
    expect(ids.map((id) => id.textContent)).toEqual(['H7171/26', 'H7171/26', 'H7172/26', 'V32/26'])
    await user.click(ids[0])
    const popover = await screen.findByRole('dialog', { name: 'NOTAM H7171/26' })
    expect(popover).toHaveTextContent('01632 960003')

    expect(shapes(map)).toHaveLength(4)
    const weights = shapes(map).map((layer) => layer.options.weight!)
    expect(weights.filter((weight) => weight === Math.max(...weights))).toHaveLength(1)
    expect(shapes(map).at(-1)!.options.color).toBe(themeColor('--action-contact-first'))
    await user.click(screen.getByRole('button', { name: 'Map layers' }))
    expect(await screen.findByRole('switch', { name: 'Answer overlays' })).toBeChecked()
  })

  it('G14/G17: a NOTAM id clicked 50 ms after the answer lands, while the badges are still entering, highlights its shape and opens its text', async () => {
    const { pick, user, map } = open(scriptOf(HIGHLIGHTS_RUN))
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)
    const badge = within(sheet).getByRole('list', { name: 'Verdict by time' }).querySelector('li')!
    const shapesBefore = shapes(map).map((layer) => layer.options.weight)

    // From here the clock is the test's: the 50 ms are advanced, and the badges, which began a moment ago, have not finished.
    vi.useFakeTimers()
    try {
      act(() => void vi.advanceTimersByTime(50))
      expect(badge.style.opacity).not.toBe('1')
      const id = within(sheet).getAllByRole('button', { name: /^[A-Z]\d+\/\d+$/ })[0]
      act(() => id.click())
      expect(screen.getByRole('dialog', { name: `NOTAM ${id.textContent}` })).toHaveTextContent('01632 960003')
      const weights = shapes(map).map((layer) => layer.options.weight!)
      expect(weights).toHaveLength(shapesBefore.length)
      expect(weights.filter((weight) => weight === Math.max(...weights))).toHaveLength(1) // the picked shape is drawn heavier
    } finally {
      vi.useRealTimers()
    }
  })

  it('GP8: the follow-up has no badges and shows only the circle; turn 1 keeps its badges, and clicking them redraws turn 1’s highlights', async () => {
    const { pick, user, map } = open(scriptOf(HIGHLIGHTS_RUN), scriptOf(FOLLOW_UP_RUN))
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)
    const before = within(sheet).getByRole('list', { name: 'Verdict by time' }).textContent // the text, not the markup: the badges' enter animation is still settling

    await ask(user, sheet, 'What does that NOTAM mean?')
    await within(sheet).findByText(/parachutists land inside/, undefined, SLOW)

    const turns = within(sheet).getAllByRole('region', { name: 'Question and answer' })
    expect(turns).toHaveLength(2)
    expect(within(turns[1]).queryByRole('list', { name: 'Verdict by time' })).toBeNull()
    expect(within(turns[0]).getByRole('list', { name: 'Verdict by time' }).textContent).toBe(before)
    expect(circles(map)).toHaveLength(1)
    expect(shapes(map)).toHaveLength(0) // the latest turn has no overlays: only the circle

    await user.click(within(turns[0]).getAllByRole('button')[0])
    await waitFor(() => expect(shapes(map)).toHaveLength(4))
    expect(circles(map)).toHaveLength(1)
  })

  it('draws the base layers the same before and after the answer highlights come and go (G16)', async () => {
    const { pick, user, map } = open(scriptOf(HIGHLIGHTS_RUN))
    pick(BLUE_MILL)
    await waitFor(() => expect(basePolygons(map).length).toBeGreaterThan(0))
    const signature = () => basePolygons(map).map((layer) => `${layer.options.color}|${layer.options.weight}|${layer.options.dashArray}|${layer.getLatLngs().flat(2).length}`).sort()
    const before = signature()
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)
    expect(shapes(map)).toHaveLength(4)
    expect(signature()).toEqual(before)
    await user.click(within(sheet).getByRole('button', { name: 'Change LZ' }))
    await waitFor(() => expect(answerPaths(map)).toHaveLength(0))
    expect(signature()).toEqual(before)
  })

  it('Change LZ clears the highlights and keeps the pin; picking a new pin clears them too', async () => {
    const { pick, user, map, view } = open(scriptOf(HIGHLIGHTS_RUN))
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)
    expect(answerPaths(map).length).toBeGreaterThan(0)

    await user.click(within(sheet).getByRole('button', { name: 'Change LZ' }))
    await waitFor(() => expect(answerPaths(map)).toHaveLength(0))
    expect(view.container.querySelector('.pick-pin')).not.toBeNull()

    const again = await chatAbout() // the old pin's thread is still its own: the answer is back with its highlights
    await within(again).findByText(/Pilot's call this afternoon/, undefined, SLOW)
    await waitFor(() => expect(shapes(map)).toHaveLength(4))
    pick(LONDON)
    await waitFor(() => expect(answerPaths(map)).toHaveLength(0))
  })

  it('E8: a run that finishes after the pilot has picked another pin draws nothing on the new LZ', async () => {
    const { pick, user, fake, map } = open(scriptOf(HIGHLIGHTS_RUN, { holdAt: 3 }))
    pick(BLUE_MILL)
    const first = await chatAbout()
    await user.click(within(first).getByRole('button', { name: FIRST_CHIP }))
    await waitFor(() => expect(within(first).getByRole('status')).toHaveTextContent('Checking airspace…'))

    pick(LONDON)
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /\d/ })).toBeNull())
    fake.runs[0].release() // the old run's result arrives late
    const second = await chatAbout()
    await new Promise((resolve) => setTimeout(resolve, 100))
    expect(within(second).getByRole('heading')).toHaveTextContent('51.5074, -0.1278')
    expect(screen.queryByText(/Pilot's call/)).toBeNull()
    expect(shapes(map)).toHaveLength(0)
    expect(circles(map)).toHaveLength(0)
  })

  it('a guard-fallback turn that carries an assessment shows its badges under the fallback header', async () => {
    const { pick, user } = open(scriptOf(FALLBACK_RUN))
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Safe summary for 10:00-14:00 GMT/, undefined, SLOW)
    const turn = within(sheet).getByRole('region', { name: 'Question and answer' })
    const header = within(turn).getByText(FALLBACK_HEADER)
    const list = within(turn).getByRole('list', { name: 'Verdict by time' })
    expect(within(list).getAllByRole('button')).toHaveLength(2)
    expect(header.compareDocumentPosition(list) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(list.compareDocumentPosition(within(turn).getByText(/Safe summary/)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})

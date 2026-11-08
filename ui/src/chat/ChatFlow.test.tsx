import L from 'leaflet'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { STARTER_CHIPS } from '@/chat/copy'
import { DRAWER_HEIGHT_FRACTION, SHEET_WIDTH_PX, lzBounds } from '@/map/fitLz'
import { BLUE_MILL_RUN, chatServer, scriptOf } from '@/test/chatServer'
import { BLUE_MILL, blueMillRoutes } from '@/test/handlers'
import { server } from '@/test/setup'
import { RAINVIEWER_INDEX_URL } from '@/map/rainIndex'
import { apiUrl } from '@/config'

const LONDON = { lat: 51.5074, lon: -0.1278 }
const WIDTH = 1200
const HEIGHT = 800
const FIRST_CHIP = STARTER_CHIPS[0]

const setWidth = (width: number) => {
  window.innerWidth = width
  window.dispatchEvent(new Event('resize'))
}

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(WIDTH)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(HEIGHT)
})

afterEach(() => {
  vi.restoreAllMocks()
  setWidth(1024)
})

function open(...script: Parameters<typeof chatServer>) {
  const fake = chatServer(...script)
  server.use(...blueMillRoutes(), ...fake.handlers)
  const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  const view = render(<App />)
  const map = fit.mock.contexts[0] as L.Map
  const pick = (point: { lat: number; lon: number }) => act(() => void map.fire('click', { latlng: L.latLng(point.lat, point.lon) }))
  return { fake, fit, map, pick, view, user: userEvent.setup() }
}

// A plain click event, not userEvent's pointer sequence: jsdom gives those no coordinates, and Leaflet's double-tap check on the map then throws.
async function chatAbout() {
  const button = await screen.findByRole('button', { name: 'Chat about this LZ' })
  act(() => button.click())
  return screen.findByRole('dialog', { name: /\d/ })
}

describe('chat from the What’s-here card', () => {
  it('opens the Sheet for the pin, fits the 2 NM circle padded for the sheet, and Change LZ closes it but keeps the pin and card (GP4, G11)', async () => {
    const { pick, fit, user, view } = open(scriptOf(BLUE_MILL_RUN))
    pick(BLUE_MILL)
    fit.mockClear()
    const sheet = await chatAbout()

    expect(sheet).toHaveAttribute('data-slot', 'sheet-content')
    expect(within(sheet).getByRole('heading')).toHaveTextContent('53.8028, -0.0866')
    await waitFor(() => expect(fit).toHaveBeenCalled())
    const [bounds, options] = fit.mock.calls.at(-1) as [L.LatLngBounds, L.FitBoundsOptions]
    expect(bounds.equals(lzBounds(BLUE_MILL))).toBe(true)
    expect((options.paddingBottomRight as L.PointTuple)[0]).toBeGreaterThanOrEqual(SHEET_WIDTH_PX)
    expect(sheet.parentElement?.tagName).toBe('MAIN') // placed in the map area, clear of the nav bar and footer
    expect(screen.getByRole('button', { name: 'Map layers' }).parentElement).toHaveClass('md:right-[27rem]') // the layer button sits beside the sheet, not under it

    await user.click(within(sheet).getByRole('button', { name: 'Change LZ' }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /\d/ })).toBeNull())
    expect(screen.getByRole('button', { name: 'Map layers' }).parentElement).not.toHaveClass('md:right-[27rem]')
    expect(view.container.querySelector('.pick-pin')).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Chat about this LZ' })).toBeInTheDocument()
  })

  it('opens a Drawer at 375 px, pads the fit for the bottom, and leaves the layer panel and card usable (E14)', async () => {
    setWidth(375)
    const { pick, fit } = open(scriptOf(BLUE_MILL_RUN))
    pick(BLUE_MILL)
    fit.mockClear()
    const drawer = await chatAbout()
    expect(drawer).toHaveAttribute('data-slot', 'drawer-content')
    await waitFor(() => expect(fit).toHaveBeenCalled())
    const options = fit.mock.calls.at(-1)![1] as L.FitBoundsOptions
    expect((options.paddingBottomRight as L.PointTuple)[1]).toBeGreaterThanOrEqual(Math.round(DRAWER_HEIGHT_FRACTION * HEIGHT))
    expect((options.paddingBottomRight as L.PointTuple)[0]).toBeLessThan(SHEET_WIDTH_PX)

    expect(screen.getByRole('button', { name: 'Map layers' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Chat about this LZ' })).toBeEnabled()
  })

  it('runs a whole conversation against the server, and the only network targets are VITE_API_URL and RainViewer’s index (F2, F4)', async () => {
    const fetched = vi.spyOn(globalThis, 'fetch')
    const { pick, user, fake } = open(scriptOf(BLUE_MILL_RUN))
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    await within(sheet).findByText(/Pilot's call this afternoon/, undefined, { timeout: 5000 })

    expect(fake.threads).toHaveLength(1)
    const targets = fetched.mock.calls.map(([input]) => new URL(input instanceof Request ? input.url : String(input)).origin)
    expect([...new Set(targets)].sort()).toEqual([new URL(apiUrl()).origin, new URL(RAINVIEWER_INDEX_URL).origin].sort())
  })

  it('warns before the page is left while a conversation has turns, and not before (G24)', async () => {
    const { pick, user } = open(scriptOf(BLUE_MILL_RUN))
    const leave = () => {
      const event = new Event('beforeunload', { cancelable: true })
      window.dispatchEvent(event)
      return event.defaultPrevented
    }
    pick(BLUE_MILL)
    const sheet = await chatAbout()
    expect(leave()).toBe(false) // open, but nothing asked yet
    await user.click(within(sheet).getByRole('button', { name: FIRST_CHIP }))
    expect(leave()).toBe(true)
    await within(sheet).findByText(/Pilot's call/, undefined, { timeout: 5000 })
    expect(leave()).toBe(true)
  })

  it('picking another LZ mid-run cancels the run, shows no old answer on the new pin, and the old pin keeps its turn (E8)', async () => {
    const { pick, user, fake } = open(scriptOf(BLUE_MILL_RUN, { holdAt: 3 }))
    pick(BLUE_MILL)
    const first = await chatAbout()
    await user.click(within(first).getByRole('button', { name: FIRST_CHIP }))
    await waitFor(() => expect(within(first).getByRole('status')).toHaveTextContent('Checking airspace…'))

    pick(LONDON)
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /\d/ })).toBeNull())
    await waitFor(() => expect(fake.runs[0].aborted).toBe(true))
    expect(fake.runs[0].body.on_disconnect).toBe('cancel') // the server is told to cancel the run when the client goes, not to carry on
    fake.runs[0].release()

    const second = await chatAbout()
    expect(within(second).getByRole('heading')).toHaveTextContent('51.5074, -0.1278')
    expect(within(second).getByRole('list', { name: 'Suggested questions' })).toBeInTheDocument()
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText(/Pilot's call/)).toBeNull()

    pick(BLUE_MILL)
    const back = await chatAbout()
    expect(within(back).getByText(FIRST_CHIP)).toBeInTheDocument()
    expect(fake.threads).toEqual(['thread-1'])
  })
})

import L from 'leaflet'
import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { STARTER_CHIPS } from '@/chat/copy'
import { LAYER_PREFS_KEY } from '@/lib/layerPrefs'
import { HIGHLIGHTS_RUN, chatServer, scriptOf } from '@/test/chatServer'
import { BLUE_MILL, blueMillRoutes } from '@/test/handlers'
import { server } from '@/test/setup'

// The Blue Mill recorded answer (phones are fiction-range): a click on a NOTAM id in it turns Answer overlays back on, and that is saved.

const SLOW = { timeout: 5000 }

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(800)
})
afterEach(() => vi.restoreAllMocks())

/** A page load with Answer overlays saved as off, the Blue Mill answer on screen. */
async function answeredWithOverlaysOff() {
  window.localStorage.setItem(LAYER_PREFS_KEY, '{"answer":false}')
  server.use(...blueMillRoutes(), ...chatServer(scriptOf(HIGHLIGHTS_RUN)).handlers)
  const fit = vi.spyOn(L.Map.prototype, 'fitBounds')
  const view = render(<App />)
  const map = fit.mock.contexts[0] as L.Map
  const user = userEvent.setup()
  act(() => void map.fire('click', { latlng: L.latLng(BLUE_MILL.lat, BLUE_MILL.lon) }))
  act(() => screen.getByRole('button', { name: 'Chat about this LZ' }).click())
  const sheet = await screen.findByRole('dialog', { name: /\d/ })
  await user.click(within(sheet).getByRole('button', { name: STARTER_CHIPS[0] }))
  await within(sheet).findByText(/Pilot's call this afternoon/, undefined, SLOW)
  return { view, sheet, user }
}

describe('G1: a NOTAM-id click that turns Answer overlays on is saved', () => {
  it('sets the stored value to true, leaves the other stored values alone, and keeps it after a reload', async () => {
    const { view, sheet, user } = await answeredWithOverlaysOff()
    expect(JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) as string)).toEqual({ answer: false })

    await user.click(within(sheet).getAllByRole('button', { name: /^[A-Z]\d+\/\d+$/ })[0])
    await screen.findByRole('dialog', { name: 'NOTAM H7171/26' })
    expect(JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) as string)).toEqual({ airspace: true, notams: true, answer: true })

    view.unmount() // the reload
    server.use(...blueMillRoutes())
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'Map layers' }))
    expect(await screen.findByRole('switch', { name: 'Answer overlays' })).toBeChecked()
  })
})

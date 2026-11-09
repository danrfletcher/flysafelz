import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { describe, expect, it } from 'vitest'

import { useMapLayers } from '@/map/useMapLayers'
import { WhatsHereCard } from '@/pick/WhatsHereCard'
import { server } from '@/test/setup'
import {
  AIRSPACE_URL,
  BLUE_MILL,
  BLUE_MILL_AIRSPACE,
  BLUE_MILL_NOTAMS,
  NOTAMS_URL,
  airspaceRoute,
  notamsRoute,
  unavailable,
} from '@/test/handlers'

// The card on the real loading hook, with MSW as the server: this is E4, the click before the layers load.
function Card() {
  const layers = useMapLayers()
  return <WhatsHereCard point={BLUE_MILL} airspace={layers.airspace} notams={layers.notams} />
}

/** A route that answers only when `release` is called, so a test can look at the card while it is still waiting. */
function held(url: string, body: object) {
  let release: () => void = () => {}
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  return { handler: http.get(url, async () => (await gate, HttpResponse.json(body))), release }
}

describe('E4: a click before the layers load', () => {
  it('shows what has arrived plus "Still loading: NOTAMs", then fills in by itself when the NOTAMs arrive', async () => {
    const notams = held(NOTAMS_URL(), BLUE_MILL_NOTAMS)
    server.use(airspaceRoute(BLUE_MILL_AIRSPACE), notams.handler)
    render(<Card />)

    expect(await screen.findByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(screen.getByText('Still loading: NOTAMs')).toBeInTheDocument()
    expect(screen.queryByText('H2001/26')).toBeNull()
    expect(screen.queryByText('Nothing in the loaded layers over this point.')).toBeNull()

    notams.release()
    expect(await screen.findByText('H2001/26')).toBeInTheDocument()
    expect(screen.queryByText(/Still loading/)).toBeNull()
    expect(screen.getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    // the order holds after the fill-in: the NOTAM rows are among the airspace rows by floor
    const names = screen.getAllByRole('listitem').map((item) => item.querySelector('button')?.textContent ?? '')
    expect(names.findIndex((name) => name.includes('H2001/26'))).toBeGreaterThan(names.findIndex((name) => name.includes('BLUE MILL TEST ATZ')))
    expect(names.findIndex((name) => name.includes('H2001/26'))).toBeLessThan(names.findIndex((name) => name.includes('BLUE MILL TEST CTA')))
  })

  it('names both layers while neither has arrived, with no list and no "Nothing…" message, then fills both in', async () => {
    const airspace = held(AIRSPACE_URL(), BLUE_MILL_AIRSPACE)
    const notams = held(NOTAMS_URL(), BLUE_MILL_NOTAMS)
    server.use(airspace.handler, notams.handler)
    render(<Card />)
    expect(screen.getByText('Still loading: Airspace, NOTAMs')).toBeInTheDocument()
    expect(screen.queryByText('Nothing in the loaded layers over this point.')).toBeNull()
    expect(screen.queryAllByRole('listitem')).toHaveLength(0)

    airspace.release()
    notams.release()
    await waitFor(() => expect(screen.queryByText(/Still loading/)).toBeNull())
    expect(screen.getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(screen.getByText('H2002/26')).toBeInTheDocument()
  })

  it('keeps an open row open when the rest of the layers arrive', async () => {
    const notams = held(NOTAMS_URL(), BLUE_MILL_NOTAMS)
    server.use(airspaceRoute(BLUE_MILL_AIRSPACE), notams.handler)
    render(<Card />)
    const atz = await screen.findByRole('button', { name: /BLUE MILL TEST ATZ/ })
    await userEvent.setup().click(atz)
    expect(await screen.findByText('MON-FRI 0800-1700')).toBeInTheDocument()
    notams.release()
    await screen.findByText('H2001/26')
    expect(screen.getByText('MON-FRI 0800-1700')).toBeInTheDocument()
  })
})

describe('E4 (503): a layer that is unavailable', () => {
  it('says "Airspace unavailable" instead of "Still loading", and the NOTAM rows still work', async () => {
    server.use(unavailable(AIRSPACE_URL()), notamsRoute(BLUE_MILL_NOTAMS))
    render(<Card />)
    expect(await screen.findByText('H2001/26')).toBeInTheDocument()
    expect(screen.getByText('Airspace unavailable')).toBeInTheDocument()
    expect(screen.queryByText(/Still loading/)).toBeNull()
    expect(screen.queryByText('BLUE MILL TEST ATZ')).toBeNull()
    expect(screen.getByText('The chat checks everything within 2 NM.')).toBeInTheDocument()
  })

  it('shows "NOTAMs unavailable" with the airspace list, and "Still loading" only for the layer that is', async () => {
    const airspace = held(AIRSPACE_URL(), BLUE_MILL_AIRSPACE)
    server.use(airspace.handler, unavailable(NOTAMS_URL()))
    render(<Card />)
    expect(await screen.findByText('NOTAMs unavailable')).toBeInTheDocument()
    expect(screen.getByText('Still loading: Airspace')).toBeInTheDocument()
    airspace.release()
    expect(await screen.findByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(screen.queryByText(/Still loading/)).toBeNull()
    expect(screen.getByText('NOTAMs unavailable')).toBeInTheDocument()
  })

  it('says unavailable for both when both fail, with no "Nothing in the loaded layers" message', async () => {
    server.use(unavailable(AIRSPACE_URL()), unavailable(NOTAMS_URL()))
    render(<Card />)
    expect(await screen.findByText('Airspace unavailable')).toBeInTheDocument()
    expect(await screen.findByText('NOTAMs unavailable')).toBeInTheDocument()
    expect(screen.queryByText('Nothing in the loaded layers over this point.')).toBeNull()
  })
})

describe('stale payloads', () => {
  it('lists a stale layer like any other and does not repeat the panel’s “stale” note', async () => {
    server.use(airspaceRoute({ ...BLUE_MILL_AIRSPACE, stale: true }), notamsRoute({ ...BLUE_MILL_NOTAMS, stale: true }))
    const { container } = render(<Card />)
    expect(await screen.findByText('H2001/26')).toBeInTheDocument()
    expect(screen.getByText('BLUE MILL TEST ATZ')).toBeInTheDocument()
    expect(container).not.toHaveTextContent(/stale/i)
    expect(screen.queryByText(/unavailable|Still loading/)).toBeNull()
  })
})

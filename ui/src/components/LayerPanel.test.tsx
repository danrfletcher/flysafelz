import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ACTIONS, ON_THE_MAP, VERDICTS } from '@/components/Legend'
import { LAYERS, LayerPanel, type LayerId, type LayerSwitches } from '@/components/LayerPanel'
import { HOW_IT_WORKS } from '@/content/howItWorks'
import type { BasemapStatus } from '@/map/basemap'
import { createRainView, type RainView } from '@/map/rainView'
import type { LayerState, LayerStatus } from '@/map/useMapLayers'
import type { RainState } from '@/map/useRain'

const UPDATED = '2026-11-08T14:32:00Z'
const ok = (over: Partial<LayerState<unknown>> = {}): LayerState<unknown> => ({ status: 'ok', features: [], updatedAt: UPDATED, stale: false, ...over })
const loading: LayerState<unknown> = { status: 'loading', features: [], updatedAt: null, stale: false }
const unavailable: LayerState<unknown> = { status: 'unavailable', features: [], updatedAt: null, stale: false }
const windOk = (over: Partial<LayerStatus> = {}): LayerStatus => ({ status: 'ok', updatedAt: UPDATED, stale: false, ...over })

const RADAR: RainState = { mode: 'radar', host: 'https://tilecache.rainviewer.com', frames: [{ time: 1_794_139_800, path: '/v2/radar/a' }], fetchedAt: 0 }

/** The panel with its switches held in state, as App holds them. */
function Panel({
  wind = windOk(),
  airspace = ok(),
  notams = ok(),
  basemap,
  onChange,
  rain = RADAR,
  rainView = createRainView(),
}: {
  wind?: LayerStatus
  airspace?: LayerState<unknown>
  notams?: LayerState<unknown>
  basemap?: BasemapStatus
  onChange?: (id: LayerId, on: boolean) => void
  rain?: RainState
  rainView?: RainView
}) {
  const [shown, setShown] = useState<LayerSwitches>({ wind: true, rain: true, airspace: true, airspaceAll: false, notams: true, notamsAll: false, answer: true })
  return (
    <LayerPanel
      states={{ wind, airspace, notams }}
      rain={{ state: rain, view: rainView }}
      basemap={basemap}
      shown={shown}
      onShownChange={(id, on) => {
        onChange?.(id, on)
        setShown((current) => ({ ...current, [id]: on }))
      }}
    />
  )
}

async function open(ui = <Panel />) {
  const user = userEvent.setup()
  const view = render(ui)
  await user.click(screen.getByRole('button', { name: 'Map layers' }))
  return { user, view, dialog: await screen.findByRole('dialog') }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('the switches', () => {
  it('has Wind, Rain, each base layer with its all-heights partner and the Answer overlays switch; the all-heights ones are off by default (G1)', async () => {
    await open()
    const switches = screen.getAllByRole('switch')
    expect(switches.map((s) => s.getAttribute('aria-checked'))).toEqual(['true', 'true', 'true', 'false', 'true', 'false', 'true'])
    expect(screen.getByRole('switch', { name: 'Airspace all heights' })).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'NOTAMs all heights' })).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'Wind' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Rain' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'NOTAMs below FL100' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Answer overlays' })).toBeChecked()
    expect(LAYERS.map((layer) => layer.id)).toEqual(['wind', 'rain', 'airspace', 'airspaceAll', 'notams', 'notamsAll', 'answer'])
    expect(LAYERS.filter((layer) => !layer.defaultOn).map((layer) => layer.id)).toEqual(['airspaceAll', 'notamsAll'])
  })

  it('switches Answer overlays on its own, and gives it no update time (it is drawn from the chat, not a route)', async () => {
    const changes: [LayerId, boolean][] = []
    const { user, dialog } = await open(<Panel onChange={(id, on) => changes.push([id, on])} />)
    await user.click(screen.getByRole('switch', { name: 'Answer overlays' }))
    expect(screen.getByRole('switch', { name: 'Answer overlays' })).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).toBeChecked()
    expect(changes).toEqual([['answer', false]])
    expect(within(dialog).getAllByText(/^updated /)).toHaveLength(5)
  })

  it('toggles each on its own', async () => {
    const changes: [LayerId, boolean][] = []
    const { user } = await open(<Panel onChange={(id, on) => changes.push([id, on])} />)
    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' }))
    expect(screen.getByRole('switch', { name: 'NOTAMs below FL100' })).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).toBeChecked()
    await user.click(screen.getByRole('switch', { name: 'Airspace below FL100' }))
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).not.toBeChecked()
    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' }))
    expect(screen.getByRole('switch', { name: 'NOTAMs below FL100' })).toBeChecked()
    expect(changes).toEqual([['notams', false], ['airspace', false], ['notams', true]])
  })

  it('is operable from the keyboard', async () => {
    const { user } = await open()
    screen.getByRole('switch', { name: 'Airspace below FL100' }).focus()
    await user.keyboard(' ')
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).not.toBeChecked()
  })

  it('puts each all-heights switch right after its partner, and says "all heights" in words (not colour alone)', async () => {
    const { dialog } = await open()
    const names = within(dialog).getAllByRole('switch').map((s) => s.closest('li')?.querySelector('label')?.textContent)
    expect(names).toEqual(['Wind', 'Rain', 'Airspace below FL100', 'Airspace all heights', 'NOTAMs below FL100', 'NOTAMs all heights', 'Answer overlays'])
  })

  it('toggles the two all-heights switches each on its own, leaving their partners as they are', async () => {
    const changes: [LayerId, boolean][] = []
    const { user } = await open(<Panel onChange={(id, on) => changes.push([id, on])} />)
    await user.click(screen.getByRole('switch', { name: 'Airspace all heights' }))
    expect(screen.getByRole('switch', { name: 'Airspace all heights' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'NOTAMs all heights' })).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).toBeChecked()
    await user.click(screen.getByRole('switch', { name: 'NOTAMs all heights' }))
    await user.click(screen.getByRole('switch', { name: 'Airspace all heights' }))
    expect(screen.getByRole('switch', { name: 'Airspace all heights' })).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'NOTAMs all heights' })).toBeChecked()
    expect(changes).toEqual([['airspaceAll', true], ['notamsAll', true], ['airspaceAll', false]])
  })

  it('has no switch for the basemap', async () => {
    const { dialog } = await open()
    expect(within(dialog).queryByRole('switch', { name: /basemap/i })).toBeNull()
  })
})

describe('the legend', () => {
  it('has both sets, with every row, and the amber line', async () => {
    const { dialog } = await open()
    expect(within(dialog).getByRole('heading', { name: 'On the map' })).toBeInTheDocument()
    expect(within(dialog).getByRole('heading', { name: 'For this field' })).toBeInTheDocument()
    for (const row of [...ON_THE_MAP, ...ACTIONS, ...VERDICTS]) expect(within(dialog).getByText(row.label)).toBeInTheDocument()
    expect(within(dialog).getByText('amber = near your limits, check the conditions too')).toBeInTheDocument()
    expect(HOW_IT_WORKS.amberLine).toBe('amber = near your limits, check the conditions too')
  })

  it('lists the four type colours, then NOTAM violet dashed and approx', () => {
    expect(ON_THE_MAP.map((row) => row.label)).toEqual([
      'Restricted or prohibited',
      'Danger',
      'CTR, CTA or ATZ',
      'Other airspace',
      'NOTAM',
      'NOTAM, approx (Q-line circle)',
    ])
    expect(ON_THE_MAP[4].swatch).toContain('border-dashed')
    expect(ON_THE_MAP[5].swatch).toContain('border-dotted')
  })

  it('lists the verdict colours most cautious first', () => {
    expect(VERDICTS.map((row) => row.label)).toEqual(['NO-GO', 'UNVERIFIED', 'APPROACHING LIMITS', 'WITH CONDITIONS', 'PILOT DISCRETION'])
  })

  it('takes every swatch colour from a theme token, never a literal', () => {
    for (const row of [...ON_THE_MAP, ...ACTIONS, ...VERDICTS]) {
      expect(row.swatch, row.label).toMatch(/\b(border|bg)-(type|action|verdict)-[a-z-]+/)
      expect(row.swatch, row.label).not.toMatch(/#[0-9a-f]{3,8}|rgb|hsl|\[/i)
    }
  })

  it('is not part of the status text: no "stale" or "unavailable" while all is well', async () => {
    const { dialog } = await open()
    expect(dialog).not.toHaveTextContent(/stale|unavailable/i)
  })
})

describe('each layer’s line', () => {
  it('shows "updated HH:MM" for wind and both layers', async () => {
    const { dialog } = await open()
    const times = within(dialog).getAllByText(/^updated \d\d:\d\d/)
    expect(times).toHaveLength(5) // wind, and each route's two layers
    const expected = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(UPDATED))
    for (const line of times) expect(line).toHaveTextContent(`updated ${expected}`)
  })

  it('shows "stale" next to the time for a stale layer only', async () => {
    const { dialog } = await open(<Panel airspace={ok()} notams={ok({ stale: true })} />)
    const rows = within(dialog).getAllByRole('listitem').slice(0, 6)
    expect(rows[0]).not.toHaveTextContent('stale') // wind
    expect(rows[1]).not.toHaveTextContent('stale') // rain
    expect(rows[2]).not.toHaveTextContent('stale') // airspace
    expect(rows[3]).not.toHaveTextContent('stale') // airspace all heights
    expect(rows[4]).toHaveTextContent(/updated \d\d:\d\d\s*stale/) // NOTAMs
    expect(rows[5]).toHaveTextContent(/updated \d\d:\d\d\s*stale/) // NOTAMs all heights: the same route, the same words
  })

  it('G6: an all-heights layer shows its route’s own time and stale, with its partner’s text exactly', async () => {
    const { dialog } = await open(<Panel airspace={ok({ stale: true, updatedAt: '2026-11-08T09:05:00Z' })} notams={ok()} />)
    const [below, all] = ['Airspace below FL100', 'Airspace all heights'].map((name) => within(dialog).getByText(name).closest('li') as HTMLElement)
    const line = (row: HTMLElement) => row.querySelector('p')?.textContent
    expect(line(all)).toBe(line(below))
    expect(line(all)).toMatch(/^updated \d\d:\d\dstale$/)
  })

  it('says "<layer> unavailable" for a 503 layer, with no time, and leaves the other alone', async () => {
    const { dialog } = await open(<Panel airspace={unavailable} notams={ok()} />)
    expect(within(dialog).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(within(dialog).queryByText('NOTAMs below FL100 unavailable')).toBeNull()
    expect(within(dialog).getAllByText(/^updated/)).toHaveLength(3) // wind, and NOTAMs with its all-heights partner
  })

  it('G6/E1: one 503 on a route gives one note for both of its layers, and no time on either; the other route’s two layers still show theirs', async () => {
    const { dialog } = await open(<Panel notams={unavailable} />)
    expect(within(dialog).getAllByText(/unavailable/)).toHaveLength(1)
    expect(within(dialog).getByText('NOTAMs below FL100 unavailable')).toBeInTheDocument()
    for (const name of ['NOTAMs below FL100', 'NOTAMs all heights']) {
      expect(within(dialog).getByText(name).closest('li')).not.toHaveTextContent(/updated|stale/)
    }
    expect(within(dialog).getAllByText(/^updated/)).toHaveLength(3) // wind, airspace and airspace all heights
    expect(within(dialog).getByRole('switch', { name: 'NOTAMs all heights' })).toBeInTheDocument()
  })

  it('shows wind’s own time, its own "stale" and its own "Wind unavailable" note, each leaving the other layers alone', async () => {
    const { dialog, view } = await open(<Panel wind={windOk({ stale: true })} />)
    const wind = within(dialog).getAllByRole('listitem')[0]
    expect(wind).toHaveTextContent(/^Wind\s*updated \d\d:\d\d\s*stale/)
    expect(within(dialog).getAllByText('stale')).toHaveLength(1)
    view.unmount()
    const gone = await open(<Panel wind={{ status: 'unavailable', updatedAt: null, stale: false }} />)
    expect(within(gone.dialog).getByText('Wind unavailable')).toBeInTheDocument()
    expect(within(gone.dialog).getAllByText(/^updated/)).toHaveLength(4) // airspace and NOTAMs, and their partners, still show their times
    expect(within(gone.dialog).queryByText('Airspace below FL100 unavailable')).toBeNull()
  })

  it('says both unavailable when both are', async () => {
    const { dialog } = await open(<Panel airspace={unavailable} notams={unavailable} />)
    expect(within(dialog).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(within(dialog).getByText('NOTAMs below FL100 unavailable')).toBeInTheDocument()
  })

  it('says loading before the first answer', async () => {
    const { dialog } = await open(<Panel wind={{ status: 'loading', updatedAt: null, stale: false }} airspace={loading} notams={loading} />)
    expect(within(dialog).getAllByText('Loading…')).toHaveLength(5)
    expect(dialog).not.toHaveTextContent(/unavailable|stale|updated/)
  })

  it('E15: an empty, fresh NOTAM layer shows its time and no "stale"', async () => {
    const { dialog } = await open(<Panel notams={ok({ features: [], stale: false })} />)
    const rows = within(dialog).getAllByRole('listitem').slice(0, 6)
    expect(rows[4]).toHaveTextContent(/updated \d\d:\d\d/) // NOTAMs below FL100
    expect(rows[5]).toHaveTextContent(/updated \d\d:\d\d/) // NOTAMs all heights: empty too, with its time
    expect(dialog).not.toHaveTextContent('stale')
  })
})

describe('the basemap entry', () => {
  const entry = (dialog: HTMLElement) => within(dialog).getByText('Basemap').closest('li') as HTMLElement

  it.each<[string, BasemapStatus | undefined, boolean]>([
    ['not told anything', undefined, false],
    ['loading', 'loading', false],
    ['ok', 'ok', false],
    ['unavailable', 'unavailable', true],
  ])('says "basemap unavailable" only when unavailable (%s)', async (_label, basemap, shows) => {
    const { dialog } = await open(<Panel basemap={basemap} />)
    expect(within(dialog).queryByText('basemap unavailable') !== null).toBe(shows)
    expect(entry(dialog)).toHaveTextContent('CARTO Dark Matter')
    expect(entry(dialog).textContent?.includes('basemap unavailable')).toBe(shows)
  })

  it('has no switch and no "updated" time for the basemap, in any state', async () => {
    const { dialog } = await open(<Panel basemap="unavailable" />)
    expect(within(entry(dialog)).queryByRole('switch')).toBeNull()
    expect(entry(dialog).textContent).not.toMatch(/updated|stale|loading/i)
    expect(within(dialog).getAllByRole('switch')).toHaveLength(7)
    expect(within(dialog).getAllByText(/^updated /)).toHaveLength(5)
  })

  it('changes nothing about the other layers: their switches, legend and stale/503 notes read the same', async () => {
    const states = { airspace: unavailable, notams: ok({ stale: true }) }
    const read = async (basemap: BasemapStatus) => {
      const { dialog, view } = await open(<Panel {...states} basemap={basemap} />)
      const text = [...dialog.querySelectorAll('li')].filter((li) => li !== entry(dialog)).map((li) => li.textContent)
      const switches = within(dialog).getAllByRole('switch').map((s) => s.getAttribute('aria-checked'))
      const legend = dialog.querySelector('hr')?.nextElementSibling?.textContent
      view.unmount()
      return { text, switches, legend }
    }
    const fine = await read('ok')
    const failed = await read('unavailable')
    expect(failed).toEqual(fine)
    expect(fine.text.join(' ')).toContain('Airspace below FL100 unavailable')
    expect(fine.text.join(' ')).toContain('stale')
    expect(fine.legend).toContain('On the map')
  })
})

describe('the panel and browser storage', () => {
  it('reads and writes no browser storage itself: it is controlled, and the saving is the page’s (src/lib/useLayerPrefs.ts)', async () => {
    const calls = [
      vi.spyOn(Storage.prototype, 'getItem'),
      vi.spyOn(Storage.prototype, 'setItem'),
      vi.spyOn(Storage.prototype, 'removeItem'),
      vi.spyOn(Storage.prototype, 'clear'),
      vi.spyOn(Storage.prototype, 'key'),
    ]
    const { user } = await open()
    await user.click(screen.getByRole('switch', { name: 'Airspace below FL100' }))
    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' }))
    for (const spy of calls) expect(spy).not.toHaveBeenCalled()
  })
})

describe('E14: a narrow screen', () => {
  it('limits the popover to the viewport width less 1 rem and the room it has, and scrolls inside', async () => {
    const { dialog } = await open()
    expect(dialog.className).toContain('max-w-[calc(100vw-1rem)]')
    expect(dialog.className).toContain('max-h-[var(--radix-popover-content-available-height)]')
    expect(dialog.className).toContain('overflow-y-auto')
    expect(dialog.className).toContain('overflow-x-hidden')
    expect(dialog.className).toContain('z-[1100]') // above Leaflet's controls, which go up to 1000
  })

  it('E14: at 390 px the two extra switches are rows like the others: text in a shrinkable column, the switch beside it, nothing that sets a width', async () => {
    vi.stubGlobal('innerWidth', 390)
    const { dialog } = await open()
    for (const name of ['Airspace all heights', 'NOTAMs all heights']) {
      const row = within(dialog).getByText(name).closest('li') as HTMLElement
      expect(row).toHaveClass('flex', 'justify-between')
      expect(within(dialog).getByText(name).parentElement).toHaveClass('min-w-0') // the label wraps instead of pushing the switch out
      expect(row.className).not.toMatch(/\b(w|min-w)-\[|whitespace-nowrap/)
      expect(within(row).getByRole('switch')).toBeVisible()
    }
    vi.unstubAllGlobals()
  })

  it('keeps both switches reachable inside the popover and the button on the map’s top right', async () => {
    const { dialog, view } = await open()
    for (const name of ['Airspace below FL100', 'Airspace all heights', 'NOTAMs below FL100', 'NOTAMs all heights']) {
      expect(within(dialog).getByRole('switch', { name })).toBeVisible()
    }
    expect(view.container.firstElementChild).toHaveClass('absolute', 'top-3', 'right-3')
  })
})

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ACTIONS, ON_THE_MAP, VERDICTS } from '@/components/Legend'
import { LAYERS, LayerPanel, type LayerId, type LayerSwitches } from '@/components/LayerPanel'
import { HOW_IT_WORKS } from '@/content/howItWorks'
import type { BasemapStatus } from '@/map/basemap'
import type { LayerState, LayerStatus } from '@/map/useMapLayers'

const UPDATED = '2026-11-08T14:32:00Z'
const ok = (over: Partial<LayerState<unknown>> = {}): LayerState<unknown> => ({ status: 'ok', features: [], updatedAt: UPDATED, stale: false, ...over })
const loading: LayerState<unknown> = { status: 'loading', features: [], updatedAt: null, stale: false }
const unavailable: LayerState<unknown> = { status: 'unavailable', features: [], updatedAt: null, stale: false }
const windOk = (over: Partial<LayerStatus> = {}): LayerStatus => ({ status: 'ok', updatedAt: UPDATED, stale: false, ...over })

/** The panel with its switches held in state, as App holds them. */
function Panel({
  wind = windOk(),
  airspace = ok(),
  notams = ok(),
  basemap,
  onChange,
}: {
  wind?: LayerStatus
  airspace?: LayerState<unknown>
  notams?: LayerState<unknown>
  basemap?: BasemapStatus
  onChange?: (id: LayerId, on: boolean) => void
}) {
  const [shown, setShown] = useState<LayerSwitches>({ wind: true, airspace: true, notams: true, answer: true })
  return (
    <LayerPanel
      states={{ wind, airspace, notams }}
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
  it('has Wind, the two base layers and the Answer overlays switch, all on by default', async () => {
    await open()
    const switches = screen.getAllByRole('switch')
    expect(switches.map((s) => s.getAttribute('aria-checked'))).toEqual(['true', 'true', 'true', 'true'])
    expect(screen.getByRole('switch', { name: 'Wind' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'NOTAMs below FL100' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Answer overlays' })).toBeChecked()
    expect(LAYERS.map((layer) => layer.id)).toEqual(['wind', 'airspace', 'notams', 'answer'])
  })

  it('switches Answer overlays on its own, and gives it no update time (it is drawn from the chat, not a route)', async () => {
    const changes: [LayerId, boolean][] = []
    const { user, dialog } = await open(<Panel onChange={(id, on) => changes.push([id, on])} />)
    await user.click(screen.getByRole('switch', { name: 'Answer overlays' }))
    expect(screen.getByRole('switch', { name: 'Answer overlays' })).not.toBeChecked()
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).toBeChecked()
    expect(changes).toEqual([['answer', false]])
    expect(within(dialog).getAllByText(/^updated /)).toHaveLength(3)
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

  it('has no switch for rain or all heights, and none for the basemap', async () => {
    const { dialog } = await open()
    expect(within(dialog).queryByText(/rain|radar|all heights/i)).toBeNull()
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
    expect(times).toHaveLength(3)
    const expected = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(UPDATED))
    for (const line of times) expect(line).toHaveTextContent(`updated ${expected}`)
  })

  it('shows "stale" next to the time for a stale layer only', async () => {
    const { dialog } = await open(<Panel airspace={ok()} notams={ok({ stale: true })} />)
    const rows = within(dialog).getAllByRole('listitem').slice(0, 3)
    expect(rows[0]).not.toHaveTextContent('stale') // wind
    expect(rows[1]).not.toHaveTextContent('stale') // airspace
    expect(rows[2]).toHaveTextContent(/updated \d\d:\d\d\s*stale/) // NOTAMs
  })

  it('says "<layer> unavailable" for a 503 layer, with no time, and leaves the other alone', async () => {
    const { dialog } = await open(<Panel airspace={unavailable} notams={ok()} />)
    expect(within(dialog).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(within(dialog).queryByText('NOTAMs below FL100 unavailable')).toBeNull()
    expect(within(dialog).getAllByText(/^updated/)).toHaveLength(2) // wind and NOTAMs
  })

  it('shows wind’s own time, its own "stale" and its own "Wind unavailable" note, each leaving the other layers alone', async () => {
    const { dialog, view } = await open(<Panel wind={windOk({ stale: true })} />)
    const wind = within(dialog).getAllByRole('listitem')[0]
    expect(wind).toHaveTextContent(/^Wind\s*updated \d\d:\d\d\s*stale/)
    expect(within(dialog).getAllByText('stale')).toHaveLength(1)
    view.unmount()
    const gone = await open(<Panel wind={{ status: 'unavailable', updatedAt: null, stale: false }} />)
    expect(within(gone.dialog).getByText('Wind unavailable')).toBeInTheDocument()
    expect(within(gone.dialog).getAllByText(/^updated/)).toHaveLength(2) // airspace and NOTAMs still show their times
    expect(within(gone.dialog).queryByText('Airspace below FL100 unavailable')).toBeNull()
  })

  it('says both unavailable when both are', async () => {
    const { dialog } = await open(<Panel airspace={unavailable} notams={unavailable} />)
    expect(within(dialog).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(within(dialog).getByText('NOTAMs below FL100 unavailable')).toBeInTheDocument()
  })

  it('says loading before the first answer', async () => {
    const { dialog } = await open(<Panel wind={{ status: 'loading', updatedAt: null, stale: false }} airspace={loading} notams={loading} />)
    expect(within(dialog).getAllByText('Loading…')).toHaveLength(3)
    expect(dialog).not.toHaveTextContent(/unavailable|stale|updated/)
  })

  it('E15: an empty, fresh NOTAM layer shows its time and no "stale"', async () => {
    const { dialog } = await open(<Panel notams={ok({ features: [], stale: false })} />)
    const rows = within(dialog).getAllByRole('listitem').slice(0, 3)
    expect(rows[2]).toHaveTextContent(/updated \d\d:\d\d/)
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
    expect(within(dialog).getAllByRole('switch')).toHaveLength(4)
    expect(within(dialog).getAllByText(/^updated /)).toHaveLength(3)
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

  it('keeps both switches reachable inside the popover and the button on the map’s top right', async () => {
    const { dialog, view } = await open()
    for (const name of ['Airspace below FL100', 'NOTAMs below FL100']) {
      expect(within(dialog).getByRole('switch', { name })).toBeVisible()
    }
    expect(view.container.firstElementChild).toHaveClass('absolute', 'top-3', 'right-3')
  })
})

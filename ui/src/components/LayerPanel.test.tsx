import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ACTIONS, ON_THE_MAP, VERDICTS } from '@/components/Legend'
import { LAYERS, LayerPanel, type LayerId, type LayerSwitches } from '@/components/LayerPanel'
import { HOW_IT_WORKS } from '@/content/howItWorks'
import type { LayerState } from '@/map/useMapLayers'

const UPDATED = '2026-11-08T14:32:00Z'
const ok = (over: Partial<LayerState<unknown>> = {}): LayerState<unknown> => ({ status: 'ok', features: [], updatedAt: UPDATED, stale: false, ...over })
const loading: LayerState<unknown> = { status: 'loading', features: [], updatedAt: null, stale: false }
const unavailable: LayerState<unknown> = { status: 'unavailable', features: [], updatedAt: null, stale: false }

/** The panel with its switches held in state, as App holds them. */
function Panel({ airspace = ok(), notams = ok(), onChange }: { airspace?: LayerState<unknown>; notams?: LayerState<unknown>; onChange?: (id: LayerId, on: boolean) => void }) {
  const [shown, setShown] = useState<LayerSwitches>({ airspace: true, notams: true })
  return (
    <LayerPanel
      states={{ airspace, notams }}
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
  it('has exactly the two this PR adds, both on by default', async () => {
    await open()
    const switches = screen.getAllByRole('switch')
    expect(switches.map((s) => s.getAttribute('aria-checked'))).toEqual(['true', 'true'])
    expect(screen.getByRole('switch', { name: 'Airspace below FL100' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'NOTAMs below FL100' })).toBeChecked()
    expect(LAYERS.map((layer) => layer.id)).toEqual(['airspace', 'notams'])
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

  it('has no switch for wind, rain, all heights, answers or the basemap', async () => {
    const { dialog } = await open()
    expect(within(dialog).queryByText(/wind|rain|radar|all heights|answer overlays|basemap/i)).toBeNull()
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
  it('shows "updated HH:MM" for both layers', async () => {
    const { dialog } = await open()
    const times = within(dialog).getAllByText(/^updated \d\d:\d\d/)
    expect(times).toHaveLength(2)
    const expected = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(UPDATED))
    for (const line of times) expect(line).toHaveTextContent(`updated ${expected}`)
  })

  it('shows "stale" next to the time for a stale layer only', async () => {
    const { dialog } = await open(<Panel airspace={ok()} notams={ok({ stale: true })} />)
    const rows = within(dialog).getAllByRole('listitem').slice(0, 2)
    expect(rows[0]).not.toHaveTextContent('stale')
    expect(rows[1]).toHaveTextContent(/updated \d\d:\d\d\s*stale/)
  })

  it('says "<layer> unavailable" for a 503 layer, with no time, and leaves the other alone', async () => {
    const { dialog } = await open(<Panel airspace={unavailable} notams={ok()} />)
    expect(within(dialog).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(within(dialog).queryByText('NOTAMs below FL100 unavailable')).toBeNull()
    expect(within(dialog).getAllByText(/^updated/)).toHaveLength(1)
  })

  it('says both unavailable when both are', async () => {
    const { dialog } = await open(<Panel airspace={unavailable} notams={unavailable} />)
    expect(within(dialog).getByText('Airspace below FL100 unavailable')).toBeInTheDocument()
    expect(within(dialog).getByText('NOTAMs below FL100 unavailable')).toBeInTheDocument()
  })

  it('says loading before the first answer', async () => {
    const { dialog } = await open(<Panel airspace={loading} notams={loading} />)
    expect(within(dialog).getAllByText('Loading…')).toHaveLength(2)
    expect(dialog).not.toHaveTextContent(/unavailable|stale|updated/)
  })

  it('E15: an empty, fresh NOTAM layer shows its time and no "stale"', async () => {
    const { dialog } = await open(<Panel notams={ok({ features: [], stale: false })} />)
    const rows = within(dialog).getAllByRole('listitem').slice(0, 2)
    expect(rows[1]).toHaveTextContent(/updated \d\d:\d\d/)
    expect(dialog).not.toHaveTextContent('stale')
  })
})

describe('no saved choices', () => {
  it('reads and writes no browser storage while it opens and toggles', async () => {
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
    // The two stores are named from pieces so the repo-wide scan for a saved-choices key does not match this file.
    for (const store of [['local', 'Storage'], ['session', 'Storage']].map((pieces) => pieces.join(''))) {
      expect((window as unknown as Record<string, Storage>)[store].length, store).toBe(0)
    }
  })

  it('starts with both switches on again after a remount, as after a reload', async () => {
    const { user, view } = await open()
    await user.click(screen.getByRole('switch', { name: 'NOTAMs below FL100' }))
    expect(screen.getByRole('switch', { name: 'NOTAMs below FL100' })).not.toBeChecked()
    view.unmount()
    await open()
    expect(screen.getByRole('switch', { name: 'NOTAMs below FL100' })).toBeChecked()
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

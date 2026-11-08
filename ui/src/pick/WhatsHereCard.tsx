import { ChevronDown, ChevronRight } from 'lucide-react'
import { useId, useMemo, useState } from 'react'

import { APPROX_LABEL } from '@/map/BaseLayers'
import type { AirspaceProps, NotamProps } from '@/map/types'
import type { LayerState } from '@/map/useMapLayers'
import { splitAtFl100 } from '@/pick/floor'
import { whatsHere, type HereRow, type LatLon } from '@/pick/whatsHere'

// The card's fixed words (G10).
export const CARD_TITLE = 'Over this point'
export const CARD_FOOTNOTE = 'The chat checks everything within 2 NM.'
export const CARD_EMPTY = 'Nothing in the loaded layers over this point.'

const LAYER_NAMES = { airspace: 'Airspace', notams: 'NOTAMs' } as const

function Chip({ row }: { row: HereRow }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-sm border-[length:var(--border-width)] px-1.5 py-0.5 text-[0.6875rem] leading-none font-semibold tracking-wide" style={{ borderColor: `var(${row.token})` }}>
      <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: `var(${row.token})` }} aria-hidden="true" />
      {row.chip}
    </span>
  )
}

/** One item over the point: a button showing the chip, the name or id and the limits; open, it adds the active times and a NOTAM's raw text. */
function Row({ row }: { row: HereRow }) {
  const [open, setOpen] = useState(false)
  const detailsId = useId()
  const Chevron = open ? ChevronDown : ChevronRight
  return (
    <li className="border-b border-border last:border-b-0">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={detailsId}
        onClick={() => setOpen((was) => !was)}
        className="flex w-full items-start gap-2 rounded-sm px-1 py-2 text-left hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
      >
        <Chevron className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <Chip row={row} />
            {row.approx && <span className="text-xs font-medium text-type-notam">{APPROX_LABEL}</span>}
          </span>
          <span className="mt-1 block font-medium [overflow-wrap:anywhere]">{row.name}</span>
          <span className="block text-xs text-muted-foreground [overflow-wrap:anywhere]">{row.limits}</span>
        </span>
      </button>
      {open && (
        <dl id={detailsId} className="mb-2 ml-7 space-y-1.5 text-xs">
          <div>
            <dt className="text-muted-foreground">Active</dt>
            <dd className="[overflow-wrap:anywhere]">{row.times}</dd>
          </div>
          {row.text !== null && (
            <div>
              <dt className="text-muted-foreground">NOTAM text</dt>
              <dd className="font-mono whitespace-pre-wrap [overflow-wrap:anywhere]">{row.text}</dd>
            </div>
          )}
        </dl>
      )}
    </li>
  )
}

function RowList({ rows }: { rows: readonly HereRow[] }) {
  return (
    <ul>
      {rows.map((row) => (
        <Row key={row.key} row={row} />
      ))}
    </ul>
  )
}

/** "Above FL100 (n)": collapsed until clicked. */
function AboveGroup({ rows }: { rows: readonly HereRow[] }) {
  const [open, setOpen] = useState(false)
  const listId = useId()
  const Chevron = open ? ChevronDown : ChevronRight
  return (
    <div className="mt-2 border-t border-border pt-1">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={listId}
        onClick={() => setOpen((was) => !was)}
        className="flex w-full items-center gap-2 rounded-sm px-1 py-2 text-left text-sm font-medium hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
      >
        <Chevron className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
        Above FL100 ({rows.length})
      </button>
      {open && (
        <div id={listId}>
          <RowList rows={rows} />
        </div>
      )}
    </div>
  )
}

export interface WhatsHereCardProps {
  point: LatLon
  airspace: LayerState<AirspaceProps>
  notams: LayerState<NotamProps>
}

/**
 * The "What's here" card for one picked point: everything loaded whose drawn shape contains it, at any height, whether or not its
 * layer is switched on. Rows are sorted by floor; those at or above FL100 sit in a collapsed group. A layer still loading is named
 * ("Still loading: ...") and a layer that answered 503 says "<layer> unavailable"; the card fills in when the data arrives.
 * A layer with `stale: true` is listed like any other: the layer panel owns the stale note.
 */
export function WhatsHereCard({ point, airspace, notams }: WhatsHereCardProps) {
  const rows = useMemo(
    () => splitAtFl100(whatsHere(point, { airspace: airspace.features, notams: notams.features })),
    [point, airspace.features, notams.features],
  )
  const states = [
    ['airspace', airspace],
    ['notams', notams],
  ] as const
  const loading = states.filter(([, state]) => state.status === 'loading').map(([id]) => LAYER_NAMES[id])
  const unavailable = states.filter(([, state]) => state.status === 'unavailable').map(([id]) => LAYER_NAMES[id])
  const anyLoaded = states.some(([, state]) => state.status === 'ok')
  const nothing = rows.below.length === 0 && rows.above.length === 0

  return (
    <section aria-label="What's here" className="text-sm text-popover-foreground">
      <h2 className="mb-1 text-base font-semibold">{CARD_TITLE}</h2>
      <div className="max-h-[min(20rem,50dvh)] overflow-x-hidden overflow-y-auto">
        {loading.length > 0 && <p className="py-1 text-xs text-muted-foreground">Still loading: {loading.join(', ')}</p>}
        {unavailable.map((name) => (
          <p key={name} className="py-1 text-xs text-destructive">
            {name} unavailable
          </p>
        ))}
        {nothing && anyLoaded && <p className="py-2 text-muted-foreground">{CARD_EMPTY}</p>}
        <RowList rows={rows.below} />
        {rows.above.length > 0 && <AboveGroup rows={rows.above} />}
      </div>
      <p className="mt-2 border-t border-border pt-2 text-xs text-muted-foreground">{CARD_FOOTNOTE}</p>
    </section>
  )
}

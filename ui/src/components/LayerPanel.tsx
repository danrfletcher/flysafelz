import { Layers } from 'lucide-react'

import { Legend } from '@/components/Legend'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { formatUpdated, type LayerState } from '@/map/useMapLayers'

export type LayerId = 'airspace' | 'notams'

/** The two switches this PR adds, in panel order. */
export const LAYERS: readonly { id: LayerId; name: string }[] = [
  { id: 'airspace', name: 'Airspace below FL100' },
  { id: 'notams', name: 'NOTAMs below FL100' },
]

export type LayerSwitches = Record<LayerId, boolean>

export interface LayerPanelProps {
  states: Record<LayerId, LayerState<unknown>>
  shown: LayerSwitches
  onShownChange: (id: LayerId, on: boolean) => void
  /** The chat sheet is open: from 768 px up the button moves left to sit beside the sheet instead of under it. */
  chatOpen?: boolean
}

/** The line under a switch: loading, "updated HH:MM" (with "stale" when the route says so or a refetch failed), or "<layer> unavailable". */
function Status({ name, state }: { name: string; state: LayerState<unknown> }) {
  if (state.status === 'loading') return <p className="text-xs text-muted-foreground">Loading…</p>
  if (state.status === 'unavailable') return <p className="text-xs text-destructive">{name} unavailable</p>
  return (
    <p className="text-xs text-muted-foreground">
      updated {state.updatedAt ? formatUpdated(state.updatedAt) : ''}
      {state.stale && <span className="ml-2 rounded-sm bg-muted px-1.5 py-0.5 font-medium text-verdict-amber">stale</span>}
    </p>
  )
}

/**
 * The layer panel: a button at the top right of the map that opens a popover with one switch per layer, each layer's update time
 * and state, and the legend. The popover is never wider than the viewport, and scrolls inside when it is taller than the room it has.
 */
export function LayerPanel({ states, shown, onShownChange, chatOpen = false }: LayerPanelProps) {
  return (
    <div className={`absolute top-3 right-3 z-[1000] ${chatOpen ? 'md:right-[27rem]' : ''}`}>
      <Popover>
        <PopoverTrigger
          aria-label="Map layers"
          className="inline-flex h-10 items-center gap-2 rounded-lg border-[length:var(--border-width)] bg-card px-3 text-sm font-medium text-card-foreground shadow-panel hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <Layers className="size-4" aria-hidden="true" />
          Layers
        </PopoverTrigger>
        <PopoverContent aria-label="Map layers">
          <ul className="space-y-3">
            {LAYERS.map(({ id, name }) => (
              <li key={id} className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <label htmlFor={`layer-${id}`} className="block text-sm font-medium">
                    {name}
                  </label>
                  <Status name={name} state={states[id]} />
                </div>
                <Switch id={`layer-${id}`} checked={shown[id]} onCheckedChange={(on) => onShownChange(id, on)} className="mt-0.5" />
              </li>
            ))}
          </ul>
          <hr className="my-4 border-border" />
          <Legend />
        </PopoverContent>
      </Popover>
    </div>
  )
}

import { Layers } from 'lucide-react'

import { Legend } from '@/components/Legend'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import type { BasemapStatus } from '@/map/basemap'
import { formatFrameTime, useRainView, type RainView } from '@/map/rainView'
import type { RainState } from '@/map/useRain'
import { formatUpdated, type LayerStatus } from '@/map/useMapLayers'

/** The layers that come from a map route, and so have an update time and a state. */
export type ServerLayerId = 'wind' | 'airspace' | 'notams'
export type LayerId = ServerLayerId | 'rain' | 'answer'

/**
 * The switches, in panel order: wind, rain, the two base layers, then the latest answer's highlights (drawn from the chat, not from a route).
 * Rain has its own row text (its frame time, not a route's update time) and is not in `states`.
 * This is also the registry of saved choices (src/lib/layerPrefs.ts): `defaultOn` is a layer's state when nothing is stored.
 */
export const LAYERS: readonly { id: LayerId; name: string; defaultOn: boolean }[] = [
  { id: 'wind', name: 'Wind', defaultOn: true },
  { id: 'rain', name: 'Rain', defaultOn: true },
  { id: 'airspace', name: 'Airspace below FL100', defaultOn: true },
  { id: 'notams', name: 'NOTAMs below FL100', defaultOn: true },
  { id: 'answer', name: 'Answer overlays', defaultOn: true },
]

export type LayerSwitches = Record<LayerId, boolean>

export interface LayerPanelProps {
  states: Record<ServerLayerId, LayerStatus>
  /** The Rain layer: where its data came from, and what the map shows of it (the frame on screen, whether the zoom hides it). */
  rain: { state: RainState; view: RainView }
  shown: LayerSwitches
  onShownChange: (id: LayerId, on: boolean) => void
  /** Whether the map's tiles are loading; the panel adds a note under the basemap entry only when they are unavailable. */
  basemap?: BasemapStatus
  /** The chat sheet is open: from 768 px up the button moves left to sit beside the sheet instead of under it. */
  chatOpen?: boolean
}

/** The line under a switch: loading, "updated HH:MM" (with "stale" when the route says so or a refetch failed), or "<layer> unavailable". */
function Status({ name, state }: { name: string; state: LayerStatus }) {
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
 * The Rain row's line. Radar: the time of the frame on screen as "frame HH:MM" (the latest frame while the layer is off), never a route's update time.
 * Fallback: the grid's "updated HH:MM", and "stale" when the route says so. Both: a note while the zoom has faded the layer out. Both sources failed: "Rain unavailable".
 */
function RainStatus({ rain }: { rain: LayerPanelProps['rain'] }) {
  const { frameTime, hidden } = useRainView(rain.view)
  const { state } = rain
  if (state.mode === 'loading') return <p className="text-xs text-muted-foreground">Loading…</p>
  if (state.mode === 'unavailable') return <p className="text-xs text-destructive">Rain unavailable</p>
  if (state.mode === 'radar') {
    const shown = frameTime ?? state.frames[state.frames.length - 1].time
    return (
      <>
        <p className="text-xs text-muted-foreground">frame {formatFrameTime(shown)}</p>
        {hidden && <p className="text-xs text-muted-foreground">radar: regional view only</p>}
      </>
    )
  }
  return (
    <>
      <p className="text-xs text-muted-foreground">
        updated {formatUpdated(state.updatedAt)}
        {state.stale && <span className="ml-2 rounded-sm bg-muted px-1.5 py-0.5 font-medium text-verdict-amber">stale</span>}
      </p>
      {hidden && <p className="text-xs text-muted-foreground">rain (model): regional view only</p>}
    </>
  )
}

/**
 * The layer panel: a button at the top right of the map that opens a popover with one switch per layer, each route layer's update time
 * and state, and the legend. The popover is never wider than the viewport, and scrolls inside when it is taller than the room it has.
 */
export function LayerPanel({ states, rain, shown, onShownChange, basemap = 'loading', chatOpen = false }: LayerPanelProps) {
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
                  {id === 'answer' ? (
                    <p className="text-xs text-muted-foreground">The latest answer: the 2 NM circle and the NOTAMs and airspace it names</p>
                  ) : id === 'rain' ? (
                    <RainStatus rain={rain} />
                  ) : (
                    <Status name={name} state={states[id]} />
                  )}
                </div>
                <Switch id={`layer-${id}`} checked={shown[id]} onCheckedChange={(on) => onShownChange(id, on)} className="mt-0.5" />
              </li>
            ))}
            {/* The map under the layers: no switch and no update time; a note only when its tiles fail. */}
            <li>
              <p className="text-sm font-medium">Basemap</p>
              <p className="text-xs text-muted-foreground">CARTO Dark Matter</p>
              {basemap === 'unavailable' && <p className="text-xs text-destructive">basemap unavailable</p>}
            </li>
          </ul>
          <hr className="my-4 border-border" />
          <Legend rain={rain.state.mode} />
        </PopoverContent>
      </Popover>
    </div>
  )
}

import L from 'leaflet'
import { useEffect, useRef } from 'react'

/** The words of the toast for a click on sea or outside the UK (G9). */
export const NOT_LAND_MESSAGE = 'Pick a field on land in the UK.'

const SHOWN_MS = 4000

/**
 * A short message at the bottom of the map. It disappears after a few seconds, or is replaced by the next one (give each message its
 * own `key`). Clicks on it never reach the map, so touching it cannot move the pin.
 */
export function Toast({ message, onDone }: { message: string; onDone: () => void }) {
  const element = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (element.current) L.DomEvent.disableClickPropagation(element.current)
  }, [])

  useEffect(() => {
    const timer = setTimeout(onDone, SHOWN_MS)
    return () => clearTimeout(timer)
  }, [onDone])

  return (
    <div
      ref={element}
      role="status"
      className="absolute bottom-8 left-1/2 z-[1000] w-max max-w-[calc(100%-2rem)] -translate-x-1/2 rounded-lg border-[length:var(--border-width)] bg-card px-4 py-2.5 text-sm text-card-foreground shadow-panel"
    >
      {message}
    </div>
  )
}

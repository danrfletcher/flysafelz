import { formatFrameTime, useRainView, type RainView } from '@/map/rainView'

/** The radar's frame time, small, at the map's bottom left, in the browser's time zone. Shown only while a frame is on screen and the layer has not faded out. */
export function RainStamp({ view }: { view: RainView }) {
  const { frameTime, hidden } = useRainView(view)
  if (frameTime === null || hidden) return null
  return (
    <p
      aria-label="Radar frame time"
      className="pointer-events-none absolute bottom-3 left-3 z-[1000] rounded-md border-[length:var(--border-width)] bg-card/85 px-2 py-1 font-mono text-xs text-card-foreground shadow-panel"
    >
      Radar {formatFrameTime(frameTime)}
    </p>
  )
}

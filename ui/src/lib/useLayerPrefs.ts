import { useCallback, useRef, useState } from 'react'

import { loadLayerPrefs, saveLayerPrefs, type LayerRegistry } from '@/lib/layerPrefs'

/**
 * The layer switches: read from the browser's storage once at page load (the registry's defaults where nothing usable is stored),
 * and written back on every change. A switch changes in memory whether or not the write works.
 */
export function useLayerPrefs<K extends string>(registry: LayerRegistry<K>) {
  const [shown, setShown] = useState(() => loadLayerPrefs(registry))
  // The latest state, so two changes in one tick (a click and the NOTAM link's own switch-on) each build on the one before.
  const latest = useRef(shown)

  const setLayer = useCallback((id: K, on: boolean) => {
    if (latest.current[id] === on) return
    latest.current = { ...latest.current, [id]: on }
    setShown(latest.current)
    saveLayerPrefs(latest.current)
  }, [])

  return [shown, setLayer] as const
}

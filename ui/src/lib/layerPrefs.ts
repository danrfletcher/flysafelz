/** The one place that touches browser storage: the layer switches, saved so that a reload keeps them (G1, E13). */
export const LAYER_PREFS_KEY = 'flysafelz.layers.v1'

/** More text than a real value could be (a handful of layer ids and booleans): not parsed, read as unreadable. */
const MAX_STORED_CHARS = 10_000

/** The layers that exist, each with the state it has when nothing is stored. Each later layer's PR adds its own entry here. */
export type LayerRegistry<K extends string = string> = readonly { id: K; defaultOn: boolean }[]

// One fixed line, once per page load: what failed is never said and no stored text is ever logged (F2).
let warned = false
function warnOnce() {
  if (warned) return
  warned = true
  console.warn('Layer choices could not be read or saved in this browser; using the defaults for this session.')
}

/** The storage, or null when the browser refuses it: even reading `window.localStorage` can throw a SecurityError. */
function storage(): Storage | null {
  try {
    return window.localStorage ?? null
  } catch {
    warnOnce()
    return null
  }
}

/** The stored choices that are booleans, by layer id: {} for a missing value, unparsable text, or anything that is not a plain JSON object. */
function readStored(store: Storage): Record<string, boolean> {
  try {
    const raw = store.getItem(LAYER_PREFS_KEY)
    if (raw === null || raw.length > MAX_STORED_CHARS) return {}
    const value: unknown = JSON.parse(raw)
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return {}
    return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean'))
  } catch {
    warnOnce()
    return {}
  }
}

/**
 * The switches at page load: each registered layer's default, replaced only by a stored value that is a boolean under that layer's id.
 * Unknown keys, other value types, and storage that is blocked, throws or holds anything unreadable leave the defaults. Never throws.
 */
export function loadLayerPrefs<K extends string>(registry: LayerRegistry<K>): Record<K, boolean> {
  const defaults = Object.fromEntries(registry.map(({ id, defaultOn }) => [id, defaultOn])) as Record<K, boolean>
  const store = storage()
  if (!store) return defaults
  const stored = readStored(store)
  const merged = { ...defaults }
  for (const { id } of registry) {
    if (Object.hasOwn(stored, id)) merged[id] = stored[id]
  }
  return merged
}

/**
 * Saves the switches. A boolean stored under an id that is not in `state` (a layer whose PR has not landed yet) is kept, so its choice
 * still applies once that layer exists; anything else already stored is replaced. A failed write is swallowed: the page's own state
 * has already changed and nothing is shown. Never throws.
 */
export function saveLayerPrefs(state: Readonly<Record<string, boolean>>): void {
  const store = storage()
  if (!store) return
  try {
    const kept = Object.entries(readStored(store)).filter(([id]) => !Object.hasOwn(state, id))
    store.setItem(LAYER_PREFS_KEY, JSON.stringify({ ...Object.fromEntries(kept), ...state }))
  } catch {
    warnOnce()
  }
}

/** For tests: lets the one-time warning fire again. */
export function resetLayerPrefsWarning() {
  warned = false
}

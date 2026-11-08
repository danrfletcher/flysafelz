import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LAYER_PREFS_KEY, loadLayerPrefs, resetLayerPrefsWarning, saveLayerPrefs, type LayerRegistry } from '@/lib/layerPrefs'

// The three real layers (all on by default) and one made-up id, standing in for a layer whose PR has not landed.
const REGISTRY: LayerRegistry = [
  { id: 'airspace', defaultOn: true },
  { id: 'notams', defaultOn: true },
  { id: 'answer', defaultOn: true },
  { id: 'later_layer', defaultOn: false },
]
const DEFAULTS = { airspace: true, notams: true, answer: true, later_layer: false }

const seed = (value: string) => window.localStorage.setItem(LAYER_PREFS_KEY, value)
const stored = () => JSON.parse(window.localStorage.getItem(LAYER_PREFS_KEY) as string) as Record<string, unknown>

let warn: ReturnType<typeof vi.spyOn>
beforeEach(() => {
  resetLayerPrefsWarning()
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
})
afterEach(() => {
  vi.restoreAllMocks()
})

describe('loadLayerPrefs: the read rule', () => {
  const rows: [string, string | null, Record<string, boolean>][] = [
    ['missing (first visit)', null, DEFAULTS],
    ['an empty object', '{}', DEFAULTS],
    ['valid and full', '{"airspace":false,"notams":false,"answer":false,"later_layer":true}', { airspace: false, notams: false, answer: false, later_layer: true }],
    ['valid and partial', '{"notams":false}', { ...DEFAULTS, notams: false }],
    ['an unknown key beside a known one', '{"wind":false,"answer":false}', { ...DEFAULTS, answer: false }],
    ['a non-boolean value beside a good one', '{"airspace":"yes","notams":false}', { ...DEFAULTS, notams: false }],
    ['numbers, null and objects as values', '{"airspace":0,"notams":null,"answer":{"on":false}}', DEFAULTS],
    ['null', 'null', DEFAULTS],
    ['an array', '[]', DEFAULTS],
    ['an array of booleans', '[false,false,false]', DEFAULTS],
    ['a string', '"true"', DEFAULTS],
    ['a number', '42', DEFAULTS],
    ['a boolean', 'false', DEFAULTS],
    ['unparsable text', '{not json', DEFAULTS],
    ['an empty string', '', DEFAULTS],
    ['an inherited-looking key', '{"__proto__":false,"constructor":false,"toString":false}', DEFAULTS],
    ['an oversized string', 'x'.repeat(100_000), DEFAULTS],
    ['an oversized, otherwise valid object', `{"airspace":false,"junk":"${'y'.repeat(100_000)}"}`, DEFAULTS],
    ['emoji keys', '{"🛩️":false,"✈️✈️":true,"airspace":false}', { ...DEFAULTS, airspace: false }],
  ]

  it.each(rows)('%s', (_name, value, expected) => {
    if (value !== null) seed(value)
    expect(loadLayerPrefs(REGISTRY)).toEqual(expected)
  })

  it('reads a missing value without writing anything', () => {
    const set = vi.spyOn(Storage.prototype, 'setItem')
    loadLayerPrefs(REGISTRY)
    expect(set).not.toHaveBeenCalled()
    expect(window.localStorage.length).toBe(0)
  })

  it('reads 100 KB of unicode junk in under 50 ms and returns the defaults', () => {
    seed('🛩️'.repeat(50_000))
    const start = performance.now()
    const result = loadLayerPrefs(REGISTRY)
    expect(performance.now() - start).toBeLessThan(50)
    expect(result).toEqual(DEFAULTS)
  })

  it('never reads a different key, such as a later v2', () => {
    window.localStorage.setItem('flysafelz.layers.v2', '{"airspace":false}')
    window.localStorage.setItem('flysafelz.layers', '{"airspace":false}')
    expect(loadLayerPrefs(REGISTRY)).toEqual(DEFAULTS)
  })
})

describe('saveLayerPrefs', () => {
  it('writes the state as JSON under the v1 key', () => {
    saveLayerPrefs({ airspace: false, notams: true, answer: true })
    expect(LAYER_PREFS_KEY).toBe('flysafelz.layers.v1')
    expect(stored()).toEqual({ airspace: false, notams: true, answer: true })
  })

  it('round-trips through load', () => {
    saveLayerPrefs({ airspace: false, notams: true, answer: false, later_layer: true })
    expect(loadLayerPrefs(REGISTRY)).toEqual({ airspace: false, notams: true, answer: false, later_layer: true })
  })

  it('keeps a stored boolean for an id that is not in the state, and ignores it on load', () => {
    seed('{"wind":false,"airspace":false}')
    const registry: LayerRegistry = REGISTRY.filter(({ id }) => id !== 'later_layer')
    expect(loadLayerPrefs(registry)).toEqual({ airspace: false, notams: true, answer: true })
    saveLayerPrefs({ airspace: true, notams: true, answer: true })
    expect(stored()).toEqual({ wind: false, airspace: true, notams: true, answer: true })
  })

  it('drops stored values that are not booleans', () => {
    seed('{"wind":false,"rain":"yes","other":{"a":1}}')
    saveLayerPrefs({ airspace: true })
    expect(stored()).toEqual({ wind: false, airspace: true })
  })

  it.each(['{not json', '[]', 'null', '"true"', '42'])('overwrites %s with valid JSON', (junk) => {
    seed(junk)
    saveLayerPrefs({ airspace: false, notams: true, answer: true })
    expect(stored()).toEqual({ airspace: false, notams: true, answer: true })
  })

  it('changes only the layer that was toggled: the others keep their stored value', () => {
    seed('{"airspace":false,"notams":true,"answer":false}')
    const before = loadLayerPrefs(REGISTRY)
    saveLayerPrefs({ ...before, notams: false })
    expect(stored()).toMatchObject({ airspace: false, notams: false, answer: false })
  })

  it('ends on the final state after ten rapid changes to one switch', () => {
    let state = loadLayerPrefs(REGISTRY)
    for (let click = 1; click <= 10; click++) {
      state = { ...state, answer: !state.answer }
      saveLayerPrefs(state)
    }
    expect(state.answer).toBe(true)
    expect(stored().answer).toBe(true)
    saveLayerPrefs({ ...state, answer: false })
    expect(stored().answer).toBe(false)
  })

  it('does not write a "__proto__" key from storage into the prototype', () => {
    seed('{"__proto__":true,"airspace":true}')
    saveLayerPrefs({ airspace: false })
    expect(({} as Record<string, unknown>).polluted).toBeUndefined()
    expect(Object.getPrototypeOf(stored())).toBe(Object.prototype)
    expect(stored().airspace).toBe(false)
  })
})

describe('storage that fails (E13)', () => {
  const failure = (name: string) => Object.assign(new Error(name), { name })

  it('returns the defaults and does not throw when reading window.localStorage throws a SecurityError', () => {
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw failure('SecurityError')
    })
    expect(loadLayerPrefs(REGISTRY)).toEqual(DEFAULTS)
    expect(() => saveLayerPrefs(DEFAULTS)).not.toThrow()
  })

  it('returns the defaults when there is no localStorage at all', () => {
    vi.spyOn(window, 'localStorage', 'get').mockReturnValue(null as unknown as Storage)
    expect(loadLayerPrefs(REGISTRY)).toEqual(DEFAULTS)
    expect(() => saveLayerPrefs(DEFAULTS)).not.toThrow()
  })

  it.each(['SecurityError', 'QuotaExceededError'])('returns the defaults when getItem throws %s, and save does not throw', (name) => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw failure(name)
    })
    expect(loadLayerPrefs(REGISTRY)).toEqual(DEFAULTS)
    expect(() => saveLayerPrefs(DEFAULTS)).not.toThrow()
  })

  it.each(['SecurityError', 'QuotaExceededError'])('lets save return when setItem throws %s, and load still works', (name) => {
    seed('{"airspace":false}')
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw failure(name)
    })
    expect(() => saveLayerPrefs({ airspace: true, notams: true, answer: true })).not.toThrow()
    expect(loadLayerPrefs(REGISTRY)).toEqual({ ...DEFAULTS, airspace: false })
  })

  it('warns at most once across every failure, with a fixed line that holds no stored text', () => {
    seed('{secret junk')
    const set = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw failure('QuotaExceededError')
    })
    loadLayerPrefs(REGISTRY)
    loadLayerPrefs(REGISTRY)
    saveLayerPrefs(DEFAULTS)
    saveLayerPrefs(DEFAULTS)
    expect(set).toHaveBeenCalledTimes(2)
    expect(warn).toHaveBeenCalledTimes(1)
    expect(warn.mock.calls[0]).toHaveLength(1)
    expect(String(warn.mock.calls[0][0])).not.toMatch(/secret|junk|Error/)
  })

  it('does not warn when storage works and holds nothing or a valid value', () => {
    loadLayerPrefs(REGISTRY)
    saveLayerPrefs(DEFAULTS)
    loadLayerPrefs(REGISTRY)
    expect(warn).not.toHaveBeenCalled()
  })
})

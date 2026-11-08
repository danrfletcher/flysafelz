import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiUrl } from '@/config'
import { filesMatching, readUi, uiTextFiles } from '@/test/files'

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('apiUrl', () => {
  it('defaults to http://localhost:2024', () => {
    vi.stubEnv('VITE_API_URL', '')
    expect(apiUrl()).toBe('http://localhost:2024')
  })

  it('is overridden by VITE_API_URL, without a trailing slash', () => {
    vi.stubEnv('VITE_API_URL', 'https://agent.example.test:8443/')
    expect(apiUrl()).toBe('https://agent.example.test:8443')
  })

  it('ignores a blank VITE_API_URL', () => {
    vi.stubEnv('VITE_API_URL', '   ')
    expect(apiUrl()).toBe('http://localhost:2024')
  })
})

// The patterns are built from pieces so that this file does not match its own rules.
const ENV_READ = new RegExp(['import', 'meta', 'env'].join('\\.') + '\\.(VITE_[A-Za-z0-9_]*)', 'g')
const SECRET_NAMES = [['OPEN', 'AIP'], ['READER', '_'], ['API', '_KEY']].map((pieces) => pieces.join(''))

describe('what the browser may read (F3)', () => {
  it('reads exactly one setting, VITE_API_URL', () => {
    const names = new Set<string>()
    for (const path of uiTextFiles()) {
      for (const match of readUi(path).matchAll(ENV_READ)) names.add(match[1])
    }
    expect([...names]).toEqual(['VITE_API_URL'])
  })

  it('has no key or server-setting name anywhere under ui/ outside node_modules', () => {
    for (const name of SECRET_NAMES) {
      expect(filesMatching(new RegExp(name)), `ui/ files naming ${name}`).toEqual([])
    }
  })

  it('has the rule scans looking at the real source files', () => {
    const files = uiTextFiles()
    expect(files).toContain('src/config.ts')
    expect(files).toContain('package.json')
    expect(files.some((path) => path.startsWith('node_modules'))).toBe(false)
  })
})

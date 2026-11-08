// @vitest-environment node
import { spawnSync } from 'node:child_process'
import { readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { ESLint } from 'eslint'
import { describe, expect, it } from 'vitest'

import { UI_ROOT, filesMatching, readUi } from '@/test/files'

const pkg = JSON.parse(readUi('package.json')) as {
  dependencies: Record<string, string>
  devDependencies: Record<string, string>
}
const installed = Object.keys({ ...pkg.dependencies, ...pkg.devDependencies })

// Patterns are built from pieces so that this file does not match its own rules.
const word = (...pieces: string[]) => new RegExp(pieces.join(''))

describe('F2: no analytics, no logging of answer text', () => {
  it('has no analytics or telemetry package in package.json', () => {
    const tracking = /analytics|telemetry|sentry|segment|mixpanel|amplitude|posthog|plausible|hotjar|datadog|gtag|logrocket|fullstory|bugsnag|rollbar|newrelic|opentelemetry|clarity/i
    expect(installed.filter((name) => tracking.test(name))).toEqual([])
  })

  it('makes console.log an ESLint error and leaves console.warn allowed', async () => {
    const eslint = new ESLint({ cwd: UI_ROOT })
    const lint = async (code: string) => {
      const [result] = await eslint.lintText(code, { filePath: join(UI_ROOT, 'src/example.ts') })
      return result.messages.filter((message) => message.ruleId === 'no-console')
    }
    for (const call of ['log', 'error', 'info', 'debug']) {
      const found = await lint(`console.${call}('x')\n`)
      expect(found, `console.${call}`).toHaveLength(1)
      expect(found[0].severity).toBe(2)
    }
    expect(await lint("console.warn('x')\n")).toEqual([])
  })
})

describe('no half-wiring: nothing here belongs to a later PR', () => {
  it('has no motion, wind, chat-SDK or map-geometry dependency yet', () => {
    const later = /^(motion|framer-motion|leaflet-velocity|@langchain\/.*|@turf\/.*|react-markdown|@radix-ui\/react-(popover|switch|tooltip))$/
    expect(installed.filter((name) => later.test(name))).toEqual([])
  })

  it('has no saved choices and no CARTO key', () => {
    expect(filesMatching(word('local', 'Storage|session', 'Storage'))).toEqual([])
    expect(filesMatching(word('VITE_', 'CARTO_KEY'))).toEqual([])
  })

  it('has no layer, chat, panel or legend component', () => {
    const names = readdirSync(resolve(UI_ROOT, 'src/components'), { recursive: true }).map(String)
    expect(names.filter((name) => /layer|chat|panel|legend|sheet|drawer|toast|pin/i.test(name))).toEqual([])
  })
})

describe('build output is not committed', () => {
  it('ignores ui/node_modules and ui/dist, and tracks neither', () => {
    const run = (...args: string[]) => spawnSync('git', args, { cwd: resolve(UI_ROOT, '..'), encoding: 'utf8' })
    if (run('rev-parse', '--git-dir').status !== 0) return // not a git checkout (a tarball): nothing to ask
    const ignored = run('check-ignore', 'ui/node_modules', 'ui/dist')
    expect(ignored.stdout.trim().split('\n').sort()).toEqual(['ui/dist', 'ui/node_modules'])
    const tracked = run('ls-files', 'ui').stdout.split('\n')
    expect(tracked.filter((path) => /^ui\/(node_modules|dist)\//.test(path))).toEqual([])
  })
})

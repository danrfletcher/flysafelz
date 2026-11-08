// @vitest-environment node
import { spawnSync } from 'node:child_process'
import { readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { ESLint } from 'eslint'
import { describe, expect, it } from 'vitest'

import { UI_ROOT, filesMatching, readUi, uiTextFiles } from '@/test/files'

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
  it('has no motion, wind, chat-SDK or map-geometry dependency yet (the popover and switch are this PR’s)', () => {
    const later = /^(motion|framer-motion|leaflet-velocity|@langchain\/.*|@turf\/.*|react-markdown|@radix-ui\/react-(tooltip|toast|sheet|dropdown-menu))$/
    expect(installed.filter((name) => later.test(name))).toEqual([])
    expect(installed).toContain('@radix-ui/react-popover')
    expect(installed).toContain('@radix-ui/react-switch')
  })

  it('has no saved choices and no CARTO key', () => {
    expect(filesMatching(word('local', 'Storage|session', 'Storage'))).toEqual([])
    expect(filesMatching(word('VITE_', 'CARTO_KEY'))).toEqual([])
  })

  it('has the two layer switches and no others: no wind, rain, all-heights, answer-overlay or basemap-status switch', () => {
    const panel = readUi('src/components/LayerPanel.tsx')
    expect(panel.match(/<Switch\b/g)).toHaveLength(1) // one in the list the two layers map over
    const layers = [...panel.matchAll(/id: '(\w+)', name: '([^']+)'/g)].map((match) => [match[1], match[2]])
    expect(layers).toEqual([['airspace', 'Airspace below FL100'], ['notams', 'NOTAMs below FL100']])
    const switchLabels = /wind|rain|radar|all heights|answer overlays|basemap/i
    expect(switchLabels.test(panel)).toBe(false)
  })

  it('fetches the two map routes and no other route', () => {
    expect(filesMatching(word('/map/', 'wind'))).toEqual([])
    expect(filesMatching(word('rain', '-grid'))).toEqual([])
    expect(filesMatching(word('rainviewer', 'api'))).toEqual([])
    const routes = uiTextFiles()
      .filter((path) => path.startsWith('src/') && !/\.test\.|\/test\//.test(path))
      .flatMap((path) => [...readUi(path).matchAll(/['"`](\/map\/[a-z-]+)['"`]/g)].map((match) => match[1]))
    expect([...new Set(routes)].sort()).toEqual(['/map/airspace', '/map/notams'])
  })

  it('has no chat, sheet, drawer, toast or pin component yet', () => {
    const names = readdirSync(resolve(UI_ROOT, 'src/components'), { recursive: true }).map(String)
    expect(names.filter((name) => /chat|sheet|drawer|toast|pin/i.test(name))).toEqual([])
  })

  it('has no verdict or action colour function: the legend swatches are theme tokens', () => {
    const sources = uiTextFiles().filter((path) => path.startsWith('src/') && !/\.test\./.test(path) && /\.tsx?$/.test(path))
    expect(sources.filter((path) => /verdictColou?r|actionColou?r|colou?rFor/i.test(readUi(path)))).toEqual([])
  })
})

describe('F2/F7: NOTAM text and colour', () => {
  it('never reads a NOTAM’s `text` outside the type and the test fixtures', () => {
    const readers = uiTextFiles()
      .filter((path) => path.startsWith('src/') && !/\.test\.|\/test\//.test(path))
      .filter((path) => /properties\.text\b|\.text\b\s*[,)}]|\btext:/.test(readUi(path)))
    expect(readers).toEqual(['src/map/types.ts']) // the field's declaration only
  })

  it('decides a map colour from the airspace type and nothing else: no regex or word test on a name, text or action', () => {
    const style = readUi('src/map/airspaceStyle.ts')
    expect(style).not.toMatch(/\.action\b|\.name\b|\.text\b|\.match\(|\.test\(/)
    const build = readUi('src/map/BaseLayers.tsx')
    expect(build).not.toMatch(/\.action\b|\.text\b/)
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

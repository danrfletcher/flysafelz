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
  it('has no motion, wind or extra chat dependency: the LangGraph SDK and react-markdown are the chat PR’s, and the one Turf package is the point-in-polygon test', () => {
    const later = /^(motion|framer-motion|leaflet-velocity|vaul|rehype-raw|@radix-ui\/react-(tooltip|toast|sheet|dropdown-menu))$/
    expect(installed.filter((name) => later.test(name))).toEqual([])
    expect(installed.filter((name) => name.startsWith('@langchain/'))).toEqual(['@langchain/langgraph-sdk'])
    expect(installed).toContain('react-markdown')
    expect(installed.filter((name) => name.startsWith('@turf/'))).toEqual(['@turf/boolean-point-in-polygon'])
    expect(installed).toContain('@radix-ui/react-popover')
    expect(installed).toContain('@radix-ui/react-switch')
  })

  it('keeps browser storage to the one layer-choices module (F2: its key and no other store), and the CARTO key named in the one file that reads it, its type declaration and nothing else under src/', () => {
    const sources = (paths: string[]) => paths.filter((path) => !/\.test\.|\/test\//.test(path))
    expect(sources(filesMatching(word('local', 'Storage')))).toEqual(['src/lib/layerPrefs.ts'])
    expect(filesMatching(word('session', 'Storage'))).toEqual([])
    expect(filesMatching(word('addEventListener\\(.', 'storage'))).toEqual([]) // no cross-tab sync in v1
    expect(sources(filesMatching(word('flysafelz\\.layers\\.', 'v\\d')))).toEqual(['src/lib/layerPrefs.ts']) // the one key, v1
    const naming = uiTextFiles().filter((path) => path.startsWith('src/') && !/\.test\.|\/test\//.test(path) && word('VITE_', 'CARTO_KEY').test(readUi(path)))
    expect(naming).toEqual(['src/map/basemap.ts', 'src/vite-env.d.ts'])
  })

  it('has the base layers’ two switches and the Answer overlays switch, and no other: no wind, rain or all-heights switch, and the basemap entry has none', () => {
    const panel = readUi('src/components/LayerPanel.tsx')
    expect(panel.match(/<Switch\b/g)).toHaveLength(1) // one in the list the layers map over
    const layers = [...panel.matchAll(/id: '(\w+)', name: '([^']+)'/g)].map((match) => [match[1], match[2]])
    expect(layers).toEqual([['airspace', 'Airspace below FL100'], ['notams', 'NOTAMs below FL100'], ['answer', 'Answer overlays']])
    const switchLabels = /wind|rain|radar|all heights/i
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

  it('keeps the chat inside src/chat, with the Sheet and Drawer as the two components/ui shells and the 2 NM fit in src/map', () => {
    const names = readdirSync(resolve(UI_ROOT, 'src'), { recursive: true }).map(String).filter((name) => /chat|sheet|drawer|fitlz/i.test(name))
    const outside = names.filter((name) => !/^(chat|test[\\/]chatServer\.ts|test[\\/]fixtures[\\/]chat)/.test(name))
    expect(outside.sort()).toEqual(['components/ui/drawer.tsx', 'components/ui/sheet.tsx', 'map/fitLz.test.ts', 'map/fitLz.ts'])
  })

  it('has no pin bounce (PR-12.F1 owns it); `overlays` and `assessment` are read by the chat store, the badge and link readers and the answer-overlay map code, and by nothing else', () => {
    const sources = uiTextFiles().filter((path) => path.startsWith('src/') && !/\.test\.|\/test\//.test(path))
    const found = (pattern: RegExp) => sources.filter((path) => pattern.test(readUi(path)))
    expect(found(word('bou', 'nce|@key', 'frames|prefers-reduced-', 'motion'))).toEqual([])
    expect(found(word('overl', 'ays'))).toEqual([
      'src/App.tsx', // hands the open pin's turn to the map
      'src/chat/Answer.tsx', // says in a comment which ids are buttons
      'src/chat/ChatPanel.tsx', // hands a turn's overlays to its NOTAM links
      'src/chat/notamLinks.ts', // says in a comment which ids are linked
      'src/chat/turns.ts', // the store
      'src/chat/useLzThread.ts', // fills it
      'src/components/LayerPanel.tsx', // the label of the Answer overlays switch
      'src/map/answerFeatures.ts', // reads a turn's overlays
      'src/map/answerOverlays.tsx', // draws them
      'src/map/useAnswerHighlight.ts', // picks the turn
    ])
    expect(found(word('assess', 'ment'))).toEqual(['src/chat/Badges.tsx', 'src/chat/ChatPanel.tsx', 'src/chat/readBadges.ts', 'src/chat/turns.ts', 'src/chat/useLzThread.ts'])
  })

  it('has no verdict or action colour function: the legend swatches are theme tokens', () => {
    const sources = uiTextFiles().filter((path) => path.startsWith('src/') && !/\.test\./.test(path) && /\.tsx?$/.test(path))
    expect(sources.filter((path) => /verdictColou?r|actionColou?r|colou?rFor/i.test(readUi(path)))).toEqual([])
  })
})

describe('F2/F7: NOTAM text and colour', () => {
  it('reads a NOTAM’s `text` only in the type, the What’s-here row builder and the card that shows it, and never passes it to a console call', () => {
    const sources = uiTextFiles().filter((path) => path.startsWith('src/') && !/\.test\.|\/test\//.test(path))
    const readers = sources.filter((path) => /properties\.text\b|\.text\b\s*[,)}]|\btext:/.test(readUi(path)))
    // the declaration, the overlay reader, the card, the row builder
    expect(readers).toEqual(['src/map/answerFeatures.ts', 'src/map/types.ts', 'src/pick/WhatsHereCard.tsx', 'src/pick/whatsHere.ts'])
    const logging = sources.filter((path) => /console\.\w+\([^)]*\b(text|conditions|answer)\b/.test(readUi(path)))
    expect(logging).toEqual([])
  })

  it('decides a map colour from the airspace type and nothing else: no regex or word test on a name, text or action', () => {
    const style = readUi('src/map/airspaceStyle.ts')
    expect(style).not.toMatch(/\.action\b|\.name\b|\.text\b|\.match\(|\.test\(/)
    const build = readUi('src/map/BaseLayers.tsx')
    expect(build).not.toMatch(/\.action\b|\.text\b/)
  })
})

describe('F7: one function chooses a badge colour', () => {
  const sources = () => uiTextFiles().filter((path) => path.startsWith('src/') && !/\.test\.|\/test\//.test(path) && /\.tsx?$/.test(path))
  const VERDICT_WORDS = /NO-GO|NO_GO|UNVERIFIED|DISCRETION|APPROACHING|CONDITIONS/

  it('names a verdict word only in badgeColour.ts and in the places that show or map those words, never in the badge files', () => {
    const naming = sources().filter((path) => VERDICT_WORDS.test(readUi(path)))
    expect(naming).toEqual([
      'src/chat/badgeColour.ts', // the one colour function
      'src/components/Legend.tsx', // the legend's labels
      'src/content/howItWorks.ts', // the How-it-works words
      'src/map/actionStyle.ts', // the server's action names on an overlay, not a verdict
    ])
  })

  it('has a single caller of badgeColour, and the badge view picks classes by the colour name it returns, not by any word', () => {
    expect(sources().filter((path) => /\bbadgeColour\(/.test(readUi(path)))).toEqual(['src/chat/badgeColour.ts', 'src/chat/readBadges.ts'])
    const reader = readUi('src/chat/readBadges.ts')
    const call = reader.slice(reader.indexOf('badgeColour({'), reader.indexOf('})', reader.indexOf('badgeColour({')))
    expect(call).not.toMatch(/display|answer|label/)
    const view = readUi('src/chat/Badges.tsx')
    expect(view).toMatch(/PAINT\[badge\.colour\]/)
    expect(view).not.toMatch(/\.display\b|\.verdict\b|\.label\.(?:includes|startsWith|match|test)|label\)\.test/)
  })
})

describe('F5: the recorded chat payloads', () => {
  const recorded = readdirSync(resolve(UI_ROOT, 'src/test/fixtures/chat')).filter((name) => name.endsWith('.json'))

  it('has the recordings the chat tests use', () => {
    expect(recorded.sort()).toEqual([
      'blue-mill-assess.json',
      'blue-mill-follow-up.json',
      'blue-mill-highlights.json',
      'guard-fallback.json',
      'network-error.json',
      'outlook-too-far.json',
      'progress-sequence.json',
      'uk-only-decline.json',
    ])
  })

  it('has only fiction-range phone numbers: 01632 960xxx and 07700 900xxx, the number ranges Ofcom keeps for drama', () => {
    for (const name of recorded) {
      const text = readUi(`src/test/fixtures/chat/${name}`)
      const numbers = text.match(/\b0[\d ]{9,12}\d\b/g) ?? []
      for (const number of numbers) expect(number.replace(/ /g, ''), `${name}: ${number}`).toMatch(/^(01632960|07700900)\d{3}$/)
    }
    expect(readUi('src/test/fixtures/chat/blue-mill-assess.json')).toContain('01632 960003')
  })

  it('uses no West Sleekburn point (55.1576, -1.5699) and no real NOTAM text from its fixture', () => {
    for (const name of recorded) {
      const text = readUi(`src/test/fixtures/chat/${name}`)
      expect(text, name).not.toMatch(/55\.157|1\.569|sleekburn/i)
    }
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

// @vitest-environment node
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import { BUILD_ONLY_NAMES, KEY_NAMES, keyProblems } from './check-no-keys.mjs'

const SCRIPT = resolve(import.meta.dirname, 'check-no-keys.mjs')
const UI = resolve(import.meta.dirname, '..')
const made = []

function folder(files) {
  const root = mkdtempSync(join(tmpdir(), 'keys-'))
  made.push(root)
  for (const [name, text] of Object.entries(files)) {
    mkdirSync(join(root, name, '..'), { recursive: true })
    writeFileSync(join(root, name), text)
  }
  return root
}

afterEach(() => {
  while (made.length) rmSync(made.pop(), { recursive: true, force: true })
})

describe('keyProblems', () => {
  it('passes a bundle with an ordinary API URL', () => {
    const root = folder({ 'assets/app.js': 'fetch("http://localhost:2024/map/airspace")' })
    expect(keyProblems(root)).toEqual([])
  })

  it.each(KEY_NAMES)('finds the name %s, and does not print the text around it', (word) => {
    const root = folder({ 'assets/app.js': `var a="${word}"; var secret="hunter2"` })
    const found = keyProblems(root)
    expect(found).toEqual([`assets/app.js: names ${word}`])
    expect(found.join('')).not.toContain('hunter2')
  })

  it.each(BUILD_ONLY_NAMES)('allows the name %s inside a build output folder only, and still finds key shapes there', (word) => {
    const root = folder({ 'dist/assets/app.js': `var a="${word}"`, 'src/app.js': `var a="${word}"` })
    expect(keyProblems(join(root, 'dist'))).toEqual([])
    expect(keyProblems(join(root, 'src'))).toEqual([`app.js: names ${word}`])
    const sk = ['sk', 'abcdefghijklmnopqrstuvwxyz0123'].join('-')
    const shaped = folder({ 'dist/assets/app.js': `var a="${word}"; var k="${sk}"` })
    expect(keyProblems(join(shaped, 'dist'))).toEqual(['assets/app.js: contains an sk- key'])
  })

  it('finds a key-shaped string', () => {
    const sk = ['sk', 'abcdefghijklmnopqrstuvwxyz0123'].join('-')
    expect(keyProblems(folder({ 'a.js': `const k="${sk}"` }))).toEqual(['a.js: contains an sk- key'])
    const bearer = ['Bearer', 'abcdefghijklmnopqrstuvwxyz012345'].join(' ')
    expect(keyProblems(folder({ 'a.js': `h="${bearer}"` }))).toEqual(['a.js: contains a bearer token'])
    const assigned = ['api', 'Key: "abcdefghijklmnopqrstuvwxyz012345"'].join('')
    expect(keyProblems(folder({ 'a.js': assigned }))).toEqual(['a.js: contains a long secret assigned to a key name'])
  })

  it('reads a single file and a missing path as nothing', () => {
    const root = folder({ 'index.html': '<html></html>' })
    expect(keyProblems(join(root, 'index.html'))).toEqual([])
    expect(keyProblems(join(root, 'missing'))).toEqual([])
  })
})

describe('the script', () => {
  const run = (...args) => spawnSync('node', [SCRIPT, ...args], { encoding: 'utf8' })

  it('exits 1 and lists the problem when a path holds a key name', () => {
    const root = folder({ 'x.js': `a=${KEY_NAMES[0]}` })
    const done = run(root)
    expect(done.status).toBe(1)
    expect(done.stderr).toContain('1 problem(s)')
  })

  it('exits 0 on clean paths', () => {
    expect(run(folder({ 'x.js': 'a=1' })).status).toBe(0)
  })
})

describe('the UI as it is now', () => {
  it('has no key and no key name in src, public, index.html or the build output (when built)', () => {
    for (const path of ['src', 'public', 'index.html', 'dist']) {
      expect(keyProblems(join(UI, path)), path).toEqual([])
    }
    expect(existsSync(join(UI, 'src'))).toBe(true)
  })
})

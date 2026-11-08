// @vitest-environment node
import { spawnSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import { fontProblems, urlProblems } from './check-external-urls.mjs'

const SCRIPT = resolve(import.meta.dirname, 'check-external-urls.mjs')
const made = []

/** A dist folder with the given files (name -> text), plus the bundled font file unless told not to. */
function dist(files, { font = true } = {}) {
  const folder = mkdtempSync(join(tmpdir(), 'dist-'))
  made.push(folder)
  mkdirSync(join(folder, 'assets'))
  if (font) writeFileSync(join(folder, 'assets', 'Geist.woff2'), 'x')
  for (const [name, text] of Object.entries(files)) writeFileSync(join(folder, name), text)
  return folder
}

const GOOD_CSS = "@font-face{font-family:Geist;src:url(/assets/Geist.woff2) format('woff2')}"

afterEach(() => {
  while (made.length) rmSync(made.pop(), { recursive: true, force: true })
})

describe('urlProblems', () => {
  it('passes the CARTO tile host (with its {s} placeholder), the attribution hosts and localhost', () => {
    const js = [
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      'https://a.basemaps.cartocdn.com/x.png',
      'https://www.openstreetmap.org/copyright',
      'https://carto.com/attributions',
      'https://www.openaip.net',
      'https://open-meteo.com',
      'https://api.rainviewer.com/public/weather-maps.json',
      'https://tilecache.rainviewer.com/v2/radar/1/256/{z}/{x}/{y}/2/1_0.png',
      'https://www.rainviewer.com',
      'http://localhost:2024',
    ]
      .map((url) => `fetch("${url}")`)
      .join(';')
    expect(urlProblems(dist({ 'index.js': js }))).toEqual([])
  })

  it('skips the namespace, React error and Leaflet link strings that are never requested', () => {
    const js = '"http://www.w3.org/2000/svg" "https://react.dev/errors/31" \'<a href="https://leafletjs.com">\''
    expect(urlProblems(dist({ 'index.js': js }))).toEqual([])
  })

  it('skips the link and endpoint strings of the bundled LangGraph SDK, react-markdown and langsmith client that are never requested', () => {
    const js = [
      'https://github.com/syntax-tree/hast-util-to-jsx-runtime',
      'https://docs.langchain.com/oss/javascript/langchain/errors/X/',
      'https://docs.expo.dev/versions/latest/sdk/expo/#expofetch-api',
      'https://api.smith.langchain.com/',
      'https://api.smith.langchain.com',
      'https://eu.smith.langchain.com',
      'https://smith.langchain.com/prompts',
    ]
      .map((url) => `a("${url}")`)
      .join(';')
    expect(urlProblems(dist({ 'index.js': js }))).toEqual([])
  })

  it.each([
    'https://smith.langchain.com.evil.example/x',
    'https://evil-smith.langchain.com/x',
    'http://api.smith.langchain.com/x',
    'https://other.smith.langchain.com/x',
    'https://docs.langchain.com.evil.example/x',
    'https://example.com/track.js',
    'https://www.google-analytics.com/collect',
    'https://carto.com.evil.example/x',
    'https://evil-carto.com/x',
    'https://carto.com@evil.example/x',
    'https://rainviewer.com.evil.example/x',
    'https://evil-rainviewer.com/x',
    'http://api.rainviewer.com.evil.example/x',
  ])('fails on %s', (url) => {
    const problems = urlProblems(dist({ 'index.js': `fetch("${url}")` }))
    expect(problems).toHaveLength(1)
    expect(problems[0]).toContain('not on the allowlist')
  })

  it.each(['fonts.googleapis.com/css2?family=Geist', 'fonts.gstatic.com/s/geist/v1/a.woff2'])('fails on %s by name', (where) => {
    const problems = urlProblems(dist({ 'index.css': `@import url(https://${where});` }))
    expect(problems).toHaveLength(1)
    expect(problems[0]).toContain('fonts must be bundled')
  })

  it('reads CSS and HTML as well as JS', () => {
    expect(urlProblems(dist({ 'index.html': '<script src="https://cdn.example/a.js"></script>' }))).toHaveLength(1)
    expect(urlProblems(dist({ 'index.css': 'a{background:url(https://cdn.example/a.png)}' }))).toHaveLength(1)
  })
})

describe('fontProblems', () => {
  it('passes an @font-face whose src is a bundled file', () => {
    expect(fontProblems(dist({ 'index.css': GOOD_CSS }))).toEqual([])
  })

  it('fails on a remote src', () => {
    const css = '@font-face{font-family:Geist;src:url(https://cdn.example/Geist.woff2)}'
    expect(fontProblems(dist({ 'index.css': css }))).toEqual([expect.stringContaining('is not a bundled file')])
  })

  it('fails on a src that is not in dist', () => {
    const problems = fontProblems(dist({ 'index.css': GOOD_CSS }, { font: false }))
    expect(problems).toEqual([expect.stringContaining('is not in dist')])
  })

  it('fails when there is no @font-face at all', () => {
    expect(fontProblems(dist({ 'index.css': 'a{color:red}' }))).toEqual([expect.stringContaining('Geist is not bundled')])
  })
})

describe('the command line', () => {
  it('exits 0 for a clean dist, 1 with the problems on stderr for a bad one, 2 with no dist', () => {
    const clean = spawnSync('node', [SCRIPT, dist({ 'index.css': GOOD_CSS })], { encoding: 'utf8' })
    expect(clean.status).toBe(0)
    const bad = spawnSync('node', [SCRIPT, dist({ 'index.css': GOOD_CSS, 'index.js': 'fetch("https://example.com/")' })], {
      encoding: 'utf8',
    })
    expect(bad.status).toBe(1)
    expect(bad.stderr).toContain('example.com')
    expect(spawnSync('node', [SCRIPT, join(tmpdir(), 'no-such-dist-folder')], { encoding: 'utf8' }).status).toBe(2)
  })
})

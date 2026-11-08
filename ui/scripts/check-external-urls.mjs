// F2: the built app (ui/dist) may call out to CARTO tiles and nothing else, and its fonts are bundled.
//
//   node ui/scripts/check-external-urls.mjs [dist-folder]      (make test-ui runs it after the build)
//
// 1. Every absolute URL in a text file under dist must be on an allowlist, by host:
//      basemaps.cartocdn.com         the tile host (the {s} subdomain placeholder is allowed)
//      openstreetmap.org, carto.com, openaip.net, open-meteo.com  the attribution link hosts (the browser never calls open-meteo.com: wind comes from /map/wind)
//      localhost
//    fonts.googleapis.com and fonts.gstatic.com fail with their own message, wherever they appear.
// 2. Every @font-face `src` in the built CSS is a bundled file that exists in dist.
//
// Some strings are in the bundle that are not requests, and are skipped by their full prefix, not by host:
//    http://www.w3.org/                 XML namespace names (svg, xlink, MathML); an identifier, never fetched
//    https://react.dev/errors/          text inside React's production error messages
//    https://leafletjs.com              the "Leaflet" link in the map's default attribution; shown, never fetched
//    https://github.com/syntax-tree/    a link in react-markdown's error text
//    https://docs.langchain.com/, https://docs.expo.dev/    links in the LangGraph SDK's error and warning text
//    https://[api.|dev.|eu.|aws.|apac.|beta.]smith.langchain.com    the default endpoints inside the langsmith client, which the LangGraph SDK's
//                                       peer @langchain/core bundles. It reads its settings from process.env, and Vite builds that as {} (the
//                                       built file holds `typeof process<"u"?{}[name]`), so tracing is off by construction and nothing is sent
//                                       there; the chat test asserts that a whole run fetches from the API origin only.
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const ALLOWED_HOSTS = ['basemaps.cartocdn.com', 'openstreetmap.org', 'carto.com', 'openaip.net', 'open-meteo.com', 'localhost']
export const FORBIDDEN_FONT_HOSTS = ['fonts.googleapis.com', 'fonts.gstatic.com']
export const NOT_REQUESTS = [
  'http://www.w3.org/',
  'https://react.dev/errors/',
  'https://leafletjs.com',
  'https://github.com/syntax-tree/',
  'https://docs.langchain.com/',
  'https://docs.expo.dev/',
]
// Those langsmith hosts, matched whole (so smith.langchain.com.evil.example is still reported), over https only.
export const INERT_LANGSMITH_HOSTS = ['', 'api.', 'dev.', 'eu.', 'aws.', 'apac.', 'beta.'].map((prefix) => `${prefix}smith.langchain.com`)

const BINARY = new Set(['.woff', '.woff2', '.ttf', '.otf', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.webp', '.map'])
const URL_IN_TEXT = /https?:\/\/[^\s"'`<>()\\]+/g

function hostAllowed(host) {
  const bare = host.replace(/^\{s\}\./, '').replace(/:\d+$/, '')
  return ALLOWED_HOSTS.some((allowed) => bare === allowed || bare.endsWith(`.${allowed}`))
}

function textFiles(folder) {
  const found = []
  for (const entry of readdirSync(folder, { withFileTypes: true })) {
    const path = join(folder, entry.name)
    if (entry.isDirectory()) found.push(...textFiles(path))
    else if (!BINARY.has(extname(entry.name).toLowerCase())) found.push(path)
  }
  return found
}

/** Problems with the absolute URLs in the text files under `dist` (an array of strings; empty when clean). */
export function urlProblems(dist) {
  const problems = []
  for (const file of textFiles(dist)) {
    const text = readFileSync(file, 'utf8')
    const name = file.slice(dist.length + 1)
    for (const host of FORBIDDEN_FONT_HOSTS) {
      if (text.includes(host)) problems.push(`${name}: reaches ${host}; fonts must be bundled`)
    }
    for (const url of text.match(URL_IN_TEXT) ?? []) {
      if (NOT_REQUESTS.some((prefix) => url.startsWith(prefix))) continue
      if (FORBIDDEN_FONT_HOSTS.some((host) => url.includes(host))) continue // reported above
      const host = url.replace(/^https?:\/\//, '').split(/[/?#]/)[0]
      if (url.startsWith('https://') && INERT_LANGSMITH_HOSTS.includes(host)) continue
      if (!hostAllowed(host)) {
        problems.push(`${name}: ${url} is on ${host}, which is not on the allowlist (${ALLOWED_HOSTS.join(', ')})`)
      }
    }
  }
  return problems
}

/** Problems with the @font-face rules in the built CSS under `dist`: a src that is not a bundled local file. */
export function fontProblems(dist) {
  const problems = []
  let faces = 0
  for (const file of textFiles(dist).filter((path) => path.endsWith('.css'))) {
    const name = file.slice(dist.length + 1)
    for (const face of readFileSync(file, 'utf8').matchAll(/@font-face\s*\{([^}]*)\}/g)) {
      faces += 1
      const src = /src:\s*([^;}]+)/.exec(face[1])?.[1] ?? ''
      const urls = [...src.matchAll(/url\(\s*["']?([^"')]+)["']?\s*\)/g)].map((match) => match[1])
      if (urls.length === 0) problems.push(`${name}: an @font-face has no url() src`)
      for (const url of urls) {
        if (/^([a-z]+:)?\/\//i.test(url) || url.startsWith('data:')) {
          problems.push(`${name}: @font-face src ${url} is not a bundled file`)
        } else if (!existsSync(join(dist, url.replace(/^\//, '')))) {
          problems.push(`${name}: @font-face src ${url} is not in dist`)
        }
      }
    }
  }
  if (faces === 0) problems.push('no @font-face rule in the built CSS: Geist is not bundled')
  return problems
}

function main() {
  const dist = resolve(process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), '..', 'dist'))
  if (!existsSync(dist)) {
    process.stderr.write(`check-external-urls: ${dist} does not exist; run the build first\n`)
    process.exit(2)
  }
  const problems = [...urlProblems(dist), ...fontProblems(dist)]
  if (problems.length > 0) {
    process.stderr.write(`check-external-urls: ${problems.length} problem(s)\n${problems.map((line) => `  ${line}`).join('\n')}\n`)
    process.exit(1)
  }
  process.stdout.write(`check-external-urls: ok (${dist})\n`)
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()

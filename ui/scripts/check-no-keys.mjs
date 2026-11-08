// F3: no openAIP or model key, and no name of one, in the UI's source or in its build output.
//
//   node ui/scripts/check-no-keys.mjs [path ...]      (make test-ui runs it after the build)
//
// With no path it reads ui/src, ui/public, ui/index.html and ui/dist. Every text file under each path is read for:
//   the openAIP key header name and the names of the server's key variables (.env.example), which the browser never needs;
//   a key that looks like one: an OpenAI-style `sk-` key, a bearer token, or a long hex or base64 string assigned to a *key* name.
// The patterns are built from pieces so that this file does not match its own rules.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { basename, dirname, extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const join2 = (...pieces) => pieces.join('')

export const KEY_NAMES = [
  join2('x-open', 'aip-api-key'),
  join2('OPEN', 'AIP', '_API', '_KEY'),
  join2('READER', '_API', '_KEY'),
  join2('AGENT', '_API', '_KEY'),
  join2('LANGSMITH', '_API', '_KEY'),
  join2('OPENAI', '_API', '_KEY'),
  join2('ANTHROPIC', '_API', '_KEY'),
  join2('VITE_OPEN', 'AIP'),
]

// The name of the langsmith key variable also appears in the build output, as text inside the langsmith client that the LangGraph SDK's peer
// @langchain/core bundles (it names the variable in an error message and in the list of variables it reads, from a process.env that Vite builds as {}).
// It is allowed in a built file only; src, public and index.html still may not name it, and the key *shapes* below are checked everywhere.
export const BUILD_ONLY_NAMES = [join2('LANGSMITH', '_API', '_KEY')]

export const KEY_SHAPES = [
  { name: 'an sk- key', pattern: /\bsk-[A-Za-z0-9_-]{20,}/ },
  { name: 'a bearer token', pattern: /\bBearer\s+[A-Za-z0-9._~+/=-]{20,}/ },
  { name: 'a long secret assigned to a key name', pattern: /\bapi[_-]?key["']?\s*[:=]\s*["'][A-Za-z0-9_-]{24,}["']/i },
]

const BINARY = new Set(['.woff', '.woff2', '.ttf', '.otf', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.webp'])

function textFiles(path) {
  if (!existsSync(path)) return []
  if (statSync(path).isFile()) return [path]
  const found = []
  for (const entry of readdirSync(path, { withFileTypes: true })) {
    const child = join(path, entry.name)
    if (entry.isDirectory()) found.push(...textFiles(child))
    else if (!BINARY.has(extname(entry.name).toLowerCase())) found.push(child)
  }
  return found
}

/** Problems in the text files under `path` (a file or a folder): one line each, naming the file and the rule, never the text matched. */
export function keyProblems(path) {
  const problems = []
  const built = basename(resolve(path)) === 'dist' // the build output folder; BUILD_ONLY_NAMES may be there
  for (const file of textFiles(path)) {
    const text = readFileSync(file, 'utf8')
    const name = file.startsWith(path) && file !== path ? file.slice(path.length + 1) : file
    for (const word of KEY_NAMES) {
      if (built && BUILD_ONLY_NAMES.includes(word)) continue
      if (text.includes(word)) problems.push(`${name}: names ${word}`)
    }
    for (const { name: shape, pattern } of KEY_SHAPES) {
      if (pattern.test(text)) problems.push(`${name}: contains ${shape}`)
    }
  }
  return problems
}

function main() {
  const ui = resolve(dirname(fileURLToPath(import.meta.url)), '..')
  const paths = process.argv.length > 2 ? process.argv.slice(2).map((p) => resolve(p)) : ['src', 'public', 'index.html', 'dist'].map((p) => join(ui, p))
  const problems = paths.flatMap((path) => keyProblems(path).map((line) => `${path}: ${line}`))
  if (problems.length > 0) {
    process.stderr.write(`check-no-keys: ${problems.length} problem(s)\n${problems.map((line) => `  ${line}`).join('\n')}\n`)
    process.exit(1)
  }
  process.stdout.write(`check-no-keys: ok (${paths.filter(existsSync).join(', ')})\n`)
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()

// A stand-in for the map routes, for looking at the UI in a real browser with no keys and no network (the tester's harness for PR-4.F1).
//
//   node ui/scripts/serve-blue-mill.mjs --port 4999 [--delay-airspace MS] [--delay-notams MS] [--delay-wind MS] [--fail-airspace] [--fail-notams]
//                                         [--fail-wind] [--stale-wind] [--delay-rain-grid MS] [--fail-rain-grid] [--stale-rain-grid]
//
// It serves the Blue Mill fixtures (ui/src/test/fixtures/map-*.blue-mill.json, fiction-range phone numbers only) at GET /map/airspace and
// GET /map/notams, the synthetic wind grid (ui/src/mocks/wind-ok.json: made-up winds, not a forecast) at GET /map/wind, and the synthetic rain grid (ui/src/mocks/rain-grid-ok.json: made-up rain)
// at GET /map/rain-grid, on 127.0.0.1,
// with CORS open to any origin, and nothing else. A delay holds that route's answer back; a fail answers 503 with a reason, as the real
// routes do; --stale-wind and --stale-rain-grid serve the grid with `stale: true`, as the real routes do after a failed refresh. It serves no RainViewer
// radar: the Rain layer's index and tiles are the browser's own calls, which a test harness answers itself (Playwright route, MSW). Point the UI at it with VITE_API_URL=http://127.0.0.1:<port> (the build reads it), e.g.
//   VITE_API_URL=http://127.0.0.1:4999 npm --prefix ui run build && npm --prefix ui run preview -- --port 4173
// The port is required: pick a free one yourself. Stop the script with the PID you started it as.
import { readFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const FIXTURES = join(HERE, '../src/test/fixtures')
const WIND_GRID = join(HERE, '../src/mocks/wind-ok.json')
const RAIN_GRID = join(HERE, '../src/mocks/rain-grid-ok.json')

/** Reads `--name value` and `--flag` options from an argument list. */
export function parseArgs(argv) {
  const options = { port: null, delayAirspace: 0, delayNotams: 0, delayWind: 0, failAirspace: false, failNotams: false, failWind: false, staleWind: false, delayRainGrid: 0, failRainGrid: false, staleRainGrid: false }
  for (let i = 0; i < argv.length; i += 1) {
    const name = argv[i]
    if (name === '--port') options.port = Number(argv[++i])
    else if (name === '--delay-airspace') options.delayAirspace = Number(argv[++i])
    else if (name === '--delay-notams') options.delayNotams = Number(argv[++i])
    else if (name === '--delay-wind') options.delayWind = Number(argv[++i])
    else if (name === '--fail-airspace') options.failAirspace = true
    else if (name === '--fail-notams') options.failNotams = true
    else if (name === '--fail-wind') options.failWind = true
    else if (name === '--stale-wind') options.staleWind = true
    else if (name === '--delay-rain-grid') options.delayRainGrid = Number(argv[++i])
    else if (name === '--fail-rain-grid') options.failRainGrid = true
    else if (name === '--stale-rain-grid') options.staleRainGrid = true
    else throw new Error(`unknown option ${name}`)
  }
  if (!Number.isInteger(options.port) || options.port < 1024 || options.port > 65535) {
    throw new Error('--port is required (1024 to 65535): pick a free port yourself')
  }
  for (const key of ['delayAirspace', 'delayNotams', 'delayWind', 'delayRainGrid']) {
    if (!Number.isFinite(options[key]) || options[key] < 0) throw new Error('a delay is a number of milliseconds')
  }
  return options
}

/** The server for the given options (not yet listening). */
export function makeServer(options) {
  const routes = {
    '/map/airspace': { file: 'map-airspace.blue-mill.json', delay: options.delayAirspace, fail: options.failAirspace },
    '/map/notams': { file: 'map-notams.blue-mill.json', delay: options.delayNotams, fail: options.failNotams },
    '/map/wind': { path: WIND_GRID, delay: options.delayWind, fail: options.failWind, stale: options.staleWind },
    '/map/rain-grid': { path: RAIN_GRID, delay: options.delayRainGrid, fail: options.failRainGrid, stale: options.staleRainGrid },
  }
  return createServer((request, response) => {
    const cors = { 'access-control-allow-origin': '*', 'access-control-allow-methods': 'GET', 'access-control-allow-headers': '*' }
    const route = request.method === 'GET' ? routes[new URL(request.url ?? '/', 'http://localhost').pathname] : undefined
    if (request.method === 'OPTIONS') {
      response.writeHead(204, cors).end()
      return
    }
    if (!route) {
      response.writeHead(404, cors).end()
      return
    }
    setTimeout(() => {
      if (route.fail) {
        response.writeHead(503, { ...cors, 'content-type': 'application/json' }).end(JSON.stringify({ reason: 'stand-in: switched off' }))
      } else {
        const body = readFileSync(route.path ?? join(FIXTURES, route.file))
        response.writeHead(200, { ...cors, 'content-type': 'application/json' }).end(route.stale ? JSON.stringify({ ...JSON.parse(body), stale: true }) : body)
      }
    }, route.delay)
  })
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const options = parseArgs(process.argv.slice(2))
  makeServer(options).listen(options.port, '127.0.0.1', () => {
    console.warn(`serve-blue-mill: pid ${process.pid}, http://127.0.0.1:${options.port}/map/airspace, /map/notams, /map/wind and /map/rain-grid`)
  })
}

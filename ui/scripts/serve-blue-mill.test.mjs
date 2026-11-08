// @vitest-environment node
import { execFile, spawn } from 'node:child_process'
import { createServer } from 'node:net'
import { promisify } from 'node:util'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import { parseArgs } from './serve-blue-mill.mjs'

const SCRIPT = resolve(import.meta.dirname, 'serve-blue-mill.mjs')
const run = promisify(execFile)
const started = []

/** A free port, found by binding port 0 and letting go of it. */
function freePort() {
  return new Promise((resolvePort) => {
    const probe = createServer()
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address()
      probe.close(() => resolvePort(port))
    })
  })
}

/** The script as its own process, so that MSW (which watches this process's requests) is not in the way. Resolves once it listens. */
async function serve(...flags) {
  const port = await freePort()
  const child = spawn(process.execPath, [SCRIPT, '--port', String(port), ...flags], { stdio: ['ignore', 'ignore', 'pipe'] })
  started.push(child)
  await new Promise((ready, fail) => {
    child.stderr.on('data', (chunk) => String(chunk).includes('serve-blue-mill: pid') && ready())
    child.on('error', fail)
    child.on('exit', (code) => fail(new Error(`exited ${code}`)))
  })
  return port
}

/** [status, body] of a GET with curl; -i would add headers, so the headers come back separately. */
async function get(port, path) {
  const { stdout } = await run('curl', ['-s', '-i', `http://127.0.0.1:${port}${path}`])
  const [head, ...body] = stdout.split('\r\n\r\n')
  return { status: Number(head.split(' ')[1]), head: head.toLowerCase(), body: body.join('\r\n\r\n') }
}

afterEach(() => {
  while (started.length) started.pop().kill('SIGTERM') // only the processes this test started
})

describe('parseArgs', () => {
  it('needs a port and takes the options', () => {
    expect(parseArgs(['--port', '4999'])).toMatchObject({ port: 4999, delayAirspace: 0, failNotams: false })
    expect(parseArgs(['--port', '4999', '--delay-notams', '500', '--fail-airspace'])).toMatchObject({ delayNotams: 500, failAirspace: true })
  })

  it.each([[[]], [['--port', 'x']], [['--port', '80']], [['--port', '4999', '--delay-notams', '-1']], [['--port', '4999', '--nope']]])('rejects %j', (argv) => {
    expect(() => parseArgs(argv)).toThrow()
  })
})

describe('the stand-in server', () => {
  it('serves the two Blue Mill payloads with CORS and nothing else', async () => {
    const port = await serve()
    const airspace = await get(port, '/map/airspace')
    const notams = await get(port, '/map/notams')
    expect([airspace.status, notams.status]).toEqual([200, 200])
    expect(airspace.head).toContain('access-control-allow-origin: *')
    expect(JSON.parse(airspace.body).features.map((f) => f.properties.name)).toContain('BLUE MILL TEST ATZ')
    expect(JSON.parse(notams.body).features.map((f) => f.properties.id)).toContain('H2001/26')
    expect((await get(port, '/other')).status).toBe(404)
  })

  it('answers 503 for a route told to fail, and holds a route told to wait', async () => {
    const port = await serve('--fail-notams', '--delay-airspace', '700')
    const notams = await get(port, '/map/notams')
    expect(notams.status).toBe(503)
    expect(JSON.parse(notams.body)).toHaveProperty('reason')
    const began = Date.now()
    expect((await get(port, '/map/airspace')).status).toBe(200)
    expect(Date.now() - began).toBeGreaterThanOrEqual(650)
  })
})

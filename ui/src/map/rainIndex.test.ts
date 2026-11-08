import { http, HttpResponse } from 'msw'
import { describe, expect, it } from 'vitest'

import { fetchRainIndex, MAX_NATIVE_ZOOM, parseRainIndex, RAINVIEWER_INDEX_URL, tileTemplate } from '@/map/rainIndex'
import { rainIndexBody } from '@/mocks/rain'
import { server } from '@/test/setup'

const HOST = 'https://tilecache.rainviewer.com'
const T = 1_794_140_000
const good = (times = [T - 1200, T - 600, T]) => rainIndexBody(times)

describe('parseRainIndex', () => {
  it('reads a valid index: the host, and the frames oldest first', () => {
    const result = parseRainIndex(good([T, T - 1200, T - 600]))
    expect(result).toEqual({
      status: 'ok',
      host: HOST,
      frames: [
        { time: T - 1200, path: `/v2/radar/${(T - 1200).toString(16)}` },
        { time: T - 600, path: `/v2/radar/${(T - 600).toString(16)}` },
        { time: T, path: `/v2/radar/${T.toString(16)}` },
      ],
    })
  })

  it('takes a single frame as a valid index', () => {
    const result = parseRainIndex(good([T]))
    expect(result.status).toBe('ok')
    if (result.status === 'ok') expect(result.frames).toHaveLength(1)
  })

  it('keeps one frame per time', () => {
    const body = good([T, T])
    const result = parseRainIndex(body)
    expect(result.status === 'ok' && result.frames).toHaveLength(1)
  })

  it('leaves out a malformed frame and keeps the good ones', () => {
    const body = good([T - 600, T])
    body.radar.past.push(
      { time: 'x' as unknown as number, path: '/v2/radar/a' },
      { time: T + 600, path: 'https://evil.example/x' },
      { time: T + 1200, path: '/v2/radar/../../x' },
      { time: -5, path: '/v2/radar/b' },
      { time: 1.5, path: '/v2/radar/b' },
      null as never,
      'text' as never,
    )
    const result = parseRainIndex(body)
    expect(result.status === 'ok' && result.frames.map((frame) => frame.time)).toEqual([T - 600, T])
  })

  it.each<[string, unknown]>([
    ['empty past', { ...good(), radar: { past: [], nowcast: [] } }],
    ['no frame usable', { ...good(), radar: { past: [{ time: 'x', path: 5 }, null] } }],
    ['past missing', { ...good(), radar: { nowcast: [] } }],
    ['past not a list', { ...good(), radar: { past: { time: T } } }],
    ['radar missing', { version: '2.0', host: HOST }],
    ['radar not an object', { host: HOST, radar: 'wet' }],
    ['host missing', { radar: good().radar }],
    ['host not text', { ...good(), host: 5 }],
    ['host not RainViewer', { ...good(), host: 'https://evil.example' }],
    ['host a lookalike', { ...good(), host: 'https://rainviewer.com.evil.example' }],
    ['host over http', { ...good(), host: 'http://tilecache.rainviewer.com' }],
    ['host with a path', { ...good(), host: 'https://tilecache.rainviewer.com/x' }],
    ['host with credentials', { ...good(), host: 'https://a@tilecache.rainviewer.com' }],
    ['a list', []],
    ['text', 'hello'],
    ['null', null],
    ['a number', 7],
    ['an empty object', {}],
  ])('gives failed for %s', (_name, body) => {
    expect(parseRainIndex(body)).toEqual({ status: 'failed' })
  })
})

describe('fetchRainIndex', () => {
  const ask = () => fetchRainIndex(new AbortController().signal)

  it('fetches the index URL and parses it', async () => {
    let asked = ''
    server.use(
      http.get(RAINVIEWER_INDEX_URL, ({ request }) => {
        asked = request.url
        return HttpResponse.json(good())
      }),
    )
    const result = await ask()
    expect(asked).toBe('https://api.rainviewer.com/public/weather-maps.json')
    expect(result.status).toBe('ok')
  })

  it.each<[string, () => Response]>([
    ['a 500', () => new HttpResponse('down', { status: 500 })],
    ['a 429', () => new HttpResponse('slow down', { status: 429 })],
    ['a 404', () => new HttpResponse(null, { status: 404 })],
    ['a 204', () => new HttpResponse(null, { status: 204 })],
    ['a body that is not JSON', () => new HttpResponse('<html>', { status: 200 })],
    ['a JSON body that is not an index', () => HttpResponse.json({ ok: true })],
    ['an empty past', () => HttpResponse.json(rainIndexBody([]))],
    ['a network error', () => HttpResponse.error()],
  ])('gives failed for %s, and does not throw', async (_name, answer) => {
    server.use(http.get(RAINVIEWER_INDEX_URL, answer))
    await expect(ask()).resolves.toEqual({ status: 'failed' })
  })

  it('gives failed when the request is aborted', async () => {
    server.use(http.get(RAINVIEWER_INDEX_URL, () => HttpResponse.json(good())))
    const controller = new AbortController()
    controller.abort()
    await expect(fetchRainIndex(controller.signal)).resolves.toEqual({ status: 'failed' })
  })
})

describe('tileTemplate', () => {
  it('is the host, the frame path and the 256 px tile path with the colour scheme and options', () => {
    expect(tileTemplate(HOST, '/v2/radar/abc')).toBe('https://tilecache.rainviewer.com/v2/radar/abc/256/{z}/{x}/{y}/2/1_0.png')
  })

  it('stops at zoom 7: tiles are never asked for above RainViewer’s maximum', () => {
    expect(MAX_NATIVE_ZOOM).toBe(7)
  })
})

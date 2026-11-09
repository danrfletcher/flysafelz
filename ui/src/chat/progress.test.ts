import { describe, expect, it } from 'vitest'

import { nextLine, progressLine } from '@/chat/progress'
import { PROGRESS_SEQUENCE } from '@/test/chatServer'

describe('progressLine: the recorded custom events', () => {
  it.each(PROGRESS_SEQUENCE.events.map((event, index) => [event, PROGRESS_SEQUENCE.lines[index]]))('%j shows %j', (event, line) => {
    expect(progressLine(event)).toBe(line)
  })

  it('covers the four kinds of line the server writes', () => {
    expect(PROGRESS_SEQUENCE.lines).toEqual([
      'Checking weather…',
      'Checking airspace…',
      'Reading 2 NOTAMs (1/2)…',
      'Reading 2 NOTAMs (2/2)…',
      'Assessing…',
      'Checking the answer…',
    ])
  })
})

describe('progressLine: shapes that are not a progress line', () => {
  it.each([
    ['null', null],
    ['undefined', undefined],
    ['a number', 7],
    ['an object', { progress: 'Checking weather…' }],
    ['an array', ['Checking weather…']],
    ['an empty string', ''],
    ['a blank string', '   '],
    ['a multi-line string', 'one\ntwo'],
    ['a very long string', 'x'.repeat(500)],
  ])('ignores %s', (_name, event) => {
    expect(progressLine(event)).toBeNull()
  })

  it('trims the spaces round a line', () => {
    expect(progressLine('  Assessing…  ')).toBe('Assessing…')
  })
})

describe('nextLine: the live status line', () => {
  it('is replaced by each new event', () => {
    let line: string | null = null
    const seen: (string | null)[] = []
    for (const event of PROGRESS_SEQUENCE.events) {
      line = nextLine(line, { type: 'event', event })
      seen.push(line)
    }
    expect(seen).toEqual(PROGRESS_SEQUENCE.lines)
  })

  it('keeps the line it has when an unknown shape arrives, without a crash', () => {
    expect(nextLine('Assessing…', { type: 'event', event: { x: 1 } })).toBe('Assessing…')
    expect(nextLine(null, { type: 'event', event: 42 })).toBeNull()
  })

  it('clears when the run ends', () => {
    expect(nextLine('Checking the answer…', { type: 'end' })).toBeNull()
  })
})

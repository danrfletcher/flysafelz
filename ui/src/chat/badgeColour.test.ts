import { describe, expect, it } from 'vitest'

import { badgeColour, type BadgeColour } from '@/chat/badgeColour'

const SOME = ['Contact 01632 960003 before flying near H7171/26.']

// Every combination of verdict x approaching_limits x conditions. The rule is "most cautious first": red, then grey, then amber (with or
// without conditions), then blue, then green; no verdict is slate.
const TABLE: [string | null, boolean, readonly string[], BadgeColour][] = [
  ['NO_GO', false, [], 'no-go'],
  ['NO_GO', false, SOME, 'no-go'],
  ['NO_GO', true, [], 'no-go'],
  ['NO_GO', true, SOME, 'no-go'],
  ['UNVERIFIED', false, [], 'unverified'],
  ['UNVERIFIED', false, SOME, 'unverified'],
  ['UNVERIFIED', true, [], 'unverified'],
  ['UNVERIFIED', true, SOME, 'unverified'],
  ['PILOT_DISCRETION', true, [], 'amber'],
  ['PILOT_DISCRETION', true, SOME, 'amber'],
  ['PILOT_DISCRETION', false, SOME, 'conditions'],
  ['PILOT_DISCRETION', false, [], 'discretion'],
  [null, false, [], 'none'],
  [null, false, SOME, 'none'],
  [null, true, [], 'none'],
  [null, true, SOME, 'none'],
]

describe('badgeColour', () => {
  it.each(TABLE)('verdict %s, approaching limits %s, conditions %j -> %s', (verdict, approaching_limits, conditions, expected) => {
    expect(badgeColour({ verdict, approaching_limits, conditions })).toBe(expected)
  })

  it('covers every combination of the four verdicts, both flags and empty/non-empty conditions', () => {
    const seen = new Set(TABLE.map(([verdict, approaching, conditions]) => `${verdict}|${approaching}|${conditions.length > 0}`))
    expect(seen.size).toBe(4 * 2 * 2)
  })

  it('Blue Mill 10:00-11:00 (pilot discretion, approaching limits, with a condition) is amber', () => {
    const segment = { verdict: 'PILOT_DISCRETION', approaching_limits: true, conditions: ['Contact 01632 960003 before flying near H7171/26.'] }
    expect(badgeColour(segment)).toBe('amber')
  })

  it('treats a verdict it does not know as no verdict, never as a colour', () => {
    expect(badgeColour({ verdict: 'SOMETHING_NEW', approaching_limits: true, conditions: SOME })).toBe('none')
    expect(badgeColour({ verdict: '', approaching_limits: false, conditions: [] })).toBe('none')
  })
})

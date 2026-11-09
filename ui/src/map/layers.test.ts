import { describe, expect, it } from 'vitest'

import { airspaceBelowFl100, notamBelowFl100 } from '@/map/belowFl100'
import { drawnAirspace, drawnFeatures, drawnNotams } from '@/map/layers'
import type { AirspaceProps, MapFeature, NotamProps } from '@/map/types'
import { ALL_HEIGHTS_AIRSPACE, ALL_HEIGHTS_NOTAMS } from '@/test/handlers'

// Which features each route draws for the four on/off combinations of its two switches, over the synthetic "all heights" payloads
// (240 stacked airspaces, 60 NOTAMs). The rule for "below" is belowFl100.ts's; these tests count what comes out of it, not the rule itself.

const AIRSPACE = ALL_HEIGHTS_AIRSPACE.features
const NOTAMS = ALL_HEIGHTS_NOTAMS.features
const ids = <P extends { id: string }>(features: readonly MapFeature<P>[]) => features.map((feature) => feature.properties.id)

const LOW_AIRSPACE = AIRSPACE.filter((feature) => airspaceBelowFl100(feature.properties))
const LOW_NOTAMS = NOTAMS.filter((feature) => notamBelowFl100(feature.properties))

describe('the synthetic all-heights payloads', () => {
  it('have at least 200 airspaces, with both low and high floors, and NOTAMs with Q-line floors on both sides of FL100', () => {
    expect(AIRSPACE.length).toBeGreaterThanOrEqual(200)
    expect(LOW_AIRSPACE.length).toBeGreaterThan(50)
    expect(AIRSPACE.length - LOW_AIRSPACE.length).toBeGreaterThan(50)
    expect(LOW_NOTAMS.length).toBeGreaterThan(10)
    expect(NOTAMS.length - LOW_NOTAMS.length).toBeGreaterThan(10)
    expect(new Set(ids(AIRSPACE)).size).toBe(AIRSPACE.length)
    expect(new Set(ids(NOTAMS)).size).toBe(NOTAMS.length)
  })
})

describe.each([
  ['airspace', AIRSPACE, LOW_AIRSPACE, (f: readonly MapFeature<AirspaceProps>[], s: { below: boolean; all: boolean }) => drawnAirspace(f, s)],
  ['NOTAMs', NOTAMS, LOW_NOTAMS, (f: readonly MapFeature<NotamProps>[], s: { below: boolean; all: boolean }) => drawnNotams(f, s)],
] as const)('%s: the four combinations of the two switches', (_name, all, low, draw) => {
  const run = (below: boolean, every: boolean) => ids(draw(all as never, { below, all: every }) as readonly MapFeature<{ id: string }>[])

  it('draws nothing with both off', () => {
    expect(run(false, false)).toEqual([])
  })

  it('draws only the below-FL100 features with "below" on', () => {
    expect(run(true, false)).toEqual(ids(low as readonly MapFeature<{ id: string }>[]))
  })

  it('draws every feature with "all heights" on alone', () => {
    expect(run(false, true)).toEqual(ids(all as readonly MapFeature<{ id: string }>[]))
  })

  it('draws every feature once with both on: the same set as "all heights" alone, no id twice', () => {
    const both = run(true, true)
    expect(both).toEqual(run(false, true))
    expect(new Set(both).size).toBe(both.length)
  })

  it('leaves exactly the set the remaining switch implies when either is turned off', () => {
    expect(run(true, true).length).toBeGreaterThan(run(true, false).length) // the high ones are the all-heights layer's
    expect(run(true, false)).toEqual(ids(low as readonly MapFeature<{ id: string }>[]))
    expect(run(false, true)).toEqual(ids(all as readonly MapFeature<{ id: string }>[]))
  })
})

describe('drawnFeatures: one id is drawn once', () => {
  const feature = (id: string, lower_fl: number | null): MapFeature<NotamProps> =>
    ({ ...NOTAMS[0], properties: { ...NOTAMS[0].properties, id, lower_fl } }) as MapFeature<NotamProps>

  it('keeps the first feature of a repeated id, in either layer', () => {
    const repeated = [feature('H9001/26', 0), feature('H9002/26', 150), feature('H9001/26', 0), feature('H9002/26', 150)]
    expect(ids(drawnNotams(repeated, { below: false, all: true }))).toEqual(['H9001/26', 'H9002/26'])
    expect(ids(drawnNotams(repeated, { below: true, all: true }))).toEqual(['H9001/26', 'H9002/26'])
    expect(ids(drawnNotams(repeated, { below: true, all: false }))).toEqual(['H9001/26'])
  })

  it('takes the "below" rule as given and returns [] for an empty payload', () => {
    expect(drawnFeatures<NotamProps>([], { below: true, all: true }, () => true)).toEqual([])
    const sample = [feature('H9003/26', 150)]
    expect(drawnFeatures(sample, { below: true, all: false }, () => true)).toHaveLength(1)
    expect(drawnFeatures(sample, { below: true, all: false }, () => false)).toHaveLength(0)
  })
})

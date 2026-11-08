import { airspaceBelowFl100, notamBelowFl100 } from '@/map/belowFl100'
import type { AirspaceProps, MapFeature, NotamProps } from '@/map/types'

/** One route's two switches: its "below FL100" layer and its "all heights" layer. */
export interface HeightSwitches {
  below: boolean
  all: boolean
}

/**
 * The features a route's layers draw, by its two switches. "All heights" draws every feature in the payload; "below FL100" draws those
 * whose lower limit is below FL100 (`isBelow`, the one rule in belowFl100.ts); with both on, the set is the whole payload once, so no
 * shape is doubled. Either alone gives exactly its own set, and neither gives none. A feature id is drawn once (the first one wins).
 */
export function drawnFeatures<P extends { id: string }>(
  features: readonly MapFeature<P>[],
  switches: HeightSwitches,
  isBelow: (props: P) => boolean,
): readonly MapFeature<P>[] {
  if (!switches.below && !switches.all) return []
  const wanted = switches.all ? features : features.filter((feature) => isBelow(feature.properties))
  const seen = new Set<string>()
  return wanted.filter((feature) => {
    const id = feature.properties.id
    if (seen.has(id)) return false
    seen.add(id)
    return true
  })
}

export const drawnAirspace = (features: readonly MapFeature<AirspaceProps>[], switches: HeightSwitches) =>
  drawnFeatures(features, switches, airspaceBelowFl100)

export const drawnNotams = (features: readonly MapFeature<NotamProps>[], switches: HeightSwitches) =>
  drawnFeatures(features, switches, notamBelowFl100)

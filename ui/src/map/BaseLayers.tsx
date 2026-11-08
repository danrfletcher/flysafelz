import L from 'leaflet'
import { useEffect, useMemo } from 'react'
import { useMap } from 'react-leaflet'

import { airspaceBelowFl100, notamBelowFl100 } from '@/map/belowFl100'
import { airspaceStyle, notamStyle, pathOptions } from '@/map/airspaceStyle'
import { airspaceTooltip, notamTooltip } from '@/map/tooltips'
import type { AirspaceProps, MapFeature, NotamProps } from '@/map/types'

/** The word drawn on an approximate NOTAM shape. */
export const APPROX_LABEL = 'approx'

/**
 * One Leaflet group for the airspace features: each shape coloured by its `type` alone (an `action` property is never read),
 * with a sticky hover tooltip. Built fresh on every call; the caller adds it to the map and removes it.
 */
export function buildAirspaceGroup(features: readonly MapFeature<AirspaceProps>[]): L.FeatureGroup {
  const shapes = L.geoJSON(features as unknown as GeoJSON.Feature[], {
    style: (feature) => pathOptions(airspaceStyle((feature as MapFeature<AirspaceProps>).properties.type)),
    onEachFeature: (feature, layer) => {
      layer.bindTooltip(airspaceTooltip((feature as MapFeature<AirspaceProps>).properties), { sticky: true })
    },
  })
  return L.featureGroup([shapes])
}

/**
 * One Leaflet group for the NOTAM features: violet dashed, geometry exactly as served (no change to a coordinate), a sticky
 * tooltip, and an "approx" label at the centre of each approximate shape.
 */
export function buildNotamGroup(features: readonly MapFeature<NotamProps>[]): L.FeatureGroup {
  const group = L.featureGroup()
  const shapes = L.geoJSON(features as unknown as GeoJSON.Feature[], {
    style: (feature) => pathOptions(notamStyle((feature as MapFeature<NotamProps>).properties.approx)),
    onEachFeature: (feature, layer) => {
      const props = (feature as MapFeature<NotamProps>).properties
      layer.bindTooltip(notamTooltip(props), { sticky: true })
      if (props.approx) {
        const centre = (layer as L.Polygon).getBounds().getCenter()
        const icon = L.divIcon({
          className: 'approx-label',
          html: `<span class="approx-label__text">${APPROX_LABEL}</span>`,
          iconSize: [0, 0],
        })
        group.addLayer(L.marker(centre, { icon, interactive: false, keyboard: false }))
      }
    },
  })
  group.addLayer(shapes)
  return group
}

function useGroup<P>(features: readonly MapFeature<P>[], build: (features: readonly MapFeature<P>[]) => L.FeatureGroup) {
  const map = useMap()
  useEffect(() => {
    const group = build(features)
    group.addTo(map)
    return () => {
      group.remove()
    }
  }, [map, features, build])
}

function AirspaceLayer({ features }: { features: readonly MapFeature<AirspaceProps>[] }) {
  useGroup(features, buildAirspaceGroup)
  return null
}

function NotamLayer({ features }: { features: readonly MapFeature<NotamProps>[] }) {
  useGroup(features, buildNotamGroup)
  return null
}

export interface BaseLayersProps {
  airspace: readonly MapFeature<AirspaceProps>[]
  notams: readonly MapFeature<NotamProps>[]
  showAirspace: boolean
  showNotams: boolean
}

/**
 * The two base layers inside a MapContainer: airspace and NOTAMs whose lower limit is below FL100, each drawn only while its
 * switch is on. A layer that is off is not mounted at all, so switching on and off leaves exactly one Leaflet group, or none.
 */
export function BaseLayers({ airspace, notams, showAirspace, showNotams }: BaseLayersProps) {
  const lowAirspace = useMemo(() => airspace.filter((feature) => airspaceBelowFl100(feature.properties)), [airspace])
  const lowNotams = useMemo(() => notams.filter((feature) => notamBelowFl100(feature.properties)), [notams])
  return (
    <>
      {showAirspace && <AirspaceLayer features={lowAirspace} />}
      {showNotams && <NotamLayer features={lowNotams} />}
    </>
  )
}

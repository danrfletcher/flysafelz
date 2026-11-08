import L from 'leaflet'
import { useEffect, useMemo } from 'react'
import { useMap } from 'react-leaflet'

import { drawnAirspace, drawnNotams } from '@/map/layers'
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
  /** The "all heights" switches: every feature of the route, whatever its lower limit. */
  showAirspaceAll: boolean
  showNotamsAll: boolean
}

/**
 * The base layers inside a MapContainer: airspace and NOTAMs, each route drawn as one group by its two switches ("below FL100" and "all
 * heights", see layers.ts). Both on draws the route's features once, never twice. A route with both switches off is not mounted at all,
 * so switching leaves exactly one Leaflet group per route, or none. The groups sit in the default overlay pane, under the answer pane.
 */
export function BaseLayers({ airspace, notams, showAirspace, showNotams, showAirspaceAll, showNotamsAll }: BaseLayersProps) {
  const airspaceDrawn = useMemo(() => drawnAirspace(airspace, { below: showAirspace, all: showAirspaceAll }), [airspace, showAirspace, showAirspaceAll])
  const notamsDrawn = useMemo(() => drawnNotams(notams, { below: showNotams, all: showNotamsAll }), [notams, showNotams, showNotamsAll])
  const airspaceOn = showAirspace || showAirspaceAll
  const notamsOn = showNotams || showNotamsAll
  return (
    <>
      {airspaceOn && <AirspaceLayer features={airspaceDrawn} />}
      {notamsOn && <NotamLayer features={notamsDrawn} />}
    </>
  )
}

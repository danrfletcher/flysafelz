import L from 'leaflet'
import { useEffect, useMemo, useRef } from 'react'
import { useMap } from 'react-leaflet'

import type { Payload } from '@/chat/turns'
import { actionStyle } from '@/map/actionStyle'
import { pathOptions } from '@/map/airspaceStyle'
import { APPROX_LABEL } from '@/map/BaseLayers'
import { readAnswerFeatures, type AnswerFeature } from '@/map/answerFeatures'
import { LZ_RADIUS_NM, METRES_PER_NM, fitPadding, type ChatLayout } from '@/map/fitLz'
import type { LatLon } from '@/pick/whatsHere'

/** The map pane that holds an answer's highlights: above the base layers' pane (400) and their tooltips' targets, below the markers (600). */
export const ANSWER_PANE = 'answerOverlays'
const ANSWER_PANE_Z = '450'
const CIRCLE_OUTLINE = 2
const CIRCLE_FILL = 0.05
const POINT_RADIUS_PX = 8

/** What to draw for the turn being shown. */
export interface AnswerHighlight {
  turnId: string
  /** The turn's `overlays`, as the server sent them. */
  overlays: readonly Payload[]
  /** The NOTAM whose shape a click on its id picked out, or null. */
  focusId: string | null
}

function style(feature: AnswerFeature, focused: boolean): L.PathOptions {
  return { ...pathOptions(actionStyle(feature.action, feature.approx, focused)), pane: ANSWER_PANE, interactive: false }
}

function shapeOf(feature: AnswerFeature, focused: boolean): L.Layer | null {
  if (!feature.geometry) return null
  const options = style(feature, focused)
  try {
    return L.geoJSON({ type: 'Feature', geometry: feature.geometry, properties: {} } as GeoJSON.Feature, {
      ...options,
      style: () => options,
      pointToLayer: (_point, latlng) => L.circleMarker(latlng, { ...options, radius: POINT_RADIUS_PX }),
    })
  } catch {
    return null // a geometry Leaflet cannot build: this feature is skipped and the others are still drawn
  }
}

/**
 * One group with the 2 NM circle round the LZ and the answer's NOTAM and airspace shapes, each in its action colour, all in the answer pane.
 * A feature with no drawable geometry is skipped. An approximate NOTAM is dashed and carries the word "approx" at its centre. The picked
 * NOTAM is drawn last (on top) and heavier. Nothing here takes part in a click or hover: the map beneath still hears them.
 */
export function buildAnswerGroup(point: LatLon, features: readonly AnswerFeature[], focusId: string | null): L.FeatureGroup {
  const group = L.featureGroup()
  group.addLayer(
    L.circle([point.lat, point.lon], {
      ...pathOptions({ token: '--primary', weight: CIRCLE_OUTLINE, fillOpacity: CIRCLE_FILL }),
      radius: LZ_RADIUS_NM * METRES_PER_NM,
      pane: ANSWER_PANE,
      interactive: false,
    }),
  )
  const picked = (feature: AnswerFeature) => feature.kind === 'notam' && feature.id === focusId
  for (const feature of [...features.filter((item) => !picked(item)), ...features.filter(picked)]) {
    const shape = shapeOf(feature, picked(feature))
    if (!shape) continue
    group.addLayer(shape)
    if (feature.approx && feature.geometry?.type !== 'Point') {
      const centre = (shape as L.GeoJSON).getBounds().getCenter()
      const icon = L.divIcon({ className: 'approx-label', html: `<span class="approx-label__text">${APPROX_LABEL}</span>`, iconSize: [0, 0] })
      group.addLayer(L.marker(centre, { icon, interactive: false, keyboard: false, pane: ANSWER_PANE }))
    }
  }
  return group
}

export interface AnswerOverlaysProps {
  /** The LZ the chat is open for. With no LZ nothing is drawn. */
  point: LatLon | null
  /** The turn to show, or null for none. */
  highlight: AnswerHighlight | null
  /** The "Answer overlays" switch. */
  visible: boolean
  layout: ChatLayout
}

/**
 * Inside the map: draws one turn's highlights (the 2 NM circle plus its overlays) in the "Answer overlays" pane, and fits the map to them,
 * padded for the sheet or the drawer. A different turn replaces the whole set. The base layers are never touched. Renders nothing.
 */
export function AnswerOverlays({ point, highlight, visible, layout }: AnswerOverlaysProps) {
  const map = useMap()
  const drawn = useRef<L.FeatureGroup | null>(null)
  const overlays = highlight?.overlays
  const features = useMemo(() => readAnswerFeatures(overlays ?? []), [overlays])
  const focusId = highlight?.focusId ?? null
  const turnId = highlight?.turnId
  const lat = point?.lat
  const lon = point?.lon
  const shown = visible && turnId !== undefined && lat !== undefined && lon !== undefined

  useEffect(() => {
    if (!shown) return
    const pane = map.getPane(ANSWER_PANE) ?? map.createPane(ANSWER_PANE)
    pane.style.zIndex = ANSWER_PANE_Z
    const group = buildAnswerGroup({ lat, lon }, features, focusId)
    group.addTo(map)
    drawn.current = group
    return () => {
      group.remove()
      drawn.current = null
    }
  }, [map, shown, lat, lon, features, focusId])

  // A new turn, the switch coming on, or a new layout fits the map; picking out a NOTAM in the turn already shown does not move it.
  useEffect(() => {
    const bounds = shown ? drawn.current?.getBounds() : undefined
    if (bounds?.isValid()) map.fitBounds(bounds, fitPadding(layout, map.getSize()))
  }, [map, shown, lat, lon, turnId, layout])

  return null
}

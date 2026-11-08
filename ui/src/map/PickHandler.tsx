import L from 'leaflet'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Marker, Popup, useMap, useMapEvents } from 'react-leaflet'

import { NOT_LAND_MESSAGE, Toast } from '@/components/Toast'
import type { AirspaceProps, NotamProps } from '@/map/types'
import type { LayerState } from '@/map/useMapLayers'
import { isUkLand } from '@/pick/land'
import { WhatsHereCard } from '@/pick/WhatsHereCard'
import type { LatLon } from '@/pick/whatsHere'

// The pin: an SVG teardrop in the theme's accent, its point on the picked coordinates. Static markup, nothing from the server in it.
const PIN_ICON = L.divIcon({
  className: 'pick-pin',
  html:
    '<svg viewBox="0 0 28 38" width="28" height="38" aria-hidden="true"><path d="M14 1C7 1 1.5 6.5 1.5 13.4 1.5 22.7 14 37 14 37S26.5 22.7 26.5 13.4C26.5 6.5 21 1 14 1Z"/><circle cx="14" cy="13.5" r="4.5"/></svg>',
  iconSize: [28, 38],
  iconAnchor: [14, 37],
  popupAnchor: [0, -34],
})

/** Keeps the open card on screen when its height changes (layers arriving, rows opening): Leaflet re-measures and pans if it must. */
function KeepInView({ popup, children }: { popup: { current: L.Popup | null }; children: ReactNode }) {
  const element = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const node = element.current
    if (!node) return
    const watcher = new ResizeObserver(() => popup.current?.update())
    watcher.observe(node)
    return () => watcher.disconnect()
  }, [popup])
  return <div ref={element}>{children}</div>
}

interface PinProps {
  point: LatLon
  airspace: LayerState<AirspaceProps>
  notams: LayerState<NotamProps>
}

/** The pin and its open card. Mounted fresh (by its key) for each picked point, so a new point never shows an old card's open rows. */
function Pin({ point, airspace, notams }: PinProps) {
  const marker = useRef<L.Marker>(null)
  const popup = useRef<L.Popup>(null)

  useEffect(() => {
    marker.current?.openPopup()
  }, [])

  return (
    <Marker ref={marker} position={[point.lat, point.lon]} icon={PIN_ICON} title="Picked field" keyboard>
      <Popup ref={popup} className="pick-card" closeButton={false} closeOnClick={false} autoClose={false} minWidth={240} maxWidth={320}>
        <KeepInView popup={popup}>
          <WhatsHereCard point={point} airspace={airspace} notams={notams} />
        </KeepInView>
      </Popup>
    </Marker>
  )
}

export interface PickHandlerProps {
  airspace: LayerState<AirspaceProps>
  notams: LayerState<NotamProps>
}

/**
 * Inside the map: a click on UK land puts the one pin there and opens the card at it (replacing any earlier pin and card); a click on
 * sea or outside the UK shows the toast and changes nothing else. The card reads the layers handed in (what was loaded, not what is
 * switched on) and makes no request of its own. Clicks inside the card and on the toast stop before the map (Leaflet's own popup
 * container does this for the card).
 */
export function PickHandler({ airspace, notams }: PickHandlerProps) {
  const map = useMap()
  const [point, setPoint] = useState<LatLon | null>(null)
  const [toast, setToast] = useState<number | null>(null)

  useMapEvents({
    click(event) {
      // Panned round the world, a click can report a longitude past 180: bring it back. An ordinary one is kept exactly as given.
      const { lat, lng } = Math.abs(event.latlng.lng) > 180 ? event.latlng.wrap() : event.latlng
      if (isUkLand(lat, lng)) setPoint({ lat, lon: lng })
      else setToast((run) => (run ?? 0) + 1)
    },
  })

  const dismiss = useCallback(() => setToast(null), [])

  return (
    <>
      {point && <Pin key={`${point.lat},${point.lon}`} point={point} airspace={airspace} notams={notams} />}
      {toast !== null && createPortal(<Toast key={toast} message={NOT_LAND_MESSAGE} onDone={dismiss} />, map.getContainer())}
    </>
  )
}

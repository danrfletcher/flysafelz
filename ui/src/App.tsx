import { useState } from 'react'

import { ChatPanel } from '@/chat/ChatPanel'
import { createSessions, pinKey } from '@/chat/turns'
import { Footer } from '@/components/Footer'
import { LAYERS, LayerPanel } from '@/components/LayerPanel'
import { NavBar } from '@/components/NavBar'
import { UkMap } from '@/components/UkMap'
import { useIsDesktop } from '@/lib/useIsDesktop'
import { useLayerPrefs } from '@/lib/useLayerPrefs'
import { AnswerOverlays } from '@/map/answerOverlays'
import { BaseLayers } from '@/map/BaseLayers'
import type { BasemapStatus } from '@/map/basemap'
import { FitToLz } from '@/map/FitToLz'
import { PickHandler } from '@/map/PickHandler'
import { RainGridLayer } from '@/map/RainGridLayer'
import { RainLayer } from '@/map/RainLayer'
import { RainStamp } from '@/map/RainStamp'
import { selectTurn, useAnswerHighlight, type Selection } from '@/map/useAnswerHighlight'
import { useMapLayers } from '@/map/useMapLayers'
import { useRain } from '@/map/useRain'
import { useWind } from '@/map/useWind'
import { WindLayer } from '@/map/WindLayer'
import type { LatLon } from '@/pick/whatsHere'

/**
 * Nav bar, the map in the space between (with the base layers, the click-to-pick pin and card, the layer panel and the chat), and the
 * footer. The page itself never scrolls. The chat's conversations live here, one per pin for as long as the page stays loaded.
 * Putting down a new pin closes the chat, which ends any run still going for the old pin; its turns stay with it in `sessions`.
 * The map shows the highlights of one turn of the open chat: the latest answer's, unless the pilot picked another turn's badges or a NOTAM id.
 * Closing the chat or putting down a new pin clears them.
 */
export default function App() {
  const layers = useMapLayers()
  const wind = useWind()
  const rain = useRain()
  // Wind, rain, airspace and NOTAMs below FL100 and the answer's highlights are on at first load; a choice is kept in this browser's local storage and read back on load.
  const [shown, setShown] = useLayerPrefs(LAYERS)
  // Whether the basemap tiles are loading: only the layer panel's note and the map's background read it, so nothing else waits on it.
  const [basemap, setBasemap] = useState<BasemapStatus>('loading')
  const [sessions] = useState(createSessions)
  const [chatPoint, setChatPoint] = useState<LatLon | null>(null)
  const [mapArea, setMapArea] = useState<HTMLElement | null>(null)
  const [selection, setSelection] = useState<Selection | null>(null)
  const highlight = useAnswerHighlight(sessions, chatPoint, selection)
  const desktop = useIsDesktop()

  const closeChat = () => {
    setChatPoint(null)
    setSelection(null)
  }
  // A click on a turn's badges or on a NOTAM id in it draws that turn's highlights, and turns the Answer overlays switch back on.
  const showTurn = (turnId: string, notamId: string | null = null) => {
    if (!chatPoint) return
    setSelection(selectTurn(sessions.get(pinKey(chatPoint)), chatPoint, turnId, notamId))
    setShown('answer', true)
  }

  return (
    <div className="flex h-dvh flex-col">
      <NavBar />
      <main ref={setMapArea} className="relative min-h-0 flex-1 overflow-hidden">
        <UkMap basemap={basemap} onBasemapStatus={setBasemap}>
          {shown.wind && wind.grid && <WindLayer grid={wind.grid} />}
          {shown.rain && rain.state.mode === 'radar' && (
            <RainLayer host={rain.state.host} frames={rain.state.frames} fetchedAt={rain.state.fetchedAt} gate={rain.gate} view={rain.view} />
          )}
          {shown.rain && rain.state.mode === 'fallback' && <RainGridLayer cells={rain.state.cells} view={rain.view} />}
          <BaseLayers
            airspace={layers.airspace.features}
            notams={layers.notams.features}
            showAirspace={shown.airspace}
            showNotams={shown.notams}
          />
          <PickHandler airspace={layers.airspace} notams={layers.notams} onChat={setChatPoint} onPick={closeChat} />
          <FitToLz point={chatPoint} layout={desktop ? 'sheet' : 'drawer'} />
          <AnswerOverlays point={chatPoint} highlight={highlight} visible={shown.answer} layout={desktop ? 'sheet' : 'drawer'} />
        </UkMap>
        {shown.rain && rain.state.mode === 'radar' && <RainStamp view={rain.view} />}
        <LayerPanel
          states={{ wind, ...layers }}
          rain={{ state: rain.state, view: rain.view }}
          basemap={basemap}
          shown={shown}
          chatOpen={chatPoint !== null}
          onShownChange={setShown}
        />
        <ChatPanel
          point={chatPoint}
          sessions={sessions}
          onClose={closeChat}
          container={mapArea}
          onShowTurn={(turnId) => showTurn(turnId)}
          onShowNotam={showTurn}
        />
      </main>
      <Footer />
    </div>
  )
}

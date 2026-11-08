import { useState } from 'react'

import { ChatPanel } from '@/chat/ChatPanel'
import { createSessions } from '@/chat/turns'
import { Footer } from '@/components/Footer'
import { LayerPanel, type LayerSwitches } from '@/components/LayerPanel'
import { NavBar } from '@/components/NavBar'
import { UkMap } from '@/components/UkMap'
import { useIsDesktop } from '@/lib/useIsDesktop'
import { BaseLayers } from '@/map/BaseLayers'
import { FitToLz } from '@/map/FitToLz'
import { PickHandler } from '@/map/PickHandler'
import { useMapLayers } from '@/map/useMapLayers'
import type { LatLon } from '@/pick/whatsHere'

// Airspace and NOTAMs below FL100 are on at first load. The choice is kept in memory only: a reload puts both back on.
const DEFAULT_SWITCHES: LayerSwitches = { airspace: true, notams: true }

/**
 * Nav bar, the map in the space between (with the base layers, the click-to-pick pin and card, the layer panel and the chat), and the
 * footer. The page itself never scrolls. The chat's conversations live here, one per pin for as long as the page stays loaded.
 * Putting down a new pin closes the chat, which ends any run still going for the old pin; its turns stay with it in `sessions`.
 */
export default function App() {
  const layers = useMapLayers()
  const [shown, setShown] = useState<LayerSwitches>(DEFAULT_SWITCHES)
  const [sessions] = useState(createSessions)
  const [chatPoint, setChatPoint] = useState<LatLon | null>(null)
  const [mapArea, setMapArea] = useState<HTMLElement | null>(null)
  const desktop = useIsDesktop()

  return (
    <div className="flex h-dvh flex-col">
      <NavBar />
      <main ref={setMapArea} className="relative min-h-0 flex-1 overflow-hidden">
        <UkMap>
          <BaseLayers
            airspace={layers.airspace.features}
            notams={layers.notams.features}
            showAirspace={shown.airspace}
            showNotams={shown.notams}
          />
          <PickHandler airspace={layers.airspace} notams={layers.notams} onChat={setChatPoint} onPick={() => setChatPoint(null)} />
          <FitToLz point={chatPoint} layout={desktop ? 'sheet' : 'drawer'} />
        </UkMap>
        <LayerPanel
          states={layers}
          shown={shown}
          chatOpen={chatPoint !== null}
          onShownChange={(id, on) => setShown((current) => ({ ...current, [id]: on }))}
        />
        <ChatPanel point={chatPoint} sessions={sessions} onClose={() => setChatPoint(null)} container={mapArea} />
      </main>
      <Footer />
    </div>
  )
}

import { useState } from 'react'

import { Footer } from '@/components/Footer'
import { LayerPanel, type LayerSwitches } from '@/components/LayerPanel'
import { NavBar } from '@/components/NavBar'
import { UkMap } from '@/components/UkMap'
import { BaseLayers } from '@/map/BaseLayers'
import { PickHandler } from '@/map/PickHandler'
import { useMapLayers } from '@/map/useMapLayers'

// Airspace and NOTAMs below FL100 are on at first load. The choice is kept in memory only: a reload puts both back on.
const DEFAULT_SWITCHES: LayerSwitches = { airspace: true, notams: true }

/** Nav bar, the map in the space between (with the base layers, the click-to-pick pin and card, and the layer panel), and the footer. The page itself never scrolls. */
export default function App() {
  const layers = useMapLayers()
  const [shown, setShown] = useState<LayerSwitches>(DEFAULT_SWITCHES)

  return (
    <div className="flex h-dvh flex-col">
      <NavBar />
      <main className="relative min-h-0 flex-1">
        <UkMap>
          <BaseLayers
            airspace={layers.airspace.features}
            notams={layers.notams.features}
            showAirspace={shown.airspace}
            showNotams={shown.notams}
          />
          <PickHandler airspace={layers.airspace} notams={layers.notams} />
        </UkMap>
        <LayerPanel
          states={layers}
          shown={shown}
          onShownChange={(id, on) => setShown((current) => ({ ...current, [id]: on }))}
        />
      </main>
      <Footer />
    </div>
  )
}

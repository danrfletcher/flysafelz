import { Footer } from '@/components/Footer'
import { NavBar } from '@/components/NavBar'
import { UkMap } from '@/components/UkMap'

/** Nav bar, the map in the space between, and the footer. The page itself never scrolls. */
export default function App() {
  return (
    <div className="flex h-dvh flex-col">
      <NavBar />
      <main className="relative min-h-0 flex-1">
        <UkMap />
      </main>
      <Footer />
    </div>
  )
}

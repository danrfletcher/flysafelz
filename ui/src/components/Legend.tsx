import { HOW_IT_WORKS } from '@/content/howItWorks'
import { notamStyle } from '@/map/airspaceStyle'
import type { RainState } from '@/map/useRain'
import { WIND_BANDS, type WindBand } from '@/map/wind'

// The legend's colours are theme variables, named through Tailwind tokens (src/styles/index.css): no colour is chosen here by a function.

const OUTLINE = 'inline-block h-3.5 w-6 shrink-0 rounded-sm border-2'

interface Row {
  label: string
  /** The swatch's classes. */
  swatch: string
  /** An SVG dash pattern, for a swatch that must show the map line's own dashes: the swatch is then an outlined rectangle drawn with it. */
  dash?: string
}

/** What the base layers look like (Q10). */
export const ON_THE_MAP: readonly Row[] = [
  { label: 'Restricted or prohibited', swatch: `${OUTLINE} border-type-restricted bg-type-restricted/20` },
  { label: 'Danger', swatch: `${OUTLINE} border-type-danger` },
  { label: 'CTR, CTA or ATZ', swatch: `${OUTLINE} border-type-controlled` },
  { label: 'Other airspace', swatch: `${OUTLINE} border-type-other` },
  { label: 'NOTAM', swatch: `${OUTLINE} border-dashed border-type-notam` },
  // The map draws an approximate NOTAM with a short dash pattern (notamStyle), which a CSS border cannot copy, so this swatch draws that pattern itself.
  { label: 'NOTAM, approx (Q-line circle)', swatch: 'h-3.5 w-6 shrink-0 stroke-type-notam', dash: notamStyle(true).dashArray },
]

// Written out so Tailwind sees each class name.
const WIND_SWATCH: Record<WindBand, string> = {
  light: 'bg-wind-light',
  moderate: 'bg-wind-moderate',
  strong: 'bg-wind-strong',
  severe: 'bg-wind-severe',
}

/** The wind layer's speed bands: a short bar in the band's colour, with its speeds in words. */
export const WIND_SPEEDS: readonly Row[] = WIND_BANDS.map(({ name, label }) => ({
  label,
  swatch: `inline-block h-1 w-6 shrink-0 rounded-full ${WIND_SWATCH[name]}`,
}))

/** The Rain layer's legend line: radar, or (when RainViewer's index failed) Open-Meteo's model, said as it is. None while the layer has no data at all. */
export function rainRow(mode: RainState['mode']): Row | null {
  const swatch = 'inline-block h-3.5 w-6 shrink-0 rounded-sm bg-rain/60'
  if (mode === 'fallback') return { label: 'rain (model, not radar)', swatch }
  if (mode === 'unavailable') return null
  return { label: 'Rain radar, past 2 h', swatch }
}

/** What an answer says about a shape for the field you picked. */
export const ACTIONS: readonly Row[] = [
  { label: 'No fly', swatch: `${OUTLINE} border-action-no-fly bg-action-no-fly/20` },
  { label: 'Contact first', swatch: `${OUTLINE} border-action-contact-first` },
  { label: 'Check SUP', swatch: `${OUTLINE} border-action-check-sup` },
  { label: 'Avoid sector', swatch: `${OUTLINE} border-action-avoid-sector` },
  { label: 'Awareness', swatch: `${OUTLINE} border-action-awareness` },
]

const BLOCK = 'inline-block size-3.5 shrink-0 rounded-full'

/** The verdict badges, most cautious first. */
export const VERDICTS: readonly Row[] = [
  { label: 'NO-GO', swatch: `${BLOCK} bg-verdict-no-go` },
  { label: 'UNVERIFIED', swatch: `${BLOCK} bg-verdict-unverified` },
  { label: 'APPROACHING LIMITS', swatch: `${BLOCK} bg-verdict-amber` },
  { label: 'WITH CONDITIONS', swatch: `${BLOCK} bg-verdict-conditions` },
  { label: 'PILOT DISCRETION', swatch: `${BLOCK} bg-verdict-discretion` },
]

function Rows({ rows }: { rows: readonly Row[] }) {
  return (
    <ul className="space-y-1">
      {rows.map((row) => (
        <li key={row.label} className="flex items-center gap-2">
          {row.dash ? (
            <svg className={row.swatch} viewBox="0 0 24 14" aria-hidden="true">
              <rect x="1" y="1" width="22" height="12" rx="2" fill="none" strokeWidth="2" strokeDasharray={row.dash} />
            </svg>
          ) : (
            <span className={row.swatch} aria-hidden="true" />
          )}
          <span>{row.label}</span>
        </li>
      ))}
    </ul>
  )
}

/** The two legend sets of the layer panel. */
export function Legend({ rain = 'loading' }: { rain?: RainState['mode'] }) {
  const rainLine = rainRow(rain)
  return (
    <div className="space-y-4 text-sm">
      <section aria-labelledby="legend-map">
        <h3 id="legend-map" className="mb-1.5 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          On the map
        </h3>
        <Rows rows={ON_THE_MAP} />
        <div className="mt-2">
          <Rows rows={WIND_SPEEDS} />
        </div>
        {rainLine && (
          <div className="mt-2">
            <Rows rows={[rainLine]} />
          </div>
        )}
      </section>
      <section aria-labelledby="legend-field">
        <h3 id="legend-field" className="mb-1.5 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          For this field
        </h3>
        <p className="mb-1 text-xs text-muted-foreground">Shapes in an answer</p>
        <Rows rows={ACTIONS} />
        <p className="mt-2 mb-1 text-xs text-muted-foreground">Answer badges</p>
        <Rows rows={VERDICTS} />
        <p className="mt-2 text-xs text-verdict-amber">{HOW_IT_WORKS.amberLine}</p>
      </section>
    </div>
  )
}

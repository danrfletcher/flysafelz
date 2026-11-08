import { readBadges } from '@/chat/readBadges'
import type { BadgeColour } from '@/chat/badgeColour'
import type { Payload } from '@/chat/turns'

// The classes that paint each colour name. The name comes from badgeColour (one function over the machine values); nothing here reads a
// verdict, a `display` text or a word of the answer. Colour is never the only signal: the label is always text beside the dot.
const PAINT: Record<BadgeColour, { badge: string; dot: string }> = {
  'no-go': { badge: 'border-verdict-no-go bg-verdict-no-go/15', dot: 'bg-verdict-no-go' },
  unverified: { badge: 'border-verdict-unverified bg-verdict-unverified/15', dot: 'bg-verdict-unverified' },
  amber: { badge: 'border-verdict-amber bg-verdict-amber/15', dot: 'bg-verdict-amber' },
  conditions: { badge: 'border-verdict-conditions bg-verdict-conditions/15', dot: 'bg-verdict-conditions' },
  discretion: { badge: 'border-verdict-discretion bg-verdict-discretion/15', dot: 'bg-verdict-discretion' },
  none: { badge: 'border-type-other bg-type-other/15', dot: 'bg-type-other' },
}

export interface BadgesProps {
  /** The turn's own `assessment`, as the server sent it. Null draws nothing at all. */
  assessment: Payload | null
  /** A click on a badge: draw this turn's highlights on the map. */
  onSelect?: () => void
}

/**
 * The verdict badges above one answer: one per segment of that turn's `assessment`, its time span, its labels as text and its
 * conditions beneath. No assessment (or no readable segment) renders nothing, not even an empty container.
 */
export function Badges({ assessment, onSelect }: BadgesProps) {
  const badges = readBadges(assessment)
  if (!badges) return null
  return (
    <ul aria-label="Verdict by time" className="space-y-2">
      {badges.map((badge, index) => (
        <li key={index} className="space-y-1">
          <button
            type="button"
            onClick={onSelect}
            title="Show this answer on the map"
            className={`flex w-full flex-wrap items-center gap-x-2 gap-y-0.5 rounded-md border-2 px-2.5 py-1.5 text-left text-xs focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none ${PAINT[badge.colour].badge}`}
          >
            <span className={`size-2.5 shrink-0 rounded-full ${PAINT[badge.colour].dot}`} aria-hidden="true" />
            <span className="font-mono">{badge.span}</span>
            <span className="font-semibold [overflow-wrap:anywhere]">{badge.label}</span>
            {badge.tier && <span className="text-muted-foreground">{badge.tier}</span>}
          </button>
          {badge.conditions.length > 0 && (
            <ul className="list-disc space-y-0.5 pl-9 text-xs text-muted-foreground [overflow-wrap:anywhere]">
              {badge.conditions.map((condition, at) => (
                <li key={at}>{condition}</li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ul>
  )
}

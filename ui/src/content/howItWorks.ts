// The "How it works" modal copy (spec §8), kept as one constant so the modal and its test read the same words.
// Dan edits this copy at Review: change it here and nowhere else.

/** A run of text; `{ strong }` is shown bold. */
export type Part = string | { strong: string }

export interface Step {
  lead: string
  body: string
}

export interface HowItWorks {
  title: string
  steps: readonly Step[]
  answers: { lead: string; body: readonly Part[] }
  checks: { lead: string; body: string }
  /** The amber legend line, shown on its own so it matches the legend later PRs add. */
  amberLine: string
  closing: string
}

export const HOW_IT_WORKS: HowItWorks = {
  title: 'How FlySafeLZ works',
  steps: [
    { lead: 'Pick a field.', body: "Click any spot on land in the UK where you'd take off." },
    { lead: "See what's here.", body: 'We list the airspace and NOTAMs over that point.' },
    {
      lead: 'Ask.',
      body: 'Chat about the field: whether a time window is ruled out, what a NOTAM means, what the rules say.',
    },
  ],
  answers: {
    lead: 'What the answers mean.',
    body: [
      'FlySafeLZ only tells you when ',
      { strong: 'not' },
      " to fly. NO-GO: the data rules it out. UNVERIFIED: we couldn't check something, so treat it as a no until you can. PILOT DISCRETION: nothing we checked rules it out; the decision is yours once you're on the field, sometimes with conditions (call a number first, avoid a sector) or near your limits. Amber means near your limits: check the conditions too.",
    ],
  },
  checks: {
    lead: 'What we check.',
    body: 'The forecast against BHPA-style limits, daylight, permanent airspace (openAIP) and today\'s NOTAMs (NATS), read and applied by code, not guessed by the AI.',
  },
  amberLine: 'amber = near your limits, check the conditions too',
  closing: 'Decision support only, not a primary means of navigation. Always do your own pre-flight checks.',
}

/** The text of a run of parts, bold markers dropped. */
export function partsText(parts: readonly Part[]): string {
  return parts.map((part) => (typeof part === 'string' ? part : part.strong)).join('')
}

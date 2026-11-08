import { Info } from 'lucide-react'

import { Dialog, DialogContent, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { HOW_IT_WORKS, type Part } from '@/content/howItWorks'

function Parts({ parts }: { parts: readonly Part[] }) {
  return parts.map((part, index) =>
    typeof part === 'string' ? part : <strong key={index}>{part.strong}</strong>,
  )
}

/** The ⓘ button and the "How it works" dialog it opens. Radix closes it on Esc, the close button and an overlay click, and puts focus back on ⓘ. */
export function InfoModal() {
  const copy = HOW_IT_WORKS
  return (
    <Dialog>
      <DialogTrigger
        aria-label="How it works"
        className="inline-flex size-9 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
      >
        <Info className="size-5" aria-hidden="true" />
      </DialogTrigger>
      <DialogContent aria-describedby={undefined}>
        <DialogTitle>{copy.title}</DialogTitle>
        <ol className="mt-4 list-decimal space-y-2 pl-5 text-sm">
          {copy.steps.map((step) => (
            <li key={step.lead}>
              <strong>{step.lead}</strong> {step.body}
            </li>
          ))}
        </ol>
        <p className="mt-4 text-sm">
          <strong>{copy.answers.lead}</strong> <Parts parts={copy.answers.body} />
        </p>
        <p className="mt-2 rounded-md bg-muted px-3 py-2 text-sm text-verdict-amber">{copy.amberLine}</p>
        <p className="mt-4 text-sm">
          <strong>{copy.checks.lead}</strong> {copy.checks.body}
        </p>
        <p className="mt-4 text-sm text-muted-foreground">{copy.closing}</p>
      </DialogContent>
    </Dialog>
  )
}

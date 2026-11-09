import { useEffect, useMemo, useRef, type ComponentProps, type ReactNode } from 'react'
import Markdown, { type Components } from 'react-markdown'

import { FALLBACK_HEADER } from '@/chat/copy'
import { NOTAM_ID_ATTRIBUTE, linkNotamIds } from '@/chat/notamLinks'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import type { AnswerFeature } from '@/map/answerFeatures'

/** The NOTAMs of the turn the answer belongs to, by id, and what a click on one does. */
export interface NotamLinks {
  /** Only an id in this map is a button (E10). Keep the map the same object while the turn's overlays are the same. */
  notams: ReadonlyMap<string, AnswerFeature>
  /** Draw this turn's highlights with this NOTAM's shape picked out. */
  onShow: (id: string) => void
}

export interface AnswerProps {
  answer: string
  /** The server's `guard` value. Only the exact string "fallback" adds the header; a missing or unknown value is a normal answer. */
  guard: string | null
  /** The turn's verdict badges. They sit between the fallback header (when there is one) and the answer. */
  badges?: ReactNode
  /** The turn's NOTAMs. Without any, no id is a button. */
  links?: NotamLinks
}

const NO_TEXT = 'No text was published with this NOTAM.'

/** A NOTAM id in the answer: a button that picks the shape out on the map and opens the NOTAM's raw text, as published. */
function NotamButton({ feature, onShow }: { feature: AnswerFeature; onShow: (id: string) => void }) {
  return (
    <Popover>
      <PopoverTrigger
        onClick={() => onShow(feature.id)}
        className="cursor-pointer rounded-sm font-mono text-primary underline decoration-dotted underline-offset-2 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
      >
        {feature.id}
      </PopoverTrigger>
      <PopoverContent align="start" aria-label={`NOTAM ${feature.id}`} className="space-y-2">
        <p className="font-mono text-sm font-semibold">{feature.id}</p>
        <p className="font-mono text-xs leading-relaxed whitespace-pre-wrap [overflow-wrap:anywhere]">{feature.text ?? NO_TEXT}</p>
      </PopoverContent>
    </Popover>
  )
}

/**
 * One answer, shown whole as markdown. Raw HTML is not rendered: react-markdown turns it into text, and no rehype-raw plugin is added, so a
 * `<script>` or an `<img onerror>` in the answer is just words on the screen. Images are dropped as well, because an image address would make
 * the browser fetch from a server other than the FlySafeLZ one. A `guard` of "fallback" puts a muted header above the answer, which is shown as given.
 * NOTAM ids in plain text that are in the turn's overlays are buttons; in code, in links, and when the turn does not carry them they stay words.
 */
export function Answer({ answer, guard, badges, links }: AnswerProps) {
  const notams = links?.notams
  // The click handler is read through a ref so the components below keep one identity: a new component type would remount an open popover.
  const onShow = useRef(links?.onShow)
  useEffect(() => {
    onShow.current = links?.onShow
  })

  const plugins = useMemo(() => (notams && notams.size > 0 ? [linkNotamIds(new Set(notams.keys()))] : []), [notams])
  const components = useMemo<Components>(
    () => ({
      span: (props: ComponentProps<'span'>) => {
        const id = (props as Record<string, unknown>)[NOTAM_ID_ATTRIBUTE]
        const feature = typeof id === 'string' ? notams?.get(id) : undefined
        return feature ? <NotamButton feature={feature} onShow={(shown) => onShow.current?.(shown)} /> : <span className={props.className}>{props.children}</span>
      },
      a: ({ href, children }) => (
        <a href={href} target="_blank" rel="noreferrer noopener">
          {children}
        </a>
      ),
    }),
    [notams],
  )

  return (
    <div className="space-y-2">
      {guard === 'fallback' && <p className="text-xs text-muted-foreground">{FALLBACK_HEADER}</p>}
      {badges}
      <div className="space-y-2 text-sm leading-relaxed [overflow-wrap:anywhere] [&_a]:text-primary [&_a]:underline [&_code]:font-mono [&_ol]:list-decimal [&_ol]:pl-5 [&_ul]:list-disc [&_ul]:pl-5">
        <Markdown disallowedElements={['img']} remarkPlugins={plugins} components={components}>
          {answer}
        </Markdown>
      </div>
    </div>
  )
}

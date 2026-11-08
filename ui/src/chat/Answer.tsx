import Markdown from 'react-markdown'

import { FALLBACK_HEADER } from '@/chat/copy'

export interface AnswerProps {
  answer: string
  /** The server's `guard` value. Only the exact string "fallback" adds the header; a missing or unknown value is a normal answer. */
  guard: string | null
}

/**
 * One answer, shown whole as markdown. Raw HTML is not rendered: react-markdown turns it into text, and no rehype-raw plugin is added, so a
 * `<script>` or an `<img onerror>` in the answer is just words on the screen. Images are dropped as well, because an image address would make
 * the browser fetch from a server other than the FlySafeLZ one. A `guard` of "fallback" puts a muted header above the answer, which is shown as given.
 */
export function Answer({ answer, guard }: AnswerProps) {
  return (
    <div className="space-y-2">
      {guard === 'fallback' && <p className="text-xs text-muted-foreground">{FALLBACK_HEADER}</p>}
      <div className="space-y-2 text-sm leading-relaxed [overflow-wrap:anywhere] [&_a]:text-primary [&_a]:underline [&_code]:font-mono [&_ol]:list-decimal [&_ol]:pl-5 [&_ul]:list-disc [&_ul]:pl-5">
        <Markdown
          disallowedElements={['img']}
          components={{
            a: ({ href, children }) => (
              <a href={href} target="_blank" rel="noreferrer noopener">
                {children}
              </a>
            ),
          }}
        >
          {answer}
        </Markdown>
      </div>
    </div>
  )
}

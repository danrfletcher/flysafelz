import { notamIdMatcher } from '@/lib/notamId'

// The markdown tree as far as this plugin needs it. A node with `children` is a container; a `text` node has a `value`.
interface Node {
  type: string
  value?: string
  children?: Node[]
  data?: { hName?: string; hProperties?: Record<string, string> }
}

/** The attribute the rendered span carries; Answer turns a span with it into the button. */
export const NOTAM_ID_ATTRIBUTE = 'data-notam-id'

// Nodes whose text is not plain prose: a link's words are already a link, and code stays as written. (`inlineCode` and `code` hold a
// `value`, not text children, so they are never visited; a link's children are skipped on purpose.)
const NOT_PLAIN = new Set(['link', 'linkReference', 'inlineCode', 'code', 'html', 'definition'])

/** One piece of a sentence: words as they are, or a NOTAM id that is in the turn's overlays. */
export type Piece = { words: string } | { id: string }

/** Splits `prose` at every NOTAM id that `known` holds. An id that is not known stays inside the words. */
export function splitNotamIds(prose: string, known: ReadonlySet<string>): Piece[] {
  const pieces: Piece[] = []
  let from = 0
  for (const found of prose.matchAll(notamIdMatcher())) {
    if (!known.has(found[0])) continue
    if (found.index > from) pieces.push({ words: prose.slice(from, found.index) })
    pieces.push({ id: found[0] })
    from = found.index + found[0].length
  }
  if (from < prose.length) pieces.push({ words: prose.slice(from) })
  return pieces
}

function link(parent: Node, known: ReadonlySet<string>): void {
  if (!parent.children) return
  const next: Node[] = []
  for (const child of parent.children) {
    if (child.type === 'text' && typeof child.value === 'string') {
      const pieces = splitNotamIds(child.value, known)
      if (pieces.every((piece) => 'words' in piece)) {
        next.push(child)
        continue
      }
      for (const piece of pieces) {
        next.push(
          'id' in piece
            ? { type: 'notamId', data: { hName: 'span', hProperties: { [NOTAM_ID_ATTRIBUTE]: piece.id } }, children: [{ type: 'text', value: piece.id }] }
            : { type: 'text', value: piece.words },
        )
      }
      continue
    }
    if (!NOT_PLAIN.has(child.type)) link(child, known)
    next.push(child)
  }
  parent.children = next
}

/**
 * A remark plugin: in plain text (not in a code span, a code block, a link or raw HTML) each NOTAM id that is one of this turn's
 * overlays becomes a span that carries the id. Ids that are not in the turn's overlays are left as words (E10). Every repeat is changed.
 */
export function linkNotamIds(known: ReadonlySet<string>) {
  return () => (tree: Node) => link(tree, known)
}

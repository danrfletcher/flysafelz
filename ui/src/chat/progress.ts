// The longest status line shown. The server's lines are short fixed phrases ("Reading 12 NOTAMs (3/12)…"); anything longer is not one.
const MAX_LINE = 120

/**
 * The status line for one custom stream event. The agent writes each progress line as a plain string made by code
 * (src/flysafelz/agent/progress.py), so a string event is the line. Any other shape (an object, a number, null, an empty
 * or multi-line string) is not a progress line and gives null: the caller keeps the line it has.
 */
export function progressLine(event: unknown): string | null {
  if (typeof event !== 'string') return null
  const line = event.trim()
  if (line === '' || line.length > MAX_LINE || /[\r\n]/.test(line)) return null
  return line
}

export type LineAction = { type: 'event'; event: unknown } | { type: 'end' }

/** The live status line after an action: each progress event replaces the line, an event of any other shape leaves it, and the end of the run clears it. */
export function nextLine(current: string | null, action: LineAction): string | null {
  if (action.type === 'end') return null
  return progressLine(action.event) ?? current
}

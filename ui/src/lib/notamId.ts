/**
 * The shape of a NOTAM id (H7171/26, Q172/26, V32/26): a series letter, one to four digits, a slash, a two-digit year. This is the same
 * pattern, character for character, as `NOTAM_ID_RE` in src/flysafelz/contracts.py; tests/unit/test_notam_id_parity.py reads the line
 * below and fails when the two differ. Change both together.
 */
export const NOTAM_ID_PATTERN = String.raw`\b[A-Z]\d{1,4}/\d{2}\b`

/** A fresh global matcher for NOTAM ids (a global RegExp keeps its position, so each caller gets its own). */
export function notamIdMatcher(): RegExp {
  return new RegExp(NOTAM_ID_PATTERN, 'g')
}

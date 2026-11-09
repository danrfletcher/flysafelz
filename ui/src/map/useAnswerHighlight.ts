import { useSyncExternalStore } from 'react'

import { pinKey, type Session, type Sessions, type Turn } from '@/chat/turns'
import type { AnswerHighlight } from '@/map/answerOverlays'
import type { LatLon } from '@/pick/whatsHere'

/**
 * A pick the pilot made on the open chat: a turn's badges (or a NOTAM id in its answer) asked for that turn's highlights. It only counts
 * while that turn's pin is the open one and no newer answer has landed since: a new answer shows its own highlights, so the latest wins.
 */
export interface Selection {
  pin: string
  turnId: string
  /** The NOTAM id picked out, or null when a badge was clicked. */
  focusId: string | null
  /** The id of the latest answered turn when the pick was made. */
  latestId: string | null
}

const NO_SESSION: Session = { threadId: null, turns: [] }

/** The newest turn that has an answer. A running, failed or unanswered turn has no highlights to show. */
export function latestAnswered(session: Session): Turn | undefined {
  return session.turns.findLast((turn) => turn.status === 'done' && turn.answer !== null)
}

/** A pick of `turnId` (and optionally one of its NOTAMs) on the chat open for `point`. */
export function selectTurn(session: Session, point: LatLon, turnId: string, focusId: string | null): Selection {
  return { pin: pinKey(point), turnId, focusId, latestId: latestAnswered(session)?.id ?? null }
}

/**
 * Which turn's highlights the map shows for the chat open on `point`: the pilot's pick while it still holds, else the latest answered
 * turn. No chat open means none, so closing the chat ("Change LZ") or picking another pin clears the map of answer highlights. Only the open
 * pin's own turns are read, so a run that finishes late for a pin the pilot has left can never draw on the new one (E8).
 */
export function useAnswerHighlight(sessions: Sessions, point: LatLon | null, selection: Selection | null): AnswerHighlight | null {
  const key = point ? pinKey(point) : null
  const session = useSyncExternalStore(sessions.subscribe, () => (key === null ? NO_SESSION : sessions.get(key)))
  const latest = latestAnswered(session)
  const holds = selection !== null && selection.pin === key && selection.latestId === (latest?.id ?? null)
  const picked = holds ? session.turns.find((turn) => turn.id === selection.turnId) : undefined
  const shown = picked ?? latest
  if (!shown) return null
  return { turnId: shown.id, overlays: shown.overlays, focusId: picked && holds ? selection.focusId : null }
}

import { useReducedMotion } from '@/hooks/useReducedMotion'
import { enterFor, type Enter, type EnterKind } from '@/lib/motion'

/** The enter animation of one kind, in full motion or the reduced variant by the system setting as it is now. */
export function useEnter(kind: EnterKind): Enter {
  return enterFor(kind, useReducedMotion())
}

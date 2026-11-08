import { useEffect } from 'react'

import { LEAVE_WARNING } from '@/chat/copy'

/**
 * While `active` (a conversation with at least one turn is open), asks the browser to confirm before the page is left. Browsers show their own
 * wording for this prompt, not ours; the text is set for the ones that still show it. The handler goes when `active` ends and when the component unmounts.
 */
export function useLeaveWarning(active: boolean): void {
  useEffect(() => {
    if (!active) return
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = LEAVE_WARNING
      return LEAVE_WARNING
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [active])
}

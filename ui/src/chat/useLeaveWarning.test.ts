import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { LEAVE_WARNING } from '@/chat/copy'
import { useLeaveWarning } from '@/chat/useLeaveWarning'

/** The `beforeunload` listeners the hook adds and removes, seen at the window. */
function spyOnWindow() {
  const added = vi.spyOn(window, 'addEventListener')
  const removed = vi.spyOn(window, 'removeEventListener')
  const count = (spy: typeof added) => spy.mock.calls.filter(([type]) => type === 'beforeunload').length
  return { added: () => count(added), removed: () => count(removed) }
}

describe('useLeaveWarning', () => {
  it('adds no handler with no conversation, and the page leaves without a prompt', () => {
    const seen = spyOnWindow()
    renderHook(() => useLeaveWarning(false))
    expect(seen.added()).toBe(0)
    const event = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(false)
  })

  it('prompts with the leave text while a conversation has turns', () => {
    renderHook(() => useLeaveWarning(true))
    // The handler sets the text on the event the way a browser reads it; jsdom's plain Event only keeps a boolean there, so catch it with a probe.
    const event = new Event('beforeunload', { cancelable: true })
    Object.defineProperty(event, 'returnValue', { value: '', writable: true })
    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(true)
    expect((event as BeforeUnloadEvent).returnValue).toBe(LEAVE_WARNING)
    expect(LEAVE_WARNING).toBe('Leave? Your conversation will be lost')
  })

  it('adds the handler once, and removes it when the conversation closes', () => {
    const seen = spyOnWindow()
    const { rerender } = renderHook(({ active }) => useLeaveWarning(active), { initialProps: { active: false } })
    rerender({ active: true })
    rerender({ active: true })
    expect(seen.added()).toBe(1)
    expect(seen.removed()).toBe(0)
    rerender({ active: false })
    expect(seen.removed()).toBe(1)
    const event = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(false)
  })

  it('removes the handler on unmount', () => {
    const seen = spyOnWindow()
    const { unmount } = renderHook(() => useLeaveWarning(true))
    unmount()
    expect(seen.added()).toBe(1)
    expect(seen.removed()).toBe(1)
    const event = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(false)
  })
})

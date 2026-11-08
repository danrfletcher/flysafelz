import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useReducedMotion } from '@/hooks/useReducedMotion'
import { mockReducedMotion } from '@/test/reducedMotion'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('useReducedMotion', () => {
  it('is true when the system asks for reduced motion', () => {
    mockReducedMotion(true)
    expect(renderHook(() => useReducedMotion()).result.current).toBe(true)
  })

  it('is false when the system has no preference', () => {
    mockReducedMotion(false)
    expect(renderHook(() => useReducedMotion()).result.current).toBe(false)
  })

  it('follows a change event of the setting without a reload, both ways, and stops listening on unmount', () => {
    const media = mockReducedMotion(false)
    const { result, unmount } = renderHook(() => useReducedMotion())
    expect(result.current).toBe(false)
    media.set(true)
    expect(result.current).toBe(true)
    media.set(false)
    expect(result.current).toBe(false)
    expect(media.listeners()).toBe(1)
    unmount()
    expect(media.listeners()).toBe(0)
  })

  it('falls back to full motion, with no throw and no console error, when matchMedia is undefined', () => {
    vi.stubGlobal('matchMedia', undefined)
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(renderHook(() => useReducedMotion()).result.current).toBe(false)
    expect(error).not.toHaveBeenCalled()
  })

  it('falls back to full motion when matchMedia throws', () => {
    vi.stubGlobal('matchMedia', () => {
      throw new Error('not supported')
    })
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(renderHook(() => useReducedMotion()).result.current).toBe(false)
    expect(error).not.toHaveBeenCalled()
  })

  it('falls back to full motion when the query object has no addEventListener (an old browser)', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: false }))
    expect(renderHook(() => useReducedMotion()).result.current).toBe(false)
  })
})

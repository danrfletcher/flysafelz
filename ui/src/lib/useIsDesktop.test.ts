import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useIsDesktop } from '@/lib/useIsDesktop'

function setWidth(width: number) {
  vi.stubGlobal('innerWidth', width)
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('useIsDesktop', () => {
  it('follows the min-width query when there is one', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} }))
    setWidth(1200)
    expect(renderHook(() => useIsDesktop()).result.current).toBe(false)
  })

  it.each([
    [1024, true],
    [768, true],
    [767, false],
  ])('falls back to the window width (%i px -> %s) when matchMedia is undefined', (width, expected) => {
    vi.stubGlobal('matchMedia', undefined)
    setWidth(width)
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(renderHook(() => useIsDesktop()).result.current).toBe(expected)
    expect(error).not.toHaveBeenCalled()
  })

  it.each([
    [1024, true],
    [500, false],
  ])('falls back to the window width (%i px -> %s) when matchMedia throws, with no throw and no console error', (width, expected) => {
    vi.stubGlobal('matchMedia', () => {
      throw new Error('boom')
    })
    setWidth(width)
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})
    const { result, unmount } = renderHook(() => useIsDesktop())
    expect(result.current).toBe(expected)
    // a resize still moves it, through the window event
    setWidth(expected ? 500 : 1024)
    act(() => {
      window.dispatchEvent(new Event('resize'))
    })
    expect(result.current).toBe(!expected)
    unmount()
    expect(error).not.toHaveBeenCalled()
  })

  it('does not crash when the query object has no addEventListener (an old browser)', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: true }))
    const { result, unmount } = renderHook(() => useIsDesktop())
    expect(result.current).toBe(true)
    expect(() => unmount()).not.toThrow()
  })
})

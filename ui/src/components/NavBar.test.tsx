import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { NavBar } from '@/components/NavBar'

describe('NavBar', () => {
  it('shows the FlySafeLZ wordmark', () => {
    render(<NavBar />)
    const banner = screen.getByRole('banner')
    expect(within(banner).getByText('FlySafeLZ', { selector: 'span' })).toBeInTheDocument()
  })

  it('shows the wing mark as an svg with a FlySafeLZ title that takes the text colour', () => {
    render(<NavBar />)
    const mark = screen.getByRole('img', { name: 'FlySafeLZ' })
    expect(mark.tagName.toLowerCase()).toBe('svg')
    expect(mark.querySelector('title')).toHaveTextContent('FlySafeLZ')
    expect(mark).toHaveAttribute('fill', 'currentColor')
  })

  it('puts the wordmark before the ⓘ button, which is a button named "How it works"', () => {
    render(<NavBar />)
    const button = screen.getByRole('button', { name: 'How it works' })
    const wordmark = screen.getByText('FlySafeLZ', { selector: 'span' })
    expect(wordmark.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})

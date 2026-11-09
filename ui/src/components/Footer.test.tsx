import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { Footer } from '@/components/Footer'
import { NavBar } from '@/components/NavBar'

const TEXT = 'Decision support only, not a primary means of navigation.'

describe('Footer', () => {
  it('is one contentinfo line with exactly the footer text', () => {
    render(<Footer />)
    expect(screen.getByRole('contentinfo')).toHaveTextContent(new RegExp(`^${TEXT.replace('.', '\\.')}$`))
  })

  it('is still rendered, and not hidden, while the modal is open, and again once it closes', async () => {
    const user = userEvent.setup()
    render(
      <>
        <NavBar />
        <Footer />
      </>,
    )
    const footer = () => screen.getByText(TEXT, { selector: 'footer' })
    expect(footer()).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'How it works' }))
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(footer()).toBeVisible()
    expect(footer()).toHaveTextContent(TEXT)
    await user.keyboard('{Escape}')
    expect(footer()).toBeVisible()
  })
})

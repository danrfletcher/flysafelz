import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { InfoModal } from '@/components/InfoModal'
import { HOW_IT_WORKS, partsText } from '@/content/howItWorks'

const copy = HOW_IT_WORKS

// Radix marks everything outside an open dialog aria-hidden, ⓘ included, so look it up with hidden: true.
function openButton() {
  return screen.getByRole('button', { name: 'How it works', hidden: true })
}

describe('InfoModal', () => {
  it('is closed until ⓘ is clicked, then opens a dialog titled "How FlySafeLZ works"', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    await user.click(openButton())
    expect(screen.getByRole('dialog', { name: 'How FlySafeLZ works' })).toBeInTheDocument()
  })

  it('has the spec copy, as the howItWorks.ts constant has it', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    await user.click(openButton())
    const dialog = screen.getByRole('dialog')

    expect(copy.title).toBe('How FlySafeLZ works')
    expect(copy.steps.map((step) => step.lead)).toEqual(['Pick a field.', "See what's here.", 'Ask.'])
    expect(copy.answers.lead).toBe('What the answers mean.')
    expect(copy.checks.lead).toBe('What we check.')
    expect(copy.closing).toBe(
      'Decision support only, not a primary means of navigation. Always do your own pre-flight checks.',
    )

    const items = within(dialog).getAllByRole('listitem')
    expect(items).toHaveLength(copy.steps.length)
    copy.steps.forEach((step, index) => {
      expect(items[index]).toHaveTextContent(`${step.lead} ${step.body}`)
    })
    const text = dialog.textContent ?? ''
    expect(text).toContain(`${copy.answers.lead} ${partsText(copy.answers.body)}`)
    expect(text).toContain(`${copy.checks.lead} ${copy.checks.body}`)
    expect(within(dialog).getByText(copy.closing)).toBeInTheDocument()
  })

  it('shows "not" in bold inside the answers paragraph', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    await user.click(openButton())
    expect(within(screen.getByRole('dialog')).getByText('not', { selector: 'strong' })).toBeInTheDocument()
  })

  it('carries the amber line', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    await user.click(openButton())
    expect(copy.amberLine).toBe('amber = near your limits, check the conditions too')
    expect(within(screen.getByRole('dialog')).getByText(copy.amberLine)).toBeInTheDocument()
  })

  it('closes with the close button, and focus returns to ⓘ', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    await user.click(openButton())
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(openButton()).toHaveFocus()
  })

  it('closes on Esc, and focus returns to ⓘ', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    await user.click(openButton())
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(openButton()).toHaveFocus()
  })

  it('closes on a click on the overlay, and focus returns to ⓘ', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    await user.click(openButton())
    const overlay = document.querySelector('[data-slot="dialog-overlay"]')
    expect(overlay).not.toBeNull()
    await user.click(overlay as Element)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(openButton()).toHaveFocus()
  })

  it('is reached by keyboard: Tab to ⓘ, Enter opens, focus stays inside, Esc closes', async () => {
    const user = userEvent.setup()
    render(
      <>
        <InfoModal />
        <button>outside</button>
      </>,
    )
    await user.tab()
    expect(openButton()).toHaveFocus()
    await user.keyboard('{Enter}')
    const dialog = screen.getByRole('dialog')
    for (let press = 0; press < 5; press += 1) {
      await user.tab()
      expect(dialog).toContainElement(document.activeElement as HTMLElement)
    }
    await user.keyboard('{Escape}')
    expect(openButton()).toHaveFocus()
  })

  it('never has more than one dialog mounted, through rapid clicks and Esc spam', async () => {
    const user = userEvent.setup()
    render(<InfoModal />)
    for (let click = 0; click < 6; click += 1) {
      fireEvent.click(openButton())
      expect(screen.queryAllByRole('dialog').length).toBeLessThanOrEqual(1)
    }
    fireEvent.click(openButton()) // an odd number of clicks in all, so it is open
    expect(screen.getAllByRole('dialog')).toHaveLength(1)
    for (let press = 0; press < 5; press += 1) {
      await user.keyboard('{Escape}')
      expect(screen.queryAllByRole('dialog').length).toBeLessThanOrEqual(1)
    }
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(openButton()).toHaveFocus()
  })
})

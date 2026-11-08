import userEvent from '@testing-library/user-event'
import { render, screen } from '@testing-library/react'
import { createElement } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { Answer } from '@/chat/Answer'
import { splitNotamIds } from '@/chat/notamLinks'
import { NOTAM_ID_PATTERN, notamIdMatcher } from '@/lib/notamId'
import { notamsById } from '@/map/answerFeatures'
import { HIGHLIGHTS_RUN, type Recording } from '@/test/chatServer'

const SQUARE = [[[-0.1, 53.8], [-0.09, 53.8], [-0.09, 53.81], [-0.1, 53.81], [-0.1, 53.8]]]
const notam = (id: string, text: string, kind = 'notam') => ({
  type: 'Feature',
  geometry: { type: 'Polygon', coordinates: SQUARE },
  properties: { id, kind, action: 'CONTACT_FIRST', bearing: null, text, approx: false },
})

const OVERLAYS = [notam('H1234/26', 'FIRST NOTAM. CALL 01632 960001.'), notam('V32/26', 'CRANE 190FT AGL. CALL 01632 960045.'), notam('Q172/26', 'THIRD NOTAM.')]

function show(answer: string, overlays: object[] = OVERLAYS) {
  const onShow = vi.fn()
  const view = render(createElement(Answer, { answer, guard: 'passed', links: { notams: notamsById(overlays as never), onShow } }))
  const buttons = () => screen.queryAllByRole('button')
  return { onShow, buttons, view, user: userEvent.setup() }
}

describe('the NOTAM id pattern', () => {
  const matches = (text: string) => text.match(notamIdMatcher()) ?? []

  it('is the shared pattern: a letter, one to four digits, a slash and two digits', () => {
    expect(NOTAM_ID_PATTERN).toBe(String.raw`\b[A-Z]\d{1,4}/\d{2}\b`)
    expect(matches('H1234/26 Q172/26 V32/26 A1/26')).toEqual(['H1234/26', 'Q172/26', 'V32/26', 'A1/26'])
  })

  it('does not match XH1234/26, H12345/26, lowercase h1234/26, H1234/2 or H1234/261', () => {
    for (const text of ['XH1234/26', 'H12345/26', 'h1234/26', 'H1234/2', 'H1234/261', 'H/26', '1234/26']) expect(matches(text), text).toEqual([])
  })

  it('hands out a fresh matcher each time, so one caller’s position never leaks into another’s', () => {
    expect(notamIdMatcher()).not.toBe(notamIdMatcher())
    const first = notamIdMatcher()
    first.exec('H1234/26')
    expect(notamIdMatcher().exec('H1234/26')?.index).toBe(0)
  })
})

describe('splitNotamIds', () => {
  const known = new Set(['H1234/26', 'V32/26'])

  it('splits at every known id, keeping the words between', () => {
    expect(splitNotamIds('See H1234/26 and V32/26.', known)).toEqual([{ words: 'See ' }, { id: 'H1234/26' }, { words: ' and ' }, { id: 'V32/26' }, { words: '.' }])
  })

  it('handles the id at the very start, at the very end and alone', () => {
    expect(splitNotamIds('H1234/26 first', known)).toEqual([{ id: 'H1234/26' }, { words: ' first' }])
    expect(splitNotamIds('last H1234/26', known)).toEqual([{ words: 'last ' }, { id: 'H1234/26' }])
    expect(splitNotamIds('H1234/26', known)).toEqual([{ id: 'H1234/26' }])
  })

  it('leaves an id that is not known as words (E10), and a text with none as one piece', () => {
    expect(splitNotamIds('G9999/26 and H1234/26', known)).toEqual([{ words: 'G9999/26 and ' }, { id: 'H1234/26' }])
    expect(splitNotamIds('nothing here', known)).toEqual([{ words: 'nothing here' }])
    expect(splitNotamIds('', known)).toEqual([])
  })

  it('links the ids next to punctuation', () => {
    const pieces = splitNotamIds('(H1234/26), "V32/26"; H1234/26.', known)
    expect(pieces.filter((piece) => 'id' in piece)).toHaveLength(3)
  })
})

describe('NOTAM ids in an answer', () => {
  it('makes a button of an id that is in the turn’s overlays, and only of those (E10)', async () => {
    const { buttons } = show('H1234/26 is near; so is G9999/26 and V32/26.')
    expect(buttons().map((button) => button.textContent)).toEqual(['H1234/26', 'V32/26'])
    expect(screen.getByText(/G9999\/26/).closest('button')).toBeNull()
    expect(screen.getByText(/G9999\/26/).className).not.toMatch(/cursor-pointer/)
  })

  it('links nothing and opens no popover for an id that is not in the overlays', async () => {
    const { buttons, user } = show('Only G9999/26 here.')
    expect(buttons()).toHaveLength(0)
    await user.click(screen.getByText(/G9999\/26/))
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('links the id at the start and the end of the text and next to punctuation', () => {
    const { buttons } = show('H1234/26 starts. (V32/26), then **Q172/26** and H1234/26')
    expect(buttons().map((button) => button.textContent)).toEqual(['H1234/26', 'V32/26', 'Q172/26', 'H1234/26'])
  })

  it('links every occurrence of a repeated id, and each opens the same feature', async () => {
    const { buttons, user, onShow } = show('H1234/26 once, H1234/26 twice.')
    expect(buttons()).toHaveLength(2)
    await user.click(buttons()[0])
    expect(await screen.findByRole('dialog', { name: 'NOTAM H1234/26' })).toHaveTextContent('FIRST NOTAM. CALL 01632 960001.')
    await user.keyboard('{Escape}')
    await user.click(buttons()[1])
    expect(await screen.findByRole('dialog', { name: 'NOTAM H1234/26' })).toHaveTextContent('FIRST NOTAM. CALL 01632 960001.')
    expect(onShow.mock.calls).toEqual([['H1234/26'], ['H1234/26']])
  })

  it('does not link an id in a code span, a code block or an existing link', () => {
    const { buttons, view } = show('`H1234/26` and [V32/26](https://example.invalid/x) and H1234/26.\n\n```\nQ172/26\n```')
    expect(buttons().map((button) => button.textContent)).toEqual(['H1234/26'])
    expect(view.container.querySelector('code')).toHaveTextContent('H1234/26')
    expect(view.container.querySelector('a')).toHaveTextContent('V32/26')
    expect(view.container.querySelector('a button')).toBeNull()
  })

  it('does not link XH1234/26, H12345/26 or lowercase h1234/26 even when H1234/26 is in the overlays', () => {
    const { buttons } = show('XH1234/26 H12345/26 h1234/26')
    expect(buttons()).toHaveLength(0)
  })

  it('opens the feature of V32/26 when it is in the overlays: its raw text, as published, and tells the map', async () => {
    const { buttons, user, onShow } = show('The crane is V32/26.')
    await user.click(buttons()[0])
    const popover = await screen.findByRole('dialog', { name: 'NOTAM V32/26' })
    expect(popover).toHaveTextContent('CRANE 190FT AGL. CALL 01632 960045.')
    expect(onShow).toHaveBeenCalledWith('V32/26')
  })

  it('links nothing when the turn has no overlays or the answer is given no links', () => {
    const { buttons } = show('H1234/26 and V32/26', [])
    expect(buttons()).toHaveLength(0)
    const { container } = render(createElement(Answer, { answer: 'H1234/26', guard: null }))
    expect(container.querySelector('button')).toBeNull()
  })

  it('does not link the airspace features of the overlays, only NOTAMs', () => {
    const { buttons } = show('Q172/26 is a NOTAM.', [notam('Q172/26', 'x', 'airspace')])
    expect(buttons()).toHaveLength(0)
  })

  it('links the ids of the recorded Blue Mill answer: those in its overlays, not G9999/26, and not the one in code or in a link', () => {
    const values = (HIGHLIGHTS_RUN as Recording).events.at(-1)!.data as { answer: string; overlays: object[] }
    const { buttons } = show(values.answer, values.overlays)
    expect(buttons().map((button) => button.textContent)).toEqual(['H7171/26', 'H7171/26', 'H7172/26', 'V32/26'])
  })
})

import { describe, expect, it } from 'vitest'

import { filesMatching, readUi, uiTextFiles } from '@/test/files'

const theme = readUi('src/styles/theme.css')

function declared(name: string): string | undefined {
  return new RegExp(`^\\s*--${name}:\\s*([^;]+);`, 'm').exec(theme)?.[1].trim()
}

const SHADCN = [
  'background', 'foreground', 'card', 'card-foreground', 'popover', 'popover-foreground', 'primary', 'primary-foreground',
  'secondary', 'secondary-foreground', 'muted', 'muted-foreground', 'accent', 'accent-foreground', 'destructive',
  'destructive-foreground', 'border', 'input', 'ring', 'radius', 'border-width', 'shadow',
]
const TYPE = ['type-restricted', 'type-danger', 'type-controlled', 'type-other', 'type-notam']
const VERDICT = ['verdict-no-go', 'verdict-unverified', 'verdict-amber', 'verdict-conditions', 'verdict-discretion']

describe('theme.css', () => {
  it.each([...SHADCN, ...TYPE, ...VERDICT])('defines --%s', (name) => {
    expect(declared(name), `--${name}`).toBeTruthy()
  })

  it('sets the radius to 12px and the border to 1px', () => {
    expect(declared('radius')).toBe('12px')
    expect(declared('border-width')).toBe('1px')
  })

  it('has a sky-blue accent as --primary', () => {
    expect(declared('primary')).toBe('#38bdf8')
  })

  it('is the only stylesheet that holds a colour literal', () => {
    const literal = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\boklch\(/
    const stylesheets = uiTextFiles().filter((path) => path.endsWith('.css'))
    expect(stylesheets).toContain('src/styles/theme.css')
    expect(stylesheets.filter((path) => literal.test(readUi(path)))).toEqual(['src/styles/theme.css'])
  })

  it('is dark only: <html class="dark"> and no theme toggle', () => {
    expect(readUi('index.html')).toMatch(/<html[^>]*class="dark"/)
    // built from pieces so this file does not match its own rule
    const toggle = new RegExp(['prefers-color', '-scheme', '|theme', '-toggle|Theme', 'Toggle|set', 'Theme'].join(''))
    expect(filesMatching(toggle)).toEqual([])
  })
})

describe('colours in components', () => {
  it('appear only as variables or Tailwind tokens: no #hex, rgb( or hsl( in ui/src/**/*.tsx', () => {
    const literal = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/
    const components = uiTextFiles().filter((path) => path.startsWith('src/') && path.endsWith('.tsx'))
    expect(components.length).toBeGreaterThan(3)
    expect(components.filter((path) => literal.test(readUi(path)))).toEqual([])
  })
})

import type { AirspaceProps, NotamProps } from '@/map/types'

// The hover tooltips' words. Built as DOM nodes with textContent, never as HTML, so a name or an id from the server cannot inject markup.
// The NOTAM's raw `text` is never used here: it belongs to the What's-here card and the chat.

function line(label: string, value: string): HTMLElement {
  const row = document.createElement('div')
  const name = document.createElement('span')
  name.className = 'map-tip__label'
  name.textContent = `${label} `
  row.append(name, document.createTextNode(value))
  return row
}

function box(title: string): HTMLElement {
  const root = document.createElement('div')
  root.className = 'map-tip'
  const heading = document.createElement('strong')
  heading.className = 'map-tip__title'
  heading.textContent = title
  root.append(heading)
  return root
}

/** "29 Oct, 09:00 BST": a time in the browser's zone, with the zone named, or the raw text when it is not a time. */
export function formatWhen(iso: string): string {
  const when = new Date(iso)
  if (Number.isNaN(when.getTime())) return iso
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    timeZoneName: 'short',
  }).format(when)
}

/** Name, limits (lower to upper, as openAIP states them) and active times. No schedule published says so. */
export function airspaceTooltip(props: AirspaceProps): HTMLElement {
  const root = box(props.name)
  root.append(line('Limits', `${props.lower} to ${props.upper}`))
  root.append(line('Active', props.hours ?? 'no times published'))
  return root
}

/** The NOTAM's limits as published: item F to item G, else the Q-line flight levels. */
export function notamLimits(props: NotamProps): string {
  if (props.item_f && props.item_g) return `${props.item_f} to ${props.item_g}`
  const lower = props.lower_fl === null ? '?' : `FL${props.lower_fl}`
  const upper = props.upper_fl === null ? '?' : `FL${props.upper_fl}`
  return `${lower} to ${upper}`
}

export function notamTimes(props: NotamProps): string {
  const from = formatWhen(props.valid_from)
  if (props.permanent) return `from ${from}, permanent`
  if (props.valid_to === null) return `from ${from}, no end given`
  return `${from} to ${formatWhen(props.valid_to)}${props.est ? ' (estimated)' : ''}`
}

/** Id, limits and times. Approximate shapes are marked. */
export function notamTooltip(props: NotamProps): HTMLElement {
  const root = box(props.id)
  root.append(line('Limits', notamLimits(props)))
  root.append(line('Times', notamTimes(props)))
  if (props.approx) root.append(line('Shape', 'approx (the NOTAM\u2019s Q-line circle)'))
  return root
}

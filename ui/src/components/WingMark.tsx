import type { SVGProps } from 'react'

/** A paramotor wing seen from the front: the canopy arc, its lines and the pilot below. Drawn in the repo; takes the text colour. */
export function WingMark(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 32 32" role="img" fill="currentColor" xmlns="http://www.w3.org/2000/svg" {...props}>
      <title>FlySafeLZ</title>
      <path d="M2 13C7 4 25 4 30 13C24.5 9.5 7.5 9.5 2 13Z" />
      <path
        d="M5 11.5L16 24M16 9.5V24M27 11.5L16 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <circle cx="16" cy="26.5" r="2.8" />
    </svg>
  )
}

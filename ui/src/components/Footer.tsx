export const FOOTER_TEXT = 'Decision support only, not a primary means of navigation.'

/** One slim line, always on screen. It sits above the modal's dimmed overlay (z-[60] against the overlay's z-50) so it stays readable while the modal is open. */
export function Footer() {
  return (
    <footer className="relative z-[60] shrink-0 border-t-[length:var(--border-width)] bg-card px-4 py-1.5 text-center text-xs text-muted-foreground">
      {FOOTER_TEXT}
    </footer>
  )
}

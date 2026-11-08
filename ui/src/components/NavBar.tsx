import { InfoModal } from '@/components/InfoModal'
import { WingMark } from '@/components/WingMark'

/** The wing mark and wordmark on the left, the ⓘ button (which opens "How it works") on the right. One row that never wraps. */
export function NavBar() {
  return (
    <header className="flex h-12 shrink-0 items-center justify-between gap-2 border-b-[length:var(--border-width)] bg-card px-4">
      <div className="flex min-w-0 items-center gap-2">
        <WingMark className="size-7 shrink-0 text-primary" />
        <span className="truncate text-base font-semibold tracking-tight">FlySafeLZ</span>
      </div>
      <InfoModal />
    </header>
  )
}

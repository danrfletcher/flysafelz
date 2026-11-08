// shadcn/ui Sheet (new-york style): a panel on the right edge, over Radix Dialog. It is non-modal: no overlay and no focus trap, so the map
// beside it stays visible and clickable. It is placed inside `container` (the map area) rather than over the nav bar and footer.
import * as DialogPrimitive from '@radix-ui/react-dialog'
import type { ComponentProps } from 'react'

import { cn } from '@/lib/utils'

function Sheet(props: Omit<ComponentProps<typeof DialogPrimitive.Root>, 'modal'>) {
  return <DialogPrimitive.Root modal={false} {...props} />
}

function SheetContent({ className, children, container, ...props }: ComponentProps<typeof DialogPrimitive.Content> & { container?: HTMLElement | null }) {
  return (
    <DialogPrimitive.Portal container={container}>
      <DialogPrimitive.Content
        data-slot="sheet-content"
        // A click on the map is the pilot picking another LZ: it must not close the sheet.
        onInteractOutside={(event) => event.preventDefault()}
        className={cn(
          'absolute inset-y-0 right-0 z-[1100] flex w-[26rem] max-w-full flex-col border-l-[length:var(--border-width)] bg-popover text-popover-foreground shadow-panel',
          className,
        )}
        {...props}
      >
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export { Sheet, SheetContent }

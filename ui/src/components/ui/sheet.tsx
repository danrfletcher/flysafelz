// shadcn/ui Sheet (new-york style): a panel on the right edge, over Radix Dialog. It is non-modal: no overlay and no focus trap, so the map
// beside it stays visible and clickable. It is placed inside `container` (the map area) rather than over the nav bar and footer.
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { motion } from 'motion/react'
import type { ComponentProps } from 'react'

import { useEnter } from '@/hooks/useEnter'
import { cn } from '@/lib/utils'

function Sheet(props: Omit<ComponentProps<typeof DialogPrimitive.Root>, 'modal'>) {
  return <DialogPrimitive.Root modal={false} {...props} />
}

function SheetContent({ className, children, container, ...props }: ComponentProps<typeof DialogPrimitive.Content> & { container?: HTMLElement | null }) {
  // It springs in from the right edge (the reduced variant only fades). Nothing animates out: closing unmounts it at once, so no panel is
  // left over the map, and the panel takes clicks from the first frame.
  const enter = useEnter('sheet')
  return (
    <DialogPrimitive.Portal container={container}>
      <DialogPrimitive.Content
        asChild
        data-slot="sheet-content"
        // A click on the map is the pilot picking another LZ: it must not close the sheet.
        onInteractOutside={(event) => event.preventDefault()}
        className={cn(
          'absolute inset-y-0 right-0 z-[1100] flex w-[26rem] max-w-full flex-col border-l-[length:var(--border-width)] bg-popover text-popover-foreground shadow-panel',
          className,
        )}
        {...props}
      >
        <motion.div initial={enter.initial} animate={enter.animate} transition={enter.transition}>
          {children}
        </motion.div>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export { Sheet, SheetContent }

// shadcn/ui Drawer (new-york style): a panel on the bottom edge for narrow screens. Built on the same non-modal Radix Dialog as the Sheet, without
// vaul's drag gestures (the brief leaves gestures beyond the drawer out). The map above it stays visible and clickable.
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { motion } from 'motion/react'
import type { ComponentProps } from 'react'

import { useEnter } from '@/hooks/useEnter'
import { cn } from '@/lib/utils'

function Drawer(props: Omit<ComponentProps<typeof DialogPrimitive.Root>, 'modal'>) {
  return <DialogPrimitive.Root modal={false} {...props} />
}

function DrawerContent({ className, children, container, ...props }: ComponentProps<typeof DialogPrimitive.Content> & { container?: HTMLElement | null }) {
  // It springs up from the bottom edge (the reduced variant only fades). Nothing animates out: closing unmounts it at once.
  const enter = useEnter('drawer')
  return (
    <DialogPrimitive.Portal container={container}>
      <DialogPrimitive.Content
        asChild
        data-slot="drawer-content"
        onInteractOutside={(event) => event.preventDefault()}
        className={cn(
          'absolute inset-x-0 bottom-0 z-[1100] flex h-[55%] flex-col rounded-t-lg border-t-[length:var(--border-width)] bg-popover text-popover-foreground shadow-panel',
          className,
        )}
        {...props}
      >
        <motion.div initial={enter.initial} animate={enter.animate} transition={enter.transition}>
          <div className="mx-auto mt-2 h-1.5 w-12 shrink-0 rounded-full bg-muted" aria-hidden="true" />
          {children}
        </motion.div>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export { Drawer, DrawerContent }

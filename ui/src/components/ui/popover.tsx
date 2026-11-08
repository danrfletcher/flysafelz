// shadcn/ui Popover (new-york style) over Radix, without the enter/exit animations: motion belongs to a later PR.
import * as PopoverPrimitive from '@radix-ui/react-popover'
import type { ComponentProps } from 'react'

import { cn } from '@/lib/utils'

function Popover(props: ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root data-slot="popover" {...props} />
}

function PopoverTrigger(props: ComponentProps<typeof PopoverPrimitive.Trigger>) {
  return <PopoverPrimitive.Trigger data-slot="popover-trigger" {...props} />
}

/**
 * Above Leaflet's panes and controls (z-index up to 1000). It is never wider than the viewport less 1 rem, never taller than the
 * room Radix finds for it, and scrolls inside when it is, so at 375 px the page does not scroll sideways and every control stays reachable.
 */
function PopoverContent({ className, align = 'end', sideOffset = 8, collisionPadding = 8, ...props }: ComponentProps<typeof PopoverPrimitive.Content>) {
  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        data-slot="popover-content"
        align={align}
        sideOffset={sideOffset}
        collisionPadding={collisionPadding}
        className={cn(
          'z-[1100] w-80 max-w-[calc(100vw-1rem)] max-h-[var(--radix-popover-content-available-height)] overflow-x-hidden overflow-y-auto rounded-lg border-[length:var(--border-width)] bg-popover p-4 text-popover-foreground shadow-panel outline-none',
          className,
        )}
        {...props}
      />
    </PopoverPrimitive.Portal>
  )
}

export { Popover, PopoverContent, PopoverTrigger }

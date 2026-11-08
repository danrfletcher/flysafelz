import { motion } from 'motion/react'

import { useEnter } from '@/hooks/useEnter'

/** The live status line of a running turn. Each new line is a new element (the caller keys it by the text), so each one fades in. */
export function ProgressLine({ line }: { line: string }) {
  const enter = useEnter('progress')
  return (
    <motion.p initial={enter.initial} animate={enter.animate} transition={enter.transition} className="text-sm text-muted-foreground">
      {line}
    </motion.p>
  )
}

const warned = new Set<string>()

/** Forget the features already warned about. For tests only. */
export function resetSkipped(): void {
  warned.clear()
}

/**
 * Says once, with `console.warn`, that a feature was left out because it has no usable shape (null or empty geometry, or a type
 * other than Polygon or MultiPolygon). Only the layer name and the feature's id are logged, never any of its text.
 */
export function warnSkippedOnce(layer: string, id: unknown, position: number): void {
  const name = typeof id === 'string' && id !== '' ? id : `#${position}`
  const key = `${layer}:${name}`
  if (warned.has(key)) return
  warned.add(key)
  console.warn(`Skipped ${layer} feature ${name}: no usable shape`)
}

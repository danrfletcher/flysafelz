import { readdirSync, readFileSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'

/** The ui/ folder, from this file's own place (ui/src/test/files.ts), so the tests do not depend on the working directory. */
export const UI_ROOT = resolve(__dirname, '../..')

const SKIPPED_FOLDERS = new Set(['node_modules', 'dist'])
const BINARY = /\.(woff2?|ttf|png|jpe?g|gif|ico|svg)$/i

/** Every text file under ui/ (paths relative to ui/), except node_modules, the build output and fonts and images. */
export function uiTextFiles(): string[] {
  const found: string[] = []
  const walk = (folder: string) => {
    for (const entry of readdirSync(folder, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        if (!SKIPPED_FOLDERS.has(entry.name)) walk(join(folder, entry.name))
      } else if (!BINARY.test(entry.name) && entry.name !== 'package-lock.json') {
        found.push(relative(UI_ROOT, join(folder, entry.name)))
      }
    }
  }
  walk(UI_ROOT)
  return found.sort()
}

export function readUi(path: string): string {
  return readFileSync(join(UI_ROOT, path), 'utf8')
}

/** The ui/ files (relative paths) whose text matches the pattern. */
export function filesMatching(pattern: RegExp): string[] {
  return uiTextFiles().filter((path) => new RegExp(pattern.source, pattern.flags.replace('g', '')).test(readUi(path)))
}

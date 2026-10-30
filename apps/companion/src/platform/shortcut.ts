import type { PlatformOs } from './types.ts'

/**
 * The keyboard stop on each OS. ⌘⇧⎋ on macOS; elsewhere Ctrl+Shift+Alt+Esc, because Ctrl+Shift+Esc opens the Task
 * Manager on Windows and Ctrl+Alt+Esc is xkill on KDE.
 */
export function stopShortcut(os: PlatformOs | NodeJS.Platform): { accelerator: string; words: string } {
  return os === 'darwin' ? { accelerator: 'CommandOrControl+Shift+Escape', words: '⌘⇧⎋' } : { accelerator: 'Control+Shift+Alt+Escape', words: 'Ctrl+Shift+Alt+Esc' }
}

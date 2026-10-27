/**
 * The visible virtual cursor: a small marker drawn at the point the agent acts on, so the person sees where it is
 * working without it touching their real pointer. It is a transparent, click-through, always-on-top window that never
 * takes focus. The agent's own actions are in the background (a virtual cursor); this overlay only shows where.
 *
 * The placement and the fade are pure and tested here; the Electron window is injected as `Surface`, so the app wires
 * the real one and a test a fake. A background action shows a soft ring; a `pointer`-mode step (the person let it use
 * the real mouse) shows a filled dot, so the difference is visible.
 */
import type { InputMode } from '../../../desktop-helper/src/computer/types.ts'

export type Spot = { x: number; y: number; mode: InputMode }

/** The overlay window the cursor drives: shown at a point, hidden, and closed with the task. */
export interface Surface {
  showAt(x: number, y: number, mode: InputMode): void
  hide(): void
  close(): void
  /** Builds and loads the window now (hidden), so the first ring of a task shows at once. */
  warm?(): void
}

/** How long the marker stays after the last action before it fades, so a quiet moment does not leave it on screen. */
export const LINGER_MS = 2500

export class VirtualCursor {
  private readonly surface: Surface
  private readonly schedule: (fn: () => void, ms: number) => { cancel: () => void }
  private timer: { cancel: () => void } | null = null

  constructor(surface: Surface, options: { schedule?: (fn: () => void, ms: number) => { cancel: () => void } } = {}) {
    this.surface = surface
    this.schedule =
      options.schedule ??
      ((fn, ms) => {
        const t = setTimeout(fn, ms)
        if (typeof t === 'object' && 'unref' in t) (t as { unref: () => void }).unref()
        return { cancel: () => clearTimeout(t) }
      })
  }

  /**
   * A task started: the window is built and its page loaded before the first action (Windows run, 2026-10-27: no ring
   * 167 ms after the first background act, while the window was still being built and loaded).
   */
  ready(): void {
    this.surface.warm?.()
  }

  /** Show the marker at a point; it fades on its own after LINGER_MS unless another action moves it first. */
  show(spot: Spot): void {
    this.timer?.cancel()
    this.surface.showAt(Math.round(spot.x), Math.round(spot.y), spot.mode)
    this.timer = this.schedule(() => this.hide(), LINGER_MS)
  }

  hide(): void {
    this.timer?.cancel()
    this.timer = null
    this.surface.hide()
  }

  /** The task ended: hide and release the window. */
  end(): void {
    this.hide()
    this.surface.close()
  }

  /** Whether the marker is showing (for tests). */
  get visible(): boolean {
    return this.timer !== null
  }
}

/** The overlay page: a dot that `showAt` moves and styles. Loaded once into the click-through window. */
export function overlayHtml(): string {
  return `<!doctype html><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;pointer-events:none}
#c{position:fixed;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;transition:left .12s,top .12s;opacity:0;
   box-shadow:0 0 0 2px rgba(43,108,255,.9),0 0 12px 3px rgba(43,108,255,.5)}
#c.on{opacity:1}
#c.pointer{background:rgba(43,108,255,.9)}
#c.background{background:rgba(43,108,255,.18)}
</style><div id="c"></div><script>
const c=document.getElementById('c');
window.lrCursor=(x,y,mode)=>{c.style.left=x+'px';c.style.top=y+'px';c.className='on '+(mode==='pointer'?'pointer':'background')};
window.lrHide=()=>{c.className=''};
</script>`
}

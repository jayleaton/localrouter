import { fakeMac } from '../../../desktop-helper/src/fake.ts'
import type { Platform } from './types.ts'

export type FakeMachine = ReturnType<typeof fakeMac>

/** The scripted Notes app (tests, headless smokes, CI): it never touches a real screen, on any OS. */
export function createFakePlatform(): { platform: Platform; fake: FakeMachine } {
  const mac = fakeMac()
  const platform: Platform = {
    os: 'fake',
    backend: 'fake (scripted Notes app)',
    screen: mac.desktop,
    tabs: null,
    launcher: mac.dock,
    apps: mac.apps,
    controls: mac.controls,
    pointer: () => mac.desktop.pointer(),
    displays: () => mac.desktop.displays(),
    presence: null,
    access: async () => ({ ...(await mac.access()), input: true, backend: 'fake (scripted Notes app)', requirements: [] }),
  }
  return { platform, fake: mac }
}

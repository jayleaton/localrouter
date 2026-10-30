import { ControlError } from '../../../desktop-helper/src/computer/macControls.ts'
import { createFakePlatform } from './fake.ts'
import type { FakeMachine } from './fake.ts'
import { createLinuxPlatform } from './linux/index.ts'
import { createMacPlatform } from './mac.ts'
import { createWindowsPlatform } from './windows/index.ts'
import type { ControlSupport, Platform } from './types.ts'

/** `native`: this OS's platform; `fake`: the scripted Notes app on any OS. */
export type PlatformChoice = 'native' | 'fake'

export type PlatformOptions = {
  /** The companion's data folder (a Linux portal's restore token is kept there). */
  dataDir?: string
  os?: NodeJS.Platform
}

export type Built = { platform: Platform; fake: FakeMachine | null }

/** This computer's platform. An OS with none yet gets one that reads nothing and refuses every control, saying why. */
export async function createPlatform(choice: PlatformChoice, options: PlatformOptions = {}): Promise<Built> {
  if (choice === 'fake') return createFakePlatform()
  const os = options.os ?? process.platform
  if (os === 'darwin') return { platform: createMacPlatform(), fake: null }
  if (os === 'win32') return { platform: createWindowsPlatform(), fake: null }
  if (os === 'linux') return { platform: await createLinuxPlatform(options.dataDir ? { dataDir: options.dataDir } : {}), fake: null }
  return { platform: unsupportedPlatform(os), fake: null }
}

const KINDS = ['open_app', 'quit_app', 'volume', 'media', 'dark_mode', 'lock_screen'] as const

/** No backend for this OS: computer use is unavailable and every control is refused with the reason. */
export function unsupportedPlatform(os: string): Platform {
  const why = `Computer control is not available on ${os} yet.`
  const unsupported: ControlSupport = Object.fromEntries(KINDS.map((kind) => [kind, why]))
  return {
    os: os === 'linux' || os === 'win32' || os === 'darwin' ? os : 'other',
    backend: `none (${os})`,
    screen: null,
    tabs: null,
    launcher: null,
    apps: null,
    controls: {
      unsupported,
      run: async () => {
        throw new ControlError('unavailable', why)
      },
    },
    pointer: null,
    displays: null,
    presence: null,
    access: async () => ({ accessibility: false, screen_recording: false, input: false, backend: `none (${os})`, requirements: [] }),
  }
}

import { describe, expect, it } from 'vitest'
import { ControlError } from '../../../../desktop-helper/src/computer/macControls.ts'
import { LinuxControls, applicationDirs, ownsWindow, parseDesktopEntry, resolveEntry, unsupportedControls } from './controls.ts'
import type { DesktopEntry } from './controls.ts'
import type { Session, Tool } from './session.ts'
import { fakeRunner, fakeSession, tool } from './testing.ts'
import type { Call } from './testing.ts'

const entry = (id: string, name: string, more: Partial<DesktopEntry> = {}): DesktopEntry => ({ id, path: `/usr/share/applications/${id}`, name, keywords: [], ...more })

const ENTRIES = [
  entry('google-chrome.desktop', 'Google Chrome', { generic: 'Web Browser', keywords: ['browser'] }),
  entry('chromium.desktop', 'Chromium', { generic: 'Web Browser', keywords: ['browser'] }),
  entry('org.mozilla.firefox.desktop', 'Firefox', { generic: 'Web Browser', keywords: ['Internet', 'WWW', 'Browser'] }),
  entry('firefox.desktop', 'Firefox', { generic: 'Web Browser' }),
  entry('org.gnome.Nautilus.desktop', 'Files', { keywords: ['folder', 'manager', 'explore'] }),
  entry('code.desktop', 'Visual Studio Code', { wmClass: 'Code', command: 'code' }),
  entry('org.gnome.TextEditor.desktop', 'Text Editor'),
  entry('libreoffice-writer.desktop', 'LibreOffice Writer'),
]

const ask = (name: string) => resolveEntry(name, ENTRIES)
const question = (name: string) => {
  try {
    ask(name)
  } catch (error) {
    return error instanceof ControlError ? `${error.code}: ${error.message}` : String(error)
  }
  return 'resolved'
}

describe('desktop entries', () => {
  it('reads Name, GenericName, Keywords, StartupWMClass and the command from [Desktop Entry] only', () => {
    const text = '[Desktop Entry]\nType=Application\nName=Visual Studio Code\nName[de]=Code\nGenericName=Text Editor\nExec=env FOO=1 /usr/share/code/code --unity-launch %F\nKeywords=vscode;editor;\nStartupWMClass=Code\n\n[Desktop Action new-empty-window]\nName=New Empty Window\n'
    expect(parseDesktopEntry(text, 'code.desktop', '/p')).toEqual({ id: 'code.desktop', path: '/p', name: 'Visual Studio Code', generic: 'Text Editor', keywords: ['vscode', 'editor'], wmClass: 'Code', command: 'code' })
    expect(parseDesktopEntry('[Desktop Entry]\nType=Application\nName=Hidden\nNoDisplay=true\n', 'x.desktop', '/p')).toBeNull()
    expect(parseDesktopEntry('[Desktop Entry]\nType=Link\nName=Site\n', 'x.desktop', '/p')).toBeNull()
  })

  it('looks in XDG_DATA_HOME, then XDG_DATA_DIRS, then flatpak’s exports', () => {
    expect(applicationDirs({ XDG_DATA_DIRS: '/usr/local/share:/usr/share:/var/lib/flatpak/exports/share' }, '/home/jj')).toEqual([
      '/home/jj/.local/share/applications', '/usr/local/share/applications', '/usr/share/applications', '/var/lib/flatpak/exports/share/applications', '/home/jj/.local/share/flatpak/exports/share/applications',
    ])
  })

  it('scans every directory once, a user’s file hiding the system one with the same id, subdirectories as dashed ids', async () => {
    const tree: Record<string, { name: string; dir: boolean }[]> = {
      '/home/jj/.local/share/applications': [{ name: 'firefox.desktop', dir: false }],
      '/usr/share/applications': [{ name: 'firefox.desktop', dir: false }, { name: 'kde4', dir: true }, { name: 'gedit.desktop', dir: false }],
      '/usr/share/applications/kde4': [{ name: 'kate.desktop', dir: false }],
    }
    const files: Record<string, string> = {
      '/home/jj/.local/share/applications/firefox.desktop': '[Desktop Entry]\nType=Application\nName=Firefox (mine)\n',
      '/usr/share/applications/firefox.desktop': '[Desktop Entry]\nType=Application\nName=Firefox\n',
      '/usr/share/applications/kde4/kate.desktop': '[Desktop Entry]\nType=Application\nName=Kate\n',
      '/usr/share/applications/gedit.desktop': '[Desktop Entry]\nType=Application\nName=gedit\nNoDisplay=true\n',
    }
    const controls = new LinuxControls({ session: fakeSession({ env: { XDG_DATA_DIRS: '/usr/share' } }), runner: fakeRunner().runner, home: '/home/jj', list: async (path) => tree[path] ?? [], readText: async (path) => files[path] ?? '' })
    expect((await controls.entries()).map((e) => [e.id, e.name])).toEqual([['firefox.desktop', 'Firefox (mine)'], ['kde4-kate.desktop', 'Kate']])
  })
})

describe('resolving an app name', () => {
  it('takes an exact Name or id, then the alias table in its order of preference', () => {
    expect(ask('Files').id).toBe('org.gnome.Nautilus.desktop')
    expect(ask('nautilus').id).toBe('org.gnome.Nautilus.desktop')
    expect(ask('chrome').id).toBe('google-chrome.desktop')
    expect(ask('VS Code').id).toBe('code.desktop')
    expect(ask('textedit').id).toBe('org.gnome.TextEditor.desktop')
    expect(ask('writer').id).toBe('libreoffice-writer.desktop')
  })

  it('asks which one when several entries match equally, naming same-named ones by id', () => {
    expect(question('browser')).toBe('ambiguous: Which one do you mean: Chromium, Firefox, Google Chrome?')
    expect(question('web browser')).toBe('ambiguous: Which one do you mean: Chromium, Firefox (firefox), Firefox (org.mozilla.firefox), Google Chrome?')
    expect(question('Firefox')).toBe('ambiguous: Which one do you mean: Firefox (firefox), Firefox (org.mozilla.firefox)?')
    expect(question('Photoshop')).toBe("not_found: Photoshop isn't installed on this computer")
  })

  it('matches a window to its entry by WM class, app_id or command, at dot boundaries only', () => {
    const code = ENTRIES.find((e) => e.id === 'code.desktop')!
    const files = ENTRIES.find((e) => e.id === 'org.gnome.Nautilus.desktop')!
    expect(ownsWindow(code, 'code.Code')).toBe(true)
    expect(ownsWindow(files, 'org.gnome.Nautilus')).toBe(true)
    expect(ownsWindow(files, 'nautilus')).toBe(true)
    expect(ownsWindow(code, 'codeblocks')).toBe(false)
  })
})

/** Controls over a fake runner that answers by program and first argument. */
function controls(session: Session, answers: Record<string, string | Error> = {}) {
  const fake = fakeRunner((call: Call) => answers[`${tool(call)} ${call.args[0] ?? ''}`.trim()] ?? answers[tool(call)] ?? '')
  const linux = new LinuxControls({ session, runner: fake.runner, list: async () => [], readText: async () => '' })
  return { ...fake, linux }
}
const with_ = (...tools: Tool[]) => fakeSession({ tools })

describe('fixed controls', () => {
  it('launches an entry with gtk-launch by id, else gio launch by path; never a shell', async () => {
    const gtk = controls(with_('gtk-launch', 'gio'))
    await gtk.linux.launch(ENTRIES[4]!)
    const gio = controls(with_('gio'))
    await gio.linux.launch(ENTRIES[4]!)
    expect([...gtk.lines(), ...gio.lines()]).toEqual(['gtk-launch org.gnome.Nautilus', 'gio launch /usr/share/applications/org.gnome.Nautilus.desktop'])
  })

  it('sets the volume with wpctl, else pactl, else amixer, stepping from the current level', async () => {
    const wp = controls(with_('wpctl', 'pactl'), { 'wpctl get-volume': 'Volume: 0.45 [MUTED]\n' })
    expect(await wp.linux.run({ kind: 'volume', action: 'up', target: 'system' })).toEqual({ done: 'Volume is at 65%.' })
    await wp.linux.run({ kind: 'volume', action: 'mute', target: 'system' })
    const pa = controls(with_('pactl'), { 'pactl get-sink-volume': 'Volume: front-left: 26214 /  40% / -23.88 dB,   front-right: 26214 /  40% / -23.88 dB\n' })
    await pa.linux.run({ kind: 'volume', action: 'down', target: 'system' })
    const alsa = controls(with_('amixer'))
    await alsa.linux.run({ kind: 'volume', action: 'set', level: 130, target: 'system' })
    expect([...wp.lines(), ...pa.lines(), ...alsa.lines()]).toEqual([
      'wpctl get-volume @DEFAULT_AUDIO_SINK@', 'wpctl set-volume -l 1.0 @DEFAULT_AUDIO_SINK@ 0.65', 'wpctl set-mute @DEFAULT_AUDIO_SINK@ 1',
      'pactl get-sink-volume @DEFAULT_SINK@', 'pactl set-sink-volume @DEFAULT_SINK@ 20%',
      'amixer -M -q set Master 100%',
    ])
  })

  it('drives Spotify by its MPRIS name and Music as playerctl’s default player', async () => {
    const c = controls(with_('playerctl'), { playerctl: '0.300000\n' })
    expect(await c.linux.run({ kind: 'volume', action: 'up', target: 'Spotify' })).toEqual({ done: 'Spotify volume is at 50%.' })
    expect(await c.linux.run({ kind: 'media', action: 'next', player: 'Music' })).toEqual({ done: 'Skipped to the next track on the music player.' })
    expect(c.lines()).toEqual(['playerctl -p spotify volume', 'playerctl -p spotify volume 0.50', 'playerctl next'])
  })

  it('says a player is not playing rather than failing, and opens Spotify for play', async () => {
    const none = new Error('No players found')
    const c = controls(with_('playerctl', 'gtk-launch'), { playerctl: none })
    expect(await c.linux.run({ kind: 'media', action: 'pause', player: 'Spotify' })).toEqual({ done: "Spotify isn't playing." })
    await expect(c.linux.run({ kind: 'media', action: 'play', player: 'Music' })).rejects.toThrow('No music player is open.')
    const spotify = new LinuxControls({ session: with_('playerctl', 'gtk-launch'), runner: c.runner, list: async (path) => (path === '/usr/share/applications' ? [{ name: 'spotify.desktop', dir: false }] : []), readText: async () => '[Desktop Entry]\nType=Application\nName=Spotify\n' })
    expect(await spotify.run({ kind: 'media', action: 'play', player: 'Spotify' })).toEqual({ done: 'Opened Spotify; press play once it has loaded.' })
    expect(c.lines().at(-1)).toBe('gtk-launch spotify')
  })

  it('switches GNOME’s color-scheme (and the classic Adwaita theme), and KDE’s colour scheme', async () => {
    const gnome = controls(fakeSession({ compositor: 'gnome', tools: ['gsettings'] }), { 'gsettings get': "'prefer-dark'\n" })
    expect(await gnome.linux.run({ kind: 'dark_mode', action: 'toggle' })).toEqual({ done: 'Dark mode is off.' })
    expect(gnome.lines()).toEqual([
      'gsettings get org.gnome.desktop.interface color-scheme',
      'gsettings set org.gnome.desktop.interface color-scheme default',
      'gsettings get org.gnome.desktop.interface gtk-theme',
    ])
    const kde = controls(fakeSession({ compositor: 'kde', tools: ['plasma-apply-colorscheme', 'kreadconfig6', 'gsettings'] }), { kreadconfig6: 'BreezeLight\n' })
    expect(await kde.linux.run({ kind: 'dark_mode', action: 'toggle' })).toEqual({ done: 'Dark mode is on.' })
    expect(kde.lines()).toEqual(['kreadconfig6 --file kdeglobals --group General --key ColorScheme', 'plasma-apply-colorscheme BreezeDark'])
  })

  it('locks through logind for this session, else the XDG screensaver', async () => {
    const logind = controls(fakeSession({ env: { XDG_SESSION_ID: '3' }, tools: ['loginctl'] }))
    await logind.linux.run({ kind: 'lock_screen' })
    const fallback = controls(with_('loginctl', 'xdg-screensaver'), { loginctl: new Error('Could not lock session: No session') })
    expect(await fallback.linux.run({ kind: 'lock_screen' })).toEqual({ done: 'Locked the screen.' })
    expect([...logind.lines(), ...fallback.lines()]).toEqual(['loginctl lock-session 3', 'loginctl lock-session', 'xdg-screensaver lock'])
  })

  it('quits by asking the app’s windows to close (Hyprland), and says when it was not open', async () => {
    const clients = [{ address: '0xa', class: 'org.gnome.Nautilus', title: 'Home', pid: 5 }, { address: '0xb', class: 'kitty', title: 'shell', pid: 6 }]
    const c = controls(fakeSession({ server: 'wayland', compositor: 'hyprland', tools: ['hyprctl'] }), { 'hyprctl clients': JSON.stringify(clients) })
    c.linux.resolve = async (name) => resolveEntry(name, ENTRIES)
    expect(await c.linux.run({ kind: 'quit_app', app: 'Files' })).toEqual({ done: 'Quit Files.' })
    expect(await c.linux.run({ kind: 'quit_app', app: 'VS Code' })).toEqual({ done: "Visual Studio Code wasn't open." })
    expect(c.lines()).toEqual(['hyprctl clients -j', 'hyprctl dispatch closewindow address:0xa', 'hyprctl clients -j'])
  })

  it('closes X11 windows with wmctrl, matching WM_CLASS', async () => {
    const list = '0x03a00003  0 4242   code.Code             box Visual Studio Code\n0x04000007  0 99     Navigator.firefox     box Mozilla Firefox\n'
    const c = controls(fakeSession({ server: 'x11', tools: ['wmctrl'] }), { 'wmctrl -lpx': list })
    c.linux.resolve = async (name) => resolveEntry(name, ENTRIES)
    await c.linux.run({ kind: 'quit_app', app: 'code' })
    expect(c.lines()).toEqual(['wmctrl -lpx', 'wmctrl -i -c 0x03a00003'])
  })

  it('refuses what a session cannot do, with why, before running anything', async () => {
    const gnome = fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['gtk-launch', 'gsettings', 'loginctl'] })
    expect(unsupportedControls(gnome)).toEqual({
      quit_app: 'GNOME on Wayland does not let other apps close windows; quit it from the app itself.',
      volume: 'Volume needs wpctl (PipeWire), pactl (PulseAudio) or amixer (ALSA).',
      media: 'Media controls need playerctl.',
    })
    const c = controls(gnome)
    await expect(c.linux.run({ kind: 'quit_app', app: 'Files' })).rejects.toMatchObject({ code: 'unavailable' })
    expect(c.calls).toHaveLength(0)
  })

  it('shows notifications with notify-send, title and text after "--"', async () => {
    const c = controls(with_('notify-send'))
    await c.linux.notify('Jarvis is using this computer', '-- stop with the tray')
    expect(c.calls[0]?.args).toEqual(['--app-name=LocalRouter', '--', 'Jarvis is using this computer', '-- stop with the tray'])
  })
})

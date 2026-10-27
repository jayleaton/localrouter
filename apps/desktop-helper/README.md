# LocalRouter Desktop

LocalRouter Desktop is a small helper on your Mac that lets the assistant on your LocalRouter box (Jarvis unless it was renamed there) use the Mac:
open and quit apps, change the volume, play the next track, switch dark mode, lock the screen, and do small tasks
on screen (bring up a browser tab, type a search, click a button) one step at a time.

The box makes every decision and grants every action. The helper reads the screen through macOS accessibility,
asks the box what to do next (`decide`), asks again before each click, keystroke, scroll or app/tab switch
(`authorize`), and acts only on a grant. Consequential actions wait for your tap in the LocalRouter web UI. The
protocol and the policy are in [docs/dev/COMPUTER.md](../../docs/dev/COMPUTER.md).

**Status: this helper has not been run on a real Mac yet.** AX observation, the OCR fallback, CGEvent clicks and
keys, the corner stop and the macOS permission prompts still need a first run on real hardware. The tests use a fake
Mac (`src/fake.ts`).

## Needs a real Mac

What the tests cannot show, to check on the first real run:

1. `node src/main.ts check`: Accessibility and Screen Recording reported as granted to the node binary after granting.
2. The LaunchAgent starts at login, connects (the box's Settings → Computer lists the Mac), and reconnects after the
   box restarts or the Mac sleeps.
3. Fixed controls: open and quit an app, volume up/down/set, Music or Spotify play/pause/next, dark mode, and lock
   (which waits for a tap).
4. AX observation in Safari, Chrome, Finder, Notes and an Electron app: the controls the box is offered match the
   window; clicks land on the right control on a Retina and an external display.
5. The OCR fallback in an app that hides its window from accessibility; the screenshot is deleted afterwards.
6. Chrome tab switching (by title, site and tab group) and the tab-search fallback.
7. Typing into a search box runs without a tap; typing into a message field and the Enter that sends it each show a
   card, and Deny stops the task.
8. The local stop: the pointer to the top-left corner, moving the mouse mid-task, and `node src/main.ts stop` each end
   the task before its next step; `disable` refuses tasks until `enable`.
9. The notification at a task's start.

## Requirements

- macOS 13 (Ventura) or later.
- Node.js 22.18 or later, which runs the TypeScript sources directly: `brew install node`.
- A device token with the `desktop_helper` scope, made on the box (computer control always needs paired devices, even
  when the box's `pairing.required` is off):
  `deploy/jarvis/pair.sh token "JJ's MacBook Pro" --scopes desktop_helper` (or `localrouter pair token ... --config FILE`).
  To approve taps from the web UI or a phone, that device needs `approve` (and `computer` to ask by voice):
  `deploy/jarvis/pair.sh scopes "JJ's phone" voice,mcp,computer,approve`. Put the web UI's token in Settings →
  Advanced (API key or device token).

The helper has no runtime dependencies; `npm install` is only needed to run its tests.

## Install

```sh
git clone https://github.com/jayleaton/localrouter.git
cd localrouter/apps/desktop-helper
node src/main.ts pair --url https://dgx-01.<tailnet>.ts.net:10443 --token <token>
node src/main.ts install-agent
```

`pair` stores the box's address, the token and this Mac's name (`--name "JJ's MacBook Pro"` to choose it) in
`~/Library/Application Support/LocalRouter Desktop/config.json`, readable only by you. The URL must be `https://`
(`http://` is accepted only for a box on this Mac itself). The token is never printed or logged.

`install-agent` installs the LaunchAgent `com.localrouter.desktop`
(`~/Library/LaunchAgents/com.localrouter.desktop.plist`): the node binary you ran it with runs `src/main.ts run`
at login, and again if it exits. Its log is `~/Library/Logs/LocalRouter Desktop.log`. Install it from the checkout
you mean to keep: the agent runs the files where they are.

### Grant Accessibility and Screen Recording

macOS asks about the program that does the work. Under the LaunchAgent that is the node binary itself
(`/opt/homebrew/bin/node` with Homebrew on Apple silicon, `/usr/local/bin/node` on Intel; `which node` shows it, and
`install-agent` prints it). In System Settings › Privacy & Security:

- **Accessibility**: add the node binary and switch it on. Needed to read windows, click, type and lock the screen.
- **Screen Recording**: add the node binary. Needed only for apps that hide their window from accessibility; the
  helper then reads that window from a screenshot taken and deleted on the Mac.
- **Automation**: macOS asks the first time the helper presses keys through System Events; allow it.

Homebrew upgrades of node can replace the binary; if a grant stops working after `brew upgrade`, remove the old
entry and add the node binary again, then `launchctl kickstart -k gui/$(id -u)/com.localrouter.desktop`.

Check what is granted, and which box the helper dials (the token is not shown):

```sh
node src/main.ts check
```

When you run `node src/main.ts run` from a terminal instead, macOS asks about the terminal app (Terminal, iTerm),
not node.

## Stopping it

While a task runs, a notification says what the assistant is doing on the Mac (by the name the box gives it). You can always stop it on the Mac itself,
without the box:

- move the pointer into the top-left corner of the main display (with displays arranged around it: the top-left corner of the screen arrangement, where the pointer stops);
- move the pointer yourself while it works (more than a short distance from where the helper left it): you took
  over, and the task stops;
- `node src/main.ts stop` stops the task running now;
- `node src/main.ts disable` refuses every request (“Computer control is switched off on this Mac”) until
  `node src/main.ts enable`. This is remembered across restarts, and works when the helper is not running.

`node src/main.ts status` shows whether the helper runs, is connected, and which task it is doing. These commands
talk to the running helper through `control.sock` in its config folder, which only you can open.

## Tap or no tap

The box decides, per action, whether it runs at once or waits for your tap (Approve / Deny) in the LocalRouter web
UI. From COMPUTER.md:

- **No tap**: opening and quitting apps, volume, media keys, dark mode; on screen, scrolling, Space, Escape, Tab,
  bringing a named app or tab forward, clicks the box's policy finds benign, typing into a search or address field,
  and Enter after typing into one.
- **A tap, each time**: locking the screen; typing into any other field; Enter after typing into one; any click whose
  label reads as sending, replying, posting, sharing, buying, paying, ordering, subscribing, deleting, removing,
  submitting, confirming or publishing, or that the decision model judges could do one of those.

A tap approves one exact action on the screen it was asked about, expires after 60 s, and Deny ends the task. A
spoken “yes” is not an approval. While the helper waits for your tap, moving the pointer does not count as taking
over, so you can tap Approve in a browser on the same Mac; the corner still stops the task.

Each task is bounded: 15 steps and 90 seconds of working time (time waiting for a tap does not count), and it stops
when it would repeat the same action on an unchanged screen.

## Uninstall

```sh
node src/main.ts uninstall-agent
rm -rf ~/Library/Application\ Support/LocalRouter\ Desktop
```

Then remove node from Accessibility and Screen Recording in System Settings if nothing else needs it, and revoke the
device token in LocalRouter.

## Troubleshooting

- **Nothing happens, `status` says not running**: `launchctl print gui/$(id -u)/com.localrouter.desktop` shows the
  agent's state; the log is `~/Library/Logs/LocalRouter Desktop.log` (JSON lines, no token, no typed text).
- **`connection.dropped` again and again in the log**: the box refuses the token (wrong token, revoked, missing the
  `desktop_helper` scope, or pairing switched off on the box), or the box is not reachable at the paired URL. Pair
  again with a new token, then `launchctl kickstart -k gui/$(id -u)/com.localrouter.desktop`.
- **“needs Accessibility access”**: grant it to the binary the message names (see above), then restart the agent.
- **“… is refusing to show its window to other apps”**: that app cannot be verified by macOS any more (often after an
  update replaced it while it was open). Quit and reopen the app.
- **A task ends “You moved the pointer…”**: the helper saw the pointer move away from where it left it. Keep your
  hands off the mouse while it works, or stop it and do the rest yourself.
- **Tasks are refused with “Computer control is switched off on this Mac”**: run `node src/main.ts enable`.

## Development

```sh
npm install
npm run type-check
npm test
LOCALROUTER_DESKTOP_FAKE=1 node src/main.ts run   # a fake Notes-like Mac, against a real box
```

`run` refuses to start on anything but macOS unless `LOCALROUTER_DESKTOP_FAKE=1`. The macOS backend is ported from
SideBud's companion and reused with the owner's permission.

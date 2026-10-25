# LocalRouter companion

The reference harness that runs on the user's own computer while the models stay on the Spark
([docs/HARNESS.md](../../docs/HARNESS.md)). This package holds:

- **`src/executor/`: the executor.** The final authority over this computer, whoever asks (our harness, a voice
  session's tool call, another harness over MCP, the box's legacy `/v1/desktop`). The policy, approvals, budgets and
  audit run here: [docs/dev/COMPUTER.md](../../docs/dev/COMPUTER.md#on-the-device-the-companions-executor-the-new-placement).
  It reuses `apps/desktop-helper`'s controller, Mac backend (AX, CGEvent, OCR) and local stop.
- **`src/harness/`: the harness core.** The Spark client (pairing, `/v1/capabilities`, chat, MCP, files, realtime),
  the tool registry (computer control through the executor; `list_files`/`read_file` inside shared folders only;
  `generate_image`, `generate_video`, `decide` on the Spark), the journal (one JSON-lines file per conversation, with a
  neutral export for switching harnesses) and realtime sessions in **client tool mode**: the Spark listens, thinks and
  speaks; every tool call runs here, once, with its complete arguments.

## Headless run (no Mac)

The same runtime on the fake Mac, for a live check against a box. It answers approvals itself, which only the fake Mac
allows:

```sh
node src/harness/cli.ts pair --url https://<box>:10443 --code XXXX-XXXX          # prints the device token
LOCALROUTER_TOKEN=… node src/harness/cli.ts turns --url https://<box>:10443 --fake-mac --speak \
  --expect list_files,computer_use,generate_image --json report.json \
  "What files are in my shared folder?" "Send my groceries note." "Make a picture of a red fox in snow."
```

- **`src/app/`: the desktop app** (Electron, macOS Apple Silicon first). Its UI is the repo's `/voice` page
  (`src/voice/voice.html`), bundled with a small shim (`src/app/ui/voice-shim.js`): the page's realtime socket goes
  through the harness in client tool mode, its `/v1/computer/*` calls are answered by this computer's executor (so its
  approval and task cards are this computer's), and its other `/v1/...` calls go to the Spark with the device token,
  which the page never sees. Approving needs Touch ID where the Mac has it.

## Install on a Mac (unsigned build)

1. Download the `LocalRouter-macos-arm64` artifact from the latest green **Companion (macOS)** run in GitHub Actions,
   unzip it, open `LocalRouter-<version>-arm64.dmg` and drag LocalRouter to Applications.
2. The build is signed with LocalRouter's own certificate, not an Apple Developer ID, so macOS refuses a double-click the first time. Either
   right-click LocalRouter in Applications › **Open** › **Open**, or in Terminal:
   `xattr -dr com.apple.quarantine /Applications/LocalRouter.app`. If macOS still says it is damaged, System Settings
   › Privacy & Security › **Open Anyway**.
3. On the Spark, make a pairing code (ten minutes, single use):
   `deploy/jarvis/pair.sh add "JJ's MacBook" --scopes inference,media,read`
4. LocalRouter's first window, **Connect**, walks four steps:
   1. **Spark address**: the `https://<spark>.<tailnet>.ts.net:10443` one `tailscale serve` gives the box (the Mac must
      be on the tailnet). If Tailscale runs on the Mac, the app asks each online peer's MagicDNS name for
      `/v1/capabilities` and offers the LocalRouter boxes it finds (one found and nothing typed: filled in).
      **Continue** checks the address before any code is spent. Pasting the `localrouter://pair?...` link `pair.sh add`
      prints fills in the code too.
   2. **Pairing code**: the computer's name and the code, then **Pair**. The token is kept in the macOS keychain.
   3. **Connection**: the box (version, machine, contract), this token's scopes and what it can use (chat, calls,
      pictures, video, speech, transcription, decisions, MCP), checked when the window opens and again on focus.
      **Re-pair** trades a new code for a new token (the old one works until then; revoke its entry on the Spark).
      **Disconnect** forgets the Spark on this Mac.
   4. **Permissions**: **Microphone** (for calls), **Accessibility** (to control the Mac) and **Screen Recording** (for
      apps that hide their window from accessibility). Add LocalRouter in each System Settings list, then quit and
      reopen LocalRouter. macOS asks about **Automation** (System Events, Chrome, Music…) the first time Jarvis uses
      each app.

   Each failure says what it was and what to do: an address that does not resolve or answer (Tailscale off, wrong
   name, nothing on that port, no answer within 8 s), a certificate the Mac does not trust (an IP address instead of
   the `.ts.net` name, an expired certificate, HTTP on an HTTPS port), something that is not LocalRouter, a box too old
   for the app, a code that is not a code (checked here, so a typo never counts toward the box's five wrong codes), a
   wrong, expired or used code, and a token the box no longer accepts.
5. **Open Jarvis**: the familiar page. Call with the mic button, chat, generations appear in the reel. Ask it to do
   something on the Mac ("open Safari", "search my notes for groceries"); anything consequential shows an approval
   card in the page and Touch ID confirms it. **Stop**: the card's Stop button, ⌘⇧⎋, the menu bar's **LR** menu, or
   the pointer in the top-left corner of the screen.

**One conversation, typed and spoken.** A call's turns, and every tool it ran with its result, go into the same chat
as typed turns, so typed chat knows what was said on the call; a call started after typing gets the chat (with its
tool calls) as context. Typed chat has the same tools as a call: `computer_control`, `computer_use`, `list_files`,
`read_file`, pictures and video, under the same executor rules and approvals. The app keeps the conversation in its
journal too (`~/Library/Application Support/LocalRouter/conversations/conv_page_*.jsonl`).

**Pictures on a call, videos in the chat.** A call makes pictures (they appear in the reel). A video is never started
on a call: it takes minutes and pauses the voice models, so Jarvis asks whether to end the call and make it. This is
the harness's rule, not the prompt's: on a call `generate_video` only records the request, and
`end_call_and_make_video` works only after the person answered with a clear yes. Then the call ends (its transcript
already in the chat), the app makes room on the box (`reserve_for`), the chat makes the video with its progress card,
and the voice models are asked back when it is done. A no keeps the call and makes nothing.

**If a call stutters**, Settings › Copy debug log on the page includes `app.audio_hop` per reply: how long the app took
to hand each piece of speech to the page (`hop_*`) and the gaps between pieces as they arrived (`gap_*`). Large gaps
with a small hop mean the box sent speech slower than real time (GPU contention, a model reloading); a large hop means
the app. During a call the app keeps macOS from napping it.

**Permissions across updates.** macOS ties Accessibility and Screen Recording to the app's signature. Every build is
signed with the same self-signed certificate (`scripts/make-signing-identity.sh`, once, into the repo's
`MACOS_SIGNING_P12` and `MACOS_SIGNING_PASSWORD` secrets), so its designated requirement (`identifier
"com.localrouter.companion" and certificate leaf = H"<the certificate's SHA-1>"`) is the same in every build and
the grants carry over. The first signed build still needs one more grant after an ad-hoc one. Without the secrets,
CI signs with a throwaway certificate and says so in the job summary: such builds lose the grants on update.

If macOS kept a grant from an older build (System Settings shows LocalRouter on, but computer control says it needs
Accessibility), Settings shows *macOS kept a permission from an older LocalRouter* with buttons that open the right
pane, reset LocalRouter's entry (`tccutil reset Accessibility com.localrouter.companion`, as you, no admin), and quit
and reopen. By hand: in the list select LocalRouter, remove it with −, add it with +, reopen the app.

Settings (⌘,) also shares folders Jarvis may read, adds other harnesses (MCP, below), lets the phone app and the
Spark's own web page keep using this Mac through the same local rules (pair a second code with
`--scopes desktop_helper`), and exports conversations for switching harness.

If the old LaunchAgent helper (`apps/desktop-helper`, `com.localrouter.desktop`) is installed, remove it first
(`node src/main.ts uninstall-agent` in its checkout), so two programs never drive the same Mac.

## Other harnesses: the executor over MCP

With the app running, enroll the harness in the app (Settings › Harnesses gives a token once), then:

```json
{ "mcpServers": { "computer": { "command": "/Applications/LocalRouter.app/Contents/Resources/localrouter-executor-mcp",
  "env": { "LOCALROUTER_EXECUTOR_TOKEN": "lrx_..." } } } }
```

Tools: `computer_capabilities`, `computer_control`, `computer_use`, `computer_status`, `computer_stop`. Approvals
appear in the app on this computer. No tool can approve.

## Checks

```sh
npm ci
npm run type-check   # tsc --noEmit, this package and the helper code it imports
npm run lint         # eslint
npm test             # vitest: policy parity with the box (src/computer/policy_golden.json), approvals, the broker

npm run build        # dist/: main.cjs, preload.cjs, mcp.cjs, ui/ (voice.html built from src/voice/voice.html)
npm run package:mac  # macOS only: the .app (ad-hoc signed), .dmg and .zip in out/
```

CI runs the checks (`.github/workflows/ci.yml`, job `companion`), and on a macOS 14 runner builds, packages and
smoke-starts the app (`--version`, `--smoke`: headless window, UI loaded with its shim, the executor answering on its
socket) and the MCP launcher, then uploads the artifact (`.github/workflows/companion-mac.yml`). The box's half of the parity check is
`src/computer/policy.zig`'s golden test under `zig build test`.

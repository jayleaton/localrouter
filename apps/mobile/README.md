# LocalRouter phone app

A native phone client (Expo / React Native, iOS and Android) for a LocalRouter box: a realtime voice call, chat, and
the generations reel of the pictures and videos made in the conversation. It talks to the box's own API over the
tailnet: no account, relay or cloud service. Adapted from Sidebud's phone app (see `NOTICE`).

## What it does

- **Voice.** "Hey Jarvis" (heard on the phone by its on-device recognizer; nothing leaves the phone until a call
  opens) or **Talk** opens a call on `/v1/realtime`: PCM16 mono at 24 kHz both ways, the box's VAD ending each turn.
  **Wait before replying** (Settings, 0.3 to 5 s, default 0.8 s) is how long you can pause before it answers; talking
  again within it restarts the wait. Talk over a reply (or tap the wave) to interrupt it: playback stops on the phone
  at once and the box is told how much you heard. "Mute audio", "mute mic", "mute voice" and "end call" work hands-free.
- **The voice** is the one you pick in Settings from what the box's speech model lists; it stays until you change
  it (a voice the box no longer has falls back to its default, with a notice).
- **Chat** uses the same session as text: typed messages during a call get spoken replies.
- **Pictures and videos** are made by the box's own tools whether you ask by voice or text. Each appears as a card
  in the chat and in the generations reel; a failed one shows why, with **Retry**. Tap a card for full size or to
  play the video.

## Pair with a box

The box must be reachable from the phone: install Tailscale on the phone and join the box's tailnet. A Jarvis box
serves HTTPS on the tailnet at `https://<box>.<tailnet>.ts.net:10443` (see `INSTALL.md`).

On the box:

```bash
deploy/jarvis/pair.sh add "My phone" --url https://<box>.<tailnet>.ts.net:10443
# or, with the binary directly: localrouter pair add "My phone" --config <config.json> --url <address>
```

It prints a code like `K7QM-3XPD` (valid 10 minutes, single use) and a `localrouter://pair?…` link. In the app enter
the address and the code, or open the link on the phone. The phone keeps its token in the keychain. On the box,
`pair.sh list` shows the paired devices and `pair.sh revoke <id or name>` removes one.

Pairing is required only when the box's config sets `"pairing": {"required": true}`; until then the app also connects
without a code. Pair anyway, so it keeps working once the box turns pairing on.

## Install it on a phone

**Android:** run the **Mobile Android build** workflow (Actions tab, or `gh workflow run mobile-android.yml --ref
<branch>`), download the `localrouter-mobile-<sha>` artifact, unzip it and open `app-release.apk` on the phone (allow
installing from that source). The JavaScript is bundled in, so it needs no development server.

**iPhone:** an iPhone build has to be signed with an Apple developer account, which only its owner can do. With an
Expo account and an Apple Developer membership, from this directory:

```bash
pnpm install
pnpm dlx eas-cli login
pnpm dlx eas-cli init                         # creates the Expo project; export EAS_PROJECT_ID=<the id it prints>
pnpm dlx eas-cli device:create                # register the iPhone (open the link it prints on the phone)
pnpm dlx eas-cli build --platform ios --profile preview
```

Open the build's install link on the iPhone. `--profile development` makes a development client instead (it loads
the JavaScript from `pnpm start` on your computer). Expo Go cannot run the app: it lacks the native audio, call and
recognizer modules.

## Develop

```bash
pnpm install
pnpm lint && pnpm type-check && pnpm test
pnpm start          # with a development build on the phone
```

Run these where the project's rules allow it (for LocalRouter: CI or the GPU host, not a shared dev box).

`pnpm e2e` checks the app's protocol layer against a live box without a phone: one spoken turn (the prompt is
synthesized by the box, streamed at 24 kHz in real time) and one typed picture request, through the same
`src/lib/localrouter/` modules the app uses. It needs Node 22 or newer.

```bash
LR_URL=https://<box>.<tailnet>.ts.net:10443 LR_TOKEN=<token> pnpm e2e    # or LR_PAIR_CODE=XXXX-XXXX to pair first
```

## Structure

- `src/lib/localrouter/`: the wire protocol (events, session settings), the realtime client (reconnect, reseeding
  the conversation), the HTTP API, the conversation store, addresses and pairing links. No React Native: `e2e/` runs it
  under Node, and ESLint keeps it that way.
- `src/lib/app/`: the paired box, its session and this phone's settings and storage (the token in the keychain).
- `src/lib/voice/`: capture and playback (`voiceEngine.ts`), the reply start hold (`replyHold.ts`), Sidebud's echo
  gate, wake gate, spoken commands and PCM helpers, and when to listen (`standbyController.ts`).
- `modules/`: Sidebud's native modules: `call-session` (CallKit / Android telecom, the headset mute gesture, the
  Android background listening service) and `voice-commands` (on-device recognition).
- `src/app/`, `src/components/`: the screens (pair, home with chat, call and reel, settings) and their components.

Voice capture, playback and audio routing need a check on a real phone whenever they change.

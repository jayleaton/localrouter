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
```

CI runs the same (`.github/workflows/ci.yml`, job `companion`). The box's half of the parity check is
`src/computer/policy.zig`'s golden test under `zig build test`.

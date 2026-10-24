# LocalRouter companion

The reference harness that runs on the user's own computer while the models stay on the Spark
([docs/HARNESS.md](../../docs/HARNESS.md)). This package holds:

- **`src/executor/`: the executor.** The final authority over this computer, whoever asks (our harness, a voice
  session's tool call, another harness over MCP, the box's legacy `/v1/desktop`). The policy, approvals, budgets and
  audit run here: [docs/dev/COMPUTER.md](../../docs/dev/COMPUTER.md#on-the-device-the-companions-executor-the-new-placement).
  It reuses `apps/desktop-helper`'s controller, Mac backend (AX, CGEvent, OCR) and local stop.

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

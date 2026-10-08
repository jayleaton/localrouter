# Security

LocalRouter has no authentication or authorization. Anyone who can reach its listener can submit jobs, fetch
retained outputs and use operator endpoints to load or unload models. Bind it to a trusted private network or a
private tunnel address and restrict access with a firewall. Do not expose the listener directly to the internet.
The default listener is `127.0.0.1:8190`, reachable from the same machine only. To serve a tailnet, set `host` (or
`--host`) to `tailscale`, which listens on the machine's tailnet address (100.64.0.0/10) alone, or run the container
installer with `LOCALROUTER_BIND=tailscale`. `host: all` (`0.0.0.0`) listens on every interface and is for trusted
networks behind a firewall. In the container image the daemon listens on all of the container's interfaces and the
published port decides who can connect: `127.0.0.1` by default (`LOCALROUTER_BIND` in `docker/compose.yaml`), and the
tailscale variant runs on the host network with `--host tailscale`.

Treat adapters and checkpoint sources as trusted code and data. Workers execute configured commands with the
service's permissions. Keep secrets out of configuration and limit filesystem permissions to the required model,
job and log directories. Job inputs, outputs and logs can contain sensitive user content; outputs expire according
to `keep_outputs_s` (24 hours by default).

## URL inputs

Input images may use HTTP or HTTPS URLs, with a 32 MB limit and a 30 second timeout. By default all resolved addresses
must be public: loopback, private, link-local, carrier-grade NAT, unique-local IPv6, multicast, reserved and unspecified
addresses are refused, including IPv4 embedded in IPv6. The connection uses the checked address. Redirects are never
followed. The daemon's own output URLs are a deliberate exception and are fetched from itself.

`allow_private_urls: true` lifts the address restriction for trusted private input servers; redirects remain disabled.
Enable it only for trusted callers who may access those servers. See [Input URLs](docs/AGENTS.md#input-urls).

## Reporting

For a suspected vulnerability, contact the maintainer privately at jayleaton@gmail.com with reproduction steps and
impact. Do not post credentials, private inputs or exploit details in a public issue before a fix is available.

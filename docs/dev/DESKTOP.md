# A desktop with one consumer NVIDIA GPU

LocalRouter's compiled-in GPU engines (Qwen-Image, MiniMax H3, the chat and Drama 3 kits) are builds for the GB10.
On a desktop with a consumer card, a model runs as a `cmd` worker instead: a Python process the daemon starts and
stops ([ARCHITECTURE.md](../ARCHITECTURE.md#any-program-the-cmd-adapter)). MCP, the OpenAI routes and pairing are the
same as on the Spark. This page sets one up for video with [Wan 2.2 TI2V-5B](../../tools/video/wan/README.md) on Windows
11 through WSL2. On native Linux, skip the WSL steps.

## Measured

RTX 5070 Ti (16 GB, Blackwell sm_120, driver 621.14), 64 GB of RAM, Windows 11 26200, WSL 3.0.5 with Ubuntu 24.04
(30 GiB, the WSL default of half the RAM), torch 2.8.0+cu128, diffusers 0.43.0, 2026-11-18. Image to video, 704x1280,
5 s (120 frames at 24 fps), seed 7, called over MCP through `tailscale serve`:

| What | Measured |
| --- | --- |
| Load (cold) | 13 s |
| A denoising step (two transformer passes: guidance 5) | about 25 s |
| VAE decode (fp32, tiled) | about 190 s |
| GPU memory | 10.7 GiB peak allocated by the worker, with the transformer stored in fp8 |
| Host memory | 21.6 GiB resident while loaded, 26.5 GiB peak |

The desktop itself uses 1.5 to 3 GB of the card. Without the worker's placement fix
([the README](../../tools/video/wan/README.md#memory)), a step took 85 s: WDDM paged GPU memory once the VAE and the
transformer were both resident.

## Install (WSL2)

1. A distro on a disk with about 60 GB free (the weights are 34 GB), with systemd and a service user:

   ```powershell
   wsl --install -d Ubuntu-24.04 --no-launch
   wsl --manage Ubuntu-24.04 --move D:\WSL\Ubuntu-24.04      # optional: off a small system disk
   wsl -d Ubuntu-24.04 -u root -- bash -c "useradd -m -s /bin/bash localrouter; printf '[boot]\nsystemd=true\n[user]\ndefault=localrouter\n' > /etc/wsl.conf"
   wsl -d Ubuntu-24.04 -u root -- apt-get install -y ffmpeg git curl xz-utils python3-venv build-essential
   ```

   `nvidia-smi` inside the distro should list the card: the Windows driver provides CUDA to WSL; install no driver in
   the distro.

2. As `localrouter`: Zig 0.17.0, the daemon, the worker's Python (uv) and the weights:

   ```bash
   git clone https://github.com/jayleaton/localrouter ~/src/localrouter && cd ~/src/localrouter
   zig build -Doptimize=ReleaseSafe                      # no -Dnvcc: the worker brings its own CUDA (torch)
   tools/video/wan/fetch.sh ~/models/Wan2.2-TI2V-5B-Diffusers --venv ~/wan-venv
   ```

3. The config: [tools/video/wan/config.example.json](../../tools/video/wan/config.example.json) with its paths set, at
   `~/localrouter-desktop/config.json`. It keeps the daemon on `127.0.0.1`, turns pairing on, and sets
   `reserve_bytes` to 2 GiB: nothing else runs in the WSL VM, and the default 8 GiB would leave too little of its 30
   GiB for the worker's 27 GiB booking.

4. A system unit, `/etc/systemd/system/localrouter.service`, then `systemctl enable --now localrouter`:

   ```ini
   [Unit]
   Description=LocalRouter on 127.0.0.1:8190
   After=network-online.target

   [Service]
   User=localrouter
   WorkingDirectory=/home/localrouter/localrouter-desktop
   ExecStart=/home/localrouter/src/localrouter/zig-out/bin/localrouter serve --config /home/localrouter/localrouter-desktop/config.json
   KillMode=control-group
   Restart=on-failure
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

5. Keep the distro running. WSL stops a distro a few seconds after its last `wsl.exe` session ends, services and
   all; requests that arrive through Windows' localhost relay do not count. A per-user scheduled task holds one
   session open from logon, without a window and without administrator rights:

   ```powershell
   $a = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\conhost.exe" -Argument "--headless $env:WINDIR\System32\wsl.exe -d Ubuntu-24.04 -u localrouter --exec /bin/sleep infinity"
   $t = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
   $s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
   Register-ScheduledTask -TaskName "LocalRouter WSL keepalive" -Action $a -Trigger $t -Settings $s
   ```

   After a reboot the service comes up when that user logs on. To have it up before anyone logs on, an administrator
   registers the same task with `-AtStartup` and the user's stored credentials.

6. The tailnet: the daemon stays on loopback, and Tailscale on Windows puts HTTPS in front of it (Windows forwards its
   `127.0.0.1:8190` to the distro):

   ```powershell
   tailscale serve --bg --https=8190 http://127.0.0.1:8190
   ```

   The MCP endpoint is then `https://<machine>.<tailnet>.ts.net:8190/mcp`. `--host tailscale` does not work inside
   WSL's NAT network: the tailnet address belongs to Windows.

7. A token for each client ([MCP.md](../MCP.md#scopes)): `localrouter pair token NAME --scopes media,read --config
   ~/localrouter-desktop/config.json`. The token goes to stdout once; write it straight into the client's secret store.

## Start, stop, check

| Do | Command (Windows) |
| --- | --- |
| Status | `wsl -d Ubuntu-24.04 -u root -- systemctl status localrouter` |
| Stop / start / restart | `wsl -d Ubuntu-24.04 -u root -- systemctl stop localrouter` (`start`, `restart`) |
| Logs | `wsl -d Ubuntu-24.04 -u root -- journalctl -u localrouter -f` |
| Stop everything, until the next logon | `Stop-ScheduledTask "LocalRouter WSL keepalive"; wsl --terminate Ubuntu-24.04` |
| Start it again now | `Start-ScheduledTask "LocalRouter WSL keepalive"` |
| Off for good | `Disable-ScheduledTask "LocalRouter WSL keepalive"`, `systemctl disable localrouter` in the distro, `tailscale serve --https=8190 off` |
| Health | `curl https://<machine>.<tailnet>.ts.net:8190/health` |

The model loads on its first request and unloads after `idle_ttl_s` (900 s in the example), which returns the GPU and
the 22 GiB of host memory to Windows.

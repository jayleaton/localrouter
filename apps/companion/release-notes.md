**Beta.** LocalRouter for desktop is new: expect rough edges, and please [open an issue](https://github.com/jayleaton/localrouter/issues) when something does not work.

LocalRouter for desktop is Jarvis on your own computer: chat and voice calls with the models on your LocalRouter box,
pictures and video, and (after your approval) control of this computer. It needs a LocalRouter box you can reach,
for example over Tailscale ([README](https://github.com/jayleaton/localrouter#readme)).

## Download

| Computer | File |
| --- | --- |
| Mac with Apple silicon (M1 or later) | `LocalRouter-{{VERSION}}-arm64.dmg` |
| Mac with an Intel processor | `LocalRouter-{{VERSION}}-x64.dmg` |
| Windows 10 or 11 (x64) | `LocalRouter-{{VERSION}}-win-x64.exe` (installer), or the `.zip` to run without installing |
| Linux (x64) | `.AppImage`, or the `.deb` for Debian and Ubuntu |

`SHA256SUMS.txt` lists every file's SHA-256.

## Install

**macOS.** Open the `.dmg` and drag LocalRouter to Applications. The app is signed with LocalRouter's own certificate,
not an Apple Developer ID, and it is **not notarized**, so macOS refuses a double-click the first time:
right-click LocalRouter in Applications › **Open** › **Open**. If macOS says it cannot be opened or is damaged, open
System Settings › Privacy & Security and click **Open Anyway**, or in Terminal run
`xattr -dr com.apple.quarantine /Applications/LocalRouter.app`. Updates signed with the same certificate keep the
Accessibility and Screen Recording permissions you grant.

**Windows.** The installer is **not code-signed**, so Microsoft Defender SmartScreen may stop it: click
**More info** › **Run anyway**. The installer asks where to install, for you only (no administrator rights needed).

**Linux.** `chmod +x LocalRouter-*.AppImage` and run it, or `sudo apt install ./LocalRouter-*.deb`. Computer control
uses AT-SPI (`python3-gi`, `gir1.2-atspi-2.0`) and, on X11, `xdotool`.

## First start

On your LocalRouter box, make a pairing code with `deploy/jarvis/pair.sh add "My laptop"`. In the app's **Connect**
window, enter the box's address and the code. The app keeps its token in the system's secret store.

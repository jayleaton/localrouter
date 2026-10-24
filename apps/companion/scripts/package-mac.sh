#!/usr/bin/env bash
# Packages dist/ (from `npm run build`) as out/LocalRouter-darwin-arm64/LocalRouter.app, ad-hoc signed (no Apple
# identity: the user opens it with right-click > Open, or `xattr -dr com.apple.quarantine`), then a .dmg and a .zip.
# macOS only. `npm run package:mac` builds and packages.
set -euo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
  echo "package-mac: macOS only (codesign, hdiutil, ditto)" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
root=$(pwd)
[ -f dist/package.json ] && [ -f dist/main.cjs ] && [ -f dist/mcp.cjs ] || { echo "package-mac: run npm run build first" >&2; exit 1; }

version=$(node -p 'require("./package.json").version')
electron_version=$(node -p 'require("electron/package.json").version')
out="$root/out"
app="$out/LocalRouter-darwin-arm64/LocalRouter.app"
rm -rf "$out"
mkdir -p "$out"

echo "package-mac: LocalRouter $version on Electron $electron_version"
node --input-type=module - "$root/dist" "$out" "$electron_version" "$version" <<'JS'
import { packager } from '@electron/packager'
const [dir, out, electronVersion, appVersion] = process.argv.slice(2)
const paths = await packager({
  dir,
  out,
  electronVersion,
  appVersion,
  buildVersion: appVersion,
  name: 'LocalRouter',
  executableName: 'LocalRouter',
  platform: 'darwin',
  arch: 'arm64',
  appBundleId: 'com.localrouter.companion',
  appCategoryType: 'public.app-category.productivity',
  asar: false,
  prune: false,
  overwrite: true,
  extendInfo: {
    NSMicrophoneUsageDescription: 'LocalRouter listens to you during a voice call.',
    NSAppleEventsUsageDescription:
      'LocalRouter controls apps on this Mac when you ask it to, after your approval for anything consequential.',
    LSMinimumSystemVersion: '13.0',
    LSApplicationCategoryType: 'public.app-category.productivity',
  },
})
console.log(`package-mac: packaged ${paths.join(', ')}`)
JS

[ -x "$app/Contents/MacOS/LocalRouter" ] || { echo "package-mac: $app has no LocalRouter executable" >&2; exit 1; }
[ -f "$app/Contents/Resources/app/mcp.cjs" ] || { echo "package-mac: mcp.cjs missing from the bundle" >&2; exit 1; }

# The executor's MCP server for other harnesses: Electron's own Node runs dist/mcp.cjs.
launcher="$app/Contents/Resources/localrouter-executor-mcp"
cat > "$launcher" <<'SH'
#!/bin/sh
# LocalRouter's executor over MCP stdio (see the app's README). Runs with the app's bundled Node.
set -e
here=$(cd "$(dirname "$0")" && pwd -P)
contents=$(dirname "$here")
ELECTRON_RUN_AS_NODE=1 exec "$contents/MacOS/LocalRouter" "$contents/Resources/app/mcp.cjs" "$@"
SH
chmod 755 "$launcher"

# Ad-hoc signature over the whole bundle (after every file is in place: the launcher is a sealed resource).
codesign --force --deep --sign - --timestamp=none "$app"
codesign --verify --deep --strict --verbose=2 "$app"
codesign -dv "$app" 2>&1

dmg="$out/LocalRouter-$version-arm64.dmg"
zip="$out/LocalRouter-$version-arm64.zip"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
ditto "$app" "$stage/LocalRouter.app"
ln -s /Applications "$stage/Applications"
# hdiutil on CI runners sometimes fails with "Resource busy"; retry.
for attempt in 1 2 3; do
  if hdiutil create -volname "LocalRouter $version" -srcfolder "$stage" -fs HFS+ -format UDZO -ov "$dmg"; then break; fi
  [ "$attempt" -lt 3 ] || { echo "package-mac: hdiutil failed" >&2; exit 1; }
  sleep 5
done
ditto -c -k --sequesterRsrc --keepParent "$app" "$zip"

echo "package-mac: sizes"
du -sh "$app" "$dmg" "$zip"

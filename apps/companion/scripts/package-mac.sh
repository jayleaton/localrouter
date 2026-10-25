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

# Signing, after every file is in place (the launcher is a sealed resource). macOS ties Accessibility and Screen
# Recording to the app's designated requirement: an ad-hoc build's is its code hash, new with every build, so each
# update loses them. With LR_SIGN_P12 (a base64 .p12, scripts/make-signing-identity.sh) and LR_SIGN_PASSWORD, every
# build is signed by the same certificate and its requirement is the bundle id and that certificate: the grants
# survive updates. Without them the build is ad-hoc, as before, and says so.
bundle_id=com.localrouter.companion
signed_by=ad-hoc
if [ -n "${LR_SIGN_P12:-}" ]; then
  [ -n "${LR_SIGN_PASSWORD:-}" ] || { echo "package-mac: LR_SIGN_P12 without LR_SIGN_PASSWORD" >&2; exit 1; }
  sign_dir=$(mktemp -d)
  kc="$sign_dir/lr-signing.keychain-db"
  kc_pass=$(openssl rand -hex 16)
  cleanup_signing() { security delete-keychain "$kc" >/dev/null 2>&1 || true; rm -rf "$sign_dir"; }
  trap 'cleanup_signing; rm -rf "${stage:-}"' EXIT
  printf '%s' "$LR_SIGN_P12" | base64 --decode > "$sign_dir/id.p12"
  openssl pkcs12 -in "$sign_dir/id.p12" -passin env:LR_SIGN_PASSWORD -nokeys -clcerts -out "$sign_dir/cert.pem" 2>/dev/null \
    || openssl pkcs12 -legacy -in "$sign_dir/id.p12" -passin env:LR_SIGN_PASSWORD -nokeys -clcerts -out "$sign_dir/cert.pem"
  sha1=$(openssl x509 -in "$sign_dir/cert.pem" -noout -fingerprint -sha1 | sed 's/.*=//; s/://g')
  security create-keychain -p "$kc_pass" "$kc"
  security set-keychain-settings -lut 3600 "$kc"
  security unlock-keychain -p "$kc_pass" "$kc"
  security import "$sign_dir/id.p12" -k "$kc" -P "$LR_SIGN_PASSWORD" -T /usr/bin/codesign >/dev/null
  security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$kc_pass" "$kc" >/dev/null
  # codesign finds the identity in the search list; the user's own keychains stay after it.
  security list-keychains -d user -s "$kc" $(security list-keychains -d user | tr -d '"')
  # A self-signed certificate is trusted for code signing on a CI runner (passwordless sudo); not needed elsewhere.
  if [ -n "${CI:-}" ]; then perl -e 'alarm shift; exec @ARGV' 60 sudo -n security add-trusted-cert -d -r trustRoot -p codeSign -k /Library/Keychains/System.keychain "$sign_dir/cert.pem" || echo "package-mac: could not mark the certificate trusted (signing goes on)"; fi
  requirement="designated => identifier \"$bundle_id\" and certificate leaf = H\"$sha1\""
  # The keychain helper (native/keychain-helper.c): installed once outside the bundle by the app and kept across
  # updates, so the Spark token never brings a keychain prompt after an update. It answers only a parent that meets
  # this app's requirement, compiled in; its own requirement pins it to the same certificate.
  kh_id="$bundle_id.keychain"
  kh_build="$sign_dir/keychain"
  mkdir -p "$kh_build"
  printf '#define LR_APP_REQUIREMENT "identifier \\"%s\\" and certificate leaf = H\\"%s\\""\n' "$bundle_id" "$sha1" > "$kh_build/requirement.h"
  xcrun --sdk macosx clang -O2 -Wall -Werror -mmacosx-version-min=13.0 -arch arm64 -I "$kh_build" \
    ${LR_KEYCHAIN_CFLAGS:-} -framework Security -framework CoreFoundation -o "$kh_build/localrouter-keychain" "$root/native/keychain-helper.c"
  kh="$app/Contents/Resources/localrouter-keychain"
  cp "$kh_build/localrouter-keychain" "$kh"
  # Every nested helper and framework first, then the keychain helper and the app, each with its pinned requirement.
  codesign --force --deep --sign "$sha1" --keychain "$kc" --timestamp=none "$app"
  codesign --force --sign "$sha1" --keychain "$kc" --timestamp=none --identifier "$kh_id" \
    -r="designated => identifier \"$kh_id\" and certificate leaf = H\"$sha1\"" "$kh"
  codesign --verify --strict -R="identifier \"$kh_id\" and certificate leaf = H\"$sha1\"" "$kh"
  echo "package-mac: keychain helper $(codesign -dv "$kh" 2>&1 | grep -E '^CDHash=' || true), requirement: $(codesign -d -r- "$kh" 2>&1 | sed -n 's/^designated => //p')"
  codesign --force --sign "$sha1" --keychain "$kc" --timestamp=none --identifier "$bundle_id" -r="$requirement" "$app"
  # What TCC checks on every launch: this build satisfies the requirement every build has.
  codesign --verify --deep --strict -R="identifier \"$bundle_id\" and certificate leaf = H\"$sha1\"" "$app"
  signed_by="LocalRouter Code Signing (SHA-1 $sha1)"
  if [ -n "${LR_SIGN_THROWAWAY:-}" ]; then
    signed_by="$signed_by: a throwaway identity made for this build (no MACOS_SIGNING_P12 secret), so permissions will not survive the next update"
    echo "::warning::package-mac: $signed_by"
  fi
else
  echo "::warning::package-mac: no signing identity (MACOS_SIGNING_P12): this build is ad-hoc signed, and Accessibility and Screen Recording will not survive the next update"
  echo "::warning::package-mac: no keychain helper without a signing identity: the token is kept with safeStorage, which asks again after every update"
  codesign --force --deep --sign - --timestamp=none "$app"
fi
codesign --verify --deep --strict --verbose=2 "$app"
codesign -dv "$app" 2>&1
echo "package-mac: designated requirement: $(codesign -d -r- "$app" 2>&1 | sed -n 's/^designated => //p')"
echo "package-mac: signed by $signed_by"
printf '%s\n' "$signed_by" > "$out/signed-by.txt"

dmg="$out/LocalRouter-$version-arm64.dmg"
zip="$out/LocalRouter-$version-arm64.zip"
stage=$(mktemp -d)
trap 'rm -rf "$stage"; [ -n "${kc:-}" ] && { security delete-keychain "$kc" >/dev/null 2>&1 || true; rm -rf "$sign_dir"; }; true' EXIT
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

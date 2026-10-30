#!/usr/bin/env bash
# Makes LocalRouter's macOS code-signing identity once, and gives it to CI as two GitHub Actions secrets.
#
# Without an Apple Developer ID, each ad-hoc signed build has a new code hash, and macOS (TCC) ties Accessibility and
# Screen Recording to it: every update silently loses them. Signed with one self-signed certificate, every build has
# the same designated requirement (bundle id + this certificate), so the grants survive updates.
#
#   apps/companion/scripts/make-signing-identity.sh [--repo OWNER/NAME] [--keep DIR] [--dry-run]
#
# --dry-run makes the identity in DIR and sets no secret (to check the script).
#
# Needs openssl and an authenticated gh with admin on the repo. Sets MACOS_SIGNING_P12 (base64 .p12) and
# MACOS_SIGNING_PASSWORD. Keeps a copy of the .p12 and its password in DIR (default ~/.config/localrouter-signing,
# mode 0700): put them in a password manager. Losing them means a new certificate, and every Mac grants once more.
# Run it again only to replace the identity (it refuses while a copy exists, unless --force).
set -euo pipefail

repo=jayleaton/localrouter
keep="${HOME}/.config/localrouter-signing"
force=0
dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) repo=$2; shift 2 ;;
    --keep) keep=$2; shift 2 ;;
    --force) force=1; shift ;;
    --dry-run) dry=1; shift ;;
    *) echo "usage: $0 [--repo OWNER/NAME] [--keep DIR] [--force]" >&2; exit 2 ;;
  esac
done

name="LocalRouter Code Signing"
if [ -f "$keep/localrouter-signing.p12" ] && [ "$force" != 1 ]; then
  echo "$keep/localrouter-signing.p12 exists: that is the identity (re-run with --force to replace it, which makes every Mac grant permissions once more)" >&2
  exit 1
fi
command -v openssl >/dev/null || { echo "needs openssl" >&2; exit 1; }
[ "$dry" = 1 ] || command -v gh >/dev/null || { echo "needs gh" >&2; exit 1; }

umask 077
mkdir -p "$keep"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat > "$work/req.cnf" <<EOF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $name
O = LocalRouter
[ext]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
subjectKeyIdentifier = hash
EOF

# Ten years: the designated requirement pins this certificate, so it must outlive every build that uses it.
openssl req -new -x509 -newkey rsa:3072 -nodes -days 3650 -sha256 -config "$work/req.cnf" -keyout "$work/key.pem" -out "$work/cert.pem" 2>/dev/null
password=$(openssl rand -base64 30 | tr -d '\n/+=' | cut -c1-32)
# macOS `security import` reads only the older PKCS#12 encryption: 3DES and a SHA-1 MAC.
# (LibreSSL, macOS's own openssl, has no -macalg; its MAC is SHA-1 already.)
openssl pkcs12 -export -inkey "$work/key.pem" -in "$work/cert.pem" -name "$name" \
  -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES -macalg sha1 -passout "pass:$password" -out "$work/id.p12" 2>/dev/null \
  || openssl pkcs12 -export -inkey "$work/key.pem" -in "$work/cert.pem" -name "$name" \
    -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES -passout "pass:$password" -out "$work/id.p12"

cp "$work/id.p12" "$keep/localrouter-signing.p12"
printf '%s\n' "$password" > "$keep/localrouter-signing.password"
cp "$work/cert.pem" "$keep/localrouter-signing.cert.pem"

if [ "$dry" = 1 ]; then
  echo "Dry run: no secret set."
else
  base64 < "$work/id.p12" | tr -d '\n' | gh secret set MACOS_SIGNING_P12 --repo "$repo"
  printf '%s' "$password" | gh secret set MACOS_SIGNING_PASSWORD --repo "$repo"
  echo "Set MACOS_SIGNING_P12 and MACOS_SIGNING_PASSWORD on $repo."
fi

sha1=$(openssl x509 -in "$work/cert.pem" -noout -fingerprint -sha1 | sed 's/.*=//; s/://g')
echo "Certificate: $name, SHA-1 $sha1 (every build's designated requirement names it)."
echo "Copies in $keep (0700): move them to a password manager."

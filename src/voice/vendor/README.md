# /voice/vendor: the page's pinned libraries

The `/voice` page (`src/voice/voice.html`) shows 3D models (kind `asset3d`, a Gaussian splat) with
[three.js](https://github.com/mrdoob/three.js) and [Spark](https://github.com/sparkjsdev/spark). Both are pinned, built
into one ESM file here, embedded in the daemon (`src/api/realtime.zig`, `vendor()`) and served at
`/voice/vendor/<name>`, the same way as the page. The companion app copies this directory into its bundle
(`apps/companion/scripts/build.mjs`). Nothing is fetched from a CDN. The page imports the file the first time a 3D model
is opened.

| File | What it is |
| --- | --- |
| `splat-viewer-3527417f73f0.js` | three.js and Spark: only the exports the page uses (`tools/voice-vendor/entry.js`), minified. The name carries the first 12 hex digits of its SHA-256, so it is served with `cache-control: immutable` |
| `THIRD_PARTY_NOTICES` | The MIT licence of each package, copied verbatim from the `LICENSE` file in its npm tarball |

## Pins

| Package | Version | npm tarball integrity | Licence |
| --- | --- | --- | --- |
| `three` | 0.187.0 | `sha512-Qiu7R+Cmh8CstZsY1uJOSWnVlfo7ZtNLV7mF8Jltk3LB3AtDu9cRZlcnRKkAk8Y4z6KKw35z7Y7Urd7jWY8etA==` | MIT, `Copyright © 2010-2026 three.js authors` |
| `@sparkjsdev/spark` | 2.4.0 (latest, 2026-10-20) | `sha512-aUjgBfhwDRcF3nUwiAD6Nh7IwvxEmW9liz2qrAXIm2BlPYBmn81QbXIpGFLBSr/o0TKGvEXbNIsKujgPGV/7QQ==` | MIT, `Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.` |
| `esbuild` (build only) | 0.28.2 | `sha512-HKVLS8dvII+xoKW9kmqxbRKrnWEXfJJr/FZhhJmiqIB0e053QNYFqOBouTMO/k5sID4MvCiUCvv8b9M4h32wIA==` | MIT (not shipped) |

`tools/voice-vendor/package-lock.json` locks the same versions and hashes. Spark 2.4.0's peer dependency is
`three >=0.180.0`.

Build output, SHA-256:

```
3527417f73f0b74900b29978fddfc27130e20e2a78f698895f191c08523cfac6  splat-viewer-3527417f73f0.js   (3,259,926 bytes; 1.09 MB gzipped)
47444bf93b4e9c73e736dbab0164ff2716028ec3bb74a356b187e697bca14f7c  THIRD_PARTY_NOTICES
```

## Rebuild

```sh
cd tools/voice-vendor
npm ci            # the locked tarballs above
npm run build     # node build.mjs
```

The build is reproducible. It was built with Node 22.22.2 on macOS arm64 and gave the same bytes on two runs.
`build.mjs` checks that the installed versions match the pins, writes both files here, removes the old bundle and puts
the new name into `src/voice/voice.html` (`VIEWER_JS`) and `src/api/realtime.zig` (`vendor_files`). To upgrade a
package, change its version in `tools/voice-vendor/package.json`, run `npm install` and then `npm run build`. Update this
README's pins and hashes, and check a 3D model in the page.

## What the page's CSP needs

Spark compiles its WebAssembly from bytes inlined in the bundle (`WebAssembly.compile`, in the page and in its
worker), so `script-src` includes `'wasm-unsafe-eval'`. This is both in the daemon's header (`src/api/realtime.zig`,
`page()`) and in the companion's (`apps/companion/src/app/main.ts`, `CSP`). Its sort worker is a `blob:` URL, which
`worker-src blob:` already allows. A `data:` fallback exists for browsers without `Blob` URLs and is not used. The
bundle makes no network requests: the page fetches the asset itself with its key (or through the companion's proxy)
and passes the bytes to the bundle.

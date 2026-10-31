// Builds src/voice/vendor/: the /voice page's 3D splat viewer (three.js + Spark) as one minified ESM file named by
// its content hash, and THIRD_PARTY_NOTICES with the licences from the pinned tarballs. Run from this directory:
//   npm ci && npm run build
// The file name is written into src/voice/voice.html and src/api/realtime.zig (the only places that name it).
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, readdirSync, unlinkSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, '..', '..');
const out = join(repo, 'src', 'voice', 'vendor');
const pkg = (name) => JSON.parse(readFileSync(join(here, 'node_modules', name, 'package.json'), 'utf8'));
const pins = JSON.parse(readFileSync(join(here, 'package.json'), 'utf8')).devDependencies;
for (const name of ['three', '@sparkjsdev/spark', 'esbuild']) {
  if (pkg(name).version !== pins[name]) throw new Error(`${name} ${pkg(name).version} installed, ${pins[name]} pinned: run npm ci`);
}

const result = await build({
  entryPoints: [join(here, 'entry.js')],
  bundle: true,
  format: 'esm',
  target: 'es2022',
  platform: 'browser',
  minify: true,
  legalComments: 'none',
  charset: 'utf8',
  write: false,
  banner: { js: `/* LocalRouter /voice 3D viewer: three ${pkg('three').version}, @sparkjsdev/spark ${pkg('@sparkjsdev/spark').version} (MIT). Licences: THIRD_PARTY_NOTICES beside this file. */` },
});
const code = result.outputFiles[0].contents;
const hash = createHash('sha256').update(code).digest('hex');
const name = `splat-viewer-${hash.slice(0, 12)}.js`;

for (const f of readdirSync(out)) if (/^splat-viewer-[0-9a-f]{12}\.js$/.test(f) && f !== name) unlinkSync(join(out, f));
writeFileSync(join(out, name), code);

const licence = (name) => readFileSync(join(here, 'node_modules', name, 'LICENSE'), 'utf8').trimEnd();
writeFileSync(join(out, 'THIRD_PARTY_NOTICES'), [
  `The /voice page's 3D viewer (${name}) bundles the following software, each under the MIT License.`,
  `The licence texts below are copied verbatim from the LICENSE file of each pinned npm tarball.`,
  '',
  `================================================================================`,
  `three ${pkg('three').version} (https://github.com/mrdoob/three.js)`,
  `================================================================================`,
  '',
  licence('three'),
  '',
  `================================================================================`,
  `@sparkjsdev/spark ${pkg('@sparkjsdev/spark').version} (https://github.com/sparkjsdev/spark)`,
  `================================================================================`,
  '',
  licence('@sparkjsdev/spark'),
  '',
].join('\n'));

for (const file of [join(repo, 'src', 'voice', 'voice.html'), join(repo, 'src', 'api', 'realtime.zig')]) {
  const text = readFileSync(file, 'utf8');
  const next = text.replace(/splat-viewer-[0-9a-f]{12}\.js/g, name);
  if (next !== text) writeFileSync(file, next);
}
console.log(`${name}  ${code.length} bytes  sha256 ${hash}`);

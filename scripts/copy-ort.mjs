// Copies runtime files next to the app:
// - onnxruntime-web WebAssembly files (loaded as separate files by the AI worker, needed for threads)
// - coi-serviceworker, which turns on multi-threading on hosts that cannot set COOP/COEP headers (GitHub Pages)
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
const src = 'node_modules/onnxruntime-web/dist/';
mkdirSync('public/ort', { recursive: true });
for (const f of readdirSync(src)) if (/^ort-wasm-simd-threaded\.jsep\.(wasm|mjs)$/.test(f)) cpSync(src + f, 'public/ort/' + f);
cpSync('node_modules/coi-serviceworker/coi-serviceworker.min.js', 'public/coi-serviceworker.min.js');

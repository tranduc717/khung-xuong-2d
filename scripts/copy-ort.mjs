// Copies the onnxruntime-web WebAssembly files next to the app so they load as separate files
// (needed for multi-threaded WebAssembly inside our worker).
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
const src = 'node_modules/onnxruntime-web/dist/';
mkdirSync('public/ort', { recursive: true });
for (const f of readdirSync(src)) if (/^ort-wasm-simd-threaded\.jsep\.(wasm|mjs)$/.test(f)) cpSync(src + f, 'public/ort/' + f);

// Runs the MobileSAM segmentation model inside a Web Worker so the page never freezes.
// WebGPU is tried first (integrated GPUs work); otherwise the CPU (WebAssembly) is used.
import * as ort from 'onnxruntime-web/webgpu';

// The runtime files live in /ort/ (copied there at build time), next to /assets/ where this worker is.
ort.env.wasm.wasmPaths = new URL('../ort/', (self as unknown as { location: Location }).location.href).href;
ort.env.wasm.numThreads = (self as unknown as { crossOriginIsolated: boolean }).crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

let encoder: ort.InferenceSession | null = null;
let decoder: ort.InferenceSession | null = null;
let device = 'cpu';
let embedding: ort.Tensor | null = null;
let size = { w: 0, h: 0 };

async function fetchModel(url: string, label: string): Promise<Uint8Array> {
  const cache = 'caches' in self ? await caches.open('khung-xuong-models-v1') : null;
  let res = cache ? await cache.match(url) : undefined;
  if (!res) {
    const r = await fetch(url);
    if (!r.ok) throw new Error('Không tải được mô hình ' + url);
    const total = Number(r.headers.get('content-length')) || 0;
    const reader = r.body!.getReader();
    const chunks: Uint8Array[] = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      post({ type: 'progress', label, got, total });
    }
    const buf = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    if (cache) await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } }));
    return buf;
  }
  return new Uint8Array(await res.arrayBuffer());
}

function post(msg: unknown, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(msg, transfer);
}

async function init(modelBase: string, allowGpu: boolean) {
  const encBytes = await fetchModel(modelBase + 'mobile_sam.encoder.onnx', 'Mô hình AI (phần 1/2)');
  const decBytes = await fetchModel(modelBase + 'mobile_sam.decoder.quant.onnx', 'Mô hình AI (phần 2/2)');
  const hasGpu = allowGpu && 'gpu' in navigator && !!(await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter().catch(() => null));
  if (hasGpu) {
    try {
      encoder = await ort.InferenceSession.create(encBytes, { executionProviders: ['webgpu', 'wasm'] });
      device = 'gpu';
    } catch (e) {
      console.warn('WebGPU failed, using CPU', e);
      encoder = null;
    }
  }
  console.log('[sam] creating cpu session');
  if (!encoder) {
    encoder = await ort.InferenceSession.create(encBytes, { executionProviders: ['wasm'] });
    device = 'cpu';
  }
  console.log('[sam] encoder ok');
  decoder = await ort.InferenceSession.create(decBytes, { executionProviders: ['wasm'] });
  post({ type: 'ready', device });
}

async function encode(w: number, h: number, rgb: Float32Array) {
  if (!encoder) throw new Error('Mô hình chưa sẵn sàng');
  const t = performance.now();
  const input = new ort.Tensor('float32', rgb, [h, w, 3]);
  let out = await encoder.run({ input_image: input });
  let emb = out.image_embeddings;
  // sanity check: a broken GPU result is all zeros / NaN -> redo on CPU
  const d = (await emb.getData()) as Float32Array;
  let bad = false, sum = 0;
  for (let i = 0; i < d.length; i += 97) { if (!Number.isFinite(d[i])) { bad = true; break; } sum += Math.abs(d[i]); }
  if ((bad || sum === 0) && device === 'gpu') {
    const bytes = await fetchModel((self as unknown as { __modelBase: string }).__modelBase + 'mobile_sam.encoder.onnx', '');
    encoder = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    device = 'cpu';
    out = await encoder.run({ input_image: input });
    emb = out.image_embeddings;
  }
  embedding = new ort.Tensor('float32', (await emb.getData()) as Float32Array, [1, 256, 64, 64]);
  size = { w, h };
  post({ type: 'encoded', ms: performance.now() - t, device });
}

async function decode(id: number, points: Array<[number, number, number]>) {
  if (!decoder || !embedding) throw new Error('Chưa phân tích ảnh');
  const n = points.length;
  const coords = new Float32Array(n * 2 + 2);
  const labels = new Float32Array(n + 1);
  points.forEach(([x, y, l], i) => { coords[i * 2] = x; coords[i * 2 + 1] = y; labels[i] = l; });
  // padding point (required by SAM when no box is given)
  labels[n] = -1;
  const feeds = {
    image_embeddings: embedding,
    point_coords: new ort.Tensor('float32', coords, [1, n + 1, 2]),
    point_labels: new ort.Tensor('float32', labels, [1, n + 1]),
    mask_input: new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
    has_mask_input: new ort.Tensor('float32', new Float32Array([0]), [1]),
    orig_im_size: new ort.Tensor('float32', new Float32Array([size.h, size.w]), [2]),
  };
  const out = await decoder.run(feeds);
  const m = (await out.masks.getData()) as Float32Array;
  const dims = out.masks.dims; // [1, k, h, w]
  const k = dims[1];
  const iou = (await out.iou_predictions.getData()) as Float32Array;
  let best = 0;
  for (let i = 1; i < k; i++) if (iou[i] > iou[best]) best = i;
  const plane = size.w * size.h;
  const mask = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) mask[i] = m[best * plane + i] > 0 ? 1 : 0;
  post({ type: 'mask', id, mask, score: iou[best] }, [mask.buffer]);
}

async function decodeBox(id: number, box: [number, number, number, number]) {
  // box prompt: labels 2 (top-left) and 3 (bottom-right)
  if (!decoder || !embedding) throw new Error('Chưa phân tích ảnh');
  const feeds = {
    image_embeddings: embedding,
    point_coords: new ort.Tensor('float32', new Float32Array(box), [1, 2, 2]),
    point_labels: new ort.Tensor('float32', new Float32Array([2, 3]), [1, 2]),
    mask_input: new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
    has_mask_input: new ort.Tensor('float32', new Float32Array([0]), [1]),
    orig_im_size: new ort.Tensor('float32', new Float32Array([size.h, size.w]), [2]),
  };
  const out = await decoder.run(feeds);
  const m = (await out.masks.getData()) as Float32Array;
  const plane = size.w * size.h;
  const mask = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) mask[i] = m[i] > 0 ? 1 : 0;
  post({ type: 'mask', id, mask, score: 1 }, [mask.buffer]);
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      (self as unknown as { __modelBase: string }).__modelBase = msg.modelBase;
      await init(msg.modelBase, msg.allowGpu);
    } else if (msg.type === 'encode') await encode(msg.w, msg.h, msg.rgb);
    else if (msg.type === 'decode') await decode(msg.id, msg.points);
    else if (msg.type === 'decodeBox') await decodeBox(msg.id, msg.box);
  } catch (err) {
    post({ type: 'error', id: msg.id, message: String((err as Error)?.message || err) });
  }
};

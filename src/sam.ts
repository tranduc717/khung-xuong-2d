// Main-thread wrapper around the segmentation worker.
import type { Mask, RGBAImage } from './image';

export type SamStatus =
  | { state: 'idle' }
  | { state: 'loading'; label: string; got: number; total: number }
  | { state: 'ready'; device: string }
  | { state: 'encoding' }
  | { state: 'encoded'; device: string; ms: number }
  | { state: 'error'; message: string };

export class Sam {
  private worker: Worker;
  private next = 1;
  private pending = new Map<number, { resolve: (m: Mask) => void; reject: (e: Error) => void }>();
  private readyP: Promise<void>;
  private encodedP: Promise<void> | null = null;
  private encodedResolve: (() => void) | null = null;
  private encodedReject: ((e: Error) => void) | null = null;
  status: SamStatus = { state: 'idle' };
  private imgW = 0;
  private imgH = 0;
  private scale = 1;
  onStatus: (s: SamStatus) => void = () => {};

  constructor() {
    this.worker = new Worker(new URL('./sam.worker.ts', import.meta.url), { type: 'module' });
    let readyResolve!: () => void, readyReject!: (e: Error) => void;
    this.readyP = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });
    this.readyP.catch(() => {});
    this.worker.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === 'progress') this.set({ state: 'loading', label: m.label, got: m.got, total: m.total });
      else if (m.type === 'ready') { this.set({ state: 'ready', device: m.device }); readyResolve(); }
      else if (m.type === 'encoded') { this.set({ state: 'encoded', device: m.device, ms: m.ms }); this.encodedResolve?.(); }
      else if (m.type === 'mask') { this.pending.get(m.id)?.resolve(m.mask); this.pending.delete(m.id); }
      else if (m.type === 'error') {
        const err = new Error(m.message);
        if (m.id && this.pending.has(m.id)) { this.pending.get(m.id)!.reject(err); this.pending.delete(m.id); }
        else {
          this.set({ state: 'error', message: m.message });
          readyReject(err);
          this.encodedReject?.(err);
        }
      }
    };
    const params = new URLSearchParams(location.search);
    this.worker.postMessage({
      type: 'init',
      modelBase: new URL(import.meta.env.BASE_URL + 'models/', location.href).href,
      allowGpu: params.get('gpu') !== '0',
    });
  }

  private set(s: SamStatus) { this.status = s; this.onStatus(s); }

  ready() { return this.readyP; }

  /** Analyse the image once (the slow part, a few seconds on CPU). */
  encode(im: RGBAImage): Promise<void> {
    this.encodedP = new Promise<void>((res, rej) => { this.encodedResolve = res; this.encodedReject = rej; });
    this.encodedP.catch(() => {});
    const run = async () => {
      await this.readyP;
      this.set({ state: 'encoding' });
      // This MobileSAM export always works on a 1024x682 canvas: the picture is scaled to fit,
      // placed top-left and padded with white (RGB 0..255).
      const MW = 1024, MH = 682;
      const sc = Math.min(MW / im.w, MH / im.h);
      const c = new OffscreenCanvas(MW, MH);
      const ctx = c.getContext('2d')!;
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, MW, MH);
      const src = new OffscreenCanvas(im.w, im.h);
      src.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(im.data), im.w, im.h), 0, 0);
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(src, 0, 0, im.w * sc, im.h * sc);
      const px = ctx.getImageData(0, 0, MW, MH).data;
      const rgb = new Float32Array(MW * MH * 3);
      for (let k = 0; k < MW * MH; k++) { rgb[k * 3] = px[k * 4]; rgb[k * 3 + 1] = px[k * 4 + 1]; rgb[k * 3 + 2] = px[k * 4 + 2]; }
      this.imgW = im.w; this.imgH = im.h; this.scale = sc;
      this.worker.postMessage({ type: 'encode', w: MW, h: MH, rgb }, [rgb.buffer]);
    };
    run().catch(e => this.encodedReject?.(e));
    return this.encodedP;
  }

  encoded() { return this.encodedP ?? Promise.reject(new Error('Chưa phân tích ảnh')); }

  /** points: [x, y, label] with label 1 = belongs, 0 = does not belong. */
  segment(points: Array<[number, number, number]>): Promise<Mask> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: m => resolve(this.crop(m)), reject });
      this.worker.postMessage({ type: 'decode', id, points: points.map(([x, y, l]) => [x * this.scale, y * this.scale, l]) });
    });
  }

  segmentBox(box: [number, number, number, number]): Promise<Mask> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: m => resolve(this.crop(m)), reject });
      this.worker.postMessage({ type: 'decodeBox', id, box: box.map(v => v * this.scale) });
    });
  }

  /** Masks come back on the model's 1024x682 canvas; map them back onto the picture. */
  private crop(m: Mask): Mask {
    const MW = 1024, w = this.imgW, h = this.imgH, sc = this.scale;
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const sy = Math.min(681, Math.floor((y + 0.5) * sc));
      for (let x = 0; x < w; x++) out[y * w + x] = m[sy * MW + Math.min(MW - 1, Math.floor((x + 0.5) * sc))];
    }
    return out;
  }
}

// Image helpers: loading, background removal, masks, distance transform, inpainting.

export interface RGBAImage {
  w: number;
  h: number;
  data: Uint8ClampedArray; // RGBA, straight alpha
}

export type Mask = Uint8Array; // 0/1 per pixel, length w*h

export const WORK_SIZE = 1024;

export async function loadImageFile(src: File | string): Promise<HTMLImageElement> {
  const url = typeof src === 'string' ? src : URL.createObjectURL(src);
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  await img.decode();
  return img;
}

/** Scale so the longest side is exactly WORK_SIZE (the size the AI model expects). */
export function toWorkImage(img: HTMLImageElement | HTMLCanvasElement): RGBAImage {
  const iw = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
  const ih = img instanceof HTMLImageElement ? img.naturalHeight : img.height;
  const s = WORK_SIZE / Math.max(iw, ih);
  const w = Math.round(iw * s);
  const h = Math.round(ih * s);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h);
  return { w, h, data: d.data };
}

export function imageToCanvas(im: RGBAImage, x0 = 0, y0 = 0, w = im.w, h = im.h): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  const id = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const sy = y + y0;
    if (sy < 0 || sy >= im.h) continue;
    for (let x = 0; x < w; x++) {
      const sx = x + x0;
      if (sx < 0 || sx >= im.w) continue;
      const si = (sy * im.w + sx) * 4;
      const di = (y * w + x) * 4;
      id.data[di] = im.data[si];
      id.data[di + 1] = im.data[si + 1];
      id.data[di + 2] = im.data[si + 2];
      id.data[di + 3] = im.data[si + 3];
    }
  }
  ctx.putImageData(id, 0, 0);
  return c;
}

/** Median colour of a thin frame around the image border. */
export function borderColor(im: RGBAImage): [number, number, number] {
  const rs: number[] = [], gs: number[] = [], bs: number[] = [];
  const push = (x: number, y: number) => {
    const i = (y * im.w + x) * 4;
    rs.push(im.data[i]); gs.push(im.data[i + 1]); bs.push(im.data[i + 2]);
  };
  for (let x = 0; x < im.w; x += 2) { push(x, 1); push(x, im.h - 2); }
  for (let y = 0; y < im.h; y += 2) { push(1, y); push(im.w - 2, y); }
  const med = (a: number[]) => a.sort((p, q) => p - q)[a.length >> 1];
  return [med(rs), med(gs), med(bs)];
}

/** How uniform the border is (0 = perfectly flat). */
export function borderSpread(im: RGBAImage, bg: [number, number, number]): number {
  let n = 0, far = 0;
  const test = (x: number, y: number) => {
    const i = (y * im.w + x) * 4;
    const d = Math.abs(im.data[i] - bg[0]) + Math.abs(im.data[i + 1] - bg[1]) + Math.abs(im.data[i + 2] - bg[2]);
    n++;
    if (d > 30) far++;
  };
  for (let x = 0; x < im.w; x += 3) { test(x, 1); test(x, im.h - 2); }
  for (let y = 0; y < im.h; y += 3) { test(1, y); test(im.w - 2, y); }
  return far / n;
}

/**
 * Background removal for plain backgrounds: flood fill from the border with a colour tolerance,
 * then a soft matte on the edge (alpha from colour distance) and colour decontamination.
 * Returns a new RGBA image with alpha and the binary silhouette.
 */
export function removePlainBackground(im: RGBAImage, tolerance = 28): { img: RGBAImage; mask: Mask } {
  const { w, h, data } = im;
  const bg = borderColor(im);
  const dist = new Float32Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    const dr = data[p] - bg[0], dg = data[p + 1] - bg[1], db = data[p + 2] - bg[2];
    dist[i] = Math.sqrt(dr * dr + dg * dg + db * db);
  }
  const isBg = new Uint8Array(w * h);
  const stack: number[] = [];
  const seed = (i: number) => { if (!isBg[i] && dist[i] <= tolerance) { isBg[i] = 1; stack.push(i); } };
  for (let x = 0; x < w; x++) { seed(x); seed((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { seed(y * w); seed(y * w + w - 1); }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w, y = (i / w) | 0;
    if (x > 0) seed(i - 1);
    if (x < w - 1) seed(i + 1);
    if (y > 0) seed(i - w);
    if (y < h - 1) seed(i + w);
  }
  // Enclosed gaps (between an arm and the body, between legs) that are background-coloured and
  // not tiny are background too. Small ones (eye whites, highlights) stay.
  removeEnclosedGaps(isBg, dist, w, h, tolerance * 0.5, w * h * 0.0006);
  const mask: Mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = isBg[i] ? 0 : 1;
  keepLargestComponents(mask, w, h, 0.02);
  return { img: applyMatte(im, mask, bg, tolerance), mask };
}

/** Uses a binary mask (e.g. from the AI model) and adds a soft edge. */
export function applyMatte(im: RGBAImage, mask: Mask, bg: [number, number, number] | null, tolerance = 28): RGBAImage {
  const { w, h, data } = im;
  const out = new Uint8ClampedArray(data);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, p = i * 4;
      if (!mask[i]) { out[p + 3] = 0; continue; }
      // edge pixel? (touches background)
      const edge = (x > 0 && !mask[i - 1]) || (x < w - 1 && !mask[i + 1]) || (y > 0 && !mask[i - w]) || (y < h - 1 && !mask[i + w]);
      if (edge && bg) {
        const dr = data[p] - bg[0], dg = data[p + 1] - bg[1], db = data[p + 2] - bg[2];
        const d = Math.sqrt(dr * dr + dg * dg + db * db);
        const a = Math.min(1, Math.max(0.15, (d - tolerance * 0.5) / (tolerance * 3)));
        out[p + 3] = Math.round(a * 255);
        // un-mix background colour so edges are not white-ish
        for (let c = 0; c < 3; c++) {
          const v = (data[p + c] - bg[c] * (1 - a)) / a;
          out[p + c] = Math.max(0, Math.min(255, v));
        }
      } else {
        out[p + 3] = 255;
      }
    }
  }
  return { w, h, data: out };
}

/** Removes small islands (dust) that are less than `frac` of the largest component. */
export function keepLargestComponents(mask: Mask, w: number, h: number, frac: number) {
  const label = new Int32Array(w * h).fill(-1);
  const sizes: number[] = [];
  const stack: number[] = [];
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || label[s] >= 0) continue;
    const id = sizes.length;
    let n = 0;
    label[s] = id;
    stack.push(s);
    while (stack.length) {
      const i = stack.pop()!;
      n++;
      const x = i % w, y = (i / w) | 0;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
      for (const j of nb) if (j >= 0 && mask[j] && label[j] < 0) { label[j] = id; stack.push(j); }
    }
    sizes.push(n);
  }
  const max = Math.max(0, ...sizes);
  for (let i = 0; i < w * h; i++) if (mask[i] && sizes[label[i]] < max * frac) mask[i] = 0;
}

function removeEnclosedGaps(isBg: Uint8Array, dist: Float32Array, w: number, h: number, tol: number, minSize: number) {
  const seen = new Uint8Array(w * h);
  const stack: number[] = [];
  for (let s = 0; s < w * h; s++) {
    if (isBg[s] || seen[s] || dist[s] > tol) continue;
    const comp: number[] = [];
    seen[s] = 1;
    stack.push(s);
    while (stack.length) {
      const i = stack.pop()!;
      comp.push(i);
      const x = i % w, y = (i / w) | 0;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
      for (const j of nb) if (j >= 0 && !isBg[j] && !seen[j] && dist[j] <= tol) { seen[j] = 1; stack.push(j); }
    }
    if (comp.length >= minSize) for (const i of comp) isBg[i] = 1;
  }
}

export function maskBBox(mask: Mask, w: number, h: number): { x0: number; y0: number; x1: number; y1: number } | null {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

/** Euclidean-ish distance transform (two-pass chamfer 3-4) to nearest 0 pixel. */
export function distanceInside(mask: Mask, w: number, h: number): Float32Array {
  const INF = 1e9;
  const d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = mask[i] ? INF : 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x > 0) v = Math.min(v, d[i - 1] + 1);
    if (y > 0) {
      v = Math.min(v, d[i - w] + 1);
      if (x > 0) v = Math.min(v, d[i - w - 1] + 1.4142);
      if (x < w - 1) v = Math.min(v, d[i - w + 1] + 1.4142);
    }
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) v = Math.min(v, 1);
    d[i] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x < w - 1) v = Math.min(v, d[i + 1] + 1);
    if (y < h - 1) {
      v = Math.min(v, d[i + w] + 1);
      if (x < w - 1) v = Math.min(v, d[i + w + 1] + 1.4142);
      if (x > 0) v = Math.min(v, d[i + w - 1] + 1.4142);
    }
    d[i] = v;
  }
  return d;
}

/**
 * Push-pull inpainting: fills pixels where `hole`=1 with a smooth blend of the surrounding known
 * pixels (`known`=1). Works well for flat, cel-shaded AI drawings.
 */
export function pushPullFill(im: RGBAImage, known: Mask, hole: Mask): void {
  const { w, h } = im;
  // level 0
  type Level = { w: number; h: number; c: Float32Array; a: Float32Array };
  const levels: Level[] = [];
  let cur: Level = { w, h, c: new Float32Array(w * h * 3), a: new Float32Array(w * h) };
  for (let i = 0; i < w * h; i++) {
    if (known[i] && !hole[i]) {
      cur.a[i] = 1;
      cur.c[i * 3] = im.data[i * 4];
      cur.c[i * 3 + 1] = im.data[i * 4 + 1];
      cur.c[i * 3 + 2] = im.data[i * 4 + 2];
    }
  }
  levels.push(cur);
  while (cur.w > 1 || cur.h > 1) {
    const nw = Math.max(1, Math.ceil(cur.w / 2)), nh = Math.max(1, Math.ceil(cur.h / 2));
    const nx: Level = { w: nw, h: nh, c: new Float32Array(nw * nh * 3), a: new Float32Array(nw * nh) };
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
      let a = 0, r = 0, g = 0, b = 0;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const sx = x * 2 + dx, sy = y * 2 + dy;
        if (sx >= cur.w || sy >= cur.h) continue;
        const si = sy * cur.w + sx;
        const sa = cur.a[si];
        a += sa; r += cur.c[si * 3] * sa; g += cur.c[si * 3 + 1] * sa; b += cur.c[si * 3 + 2] * sa;
      }
      const di = y * nw + x;
      if (a > 0) { nx.c[di * 3] = r / a; nx.c[di * 3 + 1] = g / a; nx.c[di * 3 + 2] = b / a; }
      nx.a[di] = Math.min(1, a);
    }
    levels.push(nx);
    cur = nx;
  }
  // pull: fill missing from coarser level
  for (let l = levels.length - 2; l >= 0; l--) {
    const L = levels[l], C = levels[l + 1];
    for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
      const i = y * L.w + x;
      const a = L.a[i];
      if (a >= 1) continue;
      // bilinear sample of the coarser level (avoids blocky fills)
      const fx = Math.max(0, Math.min(C.w - 1, (x + 0.5) / 2 - 0.5)), fy = Math.max(0, Math.min(C.h - 1, (y + 0.5) / 2 - 0.5));
      const ix = Math.floor(fx), iy = Math.floor(fy), jx = Math.min(C.w - 1, ix + 1), jy = Math.min(C.h - 1, iy + 1);
      const tx = fx - ix, ty = fy - iy;
      for (let c = 0; c < 3; c++) {
        const v = (C.c[(iy * C.w + ix) * 3 + c] * (1 - tx) + C.c[(iy * C.w + jx) * 3 + c] * tx) * (1 - ty)
          + (C.c[(jy * C.w + ix) * 3 + c] * (1 - tx) + C.c[(jy * C.w + jx) * 3 + c] * tx) * ty;
        L.c[i * 3 + c] = L.c[i * 3 + c] * a + v * (1 - a);
      }
      L.a[i] = 1;
    }
  }
  const L0 = levels[0];
  for (let i = 0; i < w * h; i++) {
    if (!hole[i]) continue;
    im.data[i * 4] = L0.c[i * 3];
    im.data[i * 4 + 1] = L0.c[i * 3 + 1];
    im.data[i * 4 + 2] = L0.c[i * 3 + 2];
    im.data[i * 4 + 3] = 255;
  }
}

export function dilate(mask: Mask, w: number, h: number, r: number): Mask {
  if (r <= 0) return mask.slice();
  const d = distanceInside(invert(mask), w, h);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = mask[i] || d[i] <= r ? 1 : 0;
  return out;
}

export function invert(mask: Mask): Mask {
  const o = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) o[i] = mask[i] ? 0 : 1;
  return o;
}

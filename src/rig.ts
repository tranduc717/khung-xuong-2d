// Turns part pictures into deformable meshes with smooth bone weights, plus the mannequin underlay.
import { distanceInside, type RGBAImage } from './image';
import { distToSegment, matApply, type BoneId, type Mat, type RestBone } from './skeleton';
import { DETAILS, GROUP_BONES, GROUP_INDEX, GROUP_Z, GROUPS, type Assignment, type Detail, type GroupId } from './segment';

export interface Layer {
  id: string;
  name: string;
  group: GroupId | 'mannequin';
  z: number;
  canvas: HTMLCanvasElement; // texture (cropped)
  x0: number; y0: number;    // where the crop sits in work space
  rest: Float32Array;        // vertex rest positions (x,y) in work space
  uv: Float32Array;
  index: Uint16Array | Uint32Array;
  bones: Int8Array;          // 4 per vertex, index into rig.bones (-1 none)
  weights: Float32Array;     // 4 per vertex
  tint: number;              // 1 = normal, <1 darker (far limbs)
  visible: boolean;
}

export interface Rig {
  bones: RestBone[];
  layers: Layer[];
  height: number;
}

export interface RigOptions {
  cloneFarArm: boolean;
  cloneFarLeg: boolean;
  mannequin: boolean;
  weaponHand: 'N' | 'F';
}

const FAR_OF: Partial<Record<BoneId, BoneId>> = {
  upperArmN: 'upperArmF', forearmN: 'forearmF', thighN: 'thighF', shinN: 'shinF', footN: 'footF',
};

function crop(im: RGBAImage, pad: number): { canvas: HTMLCanvasElement; x0: number; y0: number; alpha: Uint8Array; cw: number; ch: number } | null {
  let x0 = im.w, y0 = im.h, x1 = -1, y1 = -1;
  for (let y = 0; y < im.h; y++) for (let x = 0; x < im.w; x++) if (im.data[(y * im.w + x) * 4 + 3] > 8) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < 0) return null;
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1;
  const c = document.createElement('canvas');
  c.width = cw; c.height = ch;
  const ctx = c.getContext('2d')!;
  const id = ctx.createImageData(cw, ch);
  const alpha = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const sx = x + x0, sy = y + y0;
    if (sx < 0 || sy < 0 || sx >= im.w || sy >= im.h) continue;
    const si = (sy * im.w + sx) * 4, di = (y * cw + x) * 4;
    id.data[di] = im.data[si]; id.data[di + 1] = im.data[si + 1]; id.data[di + 2] = im.data[si + 2]; id.data[di + 3] = im.data[si + 3];
    alpha[y * cw + x] = im.data[si + 3] > 8 ? 1 : 0;
  }
  ctx.putImageData(id, 0, 0);
  return { canvas: c, x0, y0, alpha, cw, ch };
}

/** Grid mesh covering the opaque pixels of a crop. */
function gridMesh(alpha: Uint8Array, cw: number, ch: number, x0: number, y0: number, step: number) {
  const nx = Math.ceil((cw - 1) / step) + 1, ny = Math.ceil((ch - 1) / step) + 1;
  const cellOn = new Uint8Array((nx - 1) * (ny - 1));
  for (let cy = 0; cy < ny - 1; cy++) for (let cx = 0; cx < nx - 1; cx++) {
    // a cell is used if any opaque pixel lies in it or right next to it
    const xa = Math.max(0, cx * step - 2), xb = Math.min(cw - 1, (cx + 1) * step + 2);
    const ya = Math.max(0, cy * step - 2), yb = Math.min(ch - 1, (cy + 1) * step + 2);
    let on = 0;
    for (let y = ya; y <= yb && !on; y++) for (let x = xa; x <= xb; x++) if (alpha[y * cw + x]) { on = 1; break; }
    cellOn[cy * (nx - 1) + cx] = on;
  }
  const vid = new Int32Array(nx * ny).fill(-1);
  const pos: number[] = [], uv: number[] = [], idx: number[] = [];
  const vert = (gx: number, gy: number) => {
    const k = gy * nx + gx;
    if (vid[k] >= 0) return vid[k];
    const px = Math.min(gx * step, cw - 1), py = Math.min(gy * step, ch - 1);
    vid[k] = pos.length / 2;
    pos.push(px + x0, py + y0);
    uv.push((px + 0.5) / cw, (py + 0.5) / ch);
    return vid[k];
  };
  for (let cy = 0; cy < ny - 1; cy++) for (let cx = 0; cx < nx - 1; cx++) {
    if (!cellOn[cy * (nx - 1) + cx]) continue;
    const a = vert(cx, cy), b = vert(cx + 1, cy), c = vert(cx, cy + 1), d = vert(cx + 1, cy + 1);
    idx.push(a, b, c, b, d, c);
  }
  const n = pos.length / 2;
  return {
    rest: new Float32Array(pos), uv: new Float32Array(uv),
    index: n > 65535 ? new Uint32Array(idx) : new Uint16Array(idx),
  };
}

function weightsFor(rest: Float32Array, boneList: RestBone[], groupBones: Array<[BoneId, number]>, soft: number) {
  const n = rest.length / 2;
  const bi = new Int8Array(n * 4).fill(-1);
  const bw = new Float32Array(n * 4);
  const cands = groupBones
    .map(([id, f]) => ({ i: boneList.findIndex(b => b.id === id), f }))
    .filter(c => c.i >= 0);
  for (let v = 0; v < n; v++) {
    const p = { x: rest[v * 2], y: rest[v * 2 + 1] };
    const ws = cands.map(c => {
      const b = boneList[c.i];
      const d = distToSegment(p, b.a, b.b);
      return { i: c.i, w: c.f / Math.pow(d + soft, 4) };
    }).sort((a, b) => b.w - a.w).slice(0, 4);
    const sum = ws.reduce((s, x) => s + x.w, 0) || 1;
    ws.forEach((x, k) => { bi[v * 4 + k] = x.i; bw[v * 4 + k] = x.w / sum; });
  }
  return { bi, bw };
}

function median(a: number[]) { return a.length ? a.sort((p, q) => p - q)[a.length >> 1] : 0; }

/** Mannequin: one rounded capsule per bone, coloured like the clothes around it so gaps never show white. */
function mannequinLayers(parts: Map<GroupId, RGBAImage>, bones: RestBone[], H: number, opts: RigOptions, step: number): Layer[] {
  const out: Layer[] = [];
  const plan: Array<{ bone: BoneId; group: GroupId; src: GroupId; widthMul: number }> = [
    { bone: 'spine', group: 'torso', src: 'torso', widthMul: 0.8 },
    { bone: 'upperArmN', group: 'armN', src: 'armN', widthMul: 0.8 },
    { bone: 'forearmN', group: 'armN', src: 'armN', widthMul: 0.75 },
    { bone: 'upperArmF', group: 'armF', src: opts.cloneFarArm ? 'armN' : 'armF', widthMul: 0.8 },
    { bone: 'forearmF', group: 'armF', src: opts.cloneFarArm ? 'armN' : 'armF', widthMul: 0.75 },
    { bone: 'thighN', group: 'legN', src: 'legN', widthMul: 0.82 },
    { bone: 'shinN', group: 'legN', src: 'legN', widthMul: 0.78 },
    { bone: 'thighF', group: 'legF', src: opts.cloneFarLeg ? 'legN' : 'legF', widthMul: 0.82 },
    { bone: 'shinF', group: 'legF', src: opts.cloneFarLeg ? 'legN' : 'legF', widthMul: 0.78 },
  ];
  const dtCache = new Map<GroupId, Float32Array>();
  for (const p of plan) {
    const b = bones.find(x => x.id === p.bone);
    const part = parts.get(p.src);
    if (!b || !part) continue;
    // sample thickness and colour along the (source) bone
    const srcBone = p.src !== p.group ? bones.find(x => x.id === (Object.entries(FAR_OF).find(([, v]) => v === p.bone)?.[0] as BoneId)) ?? b : b;
    let dt = dtCache.get(p.src);
    const alphaMask = new Uint8Array(part.w * part.h);
    if (!dt) {
      for (let i = 0; i < alphaMask.length; i++) alphaMask[i] = part.data[i * 4 + 3] > 100 ? 1 : 0;
      dt = distanceInside(alphaMask, part.w, part.h);
      dtCache.set(p.src, dt);
    }
    const th: number[] = [], rs: number[] = [], gs: number[] = [], bs: number[] = [];
    for (let t = 0.15; t <= 0.85; t += 0.05) {
      const x = Math.round(srcBone.a.x + (srcBone.b.x - srcBone.a.x) * t), y = Math.round(srcBone.a.y + (srcBone.b.y - srcBone.a.y) * t);
      if (x < 0 || y < 0 || x >= part.w || y >= part.h) continue;
      const i = y * part.w + x;
      if (dt[i] > 0) th.push(dt[i]);
      if (part.data[i * 4 + 3] > 200) { rs.push(part.data[i * 4]); gs.push(part.data[i * 4 + 1]); bs.push(part.data[i * 4 + 2]); }
    }
    if (!rs.length) continue;
    const r = Math.min(H * 0.09, Math.max(H * 0.012, median(th) * p.widthMul));
    const col = `rgb(${median(rs) * 0.9 | 0},${median(gs) * 0.9 | 0},${median(bs) * 0.9 | 0})`;
    // draw the capsule at the rest position of this bone
    const minX = Math.floor(Math.min(b.a.x, b.b.x) - r - 2), minY = Math.floor(Math.min(b.a.y, b.b.y) - r - 2);
    const cw = Math.ceil(Math.abs(b.a.x - b.b.x) + 2 * r + 4), ch = Math.ceil(Math.abs(b.a.y - b.b.y) + 2 * r + 4);
    const c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    const ctx = c.getContext('2d')!;
    ctx.strokeStyle = col;
    ctx.lineCap = 'round';
    ctx.lineWidth = 2 * r;
    ctx.beginPath();
    ctx.moveTo(b.a.x - minX, b.a.y - minY);
    ctx.lineTo(b.b.x - minX, b.b.y - minY);
    ctx.stroke();
    const alpha = new Uint8Array(cw * ch);
    const id = ctx.getImageData(0, 0, cw, ch).data;
    for (let i = 0; i < cw * ch; i++) alpha[i] = id[i * 4 + 3] > 8 ? 1 : 0;
    const mesh = gridMesh(alpha, cw, ch, minX, minY, step);
    const bi = bones.findIndex(x => x.id === p.bone);
    const n = mesh.rest.length / 2;
    const bIdx = new Int8Array(n * 4).fill(-1), bW = new Float32Array(n * 4);
    for (let v = 0; v < n; v++) { bIdx[v * 4] = bi; bW[v * 4] = 1; }
    out.push({
      id: 'mannequin-' + p.bone, name: 'Ma-nơ-canh', group: 'mannequin', z: GROUP_Z[p.group] - 0.5,
      canvas: c, x0: minX, y0: minY, ...mesh, bones: bIdx, weights: bW,
      tint: p.group === 'armF' || p.group === 'legF' ? 0.72 : 1, visible: opts.mannequin,
    });
  }
  return out;
}

export function buildRig(
  parts: Map<GroupId, RGBAImage>, a: Assignment, details: Detail[], bones: RestBone[], H: number, opts: RigOptions,
): Rig {
  const step = Math.max(5, Math.min(14, Math.round(H / 45)));
  const soft = H * 0.02;
  const layers: Layer[] = [];
  const w = parts.get('torso')!.w, h = parts.get('torso')!.h;

  const addLayer = (id: string, name: string, group: GroupId, z: number, im: RGBAImage, boneSet: Array<[BoneId, number]>, tint = 1) => {
    const cr = crop(im, 2);
    if (!cr) return;
    const mesh = gridMesh(cr.alpha, cr.cw, cr.ch, cr.x0, cr.y0, step);
    const { bi, bw } = weightsFor(mesh.rest, bones, boneSet, soft);
    layers.push({ id, name, group, z, canvas: cr.canvas, x0: cr.x0, y0: cr.y0, ...mesh, bones: bi, weights: bw, tint, visible: true });
  };

  // split each part into "rest of the part" + its detail layers
  for (const g of GROUPS) {
    const part = parts.get(g.id)!;
    if ((g.id === 'armF' && opts.cloneFarArm) || (g.id === 'legF' && opts.cloneFarLeg)) continue;
    const z = g.id === 'weapon' && opts.weaponHand === 'F' ? 12 : GROUP_Z[g.id];
    const base = { w, h, data: new Uint8ClampedArray(part.data) };
    const gIdx = GROUP_INDEX[g.id];
    details.forEach((d, k) => {
      const dd = new Uint8ClampedArray(w * h * 4);
      let n = 0;
      for (let i = 0; i < w * h; i++) {
        if (a.detail[i] !== k || a.owner[i] !== gIdx) continue;
        for (let c = 0; c < 4; c++) dd[i * 4 + c] = part.data[i * 4 + c];
        base.data[i * 4 + 3] = 0;
        n++;
      }
      if (n < 20) return;
      const info = DETAILS.find(x => x.id === d.kind)!;
      const dz = info.behindBody ? info.z : z + info.z;
      addLayer(`${g.id}-${d.kind}-${k}`, `${g.label}: ${info.label}`, g.id, dz, { w, h, data: dd }, GROUP_BONES[g.id]);
    });
    addLayer(g.id, g.label, g.id, z, base, GROUP_BONES[g.id]);
  }

  // far limbs cloned from near limbs: same picture, moved by the far bones, a bit darker
  const cloneOf = (src: GroupId, dst: GroupId, label: string) => {
    const srcLayers = layers.filter(l => l.group === src);
    for (const l of srcLayers) {
      const boneSet: Array<[BoneId, number]> = GROUP_BONES[src].map(([id, f]) => [FAR_OF[id] ?? id, f]);
      const { bi, bw } = weightsFor(l.rest, bones, boneSet, soft);
      layers.push({ ...l, id: l.id + '-clone', name: label, group: dst, z: GROUP_Z[dst] + (l.z - GROUP_Z[src]), bones: bi, weights: bw, tint: 0.72 });
    }
  };
  if (opts.cloneFarArm) cloneOf('armN', 'armF', 'Tay sau (nhân bản)');
  if (opts.cloneFarLeg) cloneOf('legN', 'legF', 'Chân sau (nhân bản)');

  layers.push(...mannequinLayers(parts, bones, H, opts, step));
  layers.sort((p, q) => p.z - q.z);
  return { bones, layers, height: H };
}

/** CPU skinning: posed vertex positions for one layer. */
export function skin(layer: Layer, mats: Array<Mat | undefined>, out?: Float32Array): Float32Array {
  const n = layer.rest.length / 2;
  const res = out && out.length === n * 2 ? out : new Float32Array(n * 2);
  for (let v = 0; v < n; v++) {
    const x = layer.rest[v * 2], y = layer.rest[v * 2 + 1];
    let px = 0, py = 0, ws = 0;
    for (let k = 0; k < 4; k++) {
      const bi = layer.bones[v * 4 + k];
      if (bi < 0) continue;
      const w = layer.weights[v * 4 + k];
      const m = mats[bi];
      if (!m || w === 0) continue;
      const q = matApply(m, { x, y });
      px += q.x * w; py += q.y * w; ws += w;
    }
    if (ws > 0) { res[v * 2] = px / ws; res[v * 2 + 1] = py / ws; }
    else { res[v * 2] = x; res[v * 2 + 1] = y; }
  }
  return res;
}

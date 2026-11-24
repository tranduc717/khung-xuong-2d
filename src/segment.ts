// Splits the character into moving parts (groups) and optional detail layers.
import { distanceInside, pushPullFill, type Mask, type RGBAImage } from './image';
import { distToSegment, type BoneId, type Joints, type RestBone, type Vec } from './skeleton';
import type { Sam } from './sam';

export type GroupId = 'head' | 'torso' | 'armN' | 'armF' | 'legN' | 'legF' | 'weapon';

export const GROUPS: Array<{ id: GroupId; label: string; color: string }> = [
  { id: 'head', label: 'Đầu', color: '#f4b942' },
  { id: 'torso', label: 'Thân', color: '#4f8ff7' },
  { id: 'armN', label: 'Tay trước', color: '#e8505b' },
  { id: 'armF', label: 'Tay sau', color: '#a05ad8' },
  { id: 'legN', label: 'Chân trước', color: '#2bb673' },
  { id: 'legF', label: 'Chân sau', color: '#14868c' },
  { id: 'weapon', label: 'Vũ khí', color: '#ff8a1f' },
];
export const GROUP_INDEX: Record<GroupId, number> = Object.fromEntries(GROUPS.map((g, i) => [g.id, i + 1])) as Record<GroupId, number>;

/** Bones that move each group, with an influence factor. */
export const GROUP_BONES: Record<GroupId, Array<[BoneId, number]>> = {
  head: [['head', 1], ['spine', 0.25]],
  torso: [['spine', 1]],
  armN: [['upperArmN', 1], ['forearmN', 1], ['spine', 0.35]],
  armF: [['upperArmF', 1], ['forearmF', 1], ['spine', 0.35]],
  legN: [['thighN', 1], ['shinN', 1], ['footN', 1], ['spine', 0.3]],
  legF: [['thighF', 1], ['shinF', 1], ['footF', 1], ['spine', 0.3]],
  weapon: [['weapon', 1]],
};

// Detail layers: purely visual split inside a moving part (they move with their part).
export type DetailKind = 'backHair' | 'hair' | 'bangs' | 'face' | 'eyes' | 'mouth' | 'shirt' | 'accessory' | 'pants' | 'shoes';
export const DETAILS: Array<{ id: DetailKind; label: string; z: number; behindBody?: boolean }> = [
  { id: 'backHair', label: 'Tóc sau (dưới mặt, sau lưng)', z: 25, behindBody: true },
  { id: 'face', label: 'Mặt', z: 0.1 },
  { id: 'hair', label: 'Tóc', z: 0.2 },
  { id: 'eyes', label: 'Mắt + mi', z: 0.3 },
  { id: 'mouth', label: 'Miệng', z: 0.3 },
  { id: 'bangs', label: 'Tóc mái (trên mặt)', z: 0.45 },
  { id: 'shirt', label: 'Áo', z: 0.1 },
  { id: 'accessory', label: 'Phụ kiện', z: 0.3 },
  { id: 'pants', label: 'Quần', z: 0.1 },
  { id: 'shoes', label: 'Giày', z: 0.2 },
];

export const GROUP_Z: Record<GroupId, number> = { armF: 10, legF: 20, legN: 30, torso: 40, head: 50, armN: 60, weapon: 70 };

export interface Detail {
  kind: DetailKind;
  points: Array<[number, number, number]>;
  mask: Mask | null; // SAM result (before clipping to the character)
}

export interface SegInput {
  img: RGBAImage;            // original work image (with background) for the AI
  cut: RGBAImage;            // background removed
  sil: Mask;                 // silhouette
  joints: Joints;
  bones: RestBone[];         // real (not cloned) rest bones
  hasWeapon: boolean;
  height: number;
  facing: 1 | -1;
}

export interface SegState {
  sam: Map<GroupId, Mask | null>;   // AI masks per group
  extra: Map<GroupId, Array<[number, number, number]>>; // user correction clicks per group
  details: Detail[];
}

const sampleBone = (a: Vec, b: Vec, ts: number[]): Vec[] => ts.map(t => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }));

/** Radius around each group's bones that its pixels must lie within. */
export function groupRadius(g: GroupId, H: number, bones: RestBone[]): number {
  const b = (id: BoneId) => bones.find(x => x.id === id);
  switch (g) {
    case 'head': return Math.max(H * 0.06, (b('head')?.len ?? H * 0.12) * 0.85);
    case 'armN': return H * 0.075;
    case 'armF': return H * 0.06;
    case 'legN': case 'legF': return H * 0.1;
    case 'weapon': return H * 0.07;
    default: return H;
  }
}

/** The automatic prompts given to the AI for each part (positive along own bones, negative on others). */
export function autoPrompts(inp: SegInput, g: GroupId): Array<[number, number, number]> {
  const j = inp.joints;
  const pts: Array<[number, number, number]> = [];
  const pos = (v: Vec) => pts.push([v.x, v.y, 1]);
  const neg = (v: Vec) => pts.push([v.x, v.y, 0]);
  const inside = (v: Vec) => {
    const x = Math.round(v.x), y = Math.round(v.y);
    return x >= 0 && y >= 0 && x < inp.cut.w && y < inp.cut.h && inp.sil[y * inp.cut.w + x] === 1;
  };
  const along = (a: Vec, b: Vec, ts: number[], f: (v: Vec) => void) => sampleBone(a, b, ts).filter(inside).forEach(f);
  const H = inp.height;
  const torsoSide = (): Vec[] => {
    // points on the torso away from the near arm: front and back of the chest
    const y = (j.shoulder.y + j.hip.y) / 2;
    const out: Vec[] = [];
    for (const dx of [-0.09, 0.09]) {
      const v = { x: j.hip.x + (j.neck.x - j.hip.x) * 0.5 + dx * H, y };
      if (inside(v) && distToSegment(v, j.shoulder, j.elbowN) > H * 0.06 && distToSegment(v, j.elbowN, j.wristN) > H * 0.06) out.push(v);
    }
    return out;
  };
  switch (g) {
    case 'head': {
      const hl = Math.hypot(j.head.x - j.neck.x, j.head.y - j.neck.y);
      pos({ x: (j.head.x + j.neck.x) / 2, y: j.head.y + (j.neck.y - j.head.y) * 0.45 });
      // the face (on the side the character looks at) and the top of the hair, asked separately
      [{ x: j.head.x + (j.neck.x - j.head.x) * 0.62 + inp.facing * hl * 0.22, y: j.head.y + (j.neck.y - j.head.y) * 0.62 },
       { x: j.head.x + (j.neck.x - j.head.x) * 0.15, y: j.head.y + (j.neck.y - j.head.y) * 0.15 }].filter(inside).forEach(pos);
      neg({ x: (j.shoulder.x + j.hip.x) / 2, y: (j.shoulder.y + j.hip.y) / 2 });
      break;
    }
    case 'armN':
      along(j.shoulder, j.elbowN, [0.55], pos);
      along(j.elbowN, j.wristN, [0.35, 0.8], pos);
      torsoSide().forEach(neg);
      neg({ x: (j.hip.x + j.kneeN.x) / 2, y: (j.hip.y + j.kneeN.y) / 2 });
      break;
    case 'armF':
      // the far upper arm is usually hidden behind the body, so only the forearm/hand is asked for
      along(j.elbowF, j.wristF, [0.5, 0.9], pos);
      torsoSide().forEach(neg);
      along(j.shoulder, j.elbowN, [0.55], neg);
      along(j.elbowN, j.wristN, [0.5], neg);
      break;
    case 'legN':
      along(j.hip, j.kneeN, [0.45, 0.8], pos);
      along(j.kneeN, j.ankleN, [0.3, 0.75], pos);
      along(j.ankleN, j.toeN, [0.5], pos);
      along(j.kneeF, j.ankleF, [0.5], neg);
      break;
    case 'legF':
      along(j.hip, j.kneeF, [0.7], pos);
      along(j.kneeF, j.ankleF, [0.3, 0.75], pos);
      along(j.ankleF, j.toeF, [0.5], pos);
      along(j.kneeN, j.ankleN, [0.5], neg);
      along(j.hip, j.kneeN, [0.6], neg);
      break;
    case 'weapon': {
      const wb = inp.bones.find(b => b.id === 'weapon');
      if (wb) along(wb.a, wb.b, [0.3, 0.6, 0.92], pos);
      neg(j.elbowN);
      break;
    }
    case 'torso':
      break;
  }
  return pts;
}

/**
 * Asks the AI for one part. Limbs are asked bone by bone (upper arm, forearm...) and the answers are
 * joined, because one question for a whole bent arm often returns only half of it.
 */
export async function segmentGroup(inp: SegInput, sam: Sam, st: SegState, g: GroupId): Promise<Mask | null> {
  const all = autoPrompts(inp, g);
  const extra = st.extra.get(g) ?? [];
  const negs = all.filter(p => p[2] === 0);
  const poss = all.filter(p => p[2] === 1);
  const sets: Array<Array<[number, number, number]>> = [];
  if ((g.startsWith('arm') || g.startsWith('leg') || g === 'head') && poss.length > 1) {
    // one question per positive point group (each bone), plus the user's clicks
    for (const p of poss) sets.push([p, ...negs, ...extra]);
  } else sets.push([...all, ...extra]);
  if (!sets.some(s => s.some(p => p[2] === 1))) return null;
  let union: Mask | null = null;
  for (const s of sets) {
    if (!s.some(p => p[2] === 1)) continue;
    const m = await sam.segment(s);
    if (!union) union = m;
    else for (let i = 0; i < m.length; i++) if (m[i]) union[i] = 1;
  }
  // the user's negative clicks always win
  if (union && extra.some(p => p[2] === 0)) {
    const neg = await sam.segment(extra.filter(p => p[2] === 0).map(p => [p[0], p[1], 1] as [number, number, number]));
    for (let i = 0; i < neg.length; i++) if (neg[i]) union[i] = 0;
  }
  return union;
}

export async function runAutoSegmentation(inp: SegInput, sam: Sam | null, st: SegState, onProgress: (s: string) => void) {
  const groups: GroupId[] = ['head', 'armN', 'armF', 'legN', 'legF'];
  if (inp.hasWeapon) groups.push('weapon');
  for (const g of groups) {
    if (!sam) { st.sam.set(g, null); continue; }
    onProgress('AI đang tách: ' + GROUPS.find(x => x.id === g)!.label);
    st.sam.set(g, await segmentGroup(inp, sam, st, g));
  }
}

export interface Assignment {
  owner: Uint8Array;   // per pixel: 0 = empty, else GROUP_INDEX
  detail: Int16Array;  // per pixel: -1 = none, else index into details
}

/** Decides for every pixel which part it belongs to. */
export function assign(inp: SegInput, st: SegState): Assignment {
  const { w, h } = inp.cut;
  const H = inp.height;
  const owner = new Uint8Array(w * h);
  const bones = inp.bones;
  const bone = (id: BoneId) => bones.find(b => b.id === id);
  const order: GroupId[] = ['weapon', 'armN', 'head', 'legN', 'armF', 'legF'];
  const radius = new Map(order.map(g => [g, groupRadius(g, H, bones)]));
  // forearms reach past the wrist (the hand), shins past the ankle a little
  const extend = (b: RestBone): RestBone => {
    const k = b.id.startsWith('forearm') ? 0.45 : 0;
    if (!k) return b;
    return { ...b, b: { x: b.b.x + (b.b.x - b.a.x) * k, y: b.b.y + (b.b.y - b.a.y) * k } };
  };
  const ownBones = (g: GroupId) => (GROUP_BONES[g].filter(([, f]) => f === 1).map(([id]) => bone(id)).filter(Boolean) as RestBone[]).map(extend);
  const gb = new Map(order.map(g => [g, ownBones(g)]));
  const allBones = bones.map(extend);
  const boneGroup = (id: BoneId): GroupId => {
    for (const g of order) if (GROUP_BONES[g].some(([b, f]) => b === id && f === 1)) return g;
    return 'torso';
  };
  const sil = inp.sil;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!sil[i]) continue;
    const p = { x, y };
    let chosen: GroupId | null = null;
    for (const g of order) {
      const m = st.sam.get(g);
      if (!m || !m[i]) continue;
      const bs = gb.get(g)!;
      if (!bs.length) continue;
      let d = 1e9;
      for (const b of bs) d = Math.min(d, distToSegment(p, b.a, b.b));
      if (d <= radius.get(g)!) { chosen = g; break; }
    }
    if (!chosen) {
      // nearest bone decides, with arms/weapon only very close to their bone (they often lie over the body)
      let best: RestBone | null = null, bd = 1e9;
      for (const b of allBones) {
        const d = distToSegment(p, b.a, b.b);
        if (d < bd) { bd = d; best = b; }
      }
      const g = best ? boneGroup(best.id) : 'torso';
      if (g === 'armN' || g === 'armF' || g === 'weapon') {
        // beyond the wrist (the hand) the pixel belongs to the arm; over the body only when very close
        const beyond = best!.id.startsWith('forearm') && distToSegment(p, best!.a, best!.b) < radius.get(g)! &&
          ((p.x - best!.a.x) * (best!.b.x - best!.a.x) + (p.y - best!.a.y) * (best!.b.y - best!.a.y)) > 0.75 * ((best!.b.x - best!.a.x) ** 2 + (best!.b.y - best!.a.y) ** 2);
        chosen = beyond || (bd < radius.get(g)! * 0.45 && !st.sam.get(g)) ? g : 'torso';
      }
      else if (g === 'head') chosen = 'head';
      else chosen = g;
    }
    owner[i] = GROUP_INDEX[chosen];
  }
  cleanupIslands(owner, w, h);

  const detail = new Int16Array(w * h).fill(-1);
  st.details.forEach((d, k) => {
    if (!d.mask) return;
    for (let i = 0; i < w * h; i++) if (d.mask[i] && owner[i]) detail[i] = k;
  });
  return { owner, detail };
}

/** Tiny islands of one part inside another are given to the surrounding part. */
function cleanupIslands(owner: Uint8Array, w: number, h: number) {
  const seen = new Uint8Array(w * h);
  const stack: number[] = [];
  const minSize = w * h * 0.0004;
  for (let s = 0; s < w * h; s++) {
    if (!owner[s] || seen[s]) continue;
    const g = owner[s];
    const comp: number[] = [];
    const border = new Map<number, number>();
    seen[s] = 1; stack.push(s);
    while (stack.length) {
      const i = stack.pop()!;
      comp.push(i);
      const x = i % w, y = (i / w) | 0;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
      for (const j of nb) {
        if (j < 0) continue;
        if (owner[j] === g) { if (!seen[j]) { seen[j] = 1; stack.push(j); } }
        else if (owner[j]) border.set(owner[j], (border.get(owner[j]) ?? 0) + 1);
      }
    }
    if (comp.length < minSize && border.size) {
      let best = 0, bc = -1;
      for (const [k, c] of border) if (c > bc) { bc = c; best = k; }
      for (const i of comp) owner[i] = best;
    }
  }
}

/**
 * Builds the picture of each part. The body gets the area hidden behind the front arm painted in,
 * so nothing white shows when the arm swings.
 */
export function buildPartImages(inp: SegInput, a: Assignment): Map<GroupId, RGBAImage> {
  const { w, h } = inp.cut;
  const out = new Map<GroupId, RGBAImage>();
  for (const g of GROUPS) {
    const idx = GROUP_INDEX[g.id];
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      if (a.owner[i] !== idx) continue;
      data[i * 4] = inp.cut.data[i * 4];
      data[i * 4 + 1] = inp.cut.data[i * 4 + 1];
      data[i * 4 + 2] = inp.cut.data[i * 4 + 2];
      data[i * 4 + 3] = inp.cut.data[i * 4 + 3];
    }
    out.set(g.id, { w, h, data });
  }
  // fill torso behind the near arm (and behind the head's neck area)
  const torso = out.get('torso')!;
  const tIdx = GROUP_INDEX.torso, aIdx = GROUP_INDEX.armN, wIdx = GROUP_INDEX.weapon, fIdx = GROUP_INDEX.armF;
  const hole = new Uint8Array(w * h);
  const known = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let minX = -1, maxX = -1;
    for (let x = 0; x < w; x++) if (a.owner[y * w + x] === tIdx) { if (minX < 0) minX = x; maxX = x; }
    if (minX < 0) continue;
    for (let x = minX; x <= maxX; x++) {
      const i = y * w + x;
      const o = a.owner[i];
      if (o === tIdx) known[i] = 1;
      else if (o === aIdx || o === wIdx || o === fIdx) hole[i] = 1;
    }
  }
  // only fill reasonably thin areas (an arm's width), never huge regions
  const dt = distanceInside(hole, w, h);
  const maxR = inp.height * 0.08;
  for (let i = 0; i < w * h; i++) if (hole[i] && dt[i] > maxR) hole[i] = 0;
  pushPullFill(torso, known, hole);
  for (let i = 0; i < w * h; i++) if (hole[i]) torso.data[i * 4 + 3] = 255;
  return out;
}

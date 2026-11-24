// Side-view skeleton: joints, bones, automatic first guess, forward kinematics.
import type { Mask } from './image';

export type Vec = { x: number; y: number };

export type JointId =
  | 'head' | 'neck' | 'shoulder' | 'hip'
  | 'elbowN' | 'wristN' | 'elbowF' | 'wristF'
  | 'kneeN' | 'ankleN' | 'toeN' | 'kneeF' | 'ankleF' | 'toeF'
  | 'weaponTip';

export const JOINT_LABEL: Record<JointId, string> = {
  head: 'Đỉnh đầu', neck: 'Cổ', shoulder: 'Vai', hip: 'Hông',
  elbowN: 'Khuỷu tay trước', wristN: 'Cổ tay trước', elbowF: 'Khuỷu tay sau', wristF: 'Cổ tay sau',
  kneeN: 'Gối trước', ankleN: 'Cổ chân trước', toeN: 'Mũi chân trước',
  kneeF: 'Gối sau', ankleF: 'Cổ chân sau', toeF: 'Mũi chân sau',
  weaponTip: 'Đầu vũ khí',
};

export type BoneId =
  | 'root' | 'spine' | 'head'
  | 'upperArmN' | 'forearmN' | 'upperArmF' | 'forearmF'
  | 'thighN' | 'shinN' | 'footN' | 'thighF' | 'shinF' | 'footF'
  | 'weapon';

export interface BoneDef { id: BoneId; from: JointId; to: JointId; parent: BoneId | null }

export function boneDefs(weaponHand: 'N' | 'F'): BoneDef[] {
  return [
    { id: 'spine', from: 'hip', to: 'neck', parent: null },
    { id: 'head', from: 'neck', to: 'head', parent: 'spine' },
    { id: 'upperArmN', from: 'shoulder', to: 'elbowN', parent: 'spine' },
    { id: 'forearmN', from: 'elbowN', to: 'wristN', parent: 'upperArmN' },
    { id: 'upperArmF', from: 'shoulder', to: 'elbowF', parent: 'spine' },
    { id: 'forearmF', from: 'elbowF', to: 'wristF', parent: 'upperArmF' },
    { id: 'thighN', from: 'hip', to: 'kneeN', parent: null },
    { id: 'shinN', from: 'kneeN', to: 'ankleN', parent: 'thighN' },
    { id: 'footN', from: 'ankleN', to: 'toeN', parent: 'shinN' },
    { id: 'thighF', from: 'hip', to: 'kneeF', parent: null },
    { id: 'shinF', from: 'kneeF', to: 'ankleF', parent: 'thighF' },
    { id: 'footF', from: 'ankleF', to: 'toeF', parent: 'shinF' },
    { id: 'weapon', from: weaponHand === 'N' ? 'wristN' : 'wristF', to: 'weaponTip', parent: weaponHand === 'N' ? 'forearmN' : 'forearmF' },
  ];
}

export type BodyType = 'adult' | 'child' | 'chibi';
export const BODY_LABEL: Record<BodyType, string> = { adult: 'Người lớn', child: 'Trẻ em', chibi: 'Chibi (đầu to)' };

// Fractions of the character height measured from the top of the head.
const PROPS: Record<BodyType, { neck: number; shoulder: number; hip: number; knee: number; ankle: number; upper: number; fore: number }> = {
  adult: { neck: 0.14, shoulder: 0.19, hip: 0.5, knee: 0.73, ankle: 0.94, upper: 0.16, fore: 0.14 },
  child: { neck: 0.21, shoulder: 0.26, hip: 0.57, knee: 0.77, ankle: 0.94, upper: 0.14, fore: 0.12 },
  chibi: { neck: 0.4, shoulder: 0.44, hip: 0.7, knee: 0.84, ankle: 0.94, upper: 0.11, fore: 0.09 },
};

export type Joints = Record<JointId, Vec>;

export interface Silhouette {
  mask: Mask;
  w: number;
  h: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

function rowRuns(s: Silhouette, y: number, minLen: number): Array<[number, number]> {
  const runs: Array<[number, number]> = [];
  y = Math.max(0, Math.min(s.h - 1, Math.round(y)));
  let start = -1, gap = 0;
  for (let x = s.bbox.x0; x <= s.bbox.x1 + 1; x++) {
    const on = x <= s.bbox.x1 && s.mask[y * s.w + x];
    if (on) { if (start < 0) start = x; gap = 0; }
    else if (start >= 0) {
      gap++;
      if (gap > 3 || x > s.bbox.x1) { runs.push([start, x - gap]); start = -1; gap = 0; }
    }
  }
  return runs.filter(r => r[1] - r[0] + 1 >= minLen);
}

/** Guess which way the character faces: feet stick out forward from the ankles. */
export function guessFacing(s: Silhouette): 1 | -1 {
  const H = s.bbox.y1 - s.bbox.y0;
  const ankleRuns = rowRuns(s, s.bbox.y1 - H * 0.07, H * 0.01);
  const soleRuns = rowRuns(s, s.bbox.y1 - H * 0.012, H * 0.01);
  let score = 0;
  for (const sr of soleRuns) {
    const sc = (sr[0] + sr[1]) / 2;
    // nearest ankle run
    let best: [number, number] | null = null, bd = 1e9;
    for (const ar of ankleRuns) { const d = Math.abs((ar[0] + ar[1]) / 2 - sc); if (d < bd) { bd = d; best = ar; } }
    if (best) score += sc - (best[0] + best[1]) / 2;
  }
  return score < 0 ? -1 : 1;
}

/** A first guess for every joint from the silhouette and body proportions. The user then drags them. */
export function guessJoints(s: Silhouette, body: BodyType, facing: 1 | -1): Joints {
  const { x0, y0, x1, y1 } = s.bbox;
  const H = y1 - y0;
  const P = PROPS[body];
  const Y = (f: number) => y0 + H * f;
  const minRun = H * 0.012;

  // body axis: centre of the run closest to the overall centre, between shoulder and hip
  const centreOf = (y: number, prefer: number) => {
    const runs = rowRuns(s, y, minRun);
    if (!runs.length) return prefer;
    let best = runs[0], bd = 1e9;
    for (const r of runs) {
      const c = (r[0] + r[1]) / 2;
      const d = Math.abs(c - prefer) - (r[1] - r[0]) * 0.25; // prefer wide runs (torso, not weapon)
      if (d < bd) { bd = d; best = r; }
    }
    return (best[0] + best[1]) / 2;
  };
  const mid = (x0 + x1) / 2;
  const chestX = centreOf(Y((P.shoulder + P.hip) / 2), mid);
  const hipX = centreOf(Y(P.hip), chestX);
  const neckX = centreOf(Y(P.neck), chestX);
  const headX = centreOf(Y(P.neck * 0.45), neckX);

  // legs: look for two runs at knee and ankle height
  const legX = (y: number): [number, number] => {
    const runs = rowRuns(s, y, minRun).filter(r => Math.abs((r[0] + r[1]) / 2 - hipX) < H * 0.35);
    if (runs.length >= 2) {
      runs.sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]));
      const a = (runs[0][0] + runs[0][1]) / 2, b = (runs[1][0] + runs[1][1]) / 2;
      // near leg = the one further forward (arbitrary but consistent)
      return facing > 0 ? [Math.max(a, b), Math.min(a, b)] : [Math.min(a, b), Math.max(a, b)];
    }
    if (runs.length === 1) {
      const c = (runs[0][0] + runs[0][1]) / 2;
      return [c, c - facing * H * 0.01];
    }
    return [hipX, hipX];
  };
  const [kN, kF] = legX(Y(P.knee));
  const [aN, aF] = legX(Y(P.ankle));
  const footLen = H * (body === 'chibi' ? 0.07 : 0.08);

  const shoulder = { x: neckX - facing * H * 0.01, y: Y(P.shoulder) };
  const elbowN = { x: shoulder.x + facing * H * 0.01, y: shoulder.y + H * P.upper };
  const wristN = { x: elbowN.x + facing * H * 0.035, y: elbowN.y + H * P.fore * 0.95 };
  const elbowF = { x: shoulder.x - facing * H * 0.02, y: shoulder.y + H * P.upper * 0.97 };
  const wristF = { x: elbowF.x + facing * H * 0.01, y: elbowF.y + H * P.fore * 0.95 };

  return {
    head: { x: headX, y: y0 + H * 0.01 },
    neck: { x: neckX, y: Y(P.neck) },
    shoulder,
    hip: { x: hipX, y: Y(P.hip) },
    elbowN, wristN, elbowF, wristF,
    kneeN: { x: kN, y: Y(P.knee) },
    ankleN: { x: aN, y: Y(P.ankle) },
    toeN: { x: aN + facing * footLen, y: y1 - H * 0.01 },
    kneeF: { x: kF, y: Y(P.knee) },
    ankleF: { x: aF, y: Y(P.ankle) },
    toeF: { x: aF + facing * footLen, y: y1 - H * 0.01 },
    weaponTip: { x: wristN.x + facing * H * 0.18, y: wristN.y + H * 0.12 },
  };
}

// ---------- forward kinematics ----------

export type Mat = [number, number, number, number, number, number]; // a b c d e f  (x' = a x + c y + e, y' = b x + d y + f)

export function matMul(m: Mat, n: Mat): Mat {
  return [
    m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1],
    m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
    m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5],
  ];
}
export function matApply(m: Mat, p: Vec): Vec { return { x: m[0] * p.x + m[2] * p.y + m[4], y: m[1] * p.x + m[3] * p.y + m[5] }; }
const rotAbout = (ang: number, c: Vec, t: Vec): Mat => {
  // translate c to origin, rotate, translate to t
  const co = Math.cos(ang), si = Math.sin(ang);
  return [co, si, -si, co, t.x - (co * c.x - si * c.y), t.y - (si * c.x + co * c.y)];
};

export interface RestBone extends BoneDef { a: Vec; b: Vec; len: number; angle: number }

export function restBones(j: Joints, weaponHand: 'N' | 'F', hasWeapon: boolean, mirrorFar: { arm: boolean; leg: boolean }): RestBone[] {
  // When far limbs are cloned from near limbs, their rest shape is the near limb's shape.
  const src = (id: JointId): Vec => {
    if (mirrorFar.arm) {
      if (id === 'elbowF') return j.elbowN;
      if (id === 'wristF') return j.wristN;
    }
    if (mirrorFar.leg) {
      if (id === 'kneeF') return j.kneeN;
      if (id === 'ankleF') return j.ankleN;
      if (id === 'toeF') return j.toeN;
    }
    return j[id];
  };
  return boneDefs(weaponHand)
    .filter(b => hasWeapon || b.id !== 'weapon')
    .map(b => {
      let a = src(b.from), bb = src(b.to);
      if (b.id === 'weapon') { a = j[b.from]; bb = j.weaponTip; }
      const dx = bb.x - a.x, dy = bb.y - a.y;
      return { ...b, a, b: bb, len: Math.hypot(dx, dy), angle: Math.atan2(dy, dx) };
    });
}

/**
 * A pose = for every bone, its world angle (radians, image space, atan2 convention), plus a root offset.
 * Returns skinning matrices (rest -> posed) for each bone.
 */
export function solvePose(bones: RestBone[], angles: Partial<Record<BoneId, number>>, rootOffset: Vec): Map<BoneId, Mat> {
  const out = new Map<BoneId, Mat>();
  const byId = new Map(bones.map(b => [b.id, b]));
  const order: RestBone[] = [];
  const visit = (b: RestBone) => {
    if (order.includes(b)) return;
    if (b.parent && byId.has(b.parent)) visit(byId.get(b.parent)!);
    order.push(b);
  };
  bones.forEach(visit);
  const delta = new Map<BoneId, number>();
  for (const b of order) {
    const parentM: Mat = b.parent && out.has(b.parent) ? out.get(b.parent)! : [1, 0, 0, 1, rootOffset.x, rootOffset.y];
    const start = matApply(parentM, b.a);
    // bones without a target angle simply follow their parent's rotation
    const inherited = b.parent ? delta.get(b.parent) ?? 0 : 0;
    const ang = angles[b.id] ?? b.angle + inherited;
    delta.set(b.id, ang - b.angle);
    out.set(b.id, rotAbout(ang - b.angle, b.a, start));
  }
  return out;
}

export function distToSegment(p: Vec, a: Vec, b: Vec): number {
  const dx = b.x - a.x, dy = b.y - a.y;
  const l2 = dx * dx + dy * dy || 1;
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

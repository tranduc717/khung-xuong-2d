// Actions (walk, run, idle) defined in "character space" and turned into bone angles.
import type { BoneId, RestBone, Vec } from './skeleton';

export type ActionId = 'walk' | 'run' | 'idle';
export const ACTIONS: Array<{ id: ActionId; label: string; seconds: number }> = [
  { id: 'walk', label: 'Đi bộ', seconds: 1.0 },
  { id: 'run', label: 'Chạy', seconds: 0.62 },
  { id: 'idle', label: 'Đứng thở', seconds: 2.4 },
];

const D = Math.PI / 180;

/** Periodic Catmull-Rom interpolation of keyframes [t, value] with t in [0,1). */
function track(keys: Array<[number, number]>) {
  return (p: number) => {
    p = ((p % 1) + 1) % 1;
    const n = keys.length;
    let i = n - 1;
    for (let k = 0; k < n; k++) if (keys[k][0] <= p) i = k;
    const k0 = keys[(i - 1 + n) % n], k1 = keys[i], k2 = keys[(i + 1) % n], k3 = keys[(i + 2) % n];
    const t1 = k1[0], t2 = i + 1 < n ? k2[0] : k2[0] + 1;
    const t = (p - t1) / (t2 - t1 || 1);
    const v0 = k0[1], v1 = k1[1], v2 = k2[1], v3 = k3[1];
    const t2_ = t * t, t3 = t2_ * t;
    return 0.5 * (2 * v1 + (-v0 + v2) * t + (2 * v0 - 5 * v1 + 4 * v2 - v3) * t2_ + (-v0 + 3 * v1 - 3 * v2 + v3) * t3);
  };
}

interface Gait {
  thigh: (p: number) => number; // forward = +, degrees from straight down
  knee: (p: number) => number;  // bend, + = shin goes back
  foot: (p: number) => number;  // foot angle from down; 90 = flat forward
  arm: (p: number) => number;   // upper arm forward +
  elbow: (p: number) => number; // forearm bend forward +
  bob: (p: number) => number;   // root down (+) in character heights
  lean: (p: number) => number;  // spine forward lean, degrees
  head: (p: number) => number;
}

const WALK: Gait = {
  thigh: track([[0, 24], [0.12, 20], [0.25, 8], [0.4, -8], [0.5, -16], [0.6, -8], [0.75, 14], [0.88, 26]]),
  knee: track([[0, 3], [0.1, 14], [0.25, 6], [0.42, 6], [0.55, 32], [0.66, 58], [0.78, 34], [0.9, 6]]),
  foot: track([[0, 102], [0.1, 90], [0.35, 90], [0.5, 72], [0.6, 58], [0.72, 82], [0.88, 96]]),
  arm: p => -18 * Math.cos(2 * Math.PI * p),
  elbow: p => 10 + 16 * (1 - Math.cos(2 * Math.PI * p - Math.PI)) / 2,
  bob: p => 0.011 * Math.cos(4 * Math.PI * p),
  lean: p => 3 + 1 * Math.sin(4 * Math.PI * p),
  head: p => -2 - 1 * Math.sin(4 * Math.PI * p),
};

const RUN: Gait = {
  thigh: track([[0, 30], [0.1, 18], [0.25, -5], [0.38, -28], [0.5, -20], [0.62, 10], [0.75, 42], [0.88, 40]]),
  knee: track([[0, 15], [0.1, 35], [0.25, 18], [0.38, 30], [0.5, 80], [0.62, 105], [0.75, 75], [0.88, 30]]),
  foot: track([[0, 100], [0.12, 92], [0.3, 75], [0.42, 55], [0.6, 60], [0.78, 85], [0.9, 98]]),
  arm: p => -32 * Math.cos(2 * Math.PI * p),
  elbow: p => 70 + 15 * (1 - Math.cos(2 * Math.PI * p - Math.PI)) / 2,
  bob: p => 0.025 * Math.cos(4 * Math.PI * p - 0.6),
  lean: p => 10 + 1.5 * Math.sin(4 * Math.PI * p),
  head: p => -7,
};

export interface PoseOptions {
  facing: 1 | -1;
  height: number;
  keepArmN: boolean; // arm holding a weapon keeps its drawn pose (only sways a little)
  keepArmF: boolean;
}

export interface PoseResult { angles: Partial<Record<BoneId, number>>; root: Vec }

export function poseAt(action: ActionId, p: number, bones: RestBone[], o: PoseOptions): PoseResult {
  const rest = new Map(bones.map(b => [b.id, b.angle]));
  const f = o.facing;
  // limb pointing down with forward angle a (deg) -> image angle
  const down = (a: number) => Math.atan2(Math.cos(a * D), f * Math.sin(a * D));
  const upDelta = (id: BoneId, a: number) => (rest.get(id) ?? -Math.PI / 2) + f * a * D;
  const limbDelta = (id: BoneId, a: number) => (rest.get(id) ?? Math.PI / 2) - f * a * D;
  const angles: Partial<Record<BoneId, number>> = {};

  if (action === 'idle') {
    const s = Math.sin(2 * Math.PI * p);
    angles.spine = upDelta('spine', 1.2 * s);
    angles.head = upDelta('head', -0.8 * s);
    angles.upperArmN = limbDelta('upperArmN', 2 * s);
    angles.forearmN = limbDelta('forearmN', 3 * s);
    angles.upperArmF = limbDelta('upperArmF', 2 * s);
    angles.forearmF = limbDelta('forearmF', 3 * s);
    // legs stay where they are drawn
    for (const id of ['thighN', 'shinN', 'footN', 'thighF', 'shinF', 'footF'] as BoneId[]) angles[id] = rest.get(id);
    return { angles, root: { x: 0, y: o.height * 0.004 * (1 - Math.cos(2 * Math.PI * p)) } };
  }

  const g = action === 'run' ? RUN : WALK;
  const leg = (side: 'N' | 'F', ph: number) => {
    const th = g.thigh(ph);
    const sh = th - g.knee(ph);
    angles[('thigh' + side) as BoneId] = down(th);
    angles[('shin' + side) as BoneId] = down(sh);
    angles[('foot' + side) as BoneId] = down(g.foot(ph));
  };
  leg('N', p);
  leg('F', p + 0.5);
  const arm = (side: 'N' | 'F', ph: number, keep: boolean) => {
    const ua = ('upperArm' + side) as BoneId, fa = ('forearm' + side) as BoneId;
    if (keep) {
      const s = g.arm(ph) * 0.15;
      angles[ua] = limbDelta(ua, s);
      angles[fa] = limbDelta(fa, s);
      return;
    }
    const u = g.arm(ph);
    angles[ua] = down(u);
    angles[fa] = down(u + g.elbow(ph));
  };
  arm('N', p + 0.5, o.keepArmN);
  arm('F', p, o.keepArmF);
  angles.spine = upDelta('spine', g.lean(p));
  // head follows the spine's rotation plus its own small offset
  const spineChange = (angles.spine ?? 0) - (rest.get('spine') ?? 0);
  angles.head = (rest.get('head') ?? -Math.PI / 2) + spineChange + f * g.head(p) * D;
  return { angles, root: { x: 0, y: o.height * g.bob(p) } };
}

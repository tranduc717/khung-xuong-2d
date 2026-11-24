import './style.css';
import {
  loadImageFile, toWorkImage, removePlainBackground, applyMatte, borderColor, borderSpread, maskBBox,
  keepLargestComponents, imageToCanvas, type Mask, type RGBAImage,
} from './image';
import {
  guessFacing, guessJoints, guessBodyType, restBones, JOINT_LABEL, BODY_LABEL, solvePose,
  type BodyType, type JointId, type Joints, type Vec, type RestBone, type Mat, type BoneId,
} from './skeleton';
import { Sam, type SamStatus } from './sam';
import {
  GROUPS, DETAILS, runAutoSegmentation, segmentGroup, assign, buildPartImages,
  type GroupId, type SegState, type SegInput, type Assignment, type DetailKind,
} from './segment';
import { buildRig, type Rig } from './rig';
import { Renderer } from './renderer';
import { ACTIONS, poseAt, type ActionId, type PoseOptions } from './anim';
import { animationBox, exportSpritesheet, download } from './export';
import { mountMannequin3D } from './mannequin3d';

// ---------------------------------------------------------------- state
type Step = 1 | 2 | 3 | 4 | 5 | 6;
const S = {
  step: 1 as Step,
  name: 'nhan-vat',
  img: null as RGBAImage | null,     // work image with background
  cut: null as RGBAImage | null,     // background removed
  sil: null as Mask | null,
  bbox: null as { x0: number; y0: number; x1: number; y1: number } | null,
  tolerance: 28,
  body: 'adult' as BodyType,
  facing: 1 as 1 | -1,
  joints: null as Joints | null,
  hasWeapon: false,
  weaponHand: 'N' as 'N' | 'F',
  seg: { sam: new Map(), extra: new Map(), details: [] } as SegState,
  assignment: null as Assignment | null,
  segDirty: true,
  editGroup: 'armN' as GroupId,
  editDetail: -1,
  cloneFarArm: true,
  cloneFarLeg: true,
  mannequin: true,
  showBones: false,
  keepArm: true,
  rig: null as Rig | null,
  action: 'walk' as ActionId,
  speed: 1,
  playing: true,
  overlay: 0.55,
};

let sam: Sam | null = null;
try { sam = new Sam(); } catch (e) { console.warn(e); sam = null; }

// ---------------------------------------------------------------- DOM
const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector(sel) as T;
document.querySelector('#app')!.innerHTML = `
<header>
  <div class="brand"><span class="logo">🦴</span> Khung Xương 2D <span class="ver">bản thử 0.1</span></div>
  <div id="ai" class="pill">AI: đang chuẩn bị…</div>
</header>
<nav id="steps"></nav>
<main>
  <section id="stage">
    <canvas id="edit"></canvas>
    <canvas id="gl"></canvas>
    <div id="mann3d"></div>
    <div id="busy" hidden><div class="spin"></div><span id="busyText">Đang xử lý…</span></div>
    <div id="drop" class="drop">
      <div class="dropin">
        <div class="big">Kéo thả ảnh nhân vật vào đây</div>
        <div>hoặc <label class="btn primary">Chọn ảnh<input id="file" type="file" accept="image/*" hidden></label></div>
        <div class="hint">Ảnh nhân vật nhìn nghiêng (side view), đứng hoặc đang bước, nền trơn là tốt nhất.</div>
        <div class="samples" id="samples"></div>
      </div>
    </div>
  </section>
  <aside id="panel"></aside>
</main>`;

const edit = $<HTMLCanvasElement>('#edit');
const glc = $<HTMLCanvasElement>('#gl');
const stage = $('#stage');
const panel = $('#panel');
let renderer: Renderer | null = null;
try { renderer = new Renderer(glc); } catch (e) { console.warn(e); }

const STEP_NAMES: Record<Step, string> = { 1: 'Ảnh', 2: 'Khớp xương', 3: 'Tách lớp', 4: 'Chuyển động', 5: 'Xuất file', 6: 'Ma-nơ-canh 3D' };

function renderSteps() {
  const nav = $('#steps');
  nav.innerHTML = '';
  (Object.keys(STEP_NAMES).map(Number) as Step[]).forEach(n => {
    const b = document.createElement('button');
    b.className = 'step' + (S.step === n ? ' on' : '') + (n === 6 ? ' extra' : '');
    b.innerHTML = n === 6 ? `<b>3D</b> ${STEP_NAMES[n]}` : `<b>${n}</b> ${STEP_NAMES[n]}`;
    b.disabled = n !== 6 && n > 1 && !S.cut;
    b.onclick = () => goto(n);
    nav.appendChild(b);
  });
}

function busy(text: string | null) {
  $('#busy').hidden = text === null;
  if (text) $('#busyText').textContent = text;
}

function setAiPill(s: SamStatus) {
  const el = $('#ai');
  const mb = (n: number) => (n / 1048576).toFixed(1);
  el.className = 'pill';
  switch (s.state) {
    case 'loading': el.textContent = `AI: tải mô hình ${s.total ? Math.round(100 * s.got / s.total) + '%' : mb(s.got) + ' MB'}`; break;
    case 'ready': el.textContent = `AI sẵn sàng (${s.device === 'gpu' ? 'WebGPU' : 'CPU'})`; el.classList.add('ok'); break;
    case 'encoding': el.textContent = 'AI: đang phân tích ảnh…'; break;
    case 'encoded': el.textContent = `AI đã phân tích ảnh (${(s.ms / 1000).toFixed(1)} giây, ${s.device === 'gpu' ? 'WebGPU' : 'CPU'})`; el.classList.add('ok'); break;
    case 'error': el.textContent = 'AI lỗi: ' + s.message + ' — vẫn dùng được chế độ không AI'; el.classList.add('err'); break;
    default: el.textContent = 'AI: đang chuẩn bị…';
  }
}
if (sam) sam.onStatus = setAiPill; else setAiPill({ state: 'error', message: 'không khởi động được' });

// ---------------------------------------------------------------- samples & loading
const SAMPLES = [
  ['hiep-si.jpg', 'Hiệp sĩ'], ['ao-dai.jpg', 'Áo dài'], ['cyberpunk.jpg', 'Cyberpunk'], ['nong-dan.jpg', 'Nông dân'],
];
const samplesEl = $('#samples');
samplesEl.innerHTML = '<div class="hint">Hoặc thử ảnh mẫu:</div>';
for (const [f, label] of SAMPLES) {
  const b = document.createElement('button');
  b.className = 'sample';
  b.innerHTML = `<img src="${import.meta.env.BASE_URL}samples/${f}" alt=""><span>${label}</span>`;
  b.onclick = () => loadFrom(import.meta.env.BASE_URL + 'samples/' + f, f.replace(/\.\w+$/, ''));
  samplesEl.appendChild(b);
}
$<HTMLInputElement>('#file').onchange = e => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) loadFrom(f, f.name.replace(/\.\w+$/, ''));
};
stage.addEventListener('dragover', e => { e.preventDefault(); stage.classList.add('dragging'); });
stage.addEventListener('dragleave', () => stage.classList.remove('dragging'));
stage.addEventListener('drop', e => {
  e.preventDefault();
  stage.classList.remove('dragging');
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith('image/')) loadFrom(f, f.name.replace(/\.\w+$/, ''));
});

async function loadFrom(src: File | string, name: string) {
  busy('Đang đọc ảnh…');
  try {
    const el = await loadImageFile(src);
    S.name = name.replace(/[^\w\-]+/g, '-').slice(0, 40) || 'nhan-vat';
    S.img = toWorkImage(el);
    S.seg = { sam: new Map(), extra: new Map(), details: [] };
    S.rig = null;
    S.segDirty = true;
    cutBackground();
    if (sam) sam.encode(S.img).catch(err => console.warn(err));
    $('#drop').hidden = true;
    S.joints = null;
    goto(1);
  } catch (e) {
    alert('Không đọc được ảnh: ' + (e as Error).message);
  } finally { busy(null); }
}

function cutBackground() {
  if (!S.img) return;
  const r = removePlainBackground(S.img, S.tolerance);
  setCut(r.img, r.mask);
}

function setCut(cut: RGBAImage, mask: Mask) {
  S.cut = cut;
  S.sil = mask;
  S.bbox = maskBBox(mask, cut.w, cut.h);
  S.segDirty = true;
  S.rig = null;
}

async function aiBackground() {
  if (!sam || !S.img) return;
  busy('AI đang tách nền…');
  try {
    await sam.encoded();
    const b = S.bbox ?? { x0: 0, y0: 0, x1: S.img.w - 1, y1: S.img.h - 1 };
    const m = await sam.segmentBox([b.x0, b.y0, b.x1, b.y1]);
    keepLargestComponents(m, S.img.w, S.img.h, 0.05);
    const bg = borderSpread(S.img, borderColor(S.img)) < 0.2 ? borderColor(S.img) : null;
    setCut(applyMatte(S.img, m, bg, S.tolerance), m);
  } catch (e) { alert('AI tách nền lỗi: ' + (e as Error).message); }
  busy(null);
  draw();
}

const H = () => (S.bbox ? S.bbox.y1 - S.bbox.y0 : 100);

// ---------------------------------------------------------------- view transform for the 2D editor
const view = { s: 1, ox: 0, oy: 0 };
function fitView() {
  const r = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  for (const c of [edit, glc]) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); c.style.width = r.width + 'px'; c.style.height = r.height + 'px'; }
  if (!S.img) return;
  const b = S.bbox ? { x0: S.bbox.x0, y0: S.bbox.y0, x1: S.bbox.x1, y1: S.bbox.y1 } : { x0: 0, y0: 0, x1: S.img.w, y1: S.img.h };
  const pad = (b.y1 - b.y0) * 0.08;
  const bw = b.x1 - b.x0 + 2 * pad, bh = b.y1 - b.y0 + 2 * pad;
  view.s = Math.min(edit.width / bw, edit.height / bh);
  view.ox = (edit.width - bw * view.s) / 2 - (b.x0 - pad) * view.s;
  view.oy = (edit.height - bh * view.s) / 2 - (b.y0 - pad) * view.s;
}
const toWorld = (e: PointerEvent | MouseEvent): Vec => {
  const r = edit.getBoundingClientRect();
  const dpr = edit.width / r.width;
  return { x: ((e.clientX - r.left) * dpr - view.ox) / view.s, y: ((e.clientY - r.top) * dpr - view.oy) / view.s };
};

let cutCanvas: HTMLCanvasElement | null = null, cutFor: RGBAImage | null = null;
let ownerCanvas: HTMLCanvasElement | null = null, ownerFor: Assignment | null = null;

function checker(ctx: CanvasRenderingContext2D, w: number, h: number) {
  // the checkerboard is the stage's CSS background; canvases stay transparent
  ctx.clearRect(0, 0, w, h);
}

function ownerOverlay(a: Assignment, w: number, h: number) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  const id = ctx.createImageData(w, h);
  const cols = GROUPS.map(g => [parseInt(g.color.slice(1, 3), 16), parseInt(g.color.slice(3, 5), 16), parseInt(g.color.slice(5, 7), 16)]);
  for (let i = 0; i < w * h; i++) {
    const o = a.owner[i];
    if (!o) continue;
    let [r, g, b] = cols[o - 1];
    if (a.detail[i] >= 0) { const k = a.detail[i] * 67; r = (r + 80 + k) % 256; g = (g + 140 + k * 2) % 256; b = (b + 30) % 256; }
    // outline edges between parts darker
    const x = i % w;
    const edge = (x > 0 && a.owner[i - 1] !== o) || (x < w - 1 && a.owner[i + 1] !== o) || (i >= w && a.owner[i - w] !== o) || (i + w < w * h && a.owner[i + w] !== o);
    id.data[i * 4] = edge ? r * 0.4 : r; id.data[i * 4 + 1] = edge ? g * 0.4 : g; id.data[i * 4 + 2] = edge ? b * 0.4 : b;
    id.data[i * 4 + 3] = edge ? 255 : 170;
  }
  ctx.putImageData(id, 0, 0);
  return c;
}

const BONE_LINES: Array<[JointId, JointId, string]> = [
  ['hip', 'neck', '#4f8ff7'], ['neck', 'head', '#f4b942'],
  ['shoulder', 'elbowF', '#a05ad8'], ['elbowF', 'wristF', '#a05ad8'],
  ['hip', 'kneeF', '#14868c'], ['kneeF', 'ankleF', '#14868c'], ['ankleF', 'toeF', '#14868c'],
  ['hip', 'kneeN', '#2bb673'], ['kneeN', 'ankleN', '#2bb673'], ['ankleN', 'toeN', '#2bb673'],
  ['shoulder', 'elbowN', '#e8505b'], ['elbowN', 'wristN', '#e8505b'],
];

let hoverJoint: JointId | null = null;
function draw() {
  const ctx = edit.getContext('2d')!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  checker(ctx, edit.width, edit.height);
  if (!S.cut || S.step >= 4 && S.step <= 5) return;
  if (cutFor !== S.cut) { cutCanvas = imageToCanvas(S.cut); cutFor = S.cut; }
  ctx.setTransform(view.s, 0, 0, view.s, view.ox, view.oy);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(cutCanvas!, 0, 0);
  const dpr = window.devicePixelRatio || 1;
  const px = dpr / view.s; // one screen pixel in world units
  if (S.step === 3 && S.assignment) {
    if (ownerFor !== S.assignment) { ownerCanvas = ownerOverlay(S.assignment, S.cut.w, S.cut.h); ownerFor = S.assignment; }
    ctx.globalAlpha = S.overlay;
    ctx.drawImage(ownerCanvas!, 0, 0);
    ctx.globalAlpha = 1;
    // correction clicks
    const pts = S.editDetail >= 0 ? S.seg.details[S.editDetail]?.points ?? [] : S.seg.extra.get(S.editGroup) ?? [];
    for (const [x, y, l] of pts) {
      ctx.fillStyle = l ? '#16a34a' : '#dc2626';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2 * px;
      ctx.beginPath(); ctx.arc(x, y, 6 * px, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = `${10 * px}px sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(l ? '+' : '−', x, y + 0.5 * px);
    }
  }
  if (S.joints && (S.step === 2 || S.step === 3)) {
    const j = S.joints;
    ctx.lineCap = 'round';
    const lines = [...BONE_LINES];
    if (S.hasWeapon) lines.push([S.weaponHand === 'N' ? 'wristN' : 'wristF', 'weaponTip', '#ff8a1f']);
    for (const [a, b, col] of lines) {
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 6 * px;
      ctx.beginPath(); ctx.moveTo(j[a].x, j[a].y); ctx.lineTo(j[b].x, j[b].y); ctx.stroke();
      ctx.strokeStyle = col;
      ctx.lineWidth = 3 * px;
      ctx.beginPath(); ctx.moveTo(j[a].x, j[a].y); ctx.lineTo(j[b].x, j[b].y); ctx.stroke();
    }
    if (S.step === 2) {
      for (const id of Object.keys(j) as JointId[]) {
        if (id === 'weaponTip' && !S.hasWeapon) continue;
        const p = j[id];
        const far = id.endsWith('F');
        ctx.fillStyle = id === hoverJoint || id === dragJoint ? '#111827' : far ? '#9ca3af' : '#ffffff';
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 2 * px;
        ctx.beginPath(); ctx.arc(p.x, p.y, 7 * px, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      const show = dragJoint ?? hoverJoint;
      if (show) {
        const p = j[show];
        ctx.font = `bold ${13 * px}px system-ui, sans-serif`;
        const t = JOINT_LABEL[show];
        const tw = ctx.measureText(t).width;
        ctx.fillStyle = 'rgba(17,24,39,0.85)';
        ctx.fillRect(p.x + 12 * px, p.y - 22 * px, tw + 12 * px, 20 * px);
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(t, p.x + 18 * px, p.y - 12 * px);
      }
    }
  }
}

// ---------------------------------------------------------------- pointer handling
let dragJoint: JointId | null = null;
function nearestJoint(p: Vec): JointId | null {
  if (!S.joints) return null;
  const dpr = window.devicePixelRatio || 1;
  const lim = 14 * dpr / view.s;
  let best: JointId | null = null, bd = lim;
  for (const id of Object.keys(S.joints) as JointId[]) {
    if (id === 'weaponTip' && !S.hasWeapon) continue;
    const q = S.joints[id];
    const d = Math.hypot(q.x - p.x, q.y - p.y);
    if (d < bd) { bd = d; best = id; }
  }
  return best;
}
edit.addEventListener('pointerdown', e => {
  if (!S.cut) return;
  const p = toWorld(e);
  if (S.step === 2) {
    dragJoint = nearestJoint(p);
    if (dragJoint) edit.setPointerCapture(e.pointerId);
  } else if (S.step === 3) {
    const label = e.button === 2 || e.shiftKey || e.altKey || correctionSign < 0 ? 0 : 1;
    addCorrection([Math.round(p.x), Math.round(p.y), label]);
  }
});
let correctionSign = 1;
edit.addEventListener('contextmenu', e => { if (S.step === 3) e.preventDefault(); });
edit.addEventListener('pointermove', e => {
  if (S.step !== 2 || !S.joints) return;
  const p = toWorld(e);
  if (dragJoint) {
    S.joints[dragJoint] = p;
    S.segDirty = true;
    S.rig = null;
    draw();
  } else {
    const h = nearestJoint(p);
    if (h !== hoverJoint) { hoverJoint = h; edit.style.cursor = h ? 'grab' : 'default'; draw(); }
  }
});
edit.addEventListener('pointerup', () => { dragJoint = null; draw(); });

// ---------------------------------------------------------------- segmentation
function segInput(): SegInput {
  const bones = restBones(S.joints!, S.weaponHand, S.hasWeapon, { arm: false, leg: false });
  return { img: S.img!, cut: S.cut!, sil: S.sil!, joints: S.joints!, bones, hasWeapon: S.hasWeapon, height: H(), facing: S.facing };
}

async function runSegmentation() {
  if (!S.cut || !S.joints) return;
  busy('AI đang tách các phần cơ thể…');
  try {
    let useSam: Sam | null = sam;
    if (sam) {
      try { await sam.encoded(); } catch { useSam = null; }
    }
    await runAutoSegmentation(segInput(), useSam, S.seg, t => busy(t));
    // detail layers
    for (const d of S.seg.details) {
      if (useSam && d.points.some(p => p[2] === 1)) d.mask = await useSam.segment(d.points);
    }
    S.assignment = assign(segInput(), S.seg);
    S.segDirty = false;
    S.rig = null;
  } catch (e) {
    console.error(e);
    alert('Tách lớp lỗi: ' + (e as Error).message);
  }
  busy(null);
  draw();
  renderPanel();
}

let segTimer = 0;
async function addCorrection(pt: [number, number, number]) {
  if (S.editDetail >= 0) {
    const d = S.seg.details[S.editDetail];
    d.points.push(pt);
  } else {
    const arr = S.seg.extra.get(S.editGroup) ?? [];
    arr.push(pt);
    S.seg.extra.set(S.editGroup, arr);
  }
  draw();
  clearTimeout(segTimer);
  segTimer = window.setTimeout(() => refineOne(), 150);
}

async function refineOne() {
  if (!sam) { alert('Cần AI để sửa bằng cách bấm điểm.'); return; }
  try {
    await sam.encoded();
    const inp = segInput();
    if (S.editDetail >= 0) {
      const d = S.seg.details[S.editDetail];
      d.mask = d.points.some(p => p[2] === 1) ? await sam.segment(d.points) : null;
    } else if (S.editGroup !== 'torso') {
      S.seg.sam.set(S.editGroup, await segmentGroup(inp, sam, S.seg, S.editGroup));
    }
    S.assignment = assign(inp, S.seg);
    S.rig = null;
    draw();
  } catch (e) { console.warn(e); }
}

// ---------------------------------------------------------------- rig & animation
function poseOptions(): PoseOptions {
  return { facing: S.facing, height: H(), keepArmN: S.hasWeapon && S.keepArm && S.weaponHand === 'N', keepArmF: S.hasWeapon && S.keepArm && S.weaponHand === 'F' };
}
function ensureRig(): Rig | null {
  if (S.rig) return S.rig;
  if (!S.cut || !S.joints || !S.assignment) return null;
  const cloneArm = S.cloneFarArm && !(S.hasWeapon && S.weaponHand === 'F');
  const inp = segInput();
  const parts = buildPartImages(inp, S.assignment);
  const bones: RestBone[] = restBones(S.joints, S.weaponHand, S.hasWeapon, { arm: cloneArm, leg: S.cloneFarLeg });
  S.rig = buildRig(parts, S.assignment, S.seg.details, bones, H(), { cloneFarArm: cloneArm, cloneFarLeg: S.cloneFarLeg, mannequin: S.mannequin, weaponHand: S.weaponHand });
  renderer?.setRig(S.rig);
  frameBox = null;
  return S.rig;
}

let frameBox: { x: number; y: number; w: number; h: number } | null = null;
let t0 = performance.now(), phase = 0;
function animate(now: number) {
  requestAnimationFrame(animate);
  if (S.step !== 4 || !renderer || !S.rig) { t0 = now; return; }
  const act = ACTIONS.find(a => a.id === S.action)!;
  if (S.playing) phase = (phase + (now - t0) / 1000 * S.speed / act.seconds) % 1;
  t0 = now;
  const rig = S.rig;
  if (!frameBox) {
    const b = animationBox(renderer, rig, ACTIONS.map(a => a.id), 12, { mannequin: S.mannequin, pose: poseOptions() });
    // keep canvas aspect
    const ar = glc.width / glc.height;
    const pad = b.h * 0.1;
    let w = b.w + 2 * pad, h = b.h + 2 * pad;
    if (w / h < ar) w = h * ar; else h = w / ar;
    frameBox = { x: b.x + b.w / 2 - w / 2, y: b.y + b.h / 2 - h / 2, w, h };
  }
  const p = poseAt(S.action, phase, rig.bones, poseOptions());
  const mats = solvePose(rig.bones, p.angles, p.root);
  renderer.draw(mats, frameBox, { mannequin: S.mannequin, clear: [0, 0, 0, 0] });
  drawBonesOverlay(mats);
}
requestAnimationFrame(animate);

function drawBonesOverlay(mats: Map<BoneId, Mat>) {
  const ctx = edit.getContext('2d')!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  checker(ctx, edit.width, edit.height);
  // ground line
  if (!frameBox || !S.rig) return;
  const sx = edit.width / frameBox.w, sy = edit.height / frameBox.h;
  const gy = (S.bbox!.y1 - frameBox.y) * sy;
  ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(edit.width, gy); ctx.stroke();
  if (!S.showBones) return;
  ctx.lineCap = 'round';
  for (const b of S.rig.bones) {
    const m = mats.get(b.id);
    if (!m) continue;
    const a = { x: m[0] * b.a.x + m[2] * b.a.y + m[4], y: m[1] * b.a.x + m[3] * b.a.y + m[5] };
    const c = { x: m[0] * b.b.x + m[2] * b.b.y + m[4], y: m[1] * b.b.x + m[3] * b.b.y + m[5] };
    ctx.strokeStyle = b.id.endsWith('F') ? 'rgba(120,120,140,.9)' : 'rgba(220,38,38,.95)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo((a.x - frameBox.x) * sx, (a.y - frameBox.y) * sy);
    ctx.lineTo((c.x - frameBox.x) * sx, (c.y - frameBox.y) * sy);
    ctx.stroke();
  }
}

// ---------------------------------------------------------------- navigation & panels
let unmount3d: (() => void) | null = null;
let bodyChosen = false;
async function goto(n: Step) {
  if (n !== 6 && n > 1 && !S.cut) return;
  if (n >= 2 && n <= 5 && !S.joints && S.cut) {
    S.facing = guessFacing({ mask: S.sil!, w: S.cut.w, h: S.cut.h, bbox: S.bbox! });
    if (!bodyChosen) S.body = guessBodyType({ mask: S.sil!, w: S.cut.w, h: S.cut.h, bbox: S.bbox! });
    S.joints = guessJoints({ mask: S.sil!, w: S.cut.w, h: S.cut.h, bbox: S.bbox! }, S.body, S.facing);
  }
  S.step = n;
  document.body.dataset.step = String(n);
  if (unmount3d && n !== 6) { unmount3d(); unmount3d = null; }
  renderSteps();
  fitView();
  if (n >= 3 && n <= 5 && (S.segDirty || !S.assignment)) await runSegmentation();
  if (n >= 4 && n <= 5) {
    busy('Đang dựng lưới biến dạng…');
    await new Promise(r => setTimeout(r, 20));
    try { ensureRig(); } catch (e) { console.error(e); alert('Dựng xương lỗi: ' + (e as Error).message); }
    busy(null);
  }
  renderPanel();
  if (n === 6 && !unmount3d) {
    unmount3d = mountMannequin3D($('#mann3d'), {
      getRig: () => S.rig, poseOptions, getAction: () => S.action,
      setBody: b => {
        S.body = b;
        bodyChosen = true;
        if (S.cut) { reguess(); goto(2); } else alert('Đã chọn tỉ lệ. Hãy tải ảnh nhân vật ở bước 1, app sẽ dùng tỉ lệ này để đoán khớp.');
      },
    });
  }
  draw();
}

function renderPanel() {
  const n = S.step;
  const next = (label: string, to: Step) => `<button class="btn primary wide" data-goto="${to}">${label} →</button>`;
  let html = '';
  if (n === 1) {
    html = `<h2>1. Ảnh nhân vật</h2>
      ${S.cut ? `<p>Nền đã được tách. Phần ô caro là trong suốt.</p>
      <label class="row">Độ nhạy tách nền <input type="range" id="tol" min="8" max="80" value="${S.tolerance}"></label>
      <p class="hint">Nếu mất một phần áo trắng, giảm độ nhạy. Nếu còn viền nền, tăng lên.</p>
      <button class="btn" id="aiBg" ${sam ? '' : 'disabled'}>Ảnh có nền phức tạp? Dùng AI tách nền</button>
      <button class="btn" id="newImg">Chọn ảnh khác</button>
      ${next('Tiếp: đặt khớp xương', 2)}` : `<p>Chọn một ảnh nhân vật nhìn nghiêng. Toàn bộ xử lý chạy ngay trên máy bạn, ảnh không gửi đi đâu.</p>`}`;
  } else if (n === 2) {
    html = `<h2>2. Khớp xương</h2>
      <p>App đã đoán vị trí các khớp. <b>Kéo các chấm tròn</b> tới đúng khớp trên ảnh. Chấm xám là tay/chân phía sau.</p>
      <label class="row">Dáng người <select id="body">${(Object.keys(BODY_LABEL) as BodyType[]).map(b => `<option value="${b}" ${b === S.body ? 'selected' : ''}>${BODY_LABEL[b]}</option>`).join('')}</select></label>
      <div class="row">Nhân vật nhìn sang
        <span class="seg"><button class="btn ${S.facing < 0 ? 'on' : ''}" data-face="-1">← Trái</button><button class="btn ${S.facing > 0 ? 'on' : ''}" data-face="1">Phải →</button></span></div>
      <label class="row check"><input type="checkbox" id="weapon" ${S.hasWeapon ? 'checked' : ''}> Có vũ khí / đồ cầm tay</label>
      ${S.hasWeapon ? `<div class="row">Cầm ở <span class="seg"><button class="btn ${S.weaponHand === 'N' ? 'on' : ''}" data-hand="N">Tay trước</button><button class="btn ${S.weaponHand === 'F' ? 'on' : ''}" data-hand="F">Tay sau</button></span></div>
        <p class="hint">Kéo chấm "Đầu vũ khí" tới mũi kiếm / đầu gậy.</p>` : ''}
      <button class="btn" id="reguess">Đoán lại vị trí khớp</button>
      ${next('Tiếp: tách lớp', 3)}`;
  } else if (n === 3) {
    const groups = GROUPS.filter(g => g.id !== 'weapon' || S.hasWeapon);
    html = `<h2>3. Tách lớp</h2>
      <p>Mỗi màu là một phần chuyển động riêng. Nếu phần nào sai, chọn phần đó rồi <b>bấm chuột trái</b> vào chỗ thuộc về nó, <b>chuột phải</b> (hoặc giữ Shift) vào chỗ không thuộc về nó.</p>
      <div class="layers">${groups.map(g => `<button class="layer ${S.editDetail < 0 && S.editGroup === g.id ? 'on' : ''}" data-group="${g.id}"><i style="background:${g.color}"></i>${g.label}${(S.seg.extra.get(g.id)?.length ?? 0) ? ` <small>(${S.seg.extra.get(g.id)!.length} điểm sửa)</small>` : ''}</button>`).join('')}</div>
      <div class="row"><span class="seg"><button class="btn ${correctionSign > 0 ? 'on' : ''}" data-sign="1">＋ Thuộc về</button><button class="btn ${correctionSign < 0 ? 'on' : ''}" data-sign="-1">－ Không thuộc</button></span></div>
      <button class="btn" id="clearPts">Xoá điểm sửa của phần đang chọn</button>
      <h3>Lớp chi tiết (tuỳ chọn)</h3>
      <p class="hint">Tách thêm tóc, mặt, mắt, miệng, áo, quần, giày… để sau này chỉnh riêng. Chọn loại, bấm "Thêm", rồi bấm vào chỗ đó trên ảnh.</p>
      <div class="row"><select id="detailKind">${DETAILS.map(d => `<option value="${d.id}">${d.label}</option>`).join('')}</select><button class="btn" id="addDetail">Thêm</button></div>
      <div class="layers">${S.seg.details.map((d, k) => `<button class="layer ${S.editDetail === k ? 'on' : ''}" data-detail="${k}"><i class="stripe"></i>${DETAILS.find(x => x.id === d.kind)!.label} <small>(${d.points.length} điểm)</small><span class="x" data-del="${k}">✕</span></button>`).join('')}</div>
      <h3>Tay chân phía sau</h3>
      <label class="row check"><input type="checkbox" id="cloneArm" ${S.cloneFarArm ? 'checked' : ''}> Tay sau: nhân bản từ tay trước (khuyên dùng)</label>
      <label class="row check"><input type="checkbox" id="cloneLeg" ${S.cloneFarLeg ? 'checked' : ''}> Chân sau: nhân bản từ chân trước (khuyên dùng)</label>
      <label class="row">Độ đậm màu phủ <input type="range" id="ov" min="0" max="100" value="${S.overlay * 100}"></label>
      ${next('Tiếp: xem chuyển động', 4)}`;
  } else if (n === 4) {
    html = `<h2>4. Chuyển động</h2>
      <div class="actions">${ACTIONS.map(a => `<button class="btn ${S.action === a.id ? 'on' : ''}" data-action="${a.id}">${a.label}</button>`).join('')}</div>
      <div class="row"><button class="btn" id="play">${S.playing ? '⏸ Dừng' : '▶ Chạy'}</button></div>
      <label class="row">Tốc độ <input type="range" id="speed" min="25" max="200" value="${S.speed * 100}"></label>
      <label class="row check"><input type="checkbox" id="mann" ${S.mannequin ? 'checked' : ''}> Ma-nơ-canh lót dưới quần áo (che khe hở)</label>
      <label class="row check"><input type="checkbox" id="bones" ${S.showBones ? 'checked' : ''}> Hiện xương</label>
      ${S.hasWeapon ? `<label class="row check"><input type="checkbox" id="keepArm" ${S.keepArm ? 'checked' : ''}> Tay cầm vũ khí giữ nguyên tư thế</label>` : ''}
      <p class="hint">Thấy chỗ nào sai? Quay lại bước 2 chỉnh khớp hoặc bước 3 sửa lớp.</p>
      ${next('Tiếp: xuất file', 5)}`;
  } else if (n === 5) {
    html = `<h2>5. Xuất spritesheet</h2>
      <p>Tạo một ảnh PNG chứa mọi khung hình (mỗi hành động một hàng) và file JSON mô tả khung hình, dùng được trong Phaser, Pixi, Godot, Unity…</p>
      <div class="row">Hành động ${ACTIONS.map(a => `<label class="check"><input type="checkbox" data-exp="${a.id}" checked> ${a.label}</label>`).join(' ')}</div>
      <label class="row">Số khung mỗi hành động <select id="frames"><option>8</option><option selected>12</option><option>16</option><option>24</option></select></label>
      <label class="row">Chiều cao khung <select id="cell"><option>128</option><option selected>256</option><option>384</option><option>512</option></select> px</label>
      <label class="row check"><input type="checkbox" id="flip"> Lật ngược hướng nhìn</label>
      <button class="btn primary wide" id="doExport">Xuất PNG + JSON</button>
      <div id="expOut"></div>`;
  } else if (n === 6) {
    html = `<h2>Ma-nơ-canh 3D</h2><div id="m3dPanel"></div>`;
  }
  panel.innerHTML = html;
  bindPanel();
}

function bindPanel() {
  panel.querySelectorAll<HTMLButtonElement>('[data-goto]').forEach(b => b.onclick = () => goto(Number(b.dataset.goto) as Step));
  const on = (id: string, ev: string, fn: (e: Event) => void) => { const el = panel.querySelector('#' + id); if (el) el.addEventListener(ev, fn); };
  on('tol', 'change', e => { S.tolerance = Number((e.target as HTMLInputElement).value); cutBackground(); S.joints = null; fitView(); draw(); });
  on('aiBg', 'click', () => aiBackground());
  on('newImg', 'click', () => { $('#drop').hidden = false; });
  on('body', 'change', e => { S.body = (e.target as HTMLSelectElement).value as BodyType; bodyChosen = true; reguess(); });
  on('reguess', 'click', () => reguess());
  on('weapon', 'change', e => { S.hasWeapon = (e.target as HTMLInputElement).checked; S.segDirty = true; S.rig = null; renderPanel(); draw(); });
  panel.querySelectorAll<HTMLButtonElement>('[data-face]').forEach(b => b.onclick = () => { S.facing = Number(b.dataset.face) as 1 | -1; reguess(); });
  panel.querySelectorAll<HTMLButtonElement>('[data-hand]').forEach(b => b.onclick = () => { S.weaponHand = b.dataset.hand as 'N' | 'F'; S.segDirty = true; S.rig = null; renderPanel(); draw(); });
  panel.querySelectorAll<HTMLButtonElement>('[data-group]').forEach(b => b.onclick = () => { S.editGroup = b.dataset.group as GroupId; S.editDetail = -1; renderPanel(); draw(); });
  panel.querySelectorAll<HTMLButtonElement>('[data-sign]').forEach(b => b.onclick = () => { correctionSign = Number(b.dataset.sign); renderPanel(); });
  panel.querySelectorAll<HTMLElement>('[data-detail]').forEach(b => b.onclick = (e) => {
    const del = (e.target as HTMLElement).dataset.del;
    if (del !== undefined) {
      S.seg.details.splice(Number(del), 1);
      S.editDetail = -1;
      if (S.assignment) S.assignment = assign(segInput(), S.seg);
      S.rig = null;
    } else S.editDetail = Number(b.dataset.detail);
    renderPanel(); draw();
  });
  on('addDetail', 'click', () => {
    const kind = (panel.querySelector('#detailKind') as HTMLSelectElement).value as DetailKind;
    S.seg.details.push({ kind, points: [], mask: null });
    S.editDetail = S.seg.details.length - 1;
    renderPanel(); draw();
  });
  on('clearPts', 'click', () => {
    if (S.editDetail >= 0) { S.seg.details[S.editDetail].points = []; S.seg.details[S.editDetail].mask = null; }
    else S.seg.extra.set(S.editGroup, []);
    refineOne(); renderPanel();
  });
  on('cloneArm', 'change', e => { S.cloneFarArm = (e.target as HTMLInputElement).checked; S.rig = null; });
  on('cloneLeg', 'change', e => { S.cloneFarLeg = (e.target as HTMLInputElement).checked; S.rig = null; });
  on('ov', 'input', e => { S.overlay = Number((e.target as HTMLInputElement).value) / 100; draw(); });
  panel.querySelectorAll<HTMLButtonElement>('[data-action]').forEach(b => b.onclick = () => { S.action = b.dataset.action as ActionId; renderPanel(); });
  on('play', 'click', () => { S.playing = !S.playing; renderPanel(); });
  on('speed', 'input', e => { S.speed = Number((e.target as HTMLInputElement).value) / 100; });
  on('mann', 'change', e => { S.mannequin = (e.target as HTMLInputElement).checked; frameBox = null; });
  on('bones', 'change', e => { S.showBones = (e.target as HTMLInputElement).checked; });
  on('keepArm', 'change', e => { S.keepArm = (e.target as HTMLInputElement).checked; frameBox = null; });
  on('doExport', 'click', async () => {
    const rig = ensureRig();
    if (!rig) return;
    const actions = [...panel.querySelectorAll<HTMLInputElement>('[data-exp]')].filter(i => i.checked).map(i => i.dataset.exp as ActionId);
    if (!actions.length) { alert('Chọn ít nhất một hành động'); return; }
    busy('Đang vẽ các khung hình…');
    await new Promise(r => setTimeout(r, 20));
    try {
      const res = await exportSpritesheet(rig, {
        actions, frames: Number((panel.querySelector('#frames') as HTMLSelectElement).value),
        cell: Number((panel.querySelector('#cell') as HTMLSelectElement).value),
        mannequin: S.mannequin, flipX: (panel.querySelector('#flip') as HTMLInputElement).checked,
        pose: poseOptions(), name: S.name,
      });
      download(S.name + '.png', res.png);
      download(S.name + '.json', res.json);
      $('#expOut').innerHTML = `<p>Đã tải xuống <b>${S.name}.png</b> và <b>${S.name}.json</b>.</p><img class="sheet" src="${res.preview}" alt="spritesheet">`;
    } catch (e) { alert('Xuất lỗi: ' + (e as Error).message); }
    busy(null);
  });
}

function reguess() {
  if (!S.cut || !S.bbox || !S.sil) return;
  S.joints = guessJoints({ mask: S.sil, w: S.cut.w, h: S.cut.h, bbox: S.bbox }, S.body, S.facing);
  S.segDirty = true; S.rig = null;
  renderPanel(); draw();
}

window.addEventListener('resize', () => { fitView(); frameBox = null; draw(); });
renderSteps();
renderPanel();
fitView();
draw();

// test hook for automated checks
(window as unknown as { __app: unknown }).__app = { S, goto, loadFrom, ensureRig, sam: () => sam };

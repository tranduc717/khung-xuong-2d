// 3D mannequins (male, female, child, chibi) with a real skeleton.
// - big view: orbit 360° to check every joint
// - camera angle control: where the 2D game camera looks from
// - live result panel: the flat 2D picture from that angle, with the motion playing
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ACTIONS, gaitAt, type ActionId, type PoseOptions } from './anim';
import { PROPS, type BodyType } from './skeleton';
import type { Rig } from './rig';

export interface Mannequin3DContext {
  getRig: () => Rig | null;
  poseOptions: () => PoseOptions;
  getAction: () => ActionId;
  setBody?: (b: BodyType) => void;
}

const D = Math.PI / 180;

// Body shape per type (sizes as fractions of the height). Heights of joints come from PROPS,
// the same numbers the 2D joint guesser uses, so 2D and 3D agree.
const SHAPE: Record<BodyType, {
  label: string; headW: number; shoulderW: number; hipW: number; chestD: number;
  upperArmR: number; foreArmR: number; thighR: number; shinR: number; handL: number; footL: number; color: number;
}> = {
  adult: { label: 'Nam', headW: 0.075, shoulderW: 0.125, hipW: 0.065, chestD: 0.075, upperArmR: 0.03, foreArmR: 0.025, thighR: 0.045, shinR: 0.034, handL: 0.1, footL: 0.15, color: 0xd9b48f },
  female: { label: 'Nữ', headW: 0.072, shoulderW: 0.105, hipW: 0.078, chestD: 0.068, upperArmR: 0.025, foreArmR: 0.021, thighR: 0.045, shinR: 0.031, handL: 0.095, footL: 0.14, color: 0xe6c2a0 },
  child: { label: 'Trẻ em', headW: 0.1, shoulderW: 0.11, hipW: 0.07, chestD: 0.08, upperArmR: 0.032, foreArmR: 0.028, thighR: 0.048, shinR: 0.038, handL: 0.09, footL: 0.15, color: 0xe8c6a4 },
  chibi: { label: 'Chibi', headW: 0.19, shoulderW: 0.12, hipW: 0.085, chestD: 0.1, upperArmR: 0.04, foreArmR: 0.036, thighR: 0.055, shinR: 0.048, handL: 0.08, footL: 0.15, color: 0xf0cfae },
};

type BoneName = 'root' | 'spine' | 'chest' | 'neck' | 'head'
  | 'shoulderN' | 'upperArmN' | 'forearmN' | 'handN' | 'shoulderF' | 'upperArmF' | 'forearmF' | 'handF'
  | 'hipN' | 'thighN' | 'shinN' | 'footN' | 'hipF' | 'thighF' | 'shinF' | 'footF';

const JOINT_VN: Partial<Record<BoneName, string>> = {
  root: 'Hông', chest: 'Ngực', neck: 'Cổ', head: 'Đầu', upperArmN: 'Vai trước', forearmN: 'Khuỷu tay trước', handN: 'Cổ tay trước',
  upperArmF: 'Vai sau', forearmF: 'Khuỷu tay sau', handF: 'Cổ tay sau', thighN: 'Hông trước', shinN: 'Gối trước', footN: 'Cổ chân trước',
  thighF: 'Hông sau', shinF: 'Gối sau', footF: 'Cổ chân sau',
};

interface Built { group: THREE.Group; bones: Record<BoneName, THREE.Object3D>; height: number }

/** Builds one wooden-style mannequin: ball joints + rounded segments, each piece parented to its bone. */
function buildMannequin(type: BodyType, H = 1.7): Built {
  const P = PROPS[type], S = SHAPE[type];
  const yOf = (f: number) => H * (1 - f);
  const mat = new THREE.MeshStandardMaterial({ color: S.color, roughness: 0.75, metalness: 0.0 });
  const jointMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(S.color).multiplyScalar(0.82), roughness: 0.6 });
  const group = new THREE.Group();
  const bones = {} as Record<BoneName, THREE.Object3D>;
  const mk = (name: BoneName, parent: THREE.Object3D | null, pos: THREE.Vector3) => {
    const b = new THREE.Bone();
    b.name = name;
    b.position.copy(pos);
    (parent ?? group).add(b);
    bones[name] = b;
    return b;
  };
  const capsule = (r: number, len: number, m = mat) => {
    const g = new THREE.CapsuleGeometry(r, Math.max(0.001, len - 2 * r), 6, 16);
    const mesh = new THREE.Mesh(g, m);
    mesh.position.y = -len / 2; // hangs down from the joint
    mesh.castShadow = true;
    return mesh;
  };
  const ball = (r: number) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), jointMat); m.castShadow = true; return m; };

  const hipY = yOf(P.hip), neckY = yOf(P.neck), shoulderY = yOf(P.shoulder), kneeY = yOf(P.knee), ankleY = yOf(P.ankle);
  const root = mk('root', null, new THREE.Vector3(0, hipY, 0));
  // pelvis
  const pelvis = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mat);
  pelvis.scale.set(S.chestD * H * 0.9, S.hipW * H * 0.9, S.hipW * H * 1.45);
  pelvis.position.y = S.hipW * H * 0.25;
  root.add(pelvis);
  const spine = mk('spine', root, new THREE.Vector3(0, (shoulderY - hipY) * 0.35, 0));
  const waist = ball(S.chestD * H * 0.62); spine.add(waist);
  const chest = mk('chest', spine, new THREE.Vector3(0, (shoulderY - hipY) * 0.3, 0));
  const torso = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 18), mat);
  torso.scale.set(S.chestD * H, (shoulderY - hipY) * 0.42, S.shoulderW * H * 0.95);
  torso.position.y = (shoulderY - hipY) * 0.17;
  chest.add(torso);
  const neck = mk('neck', chest, new THREE.Vector3(0, shoulderY - hipY - (shoulderY - hipY) * 0.65, 0));
  const neckLen = neckY - shoulderY + S.headW * H * 0.25;
  const neckMesh = capsule(S.upperArmR * H * 0.9, neckLen);
  neckMesh.position.y = neckLen / 2;
  neck.add(neckMesh);
  const head = mk('head', neck, new THREE.Vector3(0, neckY - shoulderY, 0));
  const headH = H * P.neck;
  const skull = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 20), mat);
  skull.scale.set(S.headW * H * 1.15, headH * 0.5, S.headW * H);
  skull.position.set(S.headW * H * 0.1, headH * 0.5, 0);
  skull.castShadow = true;
  head.add(skull);
  // nose marks which way the face looks (+X)
  const nose = new THREE.Mesh(new THREE.ConeGeometry(headH * 0.05, headH * 0.12, 10), jointMat);
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(S.headW * H * 1.25, headH * 0.42, 0);
  head.add(nose);

  for (const side of ['N', 'F'] as const) {
    const z = (side === 'N' ? 1 : -1);
    const sh = mk(('shoulder' + side) as BoneName, chest, new THREE.Vector3(0, shoulderY - hipY - (shoulderY - hipY) * 0.65 - S.upperArmR * H * 0.6, z * S.shoulderW * H));
    sh.add(ball(S.upperArmR * H * 1.25));
    const ua = mk(('upperArm' + side) as BoneName, sh, new THREE.Vector3());
    ua.add(capsule(S.upperArmR * H, P.upper * H));
    const fa = mk(('forearm' + side) as BoneName, ua, new THREE.Vector3(0, -P.upper * H, 0));
    fa.add(ball(S.upperArmR * H * 1.05));
    fa.add(capsule(S.foreArmR * H, P.fore * H));
    const hand = mk(('hand' + side) as BoneName, fa, new THREE.Vector3(0, -P.fore * H, 0));
    hand.add(ball(S.foreArmR * H * 1.05));
    const palm = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), mat);
    palm.scale.set(S.foreArmR * H * 1.4, S.handL * H * 0.5 * (type === 'chibi' ? 0.8 : 1) * 0.5 + S.foreArmR * H * 0.6, S.foreArmR * H * 0.7);
    palm.position.y = -S.handL * H * 0.3;
    hand.add(palm);

    const hp = mk(('hip' + side) as BoneName, root, new THREE.Vector3(0, -S.hipW * H * 0.15, z * S.hipW * H * 0.85));
    hp.add(ball(S.thighR * H * 1.05));
    const th = mk(('thigh' + side) as BoneName, hp, new THREE.Vector3());
    const thighLen = hipY - kneeY;
    th.add(capsule(S.thighR * H, thighLen));
    const sn = mk(('shin' + side) as BoneName, th, new THREE.Vector3(0, -thighLen, 0));
    sn.add(ball(S.thighR * H * 0.9));
    const shinLen = kneeY - ankleY;
    sn.add(capsule(S.shinR * H, shinLen));
    const ft = mk(('foot' + side) as BoneName, sn, new THREE.Vector3(0, -shinLen, 0));
    ft.add(ball(S.shinR * H * 0.95));
    const sole = new THREE.Mesh(new THREE.BoxGeometry(S.footL * H, ankleY * 0.9, S.shinR * H * 1.7), mat);
    sole.position.set(S.footL * H * 0.32, -ankleY * 0.5, 0);
    sole.castShadow = true;
    ft.add(sole);
  }
  return { group, bones, height: H };
}

/** Puts the mannequin in the pose of an action at phase p (same curves as the 2D animation). */
function applyPose(m: Built, action: ActionId | 'rest', p: number) {
  const b = m.bones;
  for (const k of Object.keys(b) as BoneName[]) b[k].rotation.set(0, 0, 0);
  const restY = (b.root.userData.restY ??= b.root.position.y) as number;
  b.root.position.y = restY;
  if (action === 'rest') return;
  const g = gaitAt(action, p);
  b.root.position.y = restY - g.bob * m.height;
  b.spine.rotation.z = -g.lean * D * 0.5;
  b.chest.rotation.z = -g.lean * D * 0.5;
  b.head.rotation.z = -g.head * D;
  // legs: rotation about Z swings in the walking plane, + = forward (+X)
  for (const s of ['N', 'F'] as const) {
    const thigh = g[('thigh' + s) as 'thighN'], knee = g[('knee' + s) as 'kneeN'], foot = g[('foot' + s) as 'footN'];
    (b[('thigh' + s) as BoneName]).rotation.z = thigh * D;
    (b[('shin' + s) as BoneName]).rotation.z = -knee * D;
    (b[('foot' + s) as BoneName]).rotation.z = ((foot - 90) - (thigh - knee)) * D;
    const arm = g[('arm' + s) as 'armN'], elbow = g[('elbow' + s) as 'elbowN'];
    (b[('upperArm' + s) as BoneName]).rotation.z = arm * D + g.lean * D;
    (b[('forearm' + s) as BoneName]).rotation.z = elbow * D;
  }
}

export function mountMannequin3D(el: HTMLElement, ctx: Mannequin3DContext): () => void {
  el.innerHTML = `
    <div class="m3d-main"></div>
    <div class="m3d-result"><div class="m3d-title">Kết quả 2D tức thì</div><canvas></canvas></div>
    <div class="m3d-hint">Kéo chuột để xoay 360°, cuộn để phóng to. Chấm đỏ là khớp xương.</div>`;
  const mainEl = el.querySelector('.m3d-main') as HTMLElement;
  const resCanvas = el.querySelector('.m3d-result canvas') as HTMLCanvasElement;

  const state = {
    type: 'adult' as BodyType, yaw: 0, pitch: 0, action: ctx.getAction() as ActionId, playing: true, animate3d: false,
    showBones: true, speed: 1, outline: true,
  };

  // ---- main 3D view
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  mainEl.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef0f4);
  const cam = new THREE.PerspectiveCamera(35, 1, 0.05, 50);
  cam.position.set(2.2, 1.4, 3.6);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.set(0, 0.85, 0);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa3b2, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(2, 4, 3);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  scene.add(sun);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(2.2, 48), new THREE.MeshStandardMaterial({ color: 0xdfe3ea }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(4, 16, 0xb8bfcc, 0xcfd5de);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.6;
  scene.add(grid);

  // camera marker: shows where the 2D game camera looks from
  const camMarker = new THREE.Group();
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, 0), 0.7, 0x4f46e5, 0.18, 0.1);
  camMarker.add(arrow);
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.16, 0.14), new THREE.MeshStandardMaterial({ color: 0x4f46e5 }));
  camMarker.add(box);
  scene.add(camMarker);

  let man = buildMannequin(state.type);
  scene.add(man.group);

  // skeleton overlay: lines between bones + red joint dots, drawn on top
  const boneLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xdc2626, depthTest: false }));
  boneLines.renderOrder = 10;
  const jointDots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: 0xdc2626, size: 9, sizeAttenuation: false, depthTest: false }));
  jointDots.renderOrder = 11;
  scene.add(boneLines, jointDots);
  const skeletonPairs: Array<[BoneName, BoneName]> = [
    ['root', 'spine'], ['spine', 'chest'], ['chest', 'neck'], ['neck', 'head'],
    ['chest', 'shoulderN'], ['upperArmN', 'forearmN'], ['forearmN', 'handN'], ['chest', 'shoulderF'], ['upperArmF', 'forearmF'], ['forearmF', 'handF'],
    ['root', 'hipN'], ['thighN', 'shinN'], ['shinN', 'footN'], ['root', 'hipF'], ['thighF', 'shinF'], ['shinF', 'footF'],
  ];
  const headTop = new THREE.Object3D();
  const toeN = new THREE.Object3D(), toeF = new THREE.Object3D();
  const attachTips = () => {
    headTop.position.set(0, PROPS[state.type].neck * man.height, 0);
    man.bones.head.add(headTop);
    const fl = SHAPE[state.type].footL * man.height * 0.8;
    toeN.position.set(fl, -PROPS[state.type].ankle * 0 - (1 - PROPS[state.type].ankle) * man.height * 0.9, 0);
    toeF.position.copy(toeN.position);
    man.bones.footN.add(toeN);
    man.bones.footF.add(toeF);
  };
  attachTips();
  const updateSkeleton = () => {
    const pts: number[] = [], dots: number[] = [];
    const w = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3());
    const pairs: Array<[THREE.Object3D, THREE.Object3D]> = skeletonPairs.map(([a, b]) => [man.bones[a], man.bones[b]]);
    pairs.push([man.bones.head, headTop], [man.bones.footN, toeN], [man.bones.footF, toeF]);
    for (const [a, b] of pairs) { const p = w(a), q = w(b); pts.push(p.x, p.y, p.z, q.x, q.y, q.z); }
    for (const k of Object.keys(JOINT_VN) as BoneName[]) { const p = w(man.bones[k]); dots.push(p.x, p.y, p.z); }
    boneLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    jointDots.geometry.setAttribute('position', new THREE.Float32BufferAttribute(dots, 3));
    boneLines.visible = jointDots.visible = state.showBones;
  };

  // hover label for joints
  const label = document.createElement('div');
  label.className = 'm3d-label';
  el.appendChild(label);
  renderer.domElement.addEventListener('pointermove', e => {
    if (!state.showBones) { label.style.display = 'none'; return; }
    const r = renderer.domElement.getBoundingClientRect();
    let best: string | null = null, bd = 14;
    for (const k of Object.keys(JOINT_VN) as BoneName[]) {
      const p = man.bones[k].getWorldPosition(new THREE.Vector3()).project(cam);
      const x = (p.x + 1) / 2 * r.width, y = (1 - p.y) / 2 * r.height;
      const d = Math.hypot(x - (e.clientX - r.left), y - (e.clientY - r.top));
      if (d < bd) { bd = d; best = JOINT_VN[k]!; }
    }
    label.style.display = best ? 'block' : 'none';
    if (best) { label.textContent = best; label.style.left = (e.clientX - el.getBoundingClientRect().left + 12) + 'px'; label.style.top = (e.clientY - el.getBoundingClientRect().top - 10) + 'px'; }
  });

  // ---- live 2D result (orthographic, flat cel look)
  const res = new THREE.WebGLRenderer({ canvas: resCanvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  res.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  const resScene = new THREE.Scene();
  resScene.add(new THREE.HemisphereLight(0xffffff, 0xb0b6c2, 1.9));
  const resSun = new THREE.DirectionalLight(0xffffff, 1.2);
  resScene.add(resSun);
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 20);
  let resMan = buildMannequin(state.type);
  const toonify = (m: Built) => {
    const meshes: THREE.Mesh[] = [];
    m.group.traverse(o => { if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh); });
    for (const mesh of meshes) {
      const c = (mesh.material as THREE.MeshStandardMaterial).color;
      mesh.material = new THREE.MeshToonMaterial({ color: c });
      if (state.outline) {
        const ol = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({ color: 0x3b2f2a, side: THREE.BackSide }));
        ol.scale.setScalar(1.06);
        ol.position.copy(new THREE.Vector3());
        mesh.add(ol);
      }
    }
  };
  toonify(resMan);
  resScene.add(resMan.group);

  const setCamera2D = () => {
    // yaw 0 = side view (camera looks at the character's right side, it faces right on screen)
    const r = 4;
    const yaw = state.yaw * D, pitch = state.pitch * D;
    const dir = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
    const target = new THREE.Vector3(0, man.height * 0.5, 0);
    ortho.position.copy(target).addScaledVector(dir, r);
    ortho.lookAt(target);
    resSun.position.copy(ortho.position).add(new THREE.Vector3(1, 3, 0));
    camMarker.position.copy(target).addScaledVector(dir, 2.1);
    camMarker.lookAt(target);
    const half = man.height * 0.62;
    const aspect = resCanvas.clientWidth / Math.max(1, resCanvas.clientHeight);
    ortho.left = -half * aspect; ortho.right = half * aspect; ortho.top = half; ortho.bottom = -half;
    ortho.updateProjectionMatrix();
  };

  const rebuild = () => {
    scene.remove(man.group);
    resScene.remove(resMan.group);
    man = buildMannequin(state.type);
    resMan = buildMannequin(state.type);
    toonify(resMan);
    scene.add(man.group);
    resScene.add(resMan.group);
    attachTips();
    setCamera2D();
  };

  const resize = () => {
    const r = mainEl.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    renderer.domElement.style.width = r.width + 'px';
    renderer.domElement.style.height = r.height + 'px';
    cam.aspect = r.width / Math.max(1, r.height);
    cam.updateProjectionMatrix();
    const rr = resCanvas.getBoundingClientRect();
    res.setSize(rr.width, rr.height, false);
    setCamera2D();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(el);

  let phase = 0, last = performance.now(), raf = 0;
  const loop = (now: number) => {
    raf = requestAnimationFrame(loop);
    const act = ACTIONS.find(a => a.id === state.action)!;
    if (state.playing) phase = (phase + (now - last) / 1000 * state.speed / act.seconds) % 1;
    last = now;
    applyPose(man, state.animate3d ? state.action : 'rest', phase);
    applyPose(resMan, state.action, phase);
    man.group.updateMatrixWorld(true);
    updateSkeleton();
    controls.update();
    renderer.render(scene, cam);
    res.render(resScene, ortho);
  };
  resize();
  raf = requestAnimationFrame(loop);

  // ---- panel controls
  const panel = document.getElementById('m3dPanel');
  const renderPanel = () => {
    if (!panel) return;
    const presets: Array<[string, number, number]> = [['Nghiêng (game 2D)', 0, 0], ['Chéo 3/4', 45, 10], ['Chính diện', 90, 0], ['Từ trên (top-down)', 0, 60], ['Sau lưng', 180, 0], ['Nghiêng trái', -180 + 0.001, 0]];
    panel.innerHTML = `
      <p>Bốn ma-nơ-canh 3D có sẵn khung xương. Xoay để kiểm tra khớp, chọn góc camera, xem ngay ảnh 2D và chuyển động ở góc đó.</p>
      <div class="actions">${(Object.keys(SHAPE) as BodyType[]).map(t => `<button class="btn ${state.type === t ? 'on' : ''}" data-type="${t}">${SHAPE[t].label}</button>`).join('')}</div>
      <label class="row check"><input type="checkbox" id="m3bones" ${state.showBones ? 'checked' : ''}> Hiện khung xương và khớp</label>
      <label class="row check"><input type="checkbox" id="m3anim" ${state.animate3d ? 'checked' : ''}> Cho ma-nơ-canh 3D chuyển động</label>
      <h3>Góc camera 2D</h3>
      <div class="actions">${presets.map(([n, y, p]) => `<button class="btn" data-yaw="${y}" data-pitch="${p}">${n}</button>`).join('')}</div>
      <label class="row">Xoay ngang <input type="range" id="m3yaw" min="-180" max="180" value="${state.yaw}"><span id="m3yawv">${Math.round(state.yaw)}°</span></label>
      <label class="row">Nhìn từ trên <input type="range" id="m3pitch" min="-20" max="80" value="${state.pitch}"><span id="m3pitchv">${Math.round(state.pitch)}°</span></label>
      <h3>Chuyển động</h3>
      <div class="actions">${ACTIONS.map(a => `<button class="btn ${state.action === a.id ? 'on' : ''}" data-act="${a.id}">${a.label}</button>`).join('')}</div>
      <div class="row"><button class="btn" id="m3play">${state.playing ? '⏸ Dừng' : '▶ Chạy'}</button></div>
      <label class="row">Tốc độ <input type="range" id="m3speed" min="25" max="200" value="${state.speed * 100}"></label>
      <button class="btn" id="m3use">Dùng tỉ lệ này để đoán khớp cho ảnh 2D</button>
      <button class="btn primary wide" id="m3sheet">Tải spritesheet góc này (PNG)</button>`;
    panel.querySelectorAll<HTMLButtonElement>('[data-type]').forEach(b => b.onclick = () => { state.type = b.dataset.type as BodyType; rebuild(); renderPanel(); });
    panel.querySelectorAll<HTMLButtonElement>('[data-yaw]').forEach(b => b.onclick = () => { state.yaw = Number(b.dataset.yaw); state.pitch = Number(b.dataset.pitch); setCamera2D(); renderPanel(); });
    panel.querySelectorAll<HTMLButtonElement>('[data-act]').forEach(b => b.onclick = () => { state.action = b.dataset.act as ActionId; renderPanel(); });
    const inp = (id: string) => panel.querySelector('#' + id) as HTMLInputElement;
    inp('m3bones').onchange = e => { state.showBones = (e.target as HTMLInputElement).checked; };
    inp('m3anim').onchange = e => { state.animate3d = (e.target as HTMLInputElement).checked; };
    inp('m3yaw').oninput = e => { state.yaw = Number((e.target as HTMLInputElement).value); (panel.querySelector('#m3yawv') as HTMLElement).textContent = state.yaw + '°'; setCamera2D(); };
    inp('m3pitch').oninput = e => { state.pitch = Number((e.target as HTMLInputElement).value); (panel.querySelector('#m3pitchv') as HTMLElement).textContent = state.pitch + '°'; setCamera2D(); };
    inp('m3speed').oninput = e => { state.speed = Number((e.target as HTMLInputElement).value) / 100; };
    (panel.querySelector('#m3play') as HTMLButtonElement).onclick = () => { state.playing = !state.playing; renderPanel(); };
    (panel.querySelector('#m3use') as HTMLButtonElement).onclick = () => { ctx.setBody?.(state.type); };
    (panel.querySelector('#m3sheet') as HTMLButtonElement).onclick = () => exportSheet();
  };
  renderPanel();

  const exportSheet = () => {
    const frames = 12, cell = 256;
    const sheet = document.createElement('canvas');
    sheet.width = cell * frames; sheet.height = cell;
    const c2 = sheet.getContext('2d')!;
    const off = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    off.setSize(cell, cell, false);
    const savedL = ortho.left, savedR = ortho.right;
    const half = man.height * 0.62;
    ortho.left = -half; ortho.right = half; ortho.updateProjectionMatrix();
    for (let f = 0; f < frames; f++) {
      applyPose(resMan, state.action, f / frames);
      off.render(resScene, ortho);
      c2.drawImage(off.domElement, f * cell, 0);
    }
    ortho.left = savedL; ortho.right = savedR; ortho.updateProjectionMatrix();
    off.dispose();
    sheet.toBlob(b => {
      if (!b) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = `manocanh-${state.type}-${state.action}-goc${Math.round(state.yaw)}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, 'image/png');
  };

  return () => {
    cancelAnimationFrame(raf);
    ro.disconnect();
    controls.dispose();
    renderer.dispose();
    res.dispose();
    el.innerHTML = '';
  };
}

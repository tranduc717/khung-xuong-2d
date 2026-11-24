// Spritesheet export: one PNG with all frames in a grid + a JSON file game engines can read.
import { poseAt, ACTIONS, type ActionId, type PoseOptions } from './anim';
import { Renderer } from './renderer';
import type { Rig } from './rig';
import { solvePose } from './skeleton';

export interface ExportOptions {
  actions: ActionId[];
  frames: number;
  cell: number;      // cell height in px (width keeps proportion)
  mannequin: boolean;
  flipX: boolean;
  pose: PoseOptions;
  name: string;
}

/** Union box of the character over every frame of the chosen actions, so all frames line up. */
export function animationBox(r: Renderer, rig: Rig, actions: ActionId[], frames: number, o: { mannequin: boolean; pose: PoseOptions }) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const a of actions) for (let f = 0; f < frames; f++) {
    const p = poseAt(a, f / frames, rig.bones, o.pose);
    const b = r.bounds(solvePose(rig.bones, p.angles, p.root), o.mannequin);
    x0 = Math.min(x0, b.x0); y0 = Math.min(y0, b.y0); x1 = Math.max(x1, b.x1); y1 = Math.max(y1, b.y1);
  }
  const pad = (y1 - y0) * 0.03;
  return { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
}

export async function exportSpritesheet(rig: Rig, o: ExportOptions): Promise<{ png: Blob; json: string; preview: string }> {
  const canvas = document.createElement('canvas');
  const r = new Renderer(canvas);
  r.setRig(rig);
  const box = animationBox(r, rig, o.actions, o.frames, o);
  const ch = o.cell;
  const cw = Math.max(8, Math.round(ch * box.w / box.h));
  canvas.width = cw; canvas.height = ch;
  const cols = o.frames;
  const rows = o.actions.length;
  const sheet = document.createElement('canvas');
  sheet.width = cw * cols; sheet.height = ch * rows;
  const ctx = sheet.getContext('2d')!;
  const frames: Record<string, unknown> = {};
  const animations: Record<string, string[]> = {};
  o.actions.forEach((a, row) => {
    animations[a] = [];
    for (let f = 0; f < o.frames; f++) {
      const p = poseAt(a, f / o.frames, rig.bones, o.pose);
      r.draw(solvePose(rig.bones, p.angles, p.root), box, { mannequin: o.mannequin, flipX: o.flipX });
      ctx.drawImage(canvas, f * cw, row * ch);
      const key = `${a}_${String(f).padStart(2, '0')}`;
      frames[key] = { frame: { x: f * cw, y: row * ch, w: cw, h: ch }, rotated: false, trimmed: false, spriteSourceSize: { x: 0, y: 0, w: cw, h: ch }, sourceSize: { w: cw, h: ch }, duration: Math.round(1000 * (ACTIONS.find(x => x.id === a)!.seconds) / o.frames) };
      animations[a].push(key);
    }
  });
  const png: Blob = await new Promise(res => sheet.toBlob(b => res(b!), 'image/png'));
  const json = JSON.stringify({
    frames, animations,
    meta: {
      app: 'khung-xuong-2d', version: '0.1', image: o.name + '.png', format: 'RGBA8888',
      size: { w: sheet.width, h: sheet.height }, scale: '1',
      fps: Object.fromEntries(o.actions.map(a => [a, Math.round(o.frames / ACTIONS.find(x => x.id === a)!.seconds)])),
      pivot: { x: 0.5, y: 1 },
    },
  }, null, 2);
  r.gl.getExtension('WEBGL_lose_context')?.loseContext();
  return { png, json, preview: sheet.toDataURL('image/png') };
}

export function download(name: string, data: Blob | string, type = 'application/json') {
  const blob = typeof data === 'string' ? new Blob([data], { type }) : data;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

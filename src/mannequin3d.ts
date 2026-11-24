import type { Rig } from './rig';
import type { PoseOptions, ActionId } from './anim';
export function mountMannequin3D(el: HTMLElement, _ctx: { getRig: () => Rig | null; poseOptions: () => PoseOptions; getAction: () => ActionId }) {
  el.textContent = '';
  return () => { el.textContent = ''; };
}

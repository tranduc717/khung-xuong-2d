// WebGL2 renderer for skinned layers (meshes deform smoothly, no seams between triangles).
import { skin, type Layer, type Rig } from './rig';
import type { BoneId, Mat } from './skeleton';

const VS = `#version 300 es
in vec2 aPos; in vec2 aUv;
uniform vec4 uView; // scale x, scale y, offset x, offset y (world -> clip)
out vec2 vUv;
void main(){ vUv = aUv; gl_Position = vec4(aPos * uView.xy + uView.zw, 0.0, 1.0); }`;
const FS = `#version 300 es
precision mediump float;
in vec2 vUv; uniform sampler2D uTex; uniform float uTint; uniform float uAlpha;
out vec4 o;
void main(){ vec4 c = texture(uTex, vUv); o = vec4(c.rgb * uTint, c.a) * uAlpha; }`;

interface GLLayer { layer: Layer; vao: WebGLVertexArrayObject; pos: WebGLBuffer; tex: WebGLTexture; count: number; idxType: number; tmp: Float32Array }

export class Renderer {
  gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private items: GLLayer[] = [];
  private rig: Rig | null = null;
  private uView: WebGLUniformLocation;
  private uTint: WebGLUniformLocation;
  private uAlpha: WebGLUniformLocation;

  constructor(public canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, preserveDrawingBuffer: true, antialias: true });
    if (!gl) throw new Error('Trình duyệt không hỗ trợ WebGL2');
    this.gl = gl;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'shader');
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.bindAttribLocation(p, 1, 'aUv');
    gl.linkProgram(p);
    this.prog = p;
    this.uView = gl.getUniformLocation(p, 'uView')!;
    this.uTint = gl.getUniformLocation(p, 'uTint')!;
    this.uAlpha = gl.getUniformLocation(p, 'uAlpha')!;
  }

  setRig(rig: Rig) {
    const gl = this.gl;
    for (const it of this.items) { gl.deleteVertexArray(it.vao); gl.deleteBuffer(it.pos); gl.deleteTexture(it.tex); }
    this.items = [];
    this.rig = rig;
    const texCache = new Map<HTMLCanvasElement, WebGLTexture>();
    for (const layer of rig.layers) {
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      const pos = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, pos);
      gl.bufferData(gl.ARRAY_BUFFER, layer.rest, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      const uvb = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, uvb);
      gl.bufferData(gl.ARRAY_BUFFER, layer.uv, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      const ib = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, layer.index, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      let tex = texCache.get(layer.canvas);
      if (!tex) {
        tex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, layer.canvas);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        texCache.set(layer.canvas, tex);
      }
      this.items.push({
        layer, vao, pos, tex, count: layer.index.length,
        idxType: layer.index instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
        tmp: new Float32Array(layer.rest.length),
      });
    }
  }

  /** Draws the rig with the given bone matrices. view = world rect shown [x, y, w, h]. */
  draw(mats: Map<BoneId, Mat>, view: { x: number; y: number; w: number; h: number }, opts: { mannequin: boolean; clear?: [number, number, number, number]; flipX?: boolean }) {
    const gl = this.gl;
    if (!this.rig) return;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const c = opts.clear ?? [0, 0, 0, 0];
    gl.clearColor(c[0], c[1], c[2], c[3]);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const sx = 2 / view.w, sy = -2 / view.h;
    if (opts.flipX) gl.uniform4f(this.uView, -sx, sy, 1 + view.x * sx, 1 + view.y * 2 / view.h);
    else gl.uniform4f(this.uView, sx, sy, -1 - view.x * sx, 1 + view.y * 2 / view.h);
    gl.uniform1f(this.uAlpha, 1);
    const matList = this.rig.bones.map(b => mats.get(b.id));
    for (const it of this.items) {
      if (!it.layer.visible) continue;
      if (it.layer.group === 'mannequin' && !opts.mannequin) continue;
      skin(it.layer, matList, it.tmp);
      gl.bindBuffer(gl.ARRAY_BUFFER, it.pos);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, it.tmp);
      gl.bindVertexArray(it.vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, it.tex);
      gl.uniform1f(this.uTint, it.layer.tint);
      gl.drawElements(gl.TRIANGLES, it.count, it.idxType, 0);
    }
    gl.bindVertexArray(null);
  }

  /** World-space bounding box of the posed rig (for framing). */
  bounds(mats: Map<BoneId, Mat>, mannequin: boolean) {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    if (!this.rig) return { x0: 0, y0: 0, x1: 1, y1: 1 };
    const matList = this.rig.bones.map(b => mats.get(b.id));
    for (const it of this.items) {
      if (!it.layer.visible || (it.layer.group === 'mannequin' && !mannequin)) continue;
      const p = skin(it.layer, matList, it.tmp);
      for (let i = 0; i < p.length; i += 2) {
        if (p[i] < x0) x0 = p[i]; if (p[i] > x1) x1 = p[i];
        if (p[i + 1] < y0) y0 = p[i + 1]; if (p[i + 1] > y1) y1 = p[i + 1];
      }
    }
    return { x0, y0, x1, y1 };
  }
}

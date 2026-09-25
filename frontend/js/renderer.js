// WebGL2 renderer for the 3D cadastre.
//
// Three passes per frame:
//   1. opaque solids, depth-tested and written
//   2. translucent solids, back-to-front, depth-tested but not written
//   3. wireframe edges for the selection and for validation highlights
//
// Picking is done by rendering object ids into an off-screen colour buffer and
// reading back the single pixel under the cursor. That is exact - it picks
// whatever the user can actually see, including a flat glimpsed through a
// translucent facade - which ray casting against prisms would not be.

const SOLID_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in float aId;
layout(location=3) in vec2 aUV;

uniform mat4 uViewProj;
uniform float uExplode;      // metres of vertical separation per storey
uniform float uExplodeRef;   // z about which the explosion is centred

out vec3 vNormal;
out vec3 vWorld;
out vec2 vUV;
flat out float vId;

void main() {
  vec3 p = aPos;
  // "Exploded" view: push each storey apart along z in proportion to how far
  // above the reference it already is, so the stack fans out but stays ordered.
  p.z += uExplode * (p.z - uExplodeRef) * 0.12;
  vWorld = p;
  vNormal = aNormal;
  vUV = aUV;
  vId = aId;
  gl_Position = uViewProj * vec4(p, 1.0);
}`;

const SOLID_FS = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vWorld;
in vec2 vUV;
flat in float vId;

uniform vec4 uColor;
uniform vec3 uLightDir;
uniform float uSelectedId;
uniform float uHoverId;
uniform int uPickMode;
uniform float uSectionZ;     // hide everything above this height (cutaway)
uniform float uOpacity;
uniform sampler2D uTexture;
uniform int uTextured;       // 1 when this batch carries aerial imagery
uniform int uFacade;         // 1 for built volumes: draw storeys and windows
uniform float uFloorH;       // storey pitch the facade pattern repeats on

out vec4 fragColor;

vec3 idToColor(float id) {
  int i = int(id) + 1;
  return vec3(float((i      ) & 255) / 255.0,
              float((i >>  8) & 255) / 255.0,
              float((i >> 16) & 255) / 255.0);
}

void main() {
  if (vWorld.z > uSectionZ) discard;

  if (uPickMode == 1) {
    fragColor = vec4(idToColor(vId), 1.0);
    return;
  }

  vec3 n = normalize(vNormal);
  float diff = max(dot(n, normalize(uLightDir)), 0.0);
  // a little sky/ground hemispheric fill so vertical faces are not flat black
  float hemi = 0.5 + 0.5 * n.z;

  vec3 albedo = uColor.rgb;
  if (uTextured == 1) {
    // Aerial imagery already contains its own baked lighting and shadows, so
    // shading it again the way an untextured surface is shaded would crush it
    // into mud. Keep a little relief from the terrain normal and no more.
    albedo = texture(uTexture, vUV).rgb;
    fragColor = vec4(albedo * (0.82 + 0.30 * diff), uColor.a * uOpacity);
    if (uSelectedId >= 0.0 && abs(vId - uSelectedId) < 0.5) {
      fragColor.rgb = mix(fragColor.rgb, vec3(1.0, 0.78, 0.22), 0.4);
    }
    return;
  }

  vec3 base = albedo * (0.30 + 0.55 * diff + 0.22 * hemi);

  // A building reads as a building because of its floors and its windows, not
  // because of its outline. Both are drawn from world coordinates, so the
  // pattern stays fixed to the structure as the camera moves and lines up
  // across neighbouring volumes of the same block.
  if (uFacade == 1) {
    float wall = 1.0 - abs(n.z);
    float roof = smoothstep(0.55, 0.92, abs(n.z));

    if (wall > 0.28) {
      // One window per storey, centred on this wall face (face U is 0→1).
      float fz = fract(vWorld.z / uFloorH);
      float fu = vUV.x;

      float band = smoothstep(0.20, 0.30, fz) * (1.0 - smoothstep(0.68, 0.80, fz));
      float bay  = smoothstep(0.28, 0.36, fu) * (1.0 - smoothstep(0.64, 0.72, fu));
      float win  = band * bay * wall;

      // Single pane — no tiled 2×2 mullion grid
      vec3 glass = vec3(0.10, 0.16, 0.24) + vec3(0.18, 0.28, 0.38) * diff;
      glass += vec3(0.45, 0.55, 0.62) * pow(max(diff, 0.0), 4.0) * 0.35;
      base = mix(base, glass, 0.78 * win);

      // Thin frame around the one opening
      float frameU = smoothstep(0.24, 0.28, fu) * (1.0 - smoothstep(0.72, 0.76, fu));
      float frameV = smoothstep(0.16, 0.20, fz) * (1.0 - smoothstep(0.80, 0.84, fz));
      float frame = max(frameU * band, frameV * bay) * wall * (1.0 - win);
      base = mix(base, albedo * 0.55, 0.65 * frame);

      // Floor slab / balcony ledge under each storey
      float slab = 1.0 - smoothstep(0.0, 0.07, fz);
      float ledge = smoothstep(0.88, 0.94, fz);
      base *= mix(1.0, 0.72, slab * wall);
      base = mix(base, albedo * 0.82, 0.55 * ledge * wall);

      // Side piers left and right of the single window
      float pier = 1.0 - smoothstep(0.0, 0.08, min(fu, 1.0 - fu));
      base *= mix(1.0, 0.90, pier * wall * (1.0 - win));
    }

    // Roof deck: warmer, slightly mottled
    if (roof > 0.4 && n.z > 0.0) {
      float mott = fract(sin(dot(vWorld.xy, vec2(12.9898, 78.233))) * 43758.5453);
      base = mix(base, albedo * (0.78 + 0.12 * mott), 0.55 * roof);
      base *= mix(1.0, 0.88, roof);
    }

    // Contact shadow near grade so volumes sit on the ground
    base *= mix(0.70, 1.0, clamp((vWorld.z + 1.5) / 9.0, 0.0, 1.0));
  }

  float a = uColor.a * uOpacity;
  // Guard on >= 0: unpickable geometry (the terrain) carries a negative id, and
  // so does "nothing selected", so an unguarded comparison highlights the
  // ground the moment the selection is cleared.
  if (uSelectedId >= 0.0 && abs(vId - uSelectedId) < 0.5) {
    base = mix(base, vec3(1.0, 0.78, 0.22), 0.55);
    a = min(1.0, a + 0.45);
  } else if (uHoverId >= 0.0 && abs(vId - uHoverId) < 0.5) {
    base = mix(base, vec3(0.45, 0.85, 1.0), 0.35);
    a = min(1.0, a + 0.22);
  }
  fragColor = vec4(base, a);
}`;

const SKY_VS = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  // fullscreen triangle, no buffers needed
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.9999, 1.0);
}`;

const SKY_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform vec3 uTop;
uniform vec3 uBottom;
out vec4 fragColor;
void main() {
  float t = clamp(vUv.y, 0.0, 1.0);
  fragColor = vec4(mix(uBottom, uTop, pow(t, 0.85)), 1.0);
}`;

const LINE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uViewProj;
uniform float uExplode;
uniform float uExplodeRef;
void main() {
  vec3 p = aPos;
  p.z += uExplode * (p.z - uExplodeRef) * 0.12;
  gl_Position = uViewProj * vec4(p, 1.0);
}`;

const LINE_FS = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader compile failed: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('program link failed: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', {
      antialias: true, alpha: false, preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl;
    this.canvas = canvas;

    this.solidProg = program(gl, SOLID_VS, SOLID_FS);
    this.lineProg = program(gl, LINE_VS, LINE_FS);
    this.skyProg = program(gl, SKY_VS, SKY_FS);
    this.skyTop = [0.045, 0.085, 0.145];
    this.skyBottom = [0.145, 0.185, 0.230];
    this.floorHeight = 3.1;
    this.batches = [];
    this.lineBatches = [];
    this.explode = 0;
    this.explodeRef = 0;
    this.sectionZ = 1e6;

    this.solidU = this._uniforms(this.solidProg, [
      'uViewProj', 'uColor', 'uLightDir', 'uSelectedId', 'uHoverId',
      'uPickMode', 'uSectionZ', 'uOpacity', 'uExplode', 'uExplodeRef',
      'uTexture', 'uTextured', 'uFacade', 'uFloorH']);
    this.lineU = this._uniforms(this.lineProg, [
      'uViewProj', 'uColor', 'uExplode', 'uExplodeRef']);
    this.skyU = this._uniforms(this.skyProg, ['uTop', 'uBottom']);
    this.emptyVao = gl.createVertexArray();

    this._initPickTarget();
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
  }

  _uniforms(prog, names) {
    const out = {};
    for (const n of names) out[n] = this.gl.getUniformLocation(prog, n);
    return out;
  }

  _initPickTarget() {
    const gl = this.gl;
    this.pickFbo = gl.createFramebuffer();
    this.pickColor = gl.createTexture();
    this.pickDepth = gl.createRenderbuffer();
    this.pickSize = [1, 1];
  }

  _resizePickTarget(w, h) {
    const gl = this.gl;
    if (this.pickSize[0] === w && this.pickSize[1] === h) return;
    gl.bindTexture(gl.TEXTURE_2D, this.pickColor);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.pickDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pickFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, this.pickColor, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
                               gl.RENDERBUFFER, this.pickDepth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.pickSize = [w, h];
  }

  // --- batches -----------------------------------------------------------
  /**
   * Upload one draw batch.
   * `mesh` is {positions, normals, ids}; `opts` carries colour, opacity,
   * layer name and whether the batch takes part in picking.
   */
  addBatch(mesh, opts = {}) {
    const gl = this.gl;
    if (!mesh.positions.length) return null;

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const pos = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pos);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const nrm = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nrm);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.normals), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const ids = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ids);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.ids), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);

    // UVs are optional. A VAO records its own attribute-enable state, so a
    // batch without them simply leaves location 3 disabled and the shader's
    // uTextured flag keeps it unused.
    let uvBuf = null;
    if (mesh.uvs && mesh.uvs.length) {
      uvBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.uvs), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 0, 0);
    }

    gl.bindVertexArray(null);

    const batch = {
      vao, count: mesh.positions.length / 3,
      texture: null,
      color: opts.color || [0.7, 0.7, 0.75, 1.0],
      opacity: opts.opacity === undefined ? 1.0 : opts.opacity,
      layer: opts.layer || 'default',
      pickable: opts.pickable !== false,
      visible: opts.visible !== false,
      depthWrite: opts.depthWrite !== false,
      facade: opts.facade === true,
      sortKey: opts.sortKey || 0,
      buffers: [pos, nrm, ids, uvBuf].filter(Boolean),
    };
    this.batches.push(batch);
    return batch;
  }

  addLines(positions, opts = {}) {
    const gl = this.gl;
    if (!positions.length) return null;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    const batch = {
      vao, count: positions.length / 3, buffer: buf,
      color: opts.color || [1, 1, 1, 0.8],
      layer: opts.layer || 'edges',
      visible: opts.visible !== false,
    };
    this.lineBatches.push(batch);
    return batch;
  }

  clear() {
    const gl = this.gl;
    for (const b of this.batches) {
      gl.deleteVertexArray(b.vao);
      for (const buf of b.buffers) gl.deleteBuffer(buf);
      // Textures are not part of `buffers`, and the register is rebuilt on
      // every pipeline re-run and plan upload - so without this the aerial
      // image leaks a few megabytes of GPU memory each time.
      if (b.texture) gl.deleteTexture(b.texture);
    }
    for (const b of this.lineBatches) {
      gl.deleteVertexArray(b.vao);
      gl.deleteBuffer(b.buffer);
    }
    this.batches = [];
    this.lineBatches = [];
  }

  setLayerVisible(layer, visible) {
    for (const b of this.batches) if (b.layer === layer) b.visible = visible;
    for (const b of this.lineBatches) if (b.layer === layer) b.visible = visible;
  }

  setLayerOpacity(layer, opacity) {
    for (const b of this.batches) if (b.layer === layer) b.opacity = opacity;
  }

  // --- drawing -----------------------------------------------------------
  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    return [w, h];
  }

  render(viewProj, { selectedId = -1, hoverId = -1, lightDir = [0.45, 0.3, 0.84],
                     background = [0.055, 0.067, 0.094] } = {}) {
    const gl = this.gl;
    const [w, h] = this.resize();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.clearColor(background[0], background[1], background[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    this._drawSky();

    gl.useProgram(this.solidProg);
    const u = this.solidU;
    gl.uniformMatrix4fv(u.uViewProj, false, viewProj);
    gl.uniform3fv(u.uLightDir, lightDir);
    gl.uniform1f(u.uSelectedId, selectedId);
    gl.uniform1f(u.uHoverId, hoverId);
    gl.uniform1i(u.uPickMode, 0);
    gl.uniform1f(u.uSectionZ, this.sectionZ);
    gl.uniform1f(u.uExplode, this.explode);
    gl.uniform1f(u.uExplodeRef, this.explodeRef);
    gl.uniform1f(u.uFloorH, this.floorHeight);

    const visible = this.batches.filter((b) => b.visible);
    const opaque = visible.filter((b) => b.color[3] * b.opacity >= 0.999);
    const alpha = visible.filter((b) => b.color[3] * b.opacity < 0.999);

    gl.disable(gl.BLEND);
    gl.depthMask(true);
    for (const b of opaque) this._drawBatch(b, u);

    // Translucent geometry is sorted by an author-supplied key rather than by
    // true depth: the layers here (ground, air rights, tunnels) have a fixed
    // and obvious stacking order, so a full per-fragment sort would cost more
    // than it is worth.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    alpha.sort((a, b) => a.sortKey - b.sortKey);
    for (const b of alpha) this._drawBatch(b, u);
    gl.depthMask(true);

    // --- edges ------------------------------------------------------------
    gl.useProgram(this.lineProg);
    gl.uniformMatrix4fv(this.lineU.uViewProj, false, viewProj);
    gl.uniform1f(this.lineU.uExplode, this.explode);
    gl.uniform1f(this.lineU.uExplodeRef, this.explodeRef);
    for (const b of this.lineBatches) {
      if (!b.visible) continue;
      gl.uniform4fv(this.lineU.uColor, b.color);
      gl.bindVertexArray(b.vao);
      gl.drawArrays(gl.LINES, 0, b.count);
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  _drawSky() {
    const gl = this.gl;
    gl.useProgram(this.skyProg);
    gl.uniform3fv(this.skyU.uTop, this.skyTop);
    gl.uniform3fv(this.skyU.uBottom, this.skyBottom);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
  }

  _drawBatch(b, u) {
    const gl = this.gl;
    gl.uniform4fv(u.uColor, b.color);
    gl.uniform1f(u.uOpacity, b.opacity);
    gl.uniform1i(u.uFacade, b.facade ? 1 : 0);
    if (b.texture) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, b.texture);
      gl.uniform1i(u.uTexture, 0);
      gl.uniform1i(u.uTextured, 1);
    } else {
      gl.uniform1i(u.uTextured, 0);
    }
    gl.bindVertexArray(b.vao);
    gl.drawArrays(gl.TRIANGLES, 0, b.count);
    if (b.texture) gl.uniform1i(u.uTextured, 0);
  }

  /**
   * Upload an image as a texture for a batch.
   *
   * Flips on upload so v=0 is the south edge, matching world y increasing
   * north - otherwise the aerial photo lands on the site upside down.
   */
  setBatchTexture(batch, image) {
    const gl = this.gl;
    if (!batch) return;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    batch.texture = tex;
  }

  /**
   * Object id under a canvas pixel, or -1.
   *
   * Solid geometry is picked in preference to translucent geometry, in two
   * passes. Without that, every click lands on the building envelope or the
   * air-rights block wrapping the thing the user was aiming at - they are
   * barely visible but perfectly opaque to a pick buffer. The second pass runs
   * only where the first found nothing, so a translucent volume with empty
   * space behind it is still selectable.
   */
  pick(viewProj, px, py) {
    const solid = this.batches.filter(
      (b) => b.visible && b.pickable && b.color[3] * b.opacity >= 0.5);
    const hit = this._pickPass(viewProj, px, py, solid);
    if (hit >= 0) return hit;

    const ghost = this.batches.filter(
      (b) => b.visible && b.pickable && b.color[3] * b.opacity < 0.5);
    return this._pickPass(viewProj, px, py, ghost);
  }

  _pickPass(viewProj, px, py, batches) {
    if (!batches.length) return -1;
    const gl = this.gl;
    const [w, h] = this.resize();
    this._resizePickTarget(w, h);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pickFbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.BLEND);
    gl.depthMask(true);

    gl.useProgram(this.solidProg);
    const u = this.solidU;
    gl.uniformMatrix4fv(u.uViewProj, false, viewProj);
    gl.uniform1i(u.uPickMode, 1);
    gl.uniform1f(u.uSectionZ, this.sectionZ);
    gl.uniform1f(u.uExplode, this.explode);
    gl.uniform1f(u.uExplodeRef, this.explodeRef);
    gl.uniform1f(u.uSelectedId, -1);
    gl.uniform1f(u.uHoverId, -1);
    gl.uniform1f(u.uOpacity, 1.0);

    for (const b of batches) {
      gl.uniform4fv(u.uColor, b.color);
      gl.bindVertexArray(b.vao);
      gl.drawArrays(gl.TRIANGLES, 0, b.count);
    }

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const x = Math.floor(px * dpr);
    const y = Math.floor(h - py * dpr);
    const pixel = new Uint8Array(4);
    if (x >= 0 && y >= 0 && x < w && y < h) {
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);

    const id = pixel[0] | (pixel[1] << 8) | (pixel[2] << 16);
    return id === 0 ? -1 : id - 1;
  }
}

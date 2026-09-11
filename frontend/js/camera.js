// Orbit camera with a plan-view mode.
//
// Two projections share one set of orbit parameters so switching between the
// 3D view and the 2D cadastral plan keeps the same target and heading - the
// user never loses their place, which matters when the whole point is comparing
// what the 2D record says with what the 3D record knows.

import { clamp, lookAt, multiply, orthographic, perspective, vec3 } from './math.js';

export class OrbitCamera {
  constructor() {
    this.target = vec3(0, 0, 0);
    this.distance = 300;
    this.azimuth = -0.9;          // radians, 0 = looking along +x
    this.elevation = 0.62;        // radians above the horizon
    this.fov = Math.PI / 4;
    this.near = 0.5;
    this.far = 8000;
    this.mode = 'perspective';    // 'perspective' | 'plan'
    this.minDistance = 6;
    this.maxDistance = 3000;
  }

  eye() {
    const ce = Math.cos(this.elevation);
    return vec3(
      this.target[0] + this.distance * ce * Math.cos(this.azimuth),
      this.target[1] + this.distance * ce * Math.sin(this.azimuth),
      this.target[2] + this.distance * Math.sin(this.elevation),
    );
  }

  viewProj(aspect) {
    if (this.mode === 'plan') {
      // straight down, north up; the orthographic half-height follows the
      // orbit distance so the zoom control means the same thing in both modes
      const halfH = this.distance * 0.42;
      const eye = vec3(this.target[0], this.target[1], this.target[2] + 1500);
      const view = lookAt(eye, this.target, vec3(0, 1, 0));
      return multiply(orthographic(halfH * aspect, halfH, 1, 4000), view);
    }
    const view = lookAt(this.eye(), this.target, vec3(0, 0, 1));
    return multiply(perspective(this.fov, aspect, this.near, this.far), view);
  }

  orbit(dx, dy) {
    this.azimuth -= dx * 0.007;
    // stop just short of the poles, where the up vector degenerates
    this.elevation = clamp(this.elevation + dy * 0.007, -1.45, 1.45);
  }

  pan(dx, dy, aspect) {
    // Pan in the camera's own screen plane, scaled so a pixel of mouse travel
    // moves the same amount of world under the cursor at any zoom level.
    const scale = (this.mode === 'plan' ? this.distance * 0.84 : this.distance)
      * Math.tan(this.fov / 2) * 2 / 600;
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    const rightX = -sa, rightY = ca;
    if (this.mode === 'plan') {
      this.target[0] -= dx * scale * 0.8;
      this.target[1] += dy * scale * 0.8;
      return;
    }
    const se = Math.sin(this.elevation);
    this.target[0] += (-dx * rightX + dy * ca * se) * scale;
    this.target[1] += (-dx * rightY + dy * sa * se) * scale;
    this.target[2] += dy * Math.cos(this.elevation) * scale;
  }

  zoom(delta) {
    this.distance = clamp(this.distance * Math.exp(delta * 0.0014),
                          this.minDistance, this.maxDistance);
  }

  frame(extent, { padding = 1.35 } = {}) {
    const cx = (extent.x_min + extent.x_max) / 2;
    const cy = (extent.y_min + extent.y_max) / 2;
    const cz = (extent.z_min + extent.z_max) / 2;
    this.target = vec3(cx, cy, cz);
    const span = Math.max(extent.x_max - extent.x_min,
                          extent.y_max - extent.y_min,
                          extent.z_max - extent.z_min);
    this.distance = clamp(span * padding, this.minDistance, this.maxDistance);
    this.maxDistance = Math.max(this.maxDistance, span * 4);
  }

  /** Ease the camera toward a target over a few frames. */
  flyTo(target, distance, { azimuth = null, elevation = null } = {}) {
    this._fly = {
      from: { t: Array.from(this.target), d: this.distance,
              a: this.azimuth, e: this.elevation },
      to: { t: Array.from(target), d: distance,
            a: azimuth === null ? this.azimuth : azimuth,
            e: elevation === null ? this.elevation : elevation },
      t0: performance.now(), ms: 620,
    };
  }

  update() {
    if (!this._fly) return false;
    const f = this._fly;
    const k = Math.min(1, (performance.now() - f.t0) / f.ms);
    // smoothstep, so the move starts and ends gently
    const s = k * k * (3 - 2 * k);
    for (let i = 0; i < 3; i++) {
      this.target[i] = f.from.t[i] + (f.to.t[i] - f.from.t[i]) * s;
    }
    this.distance = f.from.d + (f.to.d - f.from.d) * s;
    this.azimuth = f.from.a + (f.to.a - f.from.a) * s;
    this.elevation = f.from.e + (f.to.e - f.from.e) * s;
    if (k >= 1) this._fly = null;
    return true;
  }

  attach(canvas, onChange) {
    let dragging = null;
    let lastX = 0, lastY = 0;

    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      dragging = (e.button === 2 || e.shiftKey) ? 'pan' : 'orbit';
      lastX = e.clientX; lastY = e.clientY;
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
      if (dragging === 'pan' || this.mode === 'plan') this.pan(dx, dy, aspect);
      else this.orbit(dx, dy);
      this._fly = null;
      onChange();
    });

    const stop = (e) => {
      if (dragging) canvas.releasePointerCapture?.(e.pointerId);
      dragging = null;
    };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom(e.deltaY);
      this._fly = null;
      onChange();
    }, { passive: false });
  }
}

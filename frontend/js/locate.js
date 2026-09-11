// "Locate a ULPIN" - a cinematic zoom from orbit down to one flat.
//
// The sequence is not decoration bolted onto the identifier. It *reads* the
// identifier, field by field, and every stage of the zoom corresponds to one
// part of the number:
//
//     27        state      ->  admin code resolves
//     025       district   ->  admin code resolves
//     004       tehsil     ->  admin code resolves
//     tek92et..  geohash   ->  ten cell subdivisions, one per character
//     B04       stratum+level -> the storey lifts out of the stack
//     00DY      unit       -> the flat highlights
//
// So the animation is an argument: this number is self-locating. You can find
// the property from the identifier alone, with no database lookup - and the
// last ten seconds prove it on screen.
//
// Phases 1-3 render to a 2D overlay canvas (globe and geohash cascade).
// Phases 4-6 hand off to the existing WebGL renderer. The join is a cross-fade
// at the moment the geohash cell and the site footprint are the same size, so
// the two coordinate systems line up rather than cutting.

import { drawRings, loadBasemap } from './basemap.js';
import { cellSizeMetres, formatSpan, geohashBounds, prefixLadder } from './geohash.js';
import { ATTRIBUTION, drawTiles, prefetch } from './tiles.js';

const EARTH_R = 6371000;                 // mean radius, metres
const DEG = Math.PI / 180;

// Phase boundaries in seconds. Tuned for a pitch: long enough to read the
// labels, short enough that nobody shifts in their seat.
const T = {
  globe:   [0.0,  3.0],   // sphere rotates the target into view
  admin:   [3.0,  5.6],   // state / district / tehsil resolve
  geohash: [5.6,  9.8],   // ten cell subdivisions
  plan:    [9.8, 12.2],   // cross-fade into the site, top-down
  rise:   [12.2, 14.8],   // camera tilts from overhead into 3D
  target: [14.8, 18.0],   // fly to the flat, everything else dims
};
const TOTAL = T.target[1];

const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Progress through a named phase, 0 before it starts, 1 after it ends. */
function phase(now, name) {
  const [a, b] = T[name];
  return clamp01((now - a) / (b - a));
}

/**
 * Orthographic projection of a sphere.
 *
 * Used for the whole descent, not just the globe shot. As the radius grows the
 * projection flattens on its own - which is exactly what a real zoom does, so
 * there is never a moment where a "globe view" is swapped for a "map view".
 */
function project(lat, lon, lat0, lon0, R, cx, cy) {
  const p = lat * DEG, l = lon * DEG, p0 = lat0 * DEG, l0 = lon0 * DEG;
  const dl = l - l0;
  const cosc = Math.sin(p0) * Math.sin(p) + Math.cos(p0) * Math.cos(p) * Math.cos(dl);
  return {
    x: cx + R * Math.cos(p) * Math.sin(dl),
    y: cy - R * (Math.cos(p0) * Math.sin(p) - Math.sin(p0) * Math.cos(p) * Math.cos(dl)),
    visible: cosc > 0,
  };
}

/**
 * A 0-to-1 ramp as `v` travels from `a` to `b`.
 *
 * Used to cross-fade each basemap layer in over the scale range where it has
 * something to say: country borders while you are still above the subcontinent,
 * state boundaries on the way down, neither once the view is smaller than the
 * data's own resolution. Drawing 2 km-simplified coastline at street scale
 * would look like a mistake, so those layers are gone before you get there.
 */
function ramp(v, a, b) {
  return clamp01((v - a) / (b - a));
}

/** Graticule spacing that keeps roughly ten lines on screen at any zoom. */
function graticuleStep(spanDeg) {
  const ladder = [30, 15, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
                  0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001];
  for (const s of ladder) if (spanDeg / s <= 12) return s;
  return ladder[ladder.length - 1];
}

export class LocateSequence {
  /**
   * @param {object} deps  { canvas2d, renderer, camera, state, api, onPhase, onDone }
   */
  constructor(deps) {
    Object.assign(this, deps);
    this.running = false;
    // Each run carries its own generation token. A shared boolean is not
    // enough: a frame already queued by an old run fires *after* the next run
    // flips the flag back on, and would then keep animating with its own stale
    // start time - so two sequences drive the scene at once.
    this._gen = 0;
    this.ctx = this.canvas2d.getContext('2d');
  }

  cancel() {
    this._gen++;
    this.running = false;
    this.canvas2d.style.opacity = '0';
    this._osmPainted = false;
    this._mapVisible = false;
    this._restoreScene();
  }

  _restoreScene() {
    if (!this.renderer) return;
    for (const layer of ['terrain', 'parcels', 'buildings', 'storeys', 'units', 'infra']) {
      this.renderer.setLayerOpacity(layer, 1.0);
    }
    this.renderer.sectionZ = 1e6;
  }

  /**
   * Run the full sequence for a ULPIN.
   *
   * Resolves the identifier first: everything from the globe down to the
   * geohash cell comes from the number itself, so an identifier that is *not*
   * in this register still animates all the way to its cell - which is the
   * point worth making about self-locating identifiers.
   */
  async run(ulpinText) {
    this.cancel();

    let info;
    try {
      info = await this.api.ulpin(ulpinText);
    } catch (err) {
      throw new Error(err.message);
    }

    // Real coastlines and borders, bundled with the app. A missing basemap is
    // not fatal - the descent still works off the graticule and the geohash
    // cells, which is what carries the actual argument.
    if (!this.basemap) {
      try {
        this.basemap = await loadBasemap();
      } catch (err) {
        console.warn('basemap unavailable, falling back to graticule:', err.message);
      }
    }

    const u = info.ulpin;
    const ladder = prefixLadder(u.geohash);
    const [lat, lon] = [u.centroid.lat, u.centroid.lon];
    const object = info.found ? info.object : null;

    // Warm the street tiles while the globe shot plays, so the handover to
    // Mercator does not land on blank imagery.
    if (this.provider === 'osm') {
      try { prefetch(lat, lon); } catch { /* offline is fine */ }
    }

    this.running = true;
    this.canvas2d.style.opacity = '1';

    // --- zoom schedule ---------------------------------------------------
    // The descent runs from globe scale to the final cell, exponentially, so
    // each geohash character occupies a roughly equal slice of screen time.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const resize = () => {
      const w = this.canvas2d.clientWidth, h = this.canvas2d.clientHeight;
      this.canvas2d.width = Math.max(1, Math.floor(w * dpr));
      this.canvas2d.height = Math.max(1, Math.floor(h * dpr));
    };
    resize();

    const start = performance.now();
    const self = this;
    const gen = this._gen;          // this run's token; a later run invalidates it

    return new Promise((resolve) => {
      function frame() {
        if (!self.running || self._gen !== gen) { resolve(false); return; }
        const now = (performance.now() - start) / 1000;
        resize();
        self._draw(now, { lat, lon, u, ladder, object, info });
        self._drive3D(now, { object });

        if (self.onPhase) self.onPhase(self._phaseName(now), now / TOTAL);

        if (now >= TOTAL) {
          self.running = false;
          self.canvas2d.style.opacity = '0';
          if (self.onDone) self.onDone(info);
          resolve(true);
          return;
        }
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  _phaseName(now) {
    for (const [name, [a, b]] of Object.entries(T)) {
      if (now >= a && now < b) return name;
    }
    return now >= TOTAL ? 'done' : 'globe';
  }

  // --- 2D overlay: globe, admin codes, geohash cascade -------------------
  _draw(now, ctxData) {
    const { lat, lon, u, ladder, object, info } = ctxData;
    const ctx = this.ctx;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = this.canvas2d.width, H = this.canvas2d.height;
    const cx = W / 2, cy = H / 2;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // the overlay fades out as the 3D scene takes over
    const planP = phase(now, 'plan');
    const overlayAlpha = 1 - easeInOut(planP);
    if (overlayAlpha <= 0.001) return;
    ctx.globalAlpha = overlayAlpha;

    ctx.fillStyle = '#0b0d12';
    ctx.fillRect(0, 0, W, H);

    // --- radius schedule -------------------------------------------------
    const globeR = 0.34 * Math.min(W, H);
    const gP = phase(now, 'globe');
    const aP = phase(now, 'admin');
    const hP = phase(now, 'geohash');

    // span that should fill ~60% of the viewport, in metres
    const spanAtGlobe = 0.6 * H * EARTH_R / globeR;
    const finalSpan = Math.max(ladder[ladder.length - 1].width * 2.4, 3);
    const descent = clamp01((aP * 0.42) + (hP * 0.58));   // admin then geohash
    const span = spanAtGlobe * Math.pow(finalSpan / spanAtGlobe, easeInOut(descent));
    const R = descent > 0 ? (0.6 * H * EARTH_R / span) : globeR;

    // the globe spins the target to the centre during phase 1
    const spin = easeInOut(gP);
    const lon0 = lon - (1 - spin) * 58;
    const lat0 = lat - (1 - spin) * 22;

    const proj = (la, lo) => project(la, lo, lat0, lon0, R, cx, cy);
    const spanDeg = (span / EARTH_R) / DEG;
    const view = [lon0 - spanDeg * 1.6, lat0 - spanDeg * 1.6,
                  lon0 + spanDeg * 1.6, lat0 + spanDeg * 1.6];
    const spanKm = span / 1000;

    // --- ocean sphere ----------------------------------------------------
    // Only while the limb is actually on screen; once the view is smaller than
    // the planet there is no disc left to draw.
    const sphereAlpha = 1 - ramp(descent, 0.10, 0.34);
    if (sphereAlpha > 0.01) {
      ctx.save();
      ctx.globalAlpha = overlayAlpha * sphereAlpha;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      const g = ctx.createRadialGradient(cx - R * 0.32, cy - R * 0.36, R * 0.08, cx, cy, R);
      g.addColorStop(0, '#16344a');
      g.addColorStop(0.72, '#0e2334');
      g.addColorStop(1, '#081722');
      ctx.fillStyle = g;
      ctx.fill();
      ctx.restore();
    }

    // --- OpenStreetMap tiles ---------------------------------------------
    // Mercator takes over from the sphere once the view is small enough that
    // the two projections agree locally. Below ~400 km the difference across
    // the viewport is far under a pixel, so the swap is invisible.
    let tilesPainted = false;
    if (this.provider === 'osm') {
      const tileA = 1 - ramp(spanKm, 120, 420);
      if (tileA > 0.01) {
        tilesPainted = drawTiles(ctx, {
          lat, lon, span, W, H, dpr, alpha: overlayAlpha * tileA,
        });
      }
    }
    // A painted map owes OSM its credit, and needs the text scrim so the
    // captions stay readable over a light street map.
    this._osmPainted = tilesPainted;
    this._mapVisible = tilesPainted;
    this._tilesPainted = tilesPainted;

    // --- real geography ---------------------------------------------------
    const bm = this.basemap;
    if (bm) {
      const limb = sphereAlpha > 0.01 ? R : undefined;

      // World land carries the globe shot and is gone by the time India fills
      // the frame. `ramp` rises with its input, and here the input is the view
      // span - so this is 1 when zoomed out and 0 when zoomed in, which is the
      // way round this layer wants.
      const worldA = ramp(spanKm, 900, 2600);
      if (worldA > 0.01) {
        ctx.save();
        ctx.globalAlpha = overlayAlpha * worldA;
        for (const f of bm.world) {
          const home = f.in === 1;
          drawRings(ctx, f.rings, proj, {
            fill: tilesPainted ? null : (home ? '#39493f' : '#2b3a37'),
            stroke: home ? 'rgba(242,193,78,0.55)' : 'rgba(150,175,190,0.35)',
            lineWidth: (home ? 1.6 : 0.9) * dpr,
            view, limbR: limb, cx, cy,
          });
        }
        ctx.restore();
      }

      // Indian states: fade in as the country fills the frame, out again once
      // the view is tighter than the 2 km simplification they were built at
      const appears = 1 - ramp(spanKm, 2200, 5200);   // in as the country fills
      const survives = ramp(spanKm, 6, 22);           // out below its own detail
      const stateA = Math.min(appears, survives);
      if (stateA > 0.01) {
        ctx.save();
        ctx.globalAlpha = overlayAlpha * stateA;
        drawRings(ctx, bm.india, proj, {
          fill: tilesPainted ? null : '#33423a', stroke: 'rgba(242,193,78,0.5)',
          lineWidth: 1.8 * dpr, view, limbR: limb, cx, cy,
        });
        for (const f of bm.states) {
          const mh = f.mh === 1;
          drawRings(ctx, f.rings, proj, {
            fill: (mh && !tilesPainted) ? 'rgba(242,193,78,0.14)' : null,
            stroke: mh ? 'rgba(242,193,78,0.85)' : 'rgba(160,185,200,0.38)',
            lineWidth: (mh ? 2 : 0.9) * dpr,
            view, limbR: limb, cx, cy,
          });
        }
        ctx.restore();
      }

      // A plain ground tone once the vector basemap has nothing left to offer,
      // so the final approach is not happening over empty black. Skipped when
      // tiles are carrying the view - they already have ground under them.
      const groundA = tilesPainted ? 0 : ramp(spanKm, 14, 4);
      if (groundA > 0.01) {
        ctx.save();
        ctx.globalAlpha = overlayAlpha * groundA * 0.9;
        ctx.fillStyle = '#2f3a33';
        ctx.fillRect(0, 0, W, H);
        ctx.restore();
      }
    }

    // --- graticule -------------------------------------------------------
    // Dropped once tiles are drawing: a lat/lon grid over a street map is
    // clutter, and the map already carries its own sense of scale.
    const step = graticuleStep(Math.max(spanDeg, 0.0002));
    const gratA = tilesPainted ? 0.0 : 0.20;
    ctx.strokeStyle = `rgba(150,180,200,${gratA})`;
    ctx.lineWidth = 1 * dpr;

    const drawArc = (points) => {
      ctx.beginPath();
      let pen = false;
      for (const p of points) {
        if (!p.visible) { pen = false; continue; }
        if (Math.abs(p.x - cx) > W * 2 || Math.abs(p.y - cy) > H * 2) { pen = false; continue; }
        if (!pen) { ctx.moveTo(p.x, p.y); pen = true; } else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    };

    const latLo = Math.max(-90, Math.floor((lat0 - spanDeg) / step) * step);
    const latHi = Math.min(90, Math.ceil((lat0 + spanDeg) / step) * step);
    const lonLo = Math.floor((lon0 - spanDeg) / step) * step;
    const lonHi = Math.ceil((lon0 + spanDeg) / step) * step;

    for (let la = latLo; la <= latHi + 1e-9; la += step) {
      const pts = [];
      for (let i = 0; i <= 64; i++) {
        const lo = lonLo + (lonHi - lonLo) * (i / 64);
        pts.push(project(la, lo, lat0, lon0, R, cx, cy));
      }
      drawArc(pts);
    }
    for (let lo = lonLo; lo <= lonHi + 1e-9; lo += step) {
      const pts = [];
      for (let i = 0; i <= 64; i++) {
        const la = Math.max(-89.9, Math.min(89.9, latLo + (latHi - latLo) * (i / 64)));
        pts.push(project(la, lo, lat0, lon0, R, cx, cy));
      }
      drawArc(pts);
    }

    // equator and the Tropic of Cancer, which runs through India - drawn a
    // little brighter so the globe shot reads as a real coordinate frame
    if (descent < 0.3) {
      ctx.strokeStyle = 'rgba(150,185,210,0.55)';
      ctx.lineWidth = 1.4 * dpr;
      for (const special of [0, 23.4366]) {
        const pts = [];
        for (let i = 0; i <= 128; i++) {
          pts.push(project(special, -180 + 360 * (i / 128), lat0, lon0, R, cx, cy));
        }
        drawArc(pts);
      }
    }

    // --- geohash cell cascade --------------------------------------------
    // Draw every prefix whose cell is not yet larger than the screen. Older
    // (bigger) cells stay faintly visible so the nesting is legible.
    if (descent > 0.02) {
      for (let i = 0; i < ladder.length; i++) {
        const cell = ladder[i];
        const rel = cell.width / span;
        if (rel > 6 || rel < 0.02) continue;
        const b = cell.bounds;
        const corners = [
          project(b.latMin, b.lonMin, lat0, lon0, R, cx, cy),
          project(b.latMin, b.lonMax, lat0, lon0, R, cx, cy),
          project(b.latMax, b.lonMax, lat0, lon0, R, cx, cy),
          project(b.latMax, b.lonMin, lat0, lon0, R, cx, cy),
        ];
        if (!corners.every((p) => p.visible)) continue;

        const isCurrent = rel > 0.35 && rel < 3.2;
        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y);
        for (let k = 1; k < 4; k++) ctx.lineTo(corners[k].x, corners[k].y);
        ctx.closePath();
        ctx.strokeStyle = isCurrent ? 'rgba(242,193,78,0.95)' : 'rgba(242,193,78,0.22)';
        ctx.lineWidth = (isCurrent ? 2.2 : 1) * dpr;
        ctx.stroke();

        if (isCurrent) {
          ctx.fillStyle = 'rgba(242,193,78,0.07)';
          ctx.fill();
          const label = cell.prefix;
          ctx.font = `600 ${13 * dpr}px ui-monospace, "Cascadia Mono", monospace`;
          ctx.fillStyle = 'rgba(242,193,78,0.95)';
          ctx.textAlign = 'left';
          ctx.fillText(label, corners[3].x + 6 * dpr, corners[3].y - 8 * dpr);
          ctx.font = `${11 * dpr}px ui-monospace, monospace`;
          ctx.fillStyle = 'rgba(200,215,228,0.8)';
          ctx.fillText(`${formatSpan(cell.width)} × ${formatSpan(cell.height)}`,
                       corners[3].x + 6 * dpr, corners[3].y + 8 * dpr);
        }
      }
    }

    // --- target marker ----------------------------------------------------
    const m = project(lat, lon, lat0, lon0, R, cx, cy);
    if (m.visible) {
      const pulse = 1 + 0.35 * Math.sin(now * 5);
      ctx.beginPath();
      ctx.arc(m.x, m.y, 5 * dpr * pulse, 0, Math.PI * 2);
      ctx.fillStyle = '#f2c14e';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(m.x, m.y, 13 * dpr * pulse, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(242,193,78,0.5)';
      ctx.lineWidth = 1.4 * dpr;
      ctx.stroke();
    }

    // --- captions ---------------------------------------------------------
    this._captions(ctx, now, { W, H, dpr, u, span, object, info });
    ctx.globalAlpha = 1;
  }

  _captions(ctx, now, { W, H, dpr, u, span, object, info }) {
    const pad = 28 * dpr;

    // OSM's default style is light, and every caption here is set for a dark
    // ground. Rather than dimming the whole map - which would waste the detail
    // we just went to the trouble of loading - lay a gradient scrim only under
    // the text bands, top and bottom.
    if (this._mapVisible) {
      const band = 96 * dpr;
      const top = ctx.createLinearGradient(0, 0, 0, band);
      top.addColorStop(0, 'rgba(11,13,18,0.86)');
      top.addColorStop(1, 'rgba(11,13,18,0)');
      ctx.fillStyle = top;
      ctx.fillRect(0, 0, W, band);

      const bot = ctx.createLinearGradient(0, H - band, 0, H);
      bot.addColorStop(0, 'rgba(11,13,18,0)');
      bot.addColorStop(1, 'rgba(11,13,18,0.86)');
      ctx.fillStyle = bot;
      ctx.fillRect(0, H - band, W, band);
    }

    ctx.textAlign = 'left';

    // the full identifier, always on screen, fields lighting up in turn
    const gP = phase(now, 'globe'), aP = phase(now, 'admin'), hP = phase(now, 'geohash');
    const rP = phase(now, 'rise'), tP = phase(now, 'target');

    const parts = [
      { text: u.state, lit: aP > 0.15 },
      { text: u.district, lit: aP > 0.4 },
      { text: u.tehsil, lit: aP > 0.7 },
      { text: u.geohash, lit: hP > 0.05 },
      { text: u.stratum + String(u.level).padStart(2, '0'), lit: rP > 0.3 },
      { text: String(u.unit).padStart(4, '0'), lit: tP > 0.2 },
    ];

    ctx.font = `600 ${19 * dpr}px ui-monospace, "Cascadia Mono", monospace`;
    let x = pad;
    const y = H - pad;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      ctx.fillStyle = p.lit ? '#f2c14e' : 'rgba(140,155,172,0.4)';
      ctx.fillText(p.text, x, y);
      x += ctx.measureText(p.text).width;
      if (i < parts.length - 1) {
        ctx.fillStyle = 'rgba(140,155,172,0.35)';
        ctx.fillText('-', x, y);
        x += ctx.measureText('-').width;
      }
    }

    // what the current stage of the number is telling us
    let title = '', sub = '';
    if (gP < 1) {
      title = 'Reading the identifier';
      sub = 'position decoded from the number alone — no database lookup';
    } else if (aP < 1) {
      const j = info.jurisdiction || {};
      title = 'Administrative codes';
      sub = [j.state_name, j.district_name, j.tehsil_name].filter(Boolean).join('  ›  ')
            || `state ${u.state} › district ${u.district} › tehsil ${u.tehsil}`;
    } else if (hP < 1) {
      title = 'Geohash — each character halves the cell';
      sub = `now inside ${formatSpan(span)}`;
    } else if (rP < 1) {
      title = 'The site';
      sub = 'every registered volume in this column';
    } else if (tP < 1) {
      title = u.stratum_label || 'Locating the unit';
      sub = u.description || '';
    } else {
      title = object ? (object.name || object.object_id) : 'Not in this register';
      sub = object
        ? `${object.owner || 'owner not recorded'}  ·  ${object.area_m2} m²  ·  ${object.volume_m3} m³`
        : 'the identifier still locates its cell — nothing is registered there yet';
    }

    ctx.font = `600 ${26 * dpr}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.fillStyle = '#dfe4ec';
    ctx.fillText(title, pad, pad + 26 * dpr);
    ctx.font = `${14 * dpr}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.fillStyle = 'rgba(160,175,195,0.9)';
    ctx.fillText(sub, pad, pad + 50 * dpr);

    // scale readout, top right
    ctx.textAlign = 'right';
    ctx.font = `${13 * dpr}px ui-monospace, monospace`;
    ctx.fillStyle = 'rgba(160,175,195,0.75)';
    ctx.fillText(`${u.centroid.lat.toFixed(6)}, ${u.centroid.lon.toFixed(6)}`,
                 W - pad, pad + 18 * dpr);
    ctx.fillText(`view span  ${formatSpan(span)}`, W - pad, pad + 38 * dpr);

    // OSM is ODbL-licensed: credit is required wherever its tiles are shown,
    // so this appears exactly when a tile has actually been painted.
    if (this._osmPainted) {
      ctx.font = `${11 * dpr}px system-ui, -apple-system, sans-serif`;
      ctx.fillStyle = 'rgba(190,205,220,0.75)';
      ctx.fillText(ATTRIBUTION, W - pad, H - pad * 0.55);
    }
  }

  // --- 3D scene choreography --------------------------------------------
  _drive3D(now, { object }) {
    if (!this.renderer || !this.camera) return;
    const planP = phase(now, 'plan');
    const riseP = phase(now, 'rise');
    const targP = phase(now, 'target');
    const site = this.state.site;
    if (!site) return;

    if (planP <= 0) return;             // still fully behind the overlay

    // Phase 4: the site, seen from directly overhead, so it reads as a plan
    // and matches the flat geohash cell the overlay just left us on.
    if (planP > 0 && riseP <= 0) {
      this.camera.frame(site.extent, { padding: 1.5 });
      this.camera.elevation = 1.42;      // just short of vertical
      this.camera.azimuth = -0.9;
      this.renderer.setLayerOpacity('terrain', 1);
      this.renderer.setLayerOpacity('parcels', 1);
      this.renderer.setLayerOpacity('buildings', 1);
      this.renderer.sectionZ = 1e6;
    }

    // Phase 5: tilt down into a three-quarter view - the moment the flat plan
    // becomes a set of solids.
    if (riseP > 0) {
      const e = easeInOut(riseP);
      this.camera.elevation = 1.42 - e * 0.86;      // ~81 deg -> ~32 deg
      this.camera.azimuth = -0.9 + e * 0.55;
    }

    // Phase 6: close on the unit and drop everything that is not it.
    if (targP > 0 && object) {
      const e = easeInOut(targP);
      const c = object.centroid;
      const span = Math.max(
        object.bbox[3] - object.bbox[0],
        object.bbox[4] - object.bbox[1],
        object.bbox[5] - object.bbox[2]);

      const from = this._riseTarget || (this._riseTarget = [...this.camera.target]);
      const fromD = this._riseDist || (this._riseDist = this.camera.distance);
      const toD = Math.max(span * 3.4, 26);
      this.camera.target[0] = from[0] + (c.x - from[0]) * e;
      this.camera.target[1] = from[1] + (c.y - from[1]) * e;
      this.camera.target[2] = from[2] + (c.z - from[2]) * e;
      this.camera.distance = fromD + (toD - fromD) * e;

      const fade = 1 - 0.82 * e;
      this.renderer.setLayerOpacity('terrain', fade);
      this.renderer.setLayerOpacity('parcels', fade);
      this.renderer.setLayerOpacity('buildings', fade * 0.5);
    } else {
      this._riseTarget = null;
      this._riseDist = null;
    }

    this.state.needsRedraw = true;
  }
}

export const LOCATE_DURATION = TOTAL;

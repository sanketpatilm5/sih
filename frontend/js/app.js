// Bhoomi3D viewer.
//
// Assembles the register into a WebGL scene and wires up the interactions the
// demonstration needs: pick a building, isolate a floor, select a flat, read
// its 3D-ULPIN and volume, probe the vertical column at a point, and jump to
// each validation finding in turn.

import { api } from './api.js';
import { OrbitCamera } from './camera.js';
import { buildPrism, buildPrismEdges, buildTerrain } from './geom.js';
import { LocateSequence } from './locate.js';
import { Renderer } from './renderer.js';
import { vec3 } from './math.js';

// --- palette -------------------------------------------------------------------
// Colour carries meaning here: stratum (above ground / at grade / below ground)
// is the primary axis, because that is the distinction the whole project is
// about. Provenance is shown separately, in the inspector, rather than by hue.
const STYLE = {
  parcel:         { color: [0.36, 0.55, 0.38, 0.55], layer: 'parcels',  sortKey: 1 },
  building:       { color: [0.60, 0.63, 0.70, 0.20], layer: 'buildings', sortKey: 3 },
  storey:         { color: [0.45, 0.60, 0.82, 0.30], layer: 'storeys',  sortKey: 4 },
  unit:           { color: [0.85, 0.72, 0.45, 1.00], layer: 'units',    sortKey: 6 },
  infrastructure: { color: [0.90, 0.36, 0.32, 0.95], layer: 'infra',    sortKey: 5 },
  air_rights:     { color: [0.42, 0.78, 0.90, 0.12], layer: 'air',      sortKey: 2 },
};
const UNIT_USE_COLOR = {
  parking:      [0.55, 0.57, 0.62, 1.0],
  commercial:   [0.52, 0.70, 0.88, 1.0],
  shop:         [0.52, 0.70, 0.88, 1.0],
  residential:  [0.88, 0.74, 0.46, 1.0],
  unsurveyed:   [0.70, 0.45, 0.75, 1.0],
};
const INFRA_COLOR = {
  metro: [0.93, 0.33, 0.36, 0.95],
  water: [0.30, 0.63, 0.92, 0.95],
  storm: [0.55, 0.45, 0.85, 0.95],
  power: [0.96, 0.72, 0.25, 0.95],
};

const state = {
  site: null,
  objects: [],
  byId: new Map(),
  idToIndex: new Map(),
  indexToId: [],
  selected: null,
  hover: -1,
  findings: [],
  metrics: null,
  levelFilter: null,
  needsRedraw: true,
};

let renderer, camera, canvas;

// --- helpers -------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmt = (v, d = 2) =>
  v === null || v === undefined ? '-' : Number(v).toLocaleString('en-IN',
    { minimumFractionDigits: d, maximumFractionDigits: d });

function setStatus(msg, kind = '') {
  const n = $('#status');
  n.textContent = msg;
  n.className = 'status ' + kind;
}

// --- scene assembly ------------------------------------------------------------
function styleFor(obj) {
  const base = STYLE[obj.kind] || STYLE.building;
  const s = { ...base, color: [...base.color] };
  if (obj.kind === 'unit') {
    const c = UNIT_USE_COLOR[obj.use] || UNIT_USE_COLOR.residential;
    s.color = [...c];
    // basements read as a cooler, darker version of whatever they are used for
    if (obj.level < 0) s.color = s.color.map((v, i) => (i < 3 ? v * 0.62 : v));
  } else if (obj.kind === 'infrastructure') {
    s.color = [...(INFRA_COLOR[obj.use] || INFRA_COLOR.metro)];
  }
  return s;
}

function buildScene() {
  renderer.clear();
  state.idToIndex.clear();
  state.indexToId = [];

  // Group by layer so each layer is one draw call rather than one per object.
  // The per-vertex object id is what keeps individual objects selectable.
  const groups = new Map();
  const push = (key, style) => {
    if (!groups.has(key)) {
      groups.set(key, { mesh: { positions: [], normals: [], ids: [] }, style });
    }
    return groups.get(key);
  };

  for (const obj of state.objects) {
    if (!obj.geometry) continue;
    const idx = state.indexToId.length;
    state.idToIndex.set(obj.object_id, idx);
    state.indexToId.push(obj.object_id);

    const style = styleFor(obj);
    const key = `${style.layer}|${style.color.join(',')}`;
    const g = push(key, style);

    const parts = obj.geometry.type === 'CompositeSolid'
      ? obj.geometry.parts : [obj.geometry];
    for (const p of parts) {
      buildPrism(p.rings, p.z_min, p.z_max, g.mesh, idx);
    }
  }

  for (const { mesh, style } of groups.values()) {
    renderer.addBatch(mesh, {
      color: style.color, layer: style.layer, sortKey: style.sortKey,
    });
  }

  applyLevelFilter();
  state.needsRedraw = true;
}

async function addTerrain() {
  try {
    const grid = await api.terrain('dem');

    // Try for real aerial imagery to drape over the terrain. It is what turns
    // the scene from a diagram into a recognisable place - but it needs the
    // network, so a failure here is expected and simply leaves the ground its
    // plain colour.
    let ground = null;
    try {
      ground = await api.groundImage();
    } catch (err) {
      console.info('no aerial imagery, using plain ground:', err.message);
    }

    const mesh = { positions: [], normals: [], ids: [], uvs: [] };
    buildTerrain(grid, mesh, -1, ground ? ground.extent_m : null);
    const batch = renderer.addBatch(mesh, {
      // White under a texture: the shader multiplies, so any tint here would
      // stain the photograph.
      color: ground ? [1, 1, 1, 1] : [0.30, 0.33, 0.30, 1.0],
      layer: 'terrain', pickable: false, sortKey: 0,
    });

    if (ground && batch) {
      const img = new Image();
      img.onload = () => {
        renderer.setBatchTexture(batch, img);
        state.needsRedraw = true;
        setStatus(`Ground imagery: ${ground.attribution}`, 'ok');
      };
      img.onerror = () => console.warn('ground image failed to decode');
      img.src = ground.url;
    }
    state.needsRedraw = true;
  } catch (err) {
    console.warn('terrain unavailable:', err.message);
  }
}

function selectionEdges(obj) {
  const existing = renderer.lineBatches.filter((b) => b.layer === 'selection');
  for (const b of existing) {
    renderer.gl.deleteVertexArray(b.vao);
    renderer.gl.deleteBuffer(b.buffer);
  }
  renderer.lineBatches = renderer.lineBatches.filter((b) => b.layer !== 'selection');
  if (!obj || !obj.geometry) return;

  const pts = [];
  const parts = obj.geometry.type === 'CompositeSolid'
    ? obj.geometry.parts : [obj.geometry];
  for (const p of parts) buildPrismEdges(p.rings, p.z_min, p.z_max, pts);
  renderer.addLines(pts, { color: [1.0, 0.82, 0.25, 0.95], layer: 'selection' });
}

function highlightFinding(finding) {
  renderer.lineBatches = renderer.lineBatches.filter((b) => {
    if (b.layer !== 'finding') return true;
    renderer.gl.deleteVertexArray(b.vao);
    renderer.gl.deleteBuffer(b.buffer);
    return false;
  });
  if (!finding || !finding.geometry || !finding.geometry.rings?.length) return;
  const pts = [];
  buildPrismEdges(finding.geometry.rings, finding.geometry.z_min,
                  finding.geometry.z_max, pts);
  renderer.addLines(pts, { color: [1.0, 0.28, 0.30, 1.0], layer: 'finding' });
}

// --- level isolation -----------------------------------------------------------
function applyLevelFilter() {
  // Rebuilding batches per level would be wasteful, so the filter works by
  // toggling whole layers and letting the cutaway plane do the vertical part.
  const lvl = state.levelFilter;
  const label = lvl === null ? 'all levels' :
    (lvl < 0 ? `basement ${Math.abs(lvl)}` : lvl === 0 ? 'ground floor' : `floor ${lvl}`);
  $('#levelLabel').textContent = label;

  if (lvl === null) {
    renderer.sectionZ = 1e6;
    for (const b of renderer.batches) {
      if (b.layer === 'units' || b.layer === 'storeys') b.visible =
        $(`#layer-${b.layer}`).checked;
    }
    state.needsRedraw = true;
    return;
  }

  // isolate: cut away everything above the top of the chosen storey
  const storeys = state.objects.filter(
    (o) => o.kind === 'storey' && o.level === lvl);
  if (storeys.length) {
    renderer.sectionZ = Math.max(...storeys.map((s) => s.z_max)) + 0.05;
  }
  state.needsRedraw = true;
}

// --- inspector -----------------------------------------------------------------
function renderInspector(obj) {
  const box = $('#inspector');
  box.innerHTML = '';
  if (!obj) {
    box.appendChild(el('p', 'muted',
      'Click any volume in the scene, or search for a ULPIN, to inspect it.'));
    return;
  }

  const head = el('div', 'insp-head');
  head.appendChild(el('div', 'insp-kind', obj.kind.replace('_', ' ')));
  head.appendChild(el('h2', null, obj.name || obj.object_id));
  if (obj.owner) head.appendChild(el('div', 'insp-owner', obj.owner));
  box.appendChild(head);

  if (obj.ulpin) {
    const u = el('div', 'ulpin-card');
    u.appendChild(el('div', 'ulpin-label', '3D-ULPIN'));
    const code = el('div', 'ulpin-code', obj.ulpin);
    code.title = 'Click to copy';
    code.addEventListener('click', () => {
      navigator.clipboard?.writeText(obj.ulpin);
      setStatus('ULPIN copied to clipboard', 'ok');
    });
    u.appendChild(code);
    const d = obj.ulpin_detail;
    if (d) {
      u.appendChild(el('div', 'ulpin-desc', d.description));
      const grid = el('div', 'ulpin-parts');
      const parts = [
        ['State', d.state], ['District', d.district], ['Tehsil', d.tehsil],
        ['Geohash', d.geohash], ['Stratum', d.stratum + ' - ' + d.stratum_label],
        ['Level', d.signed_level], ['Unit seq', d.unit], ['Check', d.check],
      ];
      for (const [k, v] of parts) {
        const row = el('div', 'ulpin-part');
        row.appendChild(el('span', 'k', k));
        row.appendChild(el('span', 'v', String(v)));
        grid.appendChild(row);
      }
      u.appendChild(grid);
    }
    box.appendChild(u);
  }

  const facts = [
    ['Object ID', obj.object_id],
    ['Use', obj.use || '-'],
    ['Level', obj.level],
    ['Stratum', obj.stratum_label || '-'],
    ['Plan area', fmt(obj.area_m2) + ' m²'],
    ['Volume', fmt(obj.volume_m3) + ' m³'],
    ['Height', fmt(obj.height_m) + ' m'],
    ['Z range', `${fmt(obj.z_min)} – ${fmt(obj.z_max)} m`],
    ['Centroid', obj.centroid
      ? `${obj.centroid.lat.toFixed(6)}, ${obj.centroid.lon.toFixed(6)}` : '-'],
  ];
  const tbl = el('div', 'facts');
  for (const [k, v] of facts) {
    const row = el('div', 'fact');
    row.appendChild(el('span', 'k', k));
    row.appendChild(el('span', 'v', String(v)));
    tbl.appendChild(row);
  }
  box.appendChild(tbl);

  const prov = el('div', 'prov ' + (obj.authoritative ? 'auth' : 'inferred'));
  prov.appendChild(el('span', 'badge', obj.provenance.replace('_', ' ')));
  prov.appendChild(el('span', null, obj.authoritative
    ? 'Boundary from a surveyed or approved source.'
    : `Machine-derived boundary, confidence ${fmt(obj.confidence, 2)}. Requires ground verification before publication.`));
  box.appendChild(prov);

  if (obj.findings?.length) {
    box.appendChild(el('h3', null, `Findings (${obj.findings.length})`));
    for (const f of obj.findings) {
      const card = el('div', 'finding sev-' + f.severity);
      card.appendChild(el('div', 'f-title', f.title));
      card.appendChild(el('div', 'f-detail', f.detail));
      box.appendChild(card);
    }
  }

  if (obj.ancestors?.length) {
    const nav = el('div', 'breadcrumbs');
    for (const a of [...obj.ancestors].reverse()) {
      const b = el('button', 'crumb', a);
      b.addEventListener('click', () => selectObject(a));
      nav.appendChild(b);
    }
    box.appendChild(nav);
  }

  if (obj.children?.length) {
    box.appendChild(el('h3', null, `Contains (${obj.children.length})`));
    const list = el('div', 'child-list');
    for (const c of obj.children.slice(0, 60)) {
      const b = el('button', 'child', c.split('/').pop());
      b.title = c;
      b.addEventListener('click', () => selectObject(c));
      list.appendChild(b);
    }
    box.appendChild(list);
  }
}

async function selectObject(objectId, { fly = true } = {}) {
  try {
    const obj = await api.object(objectId);
    state.selected = obj;
    selectionEdges(obj);
    renderInspector(obj);
    if (fly) {
      const c = obj.centroid;
      const span = Math.max(obj.bbox[3] - obj.bbox[0], obj.bbox[4] - obj.bbox[1],
                            obj.bbox[5] - obj.bbox[2]);
      camera.flyTo(vec3(c.x, c.y, c.z), Math.max(span * 3.2, 22));
    }
    state.needsRedraw = true;
  } catch (err) {
    setStatus('Could not load ' + objectId + ': ' + err.message, 'err');
  }
}

// --- column probe --------------------------------------------------------------
async function probeColumn(x, y) {
  try {
    const res = await api.column({ x, y });
    const panel = $('#columnPanel');
    panel.innerHTML = '';
    panel.appendChild(el('h3', null, 'Vertical column'));
    panel.appendChild(el('p', 'muted',
      `at ${res.query.lat.toFixed(6)}, ${res.query.lon.toFixed(6)}  ` +
      `(${fmt(res.query.x)}, ${fmt(res.query.y)} m local)`));

    if (!res.count) {
      panel.appendChild(el('p', 'muted', 'Nothing registered at this position.'));
      return;
    }

    // draw the stack as a scale strip, which makes the vertical structure of
    // ownership legible at a glance in a way a list cannot
    const zs = res.column.flatMap((o) => o.span_m);
    const zMin = Math.min(...zs), zMax = Math.max(...zs);
    const span = Math.max(zMax - zMin, 1);

    const strip = el('div', 'column-strip');
    for (const o of res.column) {
      const band = el('div', 'col-band k-' + o.kind);
      const top = (1 - (o.span_m[1] - zMin) / span) * 100;
      const h = ((o.span_m[1] - o.span_m[0]) / span) * 100;
      band.style.top = top + '%';
      band.style.height = Math.max(h, 0.8) + '%';
      band.title = `${o.object_id}  ${o.span_m[0]} – ${o.span_m[1]} m`;
      band.addEventListener('click', () => selectObject(o.object_id));
      strip.appendChild(band);
    }
    const wrap = el('div', 'column-wrap');
    wrap.appendChild(strip);
    const scale = el('div', 'column-scale');
    scale.appendChild(el('span', null, fmt(zMax, 1) + ' m'));
    scale.appendChild(el('span', null, '0'));
    scale.appendChild(el('span', null, fmt(zMin, 1) + ' m'));
    wrap.appendChild(scale);
    panel.appendChild(wrap);

    const list = el('div', 'column-list');
    for (const o of res.column) {
      const row = el('button', 'col-row k-' + o.kind);
      row.appendChild(el('span', 'c-kind', o.kind.replace('_', ' ')));
      row.appendChild(el('span', 'c-name', o.name || o.object_id));
      row.appendChild(el('span', 'c-span',
        `${fmt(o.span_m[0], 1)} – ${fmt(o.span_m[1], 1)} m`));
      row.addEventListener('click', () => selectObject(o.object_id));
      list.appendChild(row);
    }
    panel.appendChild(list);
    showTab('column');
  } catch (err) {
    setStatus('Column query failed: ' + err.message, 'err');
  }
}

/**
 * Un-project a click into the world, then probe the column there.
 *
 * We intersect the view ray with the horizontal plane through the camera
 * target rather than with the terrain: it is stable regardless of what the
 * cursor happens to be over, and the probe is a plan-position question anyway.
 */
function screenToGround(px, py) {
  const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
  const vp = camera.viewProj(aspect);
  const inv = invertMat(vp);
  const ndcX = (px / canvas.clientWidth) * 2 - 1;
  const ndcY = 1 - (py / canvas.clientHeight) * 2;
  const near = unproject(inv, ndcX, ndcY, -1);
  const far = unproject(inv, ndcX, ndcY, 1);
  const dz = far[2] - near[2];
  if (Math.abs(dz) < 1e-9) return null;
  const t = (camera.target[2] - near[2]) / dz;
  return [near[0] + (far[0] - near[0]) * t, near[1] + (far[1] - near[1]) * t];
}

function unproject(inv, x, y, z) {
  const w = inv[3] * x + inv[7] * y + inv[11] * z + inv[15];
  const iw = Math.abs(w) > 1e-12 ? 1 / w : 1;
  return [(inv[0] * x + inv[4] * y + inv[8] * z + inv[12]) * iw,
          (inv[1] * x + inv[5] * y + inv[9] * z + inv[13]) * iw,
          (inv[2] * x + inv[6] * y + inv[10] * z + inv[14]) * iw];
}

let invertMat;   // bound from math.js at start-up

// --- panels --------------------------------------------------------------------
function showTab(name) {
  for (const t of document.querySelectorAll('.tab')) {
    t.classList.toggle('active', t.dataset.tab === name);
  }
  for (const p of document.querySelectorAll('.panel')) {
    p.classList.toggle('active', p.dataset.panel === name);
  }
}

function renderValidation(report) {
  state.findings = report.findings || [];
  const panel = $('#validationPanel');
  panel.innerHTML = '';

  const c = report.counts || {};
  const summary = el('div', 'val-summary ' + (report.valid ? 'pass' : 'fail'));
  summary.appendChild(el('div', 'val-verdict',
    report.valid ? 'REGISTER VALID' : 'REGISTER HAS DEFECTS'));
  const chips = el('div', 'val-chips');
  for (const [k, v] of [['error', c.error], ['warning', c.warning], ['info', c.info]]) {
    const chip = el('span', 'chip sev-' + k, `${v || 0} ${k}`);
    chips.appendChild(chip);
  }
  summary.appendChild(chips);
  panel.appendChild(summary);

  const chk = report.checked || {};
  panel.appendChild(el('p', 'muted',
    `${chk.rules_run} rules over ${chk.objects} objects ` +
    `(${chk.units} units, ${chk.storeys} storeys, ${chk.buildings} buildings, ` +
    `${chk.infrastructure} corridors).`));

  for (const f of state.findings) {
    const card = el('div', 'finding sev-' + f.severity);
    const head = el('div', 'f-head');
    head.appendChild(el('span', 'f-rule', f.rule));
    head.appendChild(el('span', 'f-sev', f.severity));
    card.appendChild(head);
    card.appendChild(el('div', 'f-title', f.title));
    card.appendChild(el('div', 'f-detail', f.detail));

    if (Object.keys(f.measure || {}).length) {
      const m = el('div', 'f-measure');
      for (const [k, v] of Object.entries(f.measure)) {
        const item = el('span', 'm-item');
        item.appendChild(el('span', 'k', k.replace(/_/g, ' ')));
        item.appendChild(el('span', 'v',
          typeof v === 'number' ? fmt(v, 2) : String(v)));
        m.appendChild(item);
      }
      card.appendChild(m);
    }

    const actions = el('div', 'f-actions');
    for (const oid of (f.objects || []).slice(0, 3)) {
      const b = el('button', 'link', oid.split('/').pop());
      b.title = oid;
      b.addEventListener('click', () => {
        highlightFinding(f);
        selectObject(oid);
      });
      actions.appendChild(b);
    }
    if (f.geometry) {
      const b = el('button', 'link show', 'show in 3D');
      b.addEventListener('click', () => {
        highlightFinding(f);
        const r = f.geometry.rings?.[0] || [];
        if (r.length) {
          const cx = r.reduce((s, p) => s + p[0], 0) / r.length;
          const cy = r.reduce((s, p) => s + p[1], 0) / r.length;
          camera.flyTo(vec3(cx, cy, (f.geometry.z_min + f.geometry.z_max) / 2), 55);
          state.needsRedraw = true;
        }
      });
      actions.appendChild(b);
    }
    card.appendChild(actions);
    panel.appendChild(card);
  }
}

function renderMetrics(metrics, pipeline) {
  const panel = $('#metricsPanel');
  panel.innerHTML = '';
  state.metrics = metrics;

  const section = (title) => {
    panel.appendChild(el('h3', null, title));
    const t = el('div', 'facts');
    panel.appendChild(t);
    return t;
  };
  const row = (t, k, v, good) => {
    const r = el('div', 'fact' + (good === undefined ? '' : good ? ' good' : ' bad'));
    r.appendChild(el('span', 'k', k));
    r.appendChild(el('span', 'v', String(v)));
    t.appendChild(r);
  };

  panel.appendChild(el('p', 'muted',
    'Every stage is scored against the ground truth of the synthetic site. ' +
    'A pipeline that cannot state its own accuracy has no business producing ' +
    'a legal record.'));

  if (metrics.gnss_adjustment) {
    const g = metrics.gnss_adjustment;
    const t = section('GNSS / CORS datum adjustment');
    row(t, 'residual RMS', fmt(g.rms_m, 4) + ' m', g.rms_m < 0.05);
    row(t, 'max residual', fmt(g.max_residual_m, 4) + ' m');
    row(t, 'block scale', fmt(g.scale_ppm, 1) + ' ppm');
    row(t, 'control points used', g.n_points);
    row(t, 'blunders rejected', (g.rejected || []).join(', ') || 'none',
        (g.rejected || []).length > 0);
  }

  if (metrics.ground_filter) {
    const g = metrics.ground_filter;
    const t = section('Ground filtering (SMRF)');
    row(t, "Cohen's kappa", fmt(g.kappa, 4), g.kappa > 0.95);
    row(t, 'Type I error (ground lost)', fmt(g.type_I_error_pct, 3) + ' %',
        g.type_I_error_pct < 2);
    row(t, 'Type II error (object kept)', fmt(g.type_II_error_pct, 3) + ' %',
        g.type_II_error_pct < 2);
    row(t, 'points classified', g.total_points.toLocaleString('en-IN'));
  }

  if (metrics.image_extraction) {
    const im = metrics.image_extraction;
    const cs = im.cross_sensor;
    const t = section('Drone imagery (independent extraction)');
    row(t, 'segmentation backend', im.backend);
    row(t, 'mode', im.mode);
    row(t, 'footprints from imagery', cs.image_footprints);
    row(t, 'agreeing with LiDAR', `${cs.agreed} / ${cs.lidar_footprints}`,
        cs.agreed === cs.lidar_footprints);
    row(t, 'mean IoU between sensors', fmt(cs.mean_iou, 3), cs.mean_iou > 0.6);
    row(t, 'found by imagery only', cs.image_only, cs.image_only === 0);
    row(t, 'found by LiDAR only', cs.lidar_only, cs.lidar_only === 0);
    panel.appendChild(el('p', 'muted',
      'Imagery and LiDAR are extracted independently and then matched. Two ' +
      'sensors agreeing is evidence; one sensor repeating itself is not. ' +
      'Colour alone cannot separate a roof from a paved forecourt, so the ' +
      'image path is gated on height — which is exactly the blind spot LiDAR ' +
      'covers.'));
  }

  if (metrics.building_extraction) {
    const b = metrics.building_extraction;
    const t = section('Building extraction');
    row(t, 'precision / recall', `${fmt(b.precision, 3)} / ${fmt(b.recall, 3)}`,
        b.f1 > 0.9);
    row(t, 'F1', fmt(b.f1, 3), b.f1 > 0.9);
    row(t, 'mean IoU', fmt(b.mean_iou, 3), b.mean_iou > 0.7);
    row(t, 'worst IoU', fmt(b.min_iou, 3));
    row(t, 'mean area error', fmt(b.mean_abs_area_error_pct, 2) + ' %',
        b.mean_abs_area_error_pct < 5);
    row(t, 'height RMSE', fmt(b.height_rmse_m, 3) + ' m', b.height_rmse_m < 1);
    row(t, 'height bias', fmt(b.height_bias_m, 3) + ' m');
    if (b.floor_count_accuracy !== null) {
      row(t, 'floor-count accuracy', fmt(b.floor_count_accuracy * 100, 0) + ' %',
          b.floor_count_accuracy > 0.9);
    }

    panel.appendChild(el('h3', null, 'Per building'));
    const tbl = el('table', 'metric-table');
    const thead = el('tr');
    for (const h of ['building', 'IoU', 'area truth', 'area found', 'height truth',
                     'height found', 'floors']) thead.appendChild(el('th', null, h));
    tbl.appendChild(thead);
    for (const m of b.per_building) {
      const tr = el('tr', m.iou >= 0.5 ? '' : 'bad');
      tr.appendChild(el('td', null, m.truth_id));
      tr.appendChild(el('td', null, fmt(m.iou, 3)));
      tr.appendChild(el('td', null, fmt(m.area_truth_m2, 0)));
      tr.appendChild(el('td', null, fmt(m.area_pred_m2, 0)));
      tr.appendChild(el('td', null, fmt(m.height_truth_m, 2)));
      tr.appendChild(el('td', null, fmt(m.height_pred_m, 2)));
      tr.appendChild(el('td', null,
        `${m.floors_pred ?? '-'} / ${m.floors_truth}` +
        (m.floors_correct === false ? '  ✗' : '')));
      tbl.appendChild(tr);
    }
    panel.appendChild(tbl);
  }

  const comps = (metrics.storey_comparison || []).filter((c) => c.extra_storeys_detected);
  if (comps.length) {
    panel.appendChild(el('h3', null, 'Scan versus sanctioned plan'));
    for (const c of comps) {
      const card = el('div', 'finding sev-error');
      card.appendChild(el('div', 'f-title',
        `${c.plan_id}: ${c.extra_storeys_detected} unauthorised storey(s)`));
      card.appendChild(el('div', 'f-detail',
        `The scan measures ${c.storeys_estimated} storeys above ground; the ` +
        `sanctioned plan records ${c.storeys_on_plan}. Facade periodicity ` +
        `${fmt(c.periodicity_strength, 2)} supports the scan's count.`));
      panel.appendChild(card);
    }
  }

  if (pipeline?.stages?.length) {
    panel.appendChild(el('h3', null, 'Pipeline timing'));
    const t = el('div', 'facts');
    for (const s of pipeline.stages) {
      row(t, s.stage.replace(/_/g, ' '), fmt(s.seconds, 2) + ' s');
    }
    panel.appendChild(t);
  }
}

function renderTree(tree) {
  const panel = $('#treePanel');
  panel.innerHTML = '';

  const node = (n, depth) => {
    const wrap = el('div', 'tree-node');
    const row = el('button', `tree-row k-${n.kind} d${Math.min(depth, 4)}`);
    row.appendChild(el('span', 'tk', n.kind[0].toUpperCase()));
    row.appendChild(el('span', 'tn', n.name));
    if (n.children.length) {
      row.appendChild(el('span', 'tc', String(n.children.length)));
    }
    row.addEventListener('click', (e) => {
      e.stopPropagation();
      selectObject(n.id);
      if (n.children.length) wrap.classList.toggle('collapsed');
    });
    wrap.appendChild(row);
    if (n.children.length) {
      const kids = el('div', 'tree-kids');
      for (const c of n.children) kids.appendChild(node(c, depth + 1));
      wrap.appendChild(kids);
      if (depth >= 1) wrap.classList.add('collapsed');
    }
    return wrap;
  };

  for (const r of tree.roots) panel.appendChild(node(r, 0));
}

// --- search --------------------------------------------------------------------
let searchTimer = null;
function wireSearch() {
  const input = $('#search');
  const results = $('#searchResults');

  const run = async () => {
    const q = input.value.trim();
    if (q.length < 2) { results.innerHTML = ''; results.classList.remove('open'); return; }
    try {
      const res = await api.search(q);
      results.innerHTML = '';
      if (!res.count) {
        results.appendChild(el('div', 'sr-empty', 'No match'));
      }
      for (const o of res.results.slice(0, 14)) {
        const row = el('button', 'sr-row');
        row.appendChild(el('span', 'sr-kind', o.kind[0].toUpperCase()));
        const main = el('span', 'sr-main');
        main.appendChild(el('span', 'sr-name', o.name || o.object_id));
        main.appendChild(el('span', 'sr-sub',
          (o.ulpin || o.object_id) + (o.owner ? ' · ' + o.owner : '')));
        row.appendChild(main);
        row.addEventListener('click', () => {
          selectObject(o.object_id);
          results.classList.remove('open');
          input.value = '';
        });
        results.appendChild(row);
      }
      results.classList.add('open');
    } catch (err) {
      setStatus('Search failed: ' + err.message, 'err');
    }
  };

  input.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(run, 160);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { results.classList.remove('open'); input.blur(); }
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.searchbox')) results.classList.remove('open');
  });
}

// --- locate a ULPIN from orbit -------------------------------------------------
let locator = null;

async function wireLocate() {
  const input = $('#locateInput');
  const runBtn = $('#locateRun');
  const errBox = $('#locateError');
  const chips = $('#locateChips');
  const providerSel = $('#locateProvider');
  const providerHint = $('#providerHint');

  const HINTS = {
    osm: 'Street map from OpenStreetMap. Needs internet.',
    offline: 'Bundled coastlines and borders. Works with no network at all.',
  };
  const syncHint = () => { providerHint.textContent = HINTS[providerSel.value]; };
  syncHint();

  locator = new LocateSequence({
    canvas2d: $('#locateOverlay'),
    renderer, camera, state, api,
    provider: providerSel.value,
    onPhase: (name) => {
      for (const el of document.querySelectorAll('.lst')) {
        el.classList.toggle('active', el.dataset.stage === name);
        el.classList.toggle('past', _stageOrder(el.dataset.stage) < _stageOrder(name));
      }
    },
    onDone: (info) => {
      runBtn.disabled = false;
      for (const el of document.querySelectorAll('.lst')) {
        el.classList.remove('active');
        el.classList.add('past');
      }
      if (info.found) {
        selectObject(info.object.object_id, { fly: false });
        showTab('inspector');
        setStatus(`Located ${info.object.name || info.object.object_id}`, 'ok');
      } else {
        setStatus('That identifier is valid but not registered here', 'warn');
      }
    },
  });

  providerSel.addEventListener('change', () => {
    locator.provider = providerSel.value;
    syncHint();
  });

  const run = async () => {
    const value = input.value.trim();
    if (!value) { errBox.textContent = 'Paste a ULPIN first.'; return; }
    errBox.textContent = '';
    runBtn.disabled = true;
    setStatus('Locating…', 'busy');
    try {
      await locator.run(value);
    } catch (err) {
      errBox.textContent = err.message;
      setStatus('Could not locate that identifier', 'err');
      runBtn.disabled = false;
      locator.cancel();
    }
  };

  runBtn.addEventListener('click', run);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });

  // Quick picks, chosen to show the vertical range: something high up,
  // something below ground, and a corridor that crosses the whole site.
  const wanted = [
    { kind: 'unit', level: (l) => l >= 8, label: 'a high flat' },
    { kind: 'unit', level: (l) => l < 0, label: 'below ground' },
    { kind: 'infrastructure', level: () => true, label: 'a tunnel' },
  ];
  chips.innerHTML = '';
  for (const w of wanted) {
    const hit = state.objects.find(
      (o) => o.kind === w.kind && w.level(o.level) && o.ulpin);
    if (!hit) continue;
    const b = el('button', 'locate-chip');
    b.appendChild(el('b', null, hit.name || hit.object_id));
    b.appendChild(el('span', null, w.label));
    b.addEventListener('click', () => { input.value = hit.ulpin; run(); });
    chips.appendChild(b);
  }
}

const _STAGES = ['globe', 'admin', 'geohash', 'plan', 'rise', 'target', 'done'];
const _stageOrder = (n) => _STAGES.indexOf(n);

// --- 2D plan upload ------------------------------------------------------------
function wirePlanUpload() {
  const fileInput = $('#planFile');
  const buildBtn = $('#planBuild');
  const feedback = $('#planFeedback');
  const preview = $('#planPreview');
  const nameLabel = $('#planName');
  let loadedPlan = null;
  let previewLevel = 0;

  const download = async (kind) => {
    try {
      const plan = await api.samplePlan(kind);
      const blob = new Blob([JSON.stringify(plan, null, 2)],
                            { type: 'application/json' });
      const a = el('a');
      a.href = URL.createObjectURL(blob);
      a.download = `plan-${kind}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
      setStatus(`Downloaded plan-${kind}.json — edit it and upload it back`, 'ok');
    } catch (err) {
      setStatus('Could not fetch the sample: ' + err.message, 'err');
    }
  };
  $('#dlSimple').addEventListener('click', () => download('simple'));
  $('#dlSociety').addEventListener('click', () => download('society'));

  const showError = (msg) => {
    feedback.innerHTML = '';
    preview.innerHTML = '';
    const box = el('div', 'plan-error');
    box.appendChild(el('div', 'pe-title', 'This plan cannot be built yet'));
    box.appendChild(el('div', 'pe-msg', msg));
    feedback.appendChild(box);
    buildBtn.disabled = true;
  };

  const renderPreview = async () => {
    if (!loadedPlan) return;
    try {
      const res = await api.previewPlan(loadedPlan, previewLevel);
      if (!res.ok) { showError(res.error); return; }

      previewLevel = res.level;
      feedback.innerHTML = '';
      const s = res.summary;
      const chips = el('div', 'plan-summary');
      for (const [k, v] of [['parcels', s.parcels], ['buildings', s.buildings],
                            ['storeys', s.storeys], ['units', s.units],
                            ['corridors', s.infrastructure]]) {
        const c = el('span', 'plan-chip');
        c.appendChild(el('b', null, String(v)));
        c.appendChild(el('span', null, k));
        chips.appendChild(c);
      }
      feedback.appendChild(chips);

      preview.innerHTML = '';
      const head = el('div', 'preview-head');
      head.appendChild(el('span', 'ph-title', 'Floor being drawn'));
      const sel = el('select', 'level-select');
      for (const lv of res.levels) {
        const o = el('option', null,
          lv < 0 ? `Basement ${Math.abs(lv)}` : lv === 0 ? 'Ground floor' : `Floor ${lv}`);
        o.value = String(lv);
        if (lv === previewLevel) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => {
        previewLevel = Number(sel.value);
        renderPreview();
      });
      head.appendChild(sel);
      preview.appendChild(head);

      const frame = el('div', 'preview-frame');
      frame.innerHTML = res.svg;
      preview.appendChild(frame);
      preview.appendChild(el('p', 'muted',
        'Green dashed = parcel boundary · amber = units on this floor · ' +
        'red dashed = underground corridor. If a shape looks wrong here, it ' +
        'will look wrong in 3D.'));

      buildBtn.disabled = false;
      setStatus(`Plan looks valid — ${s.units} units across ${s.storeys} storeys`, 'ok');
    } catch (err) {
      showError(err.message);
    }
  };

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    nameLabel.textContent = file.name;
    try {
      loadedPlan = JSON.parse(await file.text());
    } catch (err) {
      loadedPlan = null;
      showError(`${file.name} is not valid JSON — ${err.message}. A missing or ` +
                `trailing comma is the usual cause.`);
      return;
    }
    previewLevel = 0;
    await renderPreview();
  });

  buildBtn.addEventListener('click', async () => {
    if (!loadedPlan) return;
    buildBtn.disabled = true;
    setStatus('Building the 3D register…', 'busy');
    try {
      const res = await api.buildFromPlan(loadedPlan);
      const v = res.validation;
      setStatus(
        `Built ${res.stats.objects} objects with ${res.stats.ulpins_issued} ULPINs` +
        (v.error ? ` — ${v.error} defect${v.error > 1 ? 's' : ''} found` : ' — no defects'),
        v.error ? 'warn' : 'ok');
      await loadAll();
      showTab('inspector');
    } catch (err) {
      showError(err.message);
      setStatus('Build failed', 'err');
    } finally {
      buildBtn.disabled = false;
    }
  });
}

// --- pipeline re-run -----------------------------------------------------------
function wirePipelineRun() {
  const btn = $('#runPipeline');
  const log = $('#pipelineLog');

  btn.addEventListener('click', () => {
    btn.disabled = true;
    log.innerHTML = '';
    log.classList.add('open');
    showTab('metrics');
    setStatus('Running the pipeline…', 'busy');

    api.runPipeline(
      (evt) => {
        if (evt.phase === 'start') {
          const row = el('div', 'plog-row running');
          row.id = 'plog-' + evt.stage;
          row.appendChild(el('span', 'ps', '…'));
          row.appendChild(el('span', 'pn', evt.stage.replace(/_/g, ' ')));
          log.appendChild(row);
          log.scrollTop = log.scrollHeight;
          return;
        }
        const existing = document.getElementById('plog-' + evt.stage);
        const row = existing || el('div', 'plog-row');
        row.className = 'plog-row done';
        row.innerHTML = '';
        row.appendChild(el('span', 'ps', '✓'));
        row.appendChild(el('span', 'pn', evt.stage.replace(/_/g, ' ')));
        const d = { ...evt.detail };
        const secs = d.seconds; delete d.seconds;
        if (secs !== undefined) row.appendChild(el('span', 'pt', fmt(secs, 2) + 's'));
        const bits = Object.entries(d)
          .filter(([k]) => !['stats', 'validation', 'metrics', 'note'].includes(k))
          .map(([k, v]) => `${k.replace(/_/g, ' ')}=${v}`).join('  ');
        if (bits) row.appendChild(el('span', 'pd', bits));
        if (!existing) log.appendChild(row);
        log.scrollTop = log.scrollHeight;

        if (evt.stage === 'error') {
          row.className = 'plog-row failed';
          setStatus('Pipeline failed: ' + evt.detail.error, 'err');
        }
      },
      async () => {
        btn.disabled = false;
        setStatus('Pipeline complete — reloading register', 'ok');
        await loadAll();
      },
      (err) => {
        btn.disabled = false;
        setStatus('Pipeline stream error: ' + err.message, 'err');
      },
    );
  });
}

// --- controls ------------------------------------------------------------------
function wireControls() {
  for (const cb of document.querySelectorAll('input[id^="layer-"]')) {
    cb.addEventListener('change', () => {
      renderer.setLayerVisible(cb.id.replace('layer-', ''), cb.checked);
      state.needsRedraw = true;
    });
  }

  $('#explode').addEventListener('input', (e) => {
    renderer.explode = Number(e.target.value);
    state.needsRedraw = true;
  });

  $('#groundAlpha').addEventListener('input', (e) => {
    const v = Number(e.target.value);
    renderer.setLayerOpacity('terrain', v);
    renderer.setLayerOpacity('parcels', v);
    state.needsRedraw = true;
  });

  $('#levelUp').addEventListener('click', () => stepLevel(+1));
  $('#levelDown').addEventListener('click', () => stepLevel(-1));
  $('#levelAll').addEventListener('click', () => {
    state.levelFilter = null;
    applyLevelFilter();
  });

  $('#viewMode').addEventListener('click', (e) => {
    camera.mode = camera.mode === 'plan' ? 'perspective' : 'plan';
    e.target.textContent = camera.mode === 'plan' ? '2D plan' : '3D view';
    e.target.classList.toggle('on', camera.mode === 'plan');
    state.needsRedraw = true;
  });

  $('#resetView').addEventListener('click', () => {
    if (state.site) camera.frame(state.site.extent);
    camera.azimuth = -0.9;
    camera.elevation = 0.62;
    state.needsRedraw = true;
  });

  for (const t of document.querySelectorAll('.tab')) {
    t.addEventListener('click', () => showTab(t.dataset.tab));
  }

  $('#probeMode').addEventListener('click', (e) => {
    state.probing = !state.probing;
    e.target.classList.toggle('on', state.probing);
    canvas.style.cursor = state.probing ? 'crosshair' : 'grab';
    setStatus(state.probing
      ? 'Column probe active — click anywhere on the site'
      : 'Column probe off');
  });
}

function stepLevel(delta) {
  const levels = [...new Set(state.objects
    .filter((o) => o.kind === 'storey').map((o) => o.level))].sort((a, b) => a - b);
  if (!levels.length) return;
  if (state.levelFilter === null) {
    state.levelFilter = delta > 0 ? levels[0] : levels[levels.length - 1];
  } else {
    const i = levels.indexOf(state.levelFilter);
    state.levelFilter = levels[Math.min(levels.length - 1, Math.max(0, i + delta))];
  }
  applyLevelFilter();
}

// --- picking -------------------------------------------------------------------
function wirePicking() {
  let downAt = null;

  canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });

  canvas.addEventListener('pointerup', async (e) => {
    if (!downAt) return;
    const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
    downAt = null;
    if (moved > 4 || e.button !== 0) return;   // a drag, not a click

    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;

    if (state.probing) {
      const g = screenToGround(px, py);
      if (g) probeColumn(g[0], g[1]);
      return;
    }

    const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
    const idx = renderer.pick(camera.viewProj(aspect), px, py);
    if (idx < 0 || idx >= state.indexToId.length) {
      state.selected = null;
      selectionEdges(null);
      renderInspector(null);
      state.needsRedraw = true;
      return;
    }
    await selectObject(state.indexToId[idx], { fly: false });
  });

  let hoverTimer = null;
  canvas.addEventListener('pointermove', (e) => {
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {
      const rect = canvas.getBoundingClientRect();
      const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
      const idx = renderer.pick(camera.viewProj(aspect),
                                e.clientX - rect.left, e.clientY - rect.top);
      if (idx !== state.hover) {
        state.hover = idx;
        const label = $('#hoverLabel');
        if (idx >= 0 && idx < state.indexToId.length) {
          const id = state.indexToId[idx];
          const o = state.byId.get(id);
          label.textContent = o ? `${o.name || id}  ·  ${o.kind}` : id;
          label.style.display = 'block';
          label.style.left = (e.clientX + 14) + 'px';
          label.style.top = (e.clientY + 14) + 'px';
        } else {
          label.style.display = 'none';
        }
        state.needsRedraw = true;
      }
    }, 45);
  });

  canvas.addEventListener('pointerleave', () => {
    $('#hoverLabel').style.display = 'none';
    state.hover = -1;
    state.needsRedraw = true;
  });
}

// --- boot ----------------------------------------------------------------------
async function loadAll() {
  setStatus('Loading register…', 'busy');
  const [site, objs, validation, metrics, pipeline, tree] = await Promise.all([
    api.site(), api.objects({ geometry: true }), api.validation(),
    api.metrics(), api.pipeline(), api.tree(),
  ]);

  state.site = site;
  state.objects = objs.objects;
  state.byId = new Map(state.objects.map((o) => [o.object_id, o]));

  $('#siteName').textContent = site.name;
  const s = site.stats;
  $('#siteStats').textContent =
    `${s.objects} objects · ${s.by_kind.unit || 0} units · ` +
    `${s.by_kind.storey || 0} storeys · ${s.by_kind.infrastructure || 0} corridors · ` +
    `${s.ulpins_issued} ULPINs · z ${fmt(s.vertical_extent_m[0], 1)} to ` +
    `${fmt(s.vertical_extent_m[1], 1)} m`;

  buildScene();
  await addTerrain();
  camera.frame(site.extent);
  renderer.explodeRef = (site.extent.z_min + site.extent.z_max) / 2;

  renderValidation(validation);
  renderMetrics(metrics, pipeline);
  renderTree(tree);
  renderInspector(null);

  const errs = validation.counts?.error || 0;
  setStatus(errs
    ? `Register loaded — ${errs} defect${errs > 1 ? 's' : ''} found`
    : 'Register loaded — no defects', errs ? 'warn' : 'ok');
  state.needsRedraw = true;
}

function frame() {
  const moving = camera.update();
  if (moving || state.needsRedraw) {
    const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
    renderer.render(camera.viewProj(aspect), {
      selectedId: state.selected
        ? (state.idToIndex.get(state.selected.object_id) ?? -1) : -1,
      hoverId: state.hover,
    });
    state.needsRedraw = moving;
  }
  requestAnimationFrame(frame);
}

async function main() {
  const mathMod = await import('./math.js');
  invertMat = mathMod.invert;

  canvas = $('#gl');
  try {
    renderer = new Renderer(canvas);
  } catch (err) {
    document.body.innerHTML =
      `<div class="fatal"><h1>WebGL2 unavailable</h1><p>${err.message}</p>
       <p>The viewer needs WebGL2. The REST API is unaffected — see
       <a href="/docs">/docs</a>.</p></div>`;
    return;
  }

  camera = new OrbitCamera();
  camera.attach(canvas, () => { state.needsRedraw = true; });

  wireControls();
  wireSearch();
  wirePicking();
  wirePipelineRun();
  wirePlanUpload();
  window.addEventListener('resize', () => { state.needsRedraw = true; });

  try {
    await loadAll();
    await wireLocate();
  } catch (err) {
    setStatus('Could not load the register: ' + err.message +
      '  — run  python scripts/build_demo.py  first.', 'err');
  }
  frame();
}

main();

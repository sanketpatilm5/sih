// Turning cadastral prisms into triangle meshes.
//
// The server sends each volumetric parcel as coordinate rings plus a z-range,
// which is roughly an order of magnitude less data than a triangulated mesh
// and lets the client decide its own tessellation. Extruding it here is cheap:
// the walls are quads, and only the caps need real triangulation.

// --- polygon triangulation -----------------------------------------------------
// Ear clipping with hole support. O(n^2) worst case, which is irrelevant here -
// cadastral footprints have tens of vertices, not thousands - and it avoids
// taking a dependency on a triangulation library for the one thing we need.

function signedArea(ring) {
  let a = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    a += (ring[j][0] - ring[i][0]) * (ring[j][1] + ring[i][1]);
  }
  return a / 2;
}

function pointInTriangle(px, py, ax, ay, bx, by, cx, cy) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

// Cut each hole into the exterior with a bridge edge, producing one simple
// polygon. This is the standard way to make ear clipping handle holes: find the
// hole's rightmost vertex and connect it to a visible vertex of the outer ring.
function eliminateHoles(outer, holes) {
  let ring = outer.slice();
  const sorted = holes
    .map((h) => ({ h, idx: h.reduce((best, p, i) => (p[0] > h[best][0] ? i : best), 0) }))
    .sort((a, b) => b.h[b.idx][0] - a.h[a.idx][0]);

  for (const { h, idx } of sorted) {
    const bridgePoint = h[idx];
    let bestI = -1;
    let bestD = Infinity;
    for (let i = 0; i < ring.length; i++) {
      const d = (ring[i][0] - bridgePoint[0]) ** 2 + (ring[i][1] - bridgePoint[1]) ** 2;
      if (ring[i][0] >= bridgePoint[0] && d < bestD) { bestD = d; bestI = i; }
    }
    if (bestI < 0) {
      for (let i = 0; i < ring.length; i++) {
        const d = (ring[i][0] - bridgePoint[0]) ** 2 + (ring[i][1] - bridgePoint[1]) ** 2;
        if (d < bestD) { bestD = d; bestI = i; }
      }
    }
    const reordered = h.slice(idx).concat(h.slice(0, idx + 1));
    ring = ring.slice(0, bestI + 1).concat(reordered, ring.slice(bestI));
  }
  return ring;
}

/** Triangulate a polygon given as [exterior, ...holes]. Returns flat index triples. */
export function triangulate(rings) {
  if (!rings || !rings.length) return { verts: [], indices: [] };

  let outer = rings[0].slice();
  // rings arrive closed; drop the duplicate last vertex
  if (outer.length > 1) {
    const a = outer[0], b = outer[outer.length - 1];
    if (Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9) outer.pop();
  }
  if (outer.length < 3) return { verts: [], indices: [] };
  if (signedArea(outer) < 0) outer.reverse();

  const holes = [];
  for (let i = 1; i < rings.length; i++) {
    let h = rings[i].slice();
    if (h.length > 1) {
      const a = h[0], b = h[h.length - 1];
      if (Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9) h.pop();
    }
    if (h.length < 3) continue;
    if (signedArea(h) > 0) h.reverse();   // holes wind opposite to the exterior
    holes.push(h);
  }

  const verts = holes.length ? eliminateHoles(outer, holes) : outer;
  const n = verts.length;
  if (n < 3) return { verts: [], indices: [] };

  const indices = [];
  const avail = verts.map((_, i) => i);
  let guard = 0;

  while (avail.length > 3 && guard++ < n * n + 64) {
    let clipped = false;
    for (let i = 0; i < avail.length; i++) {
      const i0 = avail[(i + avail.length - 1) % avail.length];
      const i1 = avail[i];
      const i2 = avail[(i + 1) % avail.length];
      const [ax, ay] = verts[i0], [bx, by] = verts[i1], [cx, cy] = verts[i2];

      // convex corner? (positive cross product, since the ring is CCW)
      if ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax) <= 0) continue;

      let contains = false;
      for (const k of avail) {
        if (k === i0 || k === i1 || k === i2) continue;
        if (pointInTriangle(verts[k][0], verts[k][1], ax, ay, bx, by, cx, cy)) {
          contains = true;
          break;
        }
      }
      if (contains) continue;

      indices.push(i0, i1, i2);
      avail.splice(i, 1);
      clipped = true;
      break;
    }
    // degenerate or self-intersecting input: stop rather than spin
    if (!clipped) break;
  }
  if (avail.length === 3) indices.push(avail[0], avail[1], avail[2]);

  return { verts, indices };
}

/**
 * Extrude a prism into a triangle soup with flat-shaded normals.
 *
 * Returns interleaved position + normal arrays plus a per-vertex object id,
 * which is what the picking pass reads back.
 */
export function buildPrism(rings, zMin, zMax, out, objectId) {
  const { verts, indices } = triangulate(rings);
  if (!verts.length) return;

  if (!out.uvs) out.uvs = [];

  const push = (x, y, z, nx, ny, nz, u = 0, v = 0) => {
    out.positions.push(x, y, z);
    out.normals.push(nx, ny, nz);
    out.ids.push(objectId);
    out.uvs.push(u, v);
  };

  // --- caps -------------------------------------------------------------
  for (let i = 0; i < indices.length; i += 3) {
    const a = verts[indices[i]], b = verts[indices[i + 1]], c = verts[indices[i + 2]];
    // top face, counter-clockwise seen from above
    push(a[0], a[1], zMax, 0, 0, 1);
    push(b[0], b[1], zMax, 0, 0, 1);
    push(c[0], c[1], zMax, 0, 0, 1);
    // bottom face, wound the other way so its normal points down
    push(a[0], a[1], zMin, 0, 0, -1);
    push(c[0], c[1], zMin, 0, 0, -1);
    push(b[0], b[1], zMin, 0, 0, -1);
  }

  // --- walls ------------------------------------------------------------
  // Every ring contributes walls, including holes: the inside face of a
  // courtyard is as much a boundary of the volume as the outside face.
  // U runs 0→1 across each wall face so the facade shader can place exactly
  // one window per face per floor (instead of tiling many bays in world space).
  for (const ringRaw of rings) {
    const ring = ringRaw.slice();
    if (ring.length > 1) {
      const a = ring[0], b = ring[ring.length - 1];
      if (Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9) ring.pop();
    }
    if (ring.length < 2) continue;

    for (let i = 0; i < ring.length; i++) {
      const p = ring[i];
      const q = ring[(i + 1) % ring.length];
      const dx = q[0] - p[0], dy = q[1] - p[1];
      const len = Math.hypot(dx, dy);
      if (len < 1e-9) continue;
      const nx = dy / len, ny = -dx / len;

      push(p[0], p[1], zMin, nx, ny, 0, 0, 0);
      push(q[0], q[1], zMin, nx, ny, 0, 1, 0);
      push(q[0], q[1], zMax, nx, ny, 0, 1, 1);

      push(p[0], p[1], zMin, nx, ny, 0, 0, 0);
      push(q[0], q[1], zMax, nx, ny, 0, 1, 1);
      push(p[0], p[1], zMax, nx, ny, 0, 0, 1);
    }
  }
}

/** Wireframe edges of a prism - the top and bottom rings plus vertical corners. */
export function buildPrismEdges(rings, zMin, zMax, out) {
  for (const ringRaw of rings) {
    const ring = ringRaw.slice();
    if (ring.length > 1) {
      const a = ring[0], b = ring[ring.length - 1];
      if (Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9) ring.pop();
    }
    if (ring.length < 2) continue;
    for (let i = 0; i < ring.length; i++) {
      const p = ring[i], q = ring[(i + 1) % ring.length];
      out.push(p[0], p[1], zMax, q[0], q[1], zMax);
      out.push(p[0], p[1], zMin, q[0], q[1], zMin);
      out.push(p[0], p[1], zMin, p[0], p[1], zMax);
    }
  }
}

// --- apartment exterior detailing ----------------------------------------------
// Bare cadastral prisms read as shipping containers. These helpers hang real
// balcony slabs, railings and a roof parapet off the envelope so a mid-rise
// housing block looks like the flats people actually live in.

function _closedRing(ringRaw) {
  const ring = ringRaw.slice();
  if (ring.length > 1) {
    const a = ring[0], b = ring[ring.length - 1];
    if (Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9) ring.pop();
  }
  return ring;
}

function _pushTri(out, objectId, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const ux = bx - ax, uy = by - ay, uz = bz - az;
  const vx = cx - ax, vy = cy - ay, vz = cz - az;
  let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const l = Math.hypot(nx, ny, nz) || 1;
  nx /= l; ny /= l; nz /= l;
  out.positions.push(ax, ay, az, bx, by, bz, cx, cy, cz);
  out.normals.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
  out.ids.push(objectId, objectId, objectId);
}

function _pushQuad(out, objectId, a, b, c, d) {
  _pushTri(out, objectId, a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  _pushTri(out, objectId, a[0], a[1], a[2], c[0], c[1], c[2], d[0], d[1], d[2]);
}

/** Axis-aligned-ish box from four XY corners and a z range. */
function _pushSlab(out, objectId, corners, z0, z1) {
  const [p0, p1, p2, p3] = corners;
  // top
  _pushQuad(out, objectId,
    [p0[0], p0[1], z1], [p1[0], p1[1], z1], [p2[0], p2[1], z1], [p3[0], p3[1], z1]);
  // bottom
  _pushQuad(out, objectId,
    [p0[0], p0[1], z0], [p3[0], p3[1], z0], [p2[0], p2[1], z0], [p1[0], p1[1], z0]);
  // sides
  _pushQuad(out, objectId,
    [p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z1], [p0[0], p0[1], z1]);
  _pushQuad(out, objectId,
    [p1[0], p1[1], z0], [p2[0], p2[1], z0], [p2[0], p2[1], z1], [p1[0], p1[1], z1]);
  _pushQuad(out, objectId,
    [p2[0], p2[1], z0], [p3[0], p3[1], z0], [p3[0], p3[1], z1], [p2[0], p2[1], z1]);
  _pushQuad(out, objectId,
    [p3[0], p3[1], z0], [p0[0], p0[1], z0], [p0[0], p0[1], z1], [p3[0], p3[1], z1]);
}

function _outwardNormal(ring, i) {
  const p = ring[i], q = ring[(i + 1) % ring.length];
  const dx = q[0] - p[0], dy = q[1] - p[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return null;
  // Exterior rings are CCW → outward is right of the edge direction
  let nx = dy / len, ny = -dx / len;
  // Flip if the normal points into the polygon (holes / CW rings)
  const mid = [(p[0] + q[0]) / 2 + nx * 0.05, (p[1] + q[1]) / 2 + ny * 0.05];
  if (_pointInRing(mid[0], mid[1], ring)) { nx = -nx; ny = -ny; }
  return { nx, ny, len, p, q, ux: dx / len, uy: dy / len };
}

function _pointInRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    if (((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-12) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Balconies, railings and roof parapet only.
 *
 * The solid mass of the building is the unit volumes themselves (so 4 flats on
 * a floor read as 4 distinct boxes). Drawing opaque floor plates + full walls
 * here hid those flats and looked like an empty cage.
 */
export function buildBuildingExterior(rings, zMin, zMax, bodyOut, railOut, objectId, floorH = 3.1) {
  if (!rings?.length) return;

  const ring = _closedRing(rings[0]);
  if (ring.length < 3) return;

  const height = Math.max(zMax - zMin, floorH);
  const nFloors = Math.max(1, Math.round(height / floorH));
  const balcDepth = 1.45;
  const balcT = 0.34;
  const railH = 1.0;
  const railT = 0.05;
  const cornerInset = 0.2;

  if (nFloors >= 2) {
    for (let i = 0; i < ring.length; i++) {
      const edge = _outwardNormal(ring, i);
      if (!edge || edge.len < 4.5) continue;
      const { nx, ny, len, p, ux, uy } = edge;

      const a0 = cornerInset;
      const a1 = len - cornerInset;
      if (a1 - a0 < 2.5) continue;

      const ax = p[0] + ux * a0, ay = p[1] + uy * a0;
      const bx = p[0] + ux * a1, by = p[1] + uy * a1;
      const ox = nx * balcDepth, oy = ny * balcDepth;
      const foot = [
        [ax, ay], [bx, by],
        [bx + ox, by + oy], [ax + ox, ay + oy],
      ];

      for (let f = 1; f <= nFloors - 1; f++) {
        const zTop = zMin + f * floorH;
        if (zTop + 0.15 > zMax) continue;
        _pushSlab(bodyOut, objectId, foot, zTop - balcT, zTop);

        if (railOut) {
          const c0 = foot[0], c1 = foot[1], c2 = foot[2], c3 = foot[3];
          _pushSlab(railOut, objectId, [
            [c3[0], c3[1]], [c2[0], c2[1]],
            [c2[0] + nx * railT, c2[1] + ny * railT],
            [c3[0] + nx * railT, c3[1] + ny * railT],
          ], zTop, zTop + railH);

          const side = 0.06;
          _pushSlab(railOut, objectId, [
            [c0[0], c0[1]], [c3[0], c3[1]],
            [c3[0] + ux * side, c3[1] + uy * side],
            [c0[0] + ux * side, c0[1] + uy * side],
          ], zTop, zTop + railH * 0.9);
          _pushSlab(railOut, objectId, [
            [c1[0], c1[1]], [c2[0], c2[1]],
            [c2[0] - ux * side, c2[1] - uy * side],
            [c1[0] - ux * side, c1[1] - uy * side],
          ], zTop, zTop + railH * 0.9);
        }
      }
    }
  }

  // Thin roof parapet so the crown still reads
  const parapetH = 0.55;
  const parapetT = 0.22;
  for (let i = 0; i < ring.length; i++) {
    const edge = _outwardNormal(ring, i);
    if (!edge) continue;
    const { nx, ny, p, q } = edge;
    _pushSlab(bodyOut, objectId, [
      [p[0], p[1]], [q[0], q[1]],
      [q[0] - nx * parapetT, q[1] - ny * parapetT],
      [p[0] - nx * parapetT, p[1] - ny * parapetT],
    ], zMax, zMax + parapetH);
  }
}

/**
 * Mesh a DEM grid into a terrain surface.
 *
 * `imageExtent` is the ground rectangle an aerial photo covers, in the same
 * local metres as the grid. Given it, each vertex also gets a texture
 * coordinate, which is what lets real imagery land on the terrain in exactly
 * the right place relative to the buildings standing on it.
 */
export function buildTerrain(grid, out, objectId, imageExtent = null) {
  const { rows, cols, cell, x0, y0, values } = grid;
  const z = (r, c) => {
    const v = values[r * cols + c];
    return v === null || v === undefined ? null : v;
  };
  const P = (r, c) => [x0 + c * cell, y0 + r * cell, z(r, c)];

  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = P(r, c), b = P(r, c + 1), d = P(r + 1, c), e = P(r + 1, c + 1);
      if (a[2] === null || b[2] === null || d[2] === null || e[2] === null) continue;
      for (const tri of [[a, b, e], [a, e, d]]) {
        const [p, q, s] = tri;
        const ux = q[0] - p[0], uy = q[1] - p[1], uz = q[2] - p[2];
        const vx = s[0] - p[0], vy = s[1] - p[1], vz = s[2] - p[2];
        let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const l = Math.hypot(nx, ny, nz) || 1;
        nx /= l; ny /= l; nz /= l;
        if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
        for (const v of tri) {
          out.positions.push(v[0], v[1], v[2]);
          out.normals.push(nx, ny, nz);
          out.ids.push(objectId);
          if (imageExtent) {
            const [ix0, iy0, ix1, iy1] = imageExtent;
            out.uvs.push((v[0] - ix0) / (ix1 - ix0), (v[1] - iy0) / (iy1 - iy0));
          }
        }
      }
    }
  }
}

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

  const push = (x, y, z, nx, ny, nz) => {
    out.positions.push(x, y, z);
    out.normals.push(nx, ny, nz);
    out.ids.push(objectId);
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

      push(p[0], p[1], zMin, nx, ny, 0);
      push(q[0], q[1], zMin, nx, ny, 0);
      push(q[0], q[1], zMax, nx, ny, 0);

      push(p[0], p[1], zMin, nx, ny, 0);
      push(q[0], q[1], zMax, nx, ny, 0);
      push(p[0], p[1], zMax, nx, ny, 0);
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

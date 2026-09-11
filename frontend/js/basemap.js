// The bundled basemap: real coastlines, country borders and Indian states.
//
// Natural Earth, public domain, simplified at build time to ~10,000 vertices
// and shipped with the app. Fetching tiles at runtime would have been easier
// and would have broken the one property this viewer actually promises - that
// it works with no network at all, in a tehsil office, on the day.
//
// Each ring carries a precomputed bounding box. At street scale the view covers
// a millionth of the world, so culling by bbox turns a 10,000-vertex redraw
// into a few dozen and keeps the descent at full frame rate.

let cache = null;

function withBBox(rings) {
  return rings.map((ring) => {
    let lo0 = 180, la0 = 90, lo1 = -180, la1 = -90;
    for (const [lon, lat] of ring) {
      if (lon < lo0) lo0 = lon;
      if (lon > lo1) lo1 = lon;
      if (lat < la0) la0 = lat;
      if (lat > la1) la1 = lat;
    }
    return { ring, bbox: [lo0, la0, lo1, la1] };
  });
}

/** Load and index the basemap. Cached - safe to call on every run. */
export async function loadBasemap(url = 'data/basemap.json') {
  if (cache) return cache;
  const doc = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(`basemap ${r.status}`);
    return r.json();
  });
  cache = {
    source: doc.source,
    world: doc.world.map((f) => ({ ...f, rings: withBBox(f.r) })),
    states: doc.states.map((f) => ({ ...f, rings: withBBox(f.r) })),
    india: withBBox(doc.india),
  };
  return cache;
}

export function basemapReady() { return cache; }

/** True when a ring's bbox overlaps the view window (both in degrees). */
export function inView(bbox, view) {
  return !(bbox[2] < view[0] || bbox[0] > view[2] ||
           bbox[3] < view[1] || bbox[1] > view[3]);
}

/**
 * Draw a set of rings through an arbitrary projection.
 *
 * Vertices behind the horizon are pushed out to the limb rather than dropped,
 * so a country straddling the edge of the globe keeps a clean silhouette
 * instead of flickering in and out as it rotates.
 */
export function drawRings(ctx, rings, project, opts) {
  const { fill, stroke, lineWidth = 1, view, limbR, cx, cy } = opts;
  ctx.beginPath();
  let drew = false;

  for (const { ring, bbox } of rings) {
    if (view && !inView(bbox, view)) continue;
    let started = false;
    for (const [lon, lat] of ring) {
      let p = project(lat, lon);
      if (!p.visible) {
        if (limbR === undefined) continue;          // flat view: just skip
        const dx = p.x - cx, dy = p.y - cy;
        const len = Math.hypot(dx, dy) || 1;
        p = { x: cx + (dx / len) * limbR, y: cy + (dy / len) * limbR, visible: true };
      }
      if (!started) { ctx.moveTo(p.x, p.y); started = true; } else ctx.lineTo(p.x, p.y);
      drew = true;
    }
    if (started) ctx.closePath();
  }

  if (!drew) return;
  if (fill) { ctx.fillStyle = fill; ctx.fill('evenodd'); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
}

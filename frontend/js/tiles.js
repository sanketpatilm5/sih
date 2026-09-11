// OpenStreetMap raster tiles, for the close half of the descent.
//
// Why this sits alongside the bundled vector basemap rather than replacing it:
//
//   * Tiles are Web Mercator. Mercator cannot draw a globe - it diverges at the
//     poles and shows one hemisphere at best. The opening shot genuinely needs
//     an orthographic sphere, so the bundled Natural Earth data still carries
//     it, and tiles take over once the view is small enough that the two
//     projections agree locally (a few hundred kilometres).
//
//   * Tiles need the network. The viewer's one hard promise is that it works in
//     an office with no connectivity, so this layer is strictly an enhancement:
//     if a tile fails, or the toggle is off, the vector basemap carries the
//     whole descent exactly as before. Nothing breaks, it just looks plainer.
//
// Attribution is not optional. OSM data is ODbL-licensed and requires credit
// wherever it is shown; `ATTRIBUTION` is rendered on the canvas whenever a tile
// has actually been drawn.

const TILE = 256;
const MAX_ZOOM = 19;
const MAX_INFLIGHT = 6;          // courtesy limit on a donated tile server
const CACHE_LIMIT = 600;

export const ATTRIBUTION = '© OpenStreetMap contributors';
const URL_TEMPLATE = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

const cache = new Map();          // "z/x/y" -> HTMLImageElement (may be loading)
let inflight = 0;
const queue = [];

/** Metres per Web Mercator world pixel at a given zoom and latitude. */
export function metresPerPixel(lat, z) {
  return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, z);
}

/** Longitude/latitude to Web Mercator world pixels at a zoom level. */
export function lonLatToWorld(lat, lon, z) {
  const s = TILE * Math.pow(2, z);
  const sinLat = Math.min(0.9999, Math.max(-0.9999, Math.sin(lat * Math.PI / 180)));
  return {
    x: (lon + 180) / 360 * s,
    y: (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * s,
  };
}

/**
 * The zoom level whose native resolution is closest to what we are drawing at.
 *
 * Picking the nearest level rather than always rounding down means tiles are
 * drawn at roughly 1:1 instead of being stretched, which is the difference
 * between a crisp map and a blurry one during a continuous zoom.
 */
export function zoomFor(spanMetres, viewportPx, lat) {
  const screenPxPerMetre = viewportPx / spanMetres;
  const z = Math.log2(screenPxPerMetre * 156543.03392 * Math.cos(lat * Math.PI / 180));
  return Math.max(0, Math.min(MAX_ZOOM, Math.round(z)));
}

function pump() {
  while (inflight < MAX_INFLIGHT && queue.length) {
    const entry = queue.shift();
    if (entry.img.src) continue;
    inflight++;
    entry.img.onload = () => { inflight--; entry.img.ok = true; pump(); };
    entry.img.onerror = () => { inflight--; entry.img.failed = true; pump(); };
    entry.img.src = entry.url;
  }
}

function getTile(z, x, y) {
  const n = 1 << z;
  if (y < 0 || y >= n) return null;               // above the pole / below it
  const wrapped = ((x % n) + n) % n;              // the world repeats east-west
  const key = `${z}/${wrapped}/${y}`;

  let img = cache.get(key);
  if (img) return img;

  img = new Image();
  img.crossOrigin = 'anonymous';
  img.ok = false;
  img.failed = false;
  cache.set(key, img);

  // crude LRU: once the cache is large, drop the oldest insertions
  if (cache.size > CACHE_LIMIT) {
    const stale = cache.keys().next().value;
    if (stale !== key) cache.delete(stale);
  }

  queue.push({
    img,
    url: URL_TEMPLATE.replace('{z}', z).replace('{x}', wrapped).replace('{y}', y),
  });
  pump();
  return img;
}

/**
 * Draw the OSM tile layer.
 *
 * Returns true if at least one tile was actually painted, which is what the
 * caller uses to decide whether attribution is owed and whether the vector
 * fallback is still needed underneath.
 */
export function drawTiles(ctx, { lat, lon, span, W, H, dpr = 1, alpha = 1 }) {
  if (alpha <= 0.01) return false;

  const z = zoomFor(span, (0.6 * H) / dpr, lat);
  const mpp = metresPerPixel(lat, z);
  // screen pixels per world pixel
  const k = (0.6 * H / span) * mpp;
  if (!isFinite(k) || k <= 0) return false;

  const centre = lonLatToWorld(lat, lon, z);
  const cx = W / 2, cy = H / 2;
  const size = TILE * k;

  // world-pixel window covered by the viewport, then the tiles spanning it
  const halfW = (W / 2) / k, halfH = (H / 2) / k;
  const x0 = Math.floor((centre.x - halfW) / TILE);
  const x1 = Math.ceil((centre.x + halfW) / TILE);
  const y0 = Math.floor((centre.y - halfH) / TILE);
  const y1 = Math.ceil((centre.y + halfH) / TILE);

  // a zoom sequence can request a lot at once; refuse absurd counts rather
  // than hammering the server on a bad frame
  if ((x1 - x0) * (y1 - y0) > 240) return false;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = true;
  let painted = false;

  for (let tx = x0; tx <= x1; tx++) {
    for (let ty = y0; ty <= y1; ty++) {
      const sx = cx + (tx * TILE - centre.x) * k;
      const sy = cy + (ty * TILE - centre.y) * k;

      const img = getTile(z, tx, ty);
      if (img && img.ok) {
        ctx.drawImage(img, sx, sy, size + 1, size + 1);
        painted = true;
        continue;
      }

      // While a tile loads, borrow the matching quadrant of an ancestor that is
      // already cached. Without this a fast zoom is mostly blank squares.
      for (let up = 1; up <= 4; up++) {
        const pz = z - up;
        if (pz < 0) break;
        const f = 1 << up;
        const px = Math.floor(tx / f), py = Math.floor(ty / f);
        const parent = cache.get(`${pz}/${((px % (1 << pz)) + (1 << pz)) % (1 << pz)}/${py}`);
        if (parent && parent.ok) {
          const sub = TILE / f;
          ctx.drawImage(parent,
            (tx - px * f) * sub, (ty - py * f) * sub, sub, sub,
            sx, sy, size + 1, size + 1);
          painted = true;
          break;
        }
      }
    }
  }

  ctx.restore();
  return painted;
}

/** Warm the cache for a position, so the descent does not start blank. */
export function prefetch(lat, lon, zooms = [4, 7, 10, 13, 16]) {
  for (const z of zooms) {
    const w = lonLatToWorld(lat, lon, z);
    const tx = Math.floor(w.x / TILE), ty = Math.floor(w.y / TILE);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) getTile(z, tx + dx, ty + dy);
    }
  }
}

export function tileStats() {
  let ok = 0, failed = 0;
  for (const img of cache.values()) {
    if (img.ok) ok++; else if (img.failed) failed++;
  }
  return { cached: cache.size, loaded: ok, failed, inflight, queued: queue.length };
}

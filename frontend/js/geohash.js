// Geohash decoding, client side.
//
// A geohash is a hierarchical spatial index: every character you add splits the
// cell again, alternating longitude and latitude. That is exactly what makes it
// usable as a zoom sequence - `t` is a continent, `te` is a large region, and by
// the tenth character you are inside a single room.
//
// The locate animation leans on that: it is not decorating the identifier with a
// zoom, it is *reading* the zoom that the identifier already contains. So this
// has to run per frame, client side, with no network round-trip.
//
// Mirrors bhoomi3d/core/ulpin.py exactly.

const ALPHABET = '0123456789bcdefghjkmnpqrstuvwxyz';

/** Bounds of a geohash cell: {latMin, lonMin, latMax, lonMax}. */
export function geohashBounds(gh) {
  let latMin = -90, latMax = 90, lonMin = -180, lonMax = 180;
  let even = true;                       // longitude is refined first

  for (const ch of gh.toLowerCase()) {
    const idx = ALPHABET.indexOf(ch);
    if (idx < 0) throw new Error(`invalid geohash character: ${ch}`);
    for (const mask of [16, 8, 4, 2, 1]) {
      const bit = (idx & mask) ? 1 : 0;
      if (even) {
        const mid = (lonMin + lonMax) / 2;
        if (bit) lonMin = mid; else lonMax = mid;
      } else {
        const mid = (latMin + latMax) / 2;
        if (bit) latMin = mid; else latMax = mid;
      }
      even = !even;
    }
  }
  return { latMin, lonMin, latMax, lonMax };
}

/** Centre of a geohash cell as [lat, lon]. */
export function geohashCentre(gh) {
  const b = geohashBounds(gh);
  return [(b.latMin + b.latMax) / 2, (b.lonMin + b.lonMax) / 2];
}

/** Approximate ground size of a cell in metres, as [width, height]. */
export function cellSizeMetres(gh) {
  const b = geohashBounds(gh);
  const midLat = ((b.latMin + b.latMax) / 2) * Math.PI / 180;
  return [
    (b.lonMax - b.lonMin) * 111320 * Math.cos(midLat),
    (b.latMax - b.latMin) * 111132,
  ];
}

/** Human-readable distance: 5,000 km / 40 km / 1.1 m. */
export function formatSpan(metres) {
  if (metres >= 1000) {
    const km = metres / 1000;
    return `${km >= 100 ? Math.round(km).toLocaleString('en-IN') : km.toFixed(1)} km`;
  }
  return `${metres >= 10 ? Math.round(metres) : metres.toFixed(1)} m`;
}

/**
 * Every prefix of a geohash, with its bounds and span.
 *
 * This is the zoom storyboard: one entry per character, each roughly an order
 * of magnitude tighter than the last.
 */
export function prefixLadder(gh) {
  const out = [];
  for (let i = 1; i <= gh.length; i++) {
    const prefix = gh.slice(0, i);
    const [w, h] = cellSizeMetres(prefix);
    out.push({ prefix, bounds: geohashBounds(prefix), width: w, height: h });
  }
  return out;
}

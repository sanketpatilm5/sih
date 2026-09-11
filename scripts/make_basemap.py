"""
Turn Natural Earth GeoJSON into a compact basemap the viewer can ship.

    python scripts/make_basemap.py <natural-earth-dir> frontend/data

Expects two files in the source directory, downloadable from
https://github.com/nvkelso/natural-earth-vector/tree/master/geojson :

    ne_110m_admin_0_countries.geojson       -> countries.geojson
    ne_50m_admin_1_states_provinces.geojson -> states.geojson

The result is committed to the repo. It only needs regenerating if the
boundaries themselves change, which is roughly never.

Natural Earth is public domain, so the data can be bundled rather than fetched
at runtime - which is the whole point: the viewer keeps working with no network.

Two products, each simplified to the scale it is actually drawn at. There is no
sense carrying 100 m coastline detail for a globe where one degree is three
pixels, and no sense carrying 50 km detail for a state filling the screen.
"""
import json
import sys
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def rings_of(geom, tol, ndigits):
    """Simplified exterior rings of a (multi)polygon, as [[lon, lat], ...]."""
    g = geom.simplify(tol, preserve_topology=True)
    if g.is_empty:
        return []
    parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
    out = []
    for p in parts:
        if p.is_empty or p.area < tol * tol:      # drop specks below the tolerance
            continue
        ring = [[round(x, ndigits), round(y, ndigits)] for x, y in p.exterior.coords]
        if len(ring) >= 4:
            out.append(ring)
    return out


def build(path, tol, ndigits, name_key, keep=None, extra=None):
    doc = json.loads(path.read_text(encoding="utf-8"))
    feats = []
    for f in doc["features"]:
        props = f["properties"]
        if keep and not keep(props):
            continue
        rings = rings_of(shape(f["geometry"]), tol, ndigits)
        if not rings:
            continue
        entry = {"n": props.get(name_key) or "", "r": rings}
        if extra:
            entry.update(extra(props))
        feats.append(entry)
    return feats


# --- world, for the globe shot ------------------------------------------------
# 0.22 degrees is about 24 km, which at globe scale is well under a pixel.
world = build(SRC / "countries.geojson", tol=0.22, ndigits=2,
              name_key="ADMIN",
              extra=lambda p: {"in": 1} if p.get("ADMIN") == "India" else {})

# --- India's states, for the country and state shots --------------------------
# 0.02 degrees is about 2 km - roughly one pixel when Maharashtra fills the view.
states = build(SRC / "states.geojson", tol=0.02, ndigits=3,
               name_key="name",
               keep=lambda p: p.get("admin") == "India",
               extra=lambda p: {"mh": 1} if p.get("name") == "Maharashtra" else {})

# the national outline, dissolved from the states so the two always agree
india_geom = unary_union([shape(f["geometry"])
                          for f in json.loads((SRC / "states.geojson")
                                              .read_text(encoding="utf-8"))["features"]
                          if f["properties"].get("admin") == "India"])
india = rings_of(india_geom, 0.05, 3)

payload = {
    "source": "Natural Earth (public domain) - ne_110m_admin_0_countries, "
              "ne_50m_admin_1_states_provinces",
    "world": world,
    "states": states,
    "india": india,
}

path = OUT / "basemap.json"
path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

kb = path.stat().st_size / 1024
pts = sum(len(r) for f in world for r in f["r"]) + \
      sum(len(r) for f in states for r in f["r"]) + sum(len(r) for r in india)
print(f"world countries : {len(world)}")
print(f"India states    : {len(states)}")
print(f"national outline: {len(india)} ring(s)")
print(f"total vertices  : {pts:,}")
print(f"written         : {path}  ({kb:.0f} KB)")

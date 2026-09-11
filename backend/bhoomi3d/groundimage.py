"""
A real map of the site draped over the 3D scene's ground plane.

The viewer's terrain is a mesh derived from the DEM, and until now it was
painted a flat grey-green. Texturing it with an actual map of the site is what
turns the scene from a diagram into something recognisable - the volumes stop
being abstract boxes and start standing on a place you could walk to.

The map is the OpenStreetMap tiles covering the site, stitched together
server-side and cached on disk. Going through the backend rather than the
browser buys two things: the image is cached so a re-run costs nothing, and
the texture is same-origin - a cross-origin image without CORS headers cannot
be uploaded to a WebGL texture at all.
"""
from __future__ import annotations

import io
import json
import math
import urllib.request
from pathlib import Path
from typing import Optional

# Web Mercator, the projection OSM tiles are served in.
EARTH_CIRCUMFERENCE = 40075016.686
TILE = 256
# Deepest zoom the standard OSM tile layer renders.
OSM_MAX_ZOOM = 19


def metres_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def zoom_for(width_m: float, pixels: int, lat: float, *, max_zoom: int = OSM_MAX_ZOOM) -> int:
    """Largest zoom whose image still covers `width_m` across `pixels`."""
    for z in range(max_zoom, 0, -1):
        if metres_per_pixel(lat, z) * pixels >= width_m:
            return z
    return 1


def lonlat_to_world(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    s = TILE * (2 ** zoom)
    sin_lat = max(-0.9999, min(0.9999, math.sin(math.radians(lat))))
    return ((lon + 180.0) / 360.0 * s,
            (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * s)


def world_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    s = TILE * (2 ** zoom)
    lon = x / s * 360.0 - 180.0
    n = math.pi * (1 - 2 * y / s)
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon


def _fetch(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={
        # OSM's tile policy requires a real identifying agent and will refuse
        # a default urllib string.
        "User-Agent": "Bhoomi3D/0.9 (3D cadastre prototype)",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def build_ground_image(enu, extent: dict, out_dir: Path, *,
                       pixels: int = 640) -> Optional[dict]:
    """
    Fetch a map covering a site and report the exact ground extent it covers.

    The extent matters more than the image: the viewer maps it onto the terrain
    mesh by world position, so being a few metres out would slide the whole
    map off the buildings standing on it.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "ground.png"
    meta_path = out_dir / "ground.json"

    # site centre and the square that comfortably contains it
    cx = (extent["x_min"] + extent["x_max"]) / 2
    cy = (extent["y_min"] + extent["y_max"]) / 2
    width_m = max(extent["x_max"] - extent["x_min"],
                  extent["y_max"] - extent["y_min"]) * 1.25
    lon_c, lat_c = enu.inverse_xy(cx, cy)

    zoom = zoom_for(width_m, pixels, lat_c)
    mpp = metres_per_pixel(lat_c, zoom)
    covered = mpp * pixels

    cache_key = {
        "provider": "osm", "zoom": zoom,
        "lat": round(lat_c, 6), "lon": round(lon_c, 6), "pixels": pixels,
    }
    if meta_path.exists() and png_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("cache_key") == cache_key:
                return old
        except (json.JSONDecodeError, OSError):
            pass

    # Stitch the tiles covering the square, then crop to it exactly.
    centre = lonlat_to_world(lat_c, lon_c, zoom)
    half = pixels / 2
    x0, y0 = centre[0] - half, centre[1] - half
    tx0, ty0 = int(x0 // TILE), int(y0 // TILE)
    tx1, ty1 = int((x0 + pixels) // TILE), int((y0 + pixels) // TILE)

    n = 1 << zoom
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    fetched = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            if not (0 <= ty < n):
                continue
            url = f"https://tile.openstreetmap.org/{zoom}/{tx % n}/{ty}.png"
            try:
                tile = Image.open(io.BytesIO(_fetch(url))).convert("RGB")
            except Exception:
                continue
            canvas.paste(tile, ((tx - tx0) * TILE, (ty - ty0) * TILE))
            fetched += 1
    if fetched == 0:
        # offline: a black square would be worse than the plain terrain colour
        return None

    img = canvas.crop((
        int(x0 - tx0 * TILE), int(y0 - ty0 * TILE),
        int(x0 - tx0 * TILE) + pixels, int(y0 - ty0 * TILE) + pixels,
    ))
    img.save(png_path, "PNG", optimize=True)

    # The ground extent in local ENU metres. Web Mercator scale is very nearly
    # constant across a few hundred metres, so a linear mapping from the centre
    # is accurate to well under a pixel at this size.
    meta = {
        "url": "/api/ground-image.png",
        "provider": "osm",
        "attribution": "© OpenStreetMap contributors",
        "zoom": zoom,
        "metres_per_pixel": round(mpp, 4),
        "pixels": img.size[0],
        "extent_m": [round(cx - covered / 2, 3), round(cy - covered / 2, 3),
                     round(cx + covered / 2, 3), round(cy + covered / 2, 3)],
        "centre_latlon": [round(lat_c, 7), round(lon_c, 7)],
        "cache_key": cache_key,
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta

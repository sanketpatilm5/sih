"""
Geospatial file I/O.

The platform reads and writes the formats a survey contractor and a state GIS
department actually exchange - LAS/LAZ for point clouds, GeoTIFF for elevation
products, GeoJSON for vector layers - rather than a bespoke serialisation. That
matters for a system meant to slot into an existing land-records workflow: the
inputs arrive as LAZ and the outputs have to open in QGIS.

Every reader and writer degrades gracefully. `laspy` and `rasterio` are listed
as dependencies but are genuinely optional; without them the platform falls
back to plain-text XYZ and a NumPy sidecar, and says so, rather than failing to
start. That keeps the system runnable on a locked-down government machine where
installing GDAL is not a five-minute job.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .core.crs import LocalENU
from .core.grid import Grid

# --- optional dependencies -----------------------------------------------------
try:
    import laspy
    HAVE_LASPY = True
except ImportError:                                   # pragma: no cover
    HAVE_LASPY = False

try:
    import rasterio
    from rasterio.transform import from_origin
    HAVE_RASTERIO = True
except ImportError:                                   # pragma: no cover
    HAVE_RASTERIO = False

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:                                   # pragma: no cover
    HAVE_PIL = False


def capabilities() -> dict:
    """What file formats this installation can actually handle."""
    return {
        "las_laz": HAVE_LASPY,
        "geotiff": HAVE_RASTERIO,
        "imagery": HAVE_PIL,
        "geojson": True,
        "xyz_text": True,
    }


# --- point clouds --------------------------------------------------------------
def write_las(path, xyz: np.ndarray, classification: Optional[np.ndarray] = None,
              intensity: Optional[np.ndarray] = None,
              return_number: Optional[np.ndarray] = None,
              scale: float = 0.001) -> Path:
    """
    Write a point cloud as LAS/LAZ (by extension), or XYZ text as a fallback.

    Point format 6 is used because it carries the extended classification range
    and 16-bit intensity that ASPRS class codes and modern sensors expect.
    """
    path = Path(path)
    if not HAVE_LASPY:
        alt = path.with_suffix(".xyz")
        header = "x y z" + (" classification" if classification is not None else "")
        cols = [xyz]
        if classification is not None:
            cols.append(classification.reshape(-1, 1))
        np.savetxt(alt, np.hstack(cols), fmt="%.3f", header=header)
        return alt

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([scale, scale, scale])
    header.offsets = np.floor(xyz.min(axis=0))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if classification is not None:
        las.classification = classification.astype(np.uint8)
    if intensity is not None:
        las.intensity = intensity.astype(np.uint16)
    if return_number is not None:
        las.return_number = return_number.astype(np.uint8)
        las.number_of_returns = np.maximum(return_number, 1).astype(np.uint8)
    las.write(str(path))
    return path


def read_pointcloud(path) -> dict:
    """
    Read a point cloud from LAS/LAZ, XYZ text, CSV or NumPy.

    Returns a dict with `xyz` and whatever ancillary channels the file carried.
    Classification and return number are passed through when present because
    both materially improve extraction - a cloud that has already been
    classified by the contractor should not be re-classified from scratch.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".las", ".laz"):
        if not HAVE_LASPY:
            raise RuntimeError(
                f"reading {suffix} needs laspy; install it with "
                f"'pip install laspy[lazrs]', or supply the cloud as .xyz text")
        las = laspy.read(str(path))
        out = {"xyz": np.column_stack([las.x, las.y, las.z]).astype(float)}
        for name, key in (("classification", "classification"),
                          ("intensity", "intensity"),
                          ("return_number", "return_number")):
            if hasattr(las, name):
                out[key] = np.asarray(getattr(las, name))
        out["source"] = str(path)
        out["format"] = f"LAS/LAZ point format {las.header.point_format.id}"
        return out

    if suffix == ".npy":
        arr = np.load(path)
        return {"xyz": arr[:, :3].astype(float), "source": str(path),
                "format": "numpy"}

    delim = "," if suffix in (".csv", ".txt") else None
    arr = np.loadtxt(path, delimiter=delim, comments="#")
    out = {"xyz": arr[:, :3].astype(float), "source": str(path),
           "format": "xyz text"}
    if arr.shape[1] >= 4:
        out["classification"] = arr[:, 3].astype(np.uint8)
    return out


# --- rasters -------------------------------------------------------------------
def write_geotiff(path, grid: Grid, enu: Optional[LocalENU] = None) -> Path:
    """
    Write a Grid as a GeoTIFF.

    The grid lives in the project-local ENU frame, which is not a registered
    CRS, so the transform is written in local metres and the geodetic origin is
    recorded in the file's metadata. That keeps the product openable in QGIS and
    self-describing about where on earth it belongs, without pretending it is in
    a projected CRS it is not.
    """
    path = Path(path)
    if not HAVE_RASTERIO:
        np.save(path.with_suffix(".npy"), grid.data)
        path.with_suffix(".json").write_text(json.dumps({
            "x0": grid.x0, "y0": grid.y0, "cell": grid.cell,
            "rows": grid.rows, "cols": grid.cols, "name": grid.name,
        }, indent=1), encoding="utf-8")
        return path.with_suffix(".npy")

    # GeoTIFF rows run north-to-south; our grids run south-to-north, so flip
    data = np.flipud(grid.data).astype("float32")
    top = grid.y0 + (grid.rows - 1) * grid.cell + grid.cell / 2
    transform = from_origin(grid.x0 - grid.cell / 2, top, grid.cell, grid.cell)

    meta = {}
    if enu is not None:
        o = enu.origin
        meta = {"ORIGIN_LAT": str(o.lat), "ORIGIN_LON": str(o.lon),
                "ORIGIN_HEIGHT": str(o.height),
                "FRAME": "local ENU metres on WGS84",
                "PRODUCT": grid.name}

    with rasterio.open(
        path, "w", driver="GTiff", height=grid.rows, width=grid.cols,
        count=1, dtype="float32", transform=transform, nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, grid.name)
        if meta:
            dst.update_tags(**meta)
    return path


def read_geotiff(path, name: str = "raster") -> Grid:
    """Read a single-band GeoTIFF into a Grid, flipping to south-up order."""
    if not HAVE_RASTERIO:
        raise RuntimeError("reading GeoTIFF needs rasterio; "
                           "install it with 'pip install rasterio'")
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        t = src.transform
        cell = abs(t.a)
        # centre of the bottom-left cell, in the file's own coordinates
        x0 = t.c + cell / 2
        y0 = t.f - abs(t.e) * src.height + cell / 2
    return Grid(np.flipud(data), x0, y0, cell, name)


# --- imagery -------------------------------------------------------------------
def write_orthophoto(path, scene, resolution: float = 0.20) -> Optional[Path]:
    """
    Render a simulated drone orthophoto of a scene.

    Real orthophotos are the second half of the extraction problem: LiDAR gives
    geometry, imagery gives material and colour, and fusing them beats either
    alone. This renders a plausible one - roofs with per-building colour, tree
    canopies, roads, cast shadows and sensor noise - so the image-based
    extractor in `ai.image_segment` has something honest to work on.
    """
    if not HAVE_PIL:
        return None
    from shapely.geometry import Polygon
    import shapely

    x0, y0, x1, y1 = scene.extent
    w = int((x1 - x0) / resolution)
    h = int((y1 - y0) / resolution)
    rng = np.random.default_rng(7)

    xs = x0 + (np.arange(w) + 0.5) * resolution
    ys = y0 + (np.arange(h) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)

    # --- ground: mottled soil / paving -------------------------------------
    base = np.zeros((h, w, 3), dtype=float)
    mottle = rng.normal(0, 6, (h, w))
    from scipy.ndimage import gaussian_filter
    mottle = gaussian_filter(mottle, 3.0)
    base[..., 0] = 138 + mottle
    base[..., 1] = 129 + mottle
    base[..., 2] = 116 + mottle

    # --- roads --------------------------------------------------------------
    for p in scene.parcels:
        if p.land_use != "road":
            continue
        poly = Polygon(p.ring)
        mask = shapely.contains_xy(poly, gx, gy)
        base[mask] = np.array([86, 86, 90]) + rng.normal(0, 4, (mask.sum(), 3))

    # --- tree canopies ------------------------------------------------------
    for t in scene.trees:
        d = np.hypot(gx - t.x, gy - t.y)
        mask = d < t.radius
        if not mask.any():
            continue
        shade = 1.0 - 0.35 * (d[mask] / t.radius)
        green = np.column_stack([
            48 + rng.normal(0, 9, mask.sum()),
            96 + rng.normal(0, 14, mask.sum()),
            42 + rng.normal(0, 9, mask.sum()),
        ]) * shade[:, None]
        base[mask] = green

    # --- buildings, with a cast shadow to the north-east --------------------
    palette = [(176, 132, 108), (150, 150, 156), (188, 168, 140), (164, 118, 96)]
    sun = np.array([0.55, 0.42])          # shadow direction, in metres per metre
    for i, b in enumerate(scene.buildings):
        poly = Polygon(b.world_footprint())
        shadow = shapely.affinity.translate(
            poly, sun[0] * b.height_above_ground * 0.28,
            sun[1] * b.height_above_ground * 0.28)
        smask = shapely.contains_xy(shadow, gx, gy)
        base[smask] *= 0.55

        mask = shapely.contains_xy(poly, gx, gy)
        colour = np.array(palette[i % len(palette)], dtype=float)
        base[mask] = colour + rng.normal(0, 7, (mask.sum(), 3))
        # a bright parapet edge, which is what makes roofs pop in real imagery
        edge = shapely.contains_xy(poly, gx, gy) & ~shapely.contains_xy(
            poly.buffer(-0.8), gx, gy)
        base[edge] = np.clip(base[edge] * 1.18, 0, 255)

    base = np.clip(base + rng.normal(0, 2.5, base.shape), 0, 255).astype(np.uint8)
    # image row 0 is north, our row 0 is south
    Image.fromarray(np.flipud(base)).save(path)

    Path(str(path) + ".json").write_text(json.dumps({
        "resolution_m": resolution, "width": w, "height": h,
        "extent_local_m": [x0, y0, x1, y1],
        "row_order": "north-up (row 0 = maximum y)",
        "frame": "project-local ENU metres",
    }, indent=1), encoding="utf-8")
    return Path(path)


def read_orthophoto(path) -> tuple[np.ndarray, dict]:
    """Read an orthophoto and its sidecar georeference, in south-up order."""
    if not HAVE_PIL:
        raise RuntimeError("reading imagery needs Pillow")
    img = np.asarray(Image.open(path).convert("RGB")).astype(float)
    meta_path = Path(str(path) + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return np.flipud(img), meta


# --- vector --------------------------------------------------------------------
def read_geojson(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cadastre_to_geojson(cad, enu: LocalENU, *, kinds=None) -> dict:
    """
    Export the register as GeoJSON so it opens in any GIS.

    GeoJSON is a 2D format, so each object's footprint is written as the
    geometry and its vertical extent is carried in the properties. That is a
    genuine loss of information, which is precisely the limitation this project
    exists to overcome - so the export is offered for interoperability while
    the native format stays the volumetric one.
    """
    features = []
    for o in cad.objects:
        if kinds and o.kind.value not in kinds:
            continue
        fp = o.footprint
        if fp.is_empty:
            continue
        polys = fp.geoms if fp.geom_type == "MultiPolygon" else [fp]
        for poly in polys:
            ring = enu.inverse_ring(list(poly.exterior.coords))
            holes = [enu.inverse_ring(list(h.coords)) for h in poly.interiors]
            features.append({
                "type": "Feature",
                "properties": {
                    "ulpin": str(o.ulpin) if o.ulpin else None,
                    "object_id": o.object_id,
                    "kind": o.kind.value,
                    "name": o.name,
                    "owner": o.owner,
                    "use": o.use,
                    "level": o.level,
                    "stratum": o.stratum.value if o.stratum else None,
                    "z_min": round(o.z_min, 3),
                    "z_max": round(o.z_max, 3),
                    "area_m2": round(o.area, 2),
                    "volume_m3": round(o.volume, 2),
                    "provenance": o.provenance.value,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring, *holes]},
            })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "note": "2D projection of a volumetric register; z extents are in the "
                "properties because GeoJSON cannot express a solid",
        "features": features,
    }


def cadastre_to_citygml_lite(cad, enu: LocalENU) -> str:
    """
    Export a CityGML-flavoured XML view of the register.

    Not a conformant CityGML 3.0 document - producing one properly is a project
    in itself - but it maps our object model onto the standard's vocabulary
    (Building / BuildingStorey / BuildingUnit with lod1Solid geometry), which is
    the interchange path a real land-records department would need. It is
    labelled honestly in the output so nobody mistakes it for the real thing.
    """
    from xml.sax.saxutils import escape

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!-- CityGML-flavoured export from Bhoomi3D. Vocabulary follows '
        'CityGML 3.0 but this is NOT a conformance-checked document. -->',
        '<CityModel xmlns="http://www.opengis.net/citygml/3.0">',
        f'  <name>{escape(cad.name)}</name>',
    ]
    for o in cad.objects:
        tag = {"building": "Building", "storey": "BuildingStorey",
               "unit": "BuildingUnit", "parcel": "LandParcel",
               "infrastructure": "CityFurniture",
               "air_rights": "GenericCityObject"}.get(o.kind.value, "GenericCityObject")
        fp = o.footprint
        if fp.is_empty or fp.geom_type != "Polygon":
            continue
        ring = enu.inverse_ring(list(fp.exterior.coords))
        pos = " ".join(f"{lon:.8f} {lat:.8f} {o.z_min:.3f}" for lon, lat in ring)
        lines += [
            '  <cityObjectMember>',
            f'    <{tag} gml:id="{escape(o.object_id)}">',
            f'      <ulpin>{escape(str(o.ulpin) if o.ulpin else "")}</ulpin>',
            f'      <name>{escape(o.name)}</name>',
            f'      <owner>{escape(o.owner)}</owner>',
            f'      <provenance>{o.provenance.value}</provenance>',
            '      <lod1Solid><Solid><exterior>',
            f'        <basePolygon><posList>{pos}</posList></basePolygon>',
            f'        <height>{o.z_max - o.z_min:.3f}</height>',
            '      </exterior></Solid></lod1Solid>',
            f'    </{tag}>',
            '  </cityObjectMember>',
        ]
    lines.append('</CityModel>')
    return "\n".join(lines)

"""
Sensor simulation: turn a :class:`~bhoomi3d.data.scene.Scene` into the inputs a
real survey would deliver.

The point of simulating rather than shipping a canned cloud is that the
pipeline must be shown to *recover* structure it was not told about. So the
simulator models the specific artefacts that make automated extraction hard:

  * **trees**, which are tall and dense - the dominant false positive
  * **parapets**, which make a roof read taller than its top floor
  * **balcony bands**, the periodic facade structure that floor segmentation
    keys off (real facades show exactly this signature in a photogrammetric or
    oblique-LiDAR cloud)
  * **a floating datum** - the block carries a small unknown shift, tilt and
    scale error until GNSS/CORS control ties it down, which is what actually
    happens to an unconstrained drone block
  * **occlusion and noise** - lower storeys get fewer facade hits because the
    ones above them shadow the sensor

Nothing downstream is told the true answer; the ground truth is kept only so
`bhoomi3d.ai.evaluate` can score the result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from .scene import BuildingSpec, Scene

# LAS-style ASPRS classification codes, so exported clouds are standards-conformant
CLASS_UNCLASSIFIED = 1
CLASS_GROUND = 2
CLASS_VEGETATION = 5
CLASS_BUILDING = 6


@dataclass
class SimulatedSurvey:
    """Everything a survey contractor would hand over."""

    xyz: np.ndarray             # (n, 3) in the *observed* (un-adjusted) frame
    classification: np.ndarray  # (n,) ASPRS class - ground truth, for scoring only
    intensity: np.ndarray       # (n,) uint16
    return_number: np.ndarray   # (n,) uint8
    source_building: np.ndarray # (n,) index into scene.buildings, -1 otherwise
    datum_offset: dict          # the error GNSS control has to remove

    @property
    def n(self) -> int:
        return len(self.xyz)

    def summary(self) -> dict:
        vals, counts = np.unique(self.classification, return_counts=True)
        names = {1: "unclassified", 2: "ground", 5: "vegetation", 6: "building"}
        return {
            "points": self.n,
            "bounds": [round(float(v), 2) for v in
                       (*self.xyz.min(axis=0), *self.xyz.max(axis=0))],
            "by_class": {names.get(int(v), str(int(v))): int(c)
                         for v, c in zip(vals, counts)},
            "datum_offset": self.datum_offset,
        }


# --- helpers ------------------------------------------------------------------
def _sample_in_polygon(poly: Polygon, density: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly sample a polygon at `density` points per square metre.

    Batched rejection sampling against shapely's vectorised `contains_xy` - the
    per-point Python loop this replaces dominated the whole simulation.
    """
    n_target = int(poly.area * density)
    if n_target <= 0:
        return np.empty((0, 2))
    x0, y0, x1, y1 = poly.bounds
    # oversample by the reciprocal of the expected bounding-box hit rate so one
    # batch is almost always enough
    hit_rate = max(poly.area / max((x1 - x0) * (y1 - y0), 1e-9), 0.05)
    out: list[np.ndarray] = []
    got = 0
    for _ in range(12):
        if got >= n_target:
            break
        n_batch = int((n_target - got) / hit_rate * 1.25) + 64
        batch = rng.uniform([x0, y0], [x1, y1], size=(n_batch, 2))
        keep = shapely.contains_xy(poly, batch[:, 0], batch[:, 1])
        sel = batch[keep]
        out.append(sel)
        got += len(sel)
    if not out:
        return np.empty((0, 2))
    return np.vstack(out)[:n_target]


def _wall_segments(world_ring) -> list[tuple[np.ndarray, np.ndarray, float]]:
    ring = list(world_ring)
    segs = []
    for i in range(len(ring)):
        a = np.array(ring[i], dtype=float)
        b = np.array(ring[(i + 1) % len(ring)], dtype=float)
        length = float(np.linalg.norm(b - a))
        if length > 0.1:
            segs.append((a, b, length))
    return segs


def _simulate_building(b: BuildingSpec, terrain, rng: np.random.Generator,
                       *, roof_density: float, facade_density: float):
    """Roof, parapet and facade returns for one structure."""
    ring = b.world_footprint()
    poly = Polygon(ring)
    ground_z = float(terrain.height(*np.array(poly.centroid.coords[0])))
    roof_z = ground_z + b.height_above_ground

    pts: list[np.ndarray] = []
    classes: list[np.ndarray] = []

    # --- roof surface ------------------------------------------------------
    xy = _sample_in_polygon(poly, roof_density, rng)
    if len(xy):
        if b.roof == "gable":
            # ridge along the footprint's long axis; height varies with the
            # perpendicular distance to it
            cx, cy = poly.centroid.x, poly.centroid.y
            th = math.radians(b.rotation_deg)
            axis = np.array([math.cos(th), math.sin(th)])
            perp = np.array([-axis[1], axis[0]])
            d = (xy - np.array([cx, cy])) @ perp
            span = max(np.abs(d).max(), 1e-6)
            z = roof_z + 2.4 * (1.0 - np.abs(d) / span)
        else:
            # flat roofs are never truly flat - they are laid to falls for drainage
            z = np.full(len(xy), roof_z) + 0.05 * np.sin(xy[:, 0] * 0.3)
        pts.append(np.column_stack([xy, z]))
        classes.append(np.full(len(xy), CLASS_BUILDING))

    # --- parapet ring ------------------------------------------------------
    # A parapet is why "roof height" and "top floor ceiling" are different
    # numbers, and the pipeline has to notice that or every top flat is wrong.
    if b.roof == "flat" and b.parapet > 0:
        peri = poly.exterior.length
        n_par = int(peri * 12)
        t = rng.uniform(0, peri, n_par)
        along = shapely.line_interpolate_point(poly.exterior, t)
        par_xy = shapely.get_coordinates(along)
        par_z = roof_z + rng.uniform(0, b.parapet, n_par)
        pts.append(np.column_stack([par_xy, par_z]))
        classes.append(np.full(n_par, CLASS_BUILDING))

    # --- facades -----------------------------------------------------------
    for a, c, length in _wall_segments(ring):
        n_wall = int(length * b.height_above_ground * facade_density)
        if n_wall <= 0:
            continue
        t = rng.uniform(0, 1, n_wall)
        base = a + (c - a) * t[:, None]
        z = rng.uniform(ground_z, roof_z, n_wall)
        # occlusion: a sensor looking down sees less of the lower storeys
        vis = 0.35 + 0.65 * (z - ground_z) / max(roof_z - ground_z, 1e-6)
        keep = rng.uniform(0, 1, n_wall) < vis
        base, z = base[keep], z[keep]
        if len(z):
            pts.append(np.column_stack([base, z]))
            classes.append(np.full(len(z), CLASS_BUILDING))

        # balcony bands: extra returns just above each floor slab, pushed
        # slightly proud of the wall. This is the periodic vertical signal that
        # makes floor segmentation possible from an exterior scan alone.
        normal = np.array([-(c - a)[1], (c - a)[0]])
        normal = normal / max(np.linalg.norm(normal), 1e-9)
        for f in b.above_ground_floors:
            slab_z = ground_z + f.z_bottom
            n_bal = int(length * 9)
            if n_bal <= 0:
                continue
            tb = rng.uniform(0.08, 0.92, n_bal)
            xy_b = a + (c - a) * tb[:, None] + normal * rng.uniform(0.55, 0.95, (n_bal, 1))
            zb = slab_z + rng.uniform(-0.05, 0.45, n_bal)
            visb = 0.4 + 0.6 * (slab_z - ground_z) / max(roof_z - ground_z, 1e-6)
            keepb = rng.uniform(0, 1, n_bal) < visb
            if keepb.any():
                pts.append(np.column_stack([xy_b[keepb], zb[keepb]]))
                classes.append(np.full(int(keepb.sum()), CLASS_BUILDING))

    if not pts:
        return np.empty((0, 3)), np.empty(0, dtype=int)
    return np.vstack(pts), np.concatenate(classes)


def _simulate_tree(t, terrain, rng: np.random.Generator) -> np.ndarray:
    """A canopy as a scattered ellipsoid shell plus a thin trunk."""
    base_z = float(terrain.height(t.x, t.y))
    n_can = int(rng.integers(240, 460))
    # points scattered through the canopy volume, biased toward the shell -
    # volumetric scatter is exactly what distinguishes foliage from a roof
    u = rng.normal(size=(n_can, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(0.55, 1.0, (n_can, 1)) ** (1 / 3)
    canopy = u * r * np.array([t.radius, t.radius, t.height * 0.34])
    canopy[:, 2] += base_z + t.height * 0.66
    canopy[:, 0] += t.x
    canopy[:, 1] += t.y

    n_trunk = 24
    trunk = np.column_stack([
        t.x + rng.normal(0, 0.12, n_trunk),
        t.y + rng.normal(0, 0.12, n_trunk),
        base_z + rng.uniform(0, t.height * 0.6, n_trunk),
    ])
    return np.vstack([canopy, trunk])


def simulate_lidar(scene: Scene, *, ground_density: float = 14.0,
                   roof_density: float = 26.0, facade_density: float = 2.6,
                   noise_sigma: float = 0.025, seed: int = 2024,
                   apply_datum_error: bool = True) -> SimulatedSurvey:
    """
    Simulate an oblique drone-LiDAR survey of the scene.

    Densities are chosen to match a typical 80 m AGL survey with a mid-range
    sensor: roughly 14 pts/m2 on open ground, denser on roofs where the beam is
    near-normal, and sparse on facades where it is grazing.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = scene.extent

    chunks: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    src: list[np.ndarray] = []

    # --- ground ------------------------------------------------------------
    n_ground = int((x1 - x0) * (y1 - y0) * ground_density)
    gx = rng.uniform(x0, x1, n_ground)
    gy = rng.uniform(y0, y1, n_ground)
    gz = scene.terrain.height(gx, gy)
    ground = np.column_stack([gx, gy, gz])

    # remove ground returns that fall under a building - the sensor cannot see
    # through a roof, and leaving them in would make the ground filter's job
    # artificially easy
    keep = np.ones(len(ground), dtype=bool)
    for b in scene.buildings:
        poly = Polygon(b.world_footprint())
        keep &= ~shapely.contains_xy(poly, ground[:, 0], ground[:, 1])
    ground = ground[keep]
    chunks.append(ground)
    classes.append(np.full(len(ground), CLASS_GROUND))
    src.append(np.full(len(ground), -1))

    # --- buildings ---------------------------------------------------------
    for bi, b in enumerate(scene.buildings):
        p, c = _simulate_building(b, scene.terrain, rng,
                                  roof_density=roof_density,
                                  facade_density=facade_density)
        if len(p):
            chunks.append(p)
            classes.append(c)
            src.append(np.full(len(p), bi))

    # --- vegetation --------------------------------------------------------
    for t in scene.trees:
        p = _simulate_tree(t, scene.terrain, rng)
        chunks.append(p)
        classes.append(np.full(len(p), CLASS_VEGETATION))
        src.append(np.full(len(p), -1))

    xyz = np.vstack(chunks)
    classification = np.concatenate(classes).astype(np.uint8)
    source = np.concatenate(src).astype(int)

    # --- sensor noise ------------------------------------------------------
    xyz = xyz + rng.normal(0, noise_sigma, xyz.shape)
    # vertical accuracy is typically worse than horizontal on airborne systems
    xyz[:, 2] += rng.normal(0, noise_sigma * 1.6, len(xyz))

    # --- intensity and returns --------------------------------------------
    intensity = np.where(classification == CLASS_VEGETATION,
                         rng.integers(3000, 12000, len(xyz)),
                         rng.integers(14000, 42000, len(xyz))).astype(np.uint16)
    # foliage is the main source of multiple returns - a beam that clips a leaf
    # keeps going. Roofs and roads give a single return.
    return_number = np.where(
        classification == CLASS_VEGETATION,
        rng.integers(1, 4, len(xyz)), np.ones(len(xyz), dtype=int)).astype(np.uint8)

    # --- floating datum ----------------------------------------------------
    offset = {"shift_m": [0.0, 0.0, 0.0], "rotation_deg": 0.0, "scale_ppm": 0.0}
    if apply_datum_error:
        shift = np.array([0.842, -1.317, 0.406])
        yaw = math.radians(0.093)          # a small residual block rotation
        scale = 1.0 + 62e-6                # 62 ppm - a plausible block scale error
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        centre = xyz.mean(axis=0)
        xyz = scale * ((xyz - centre) @ R.T) + centre + shift
        offset = {"shift_m": shift.round(4).tolist(),
                  "rotation_deg": round(math.degrees(yaw), 6),
                  "scale_ppm": round((scale - 1) * 1e6, 3)}

    return SimulatedSurvey(
        xyz=xyz, classification=classification, intensity=intensity,
        return_number=return_number, source_building=source,
        datum_offset=offset,
    )


def simulate_control_network(scene: Scene, survey: SimulatedSurvey, *,
                             n_points: int = 7, sigma: float = 0.015,
                             blunder_index: int | None = 4,
                             seed: int = 99) -> list[dict]:
    """
    Simulate a GNSS/CORS ground-control network over the site.

    Control points are placed on open ground and observed twice: once in the
    drone block's floating frame (`observed`) and once against the CORS network
    (`reference`). One point is given a deliberate blunder - a transposed
    coordinate, the classic field error - so the Helmert fit's outlier rejection
    has something real to catch.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = scene.extent

    # place control well spread out, away from structures
    candidates = []
    while len(candidates) < n_points:
        x = rng.uniform(x0 + 8, x1 - 8)
        y = rng.uniform(y0 + 8, y1 - 8)
        if any(Polygon(b.world_footprint()).buffer(9).contains(Point(x, y))
               for b in scene.buildings):
            continue
        candidates.append((x, y))

    out = []
    d = survey.datum_offset
    shift = np.array(d["shift_m"])
    yaw = math.radians(d["rotation_deg"])
    scale = 1.0 + d["scale_ppm"] * 1e-6
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = survey.xyz.mean(axis=0)
    # undo the datum error to recover where the block *thinks* the point is;
    # the CORS value is the truth
    for i, (x, y) in enumerate(candidates):
        z = float(scene.terrain.height(x, y))
        truth = np.array([x, y, z])
        observed = scale * ((truth - centre) @ R.T) + centre + shift
        observed = observed + rng.normal(0, 0.008, 3)
        reference = truth + rng.normal(0, sigma, 3)
        if blunder_index is not None and i == blunder_index:
            # transposed easting/northing digits - a real and common blunder
            reference = reference + np.array([0.0, 0.62, 0.0])
        out.append({
            "name": f"GCP-{i+1:02d}",
            "observed": [round(float(v), 4) for v in observed],
            "reference": [round(float(v), 4) for v in reference],
            "sigma": sigma,
            "is_blunder": bool(blunder_index is not None and i == blunder_index),
        })
    return out


def simulate_floor_plans(scene: Scene) -> list[dict]:
    """
    Emit the registered floor plans for every building.

    A floor plan is the *legal* description of what is inside a structure. The
    point cloud can tell us where the building is and how many storeys it has,
    but only the approved plan says where one flat ends and the next begins - so
    the pipeline fuses the two rather than choosing between them.
    """
    plans = []
    for b in scene.buildings:
        ground_z = float(scene.terrain.height(*b.origin))
        floors = []
        # The plan shows only what was sanctioned. Where a building has been
        # extended beyond its approval, the extra storeys are simply absent
        # here - which is exactly why the as-built scan has to be compared
        # against the plan rather than trusted to agree with it.
        sanctioned = [f for f in b.floors
                      if b.registered_floors is None
                      or f.index < b.registered_floors]
        for f in sanctioned:
            floors.append({
                "index": f.index,
                "z_bottom_local": round(f.z_bottom, 3),
                "z_top_local": round(f.z_top, 3),
                "use": f.use,
                "units": [
                    {
                        "name": u.name,
                        "kind": u.kind,
                        "owner": u.owner,
                        "preferred_unit_no": u.preferred_unit_no,
                        "ring": [[round(x, 3), round(y, 3)]
                                 for x, y in b.world_rect(u.rect)],
                    }
                    for u in f.units
                ],
            })
        plans.append({
            "building_id": b.id,
            "building_name": b.name,
            "parcel_id": b.parcel_id,
            "rotation_deg": b.rotation_deg,
            "ground_level_z": round(ground_z, 3),
            "outline": [[round(x, 3), round(y, 3)] for x, y in b.world_footprint()],
            "floors": floors,
        })
    return plans


def simulate_parcels_geojson(scene: Scene, enu) -> dict:
    """
    Emit the existing 2D cadastral layer as GeoJSON in WGS-84.

    This is the data a land-records department already holds. The platform must
    consume it as-is rather than replacing it - the 3D cadastre is an extension
    of the existing record, not a competitor to it.
    """
    features = []
    for p in scene.parcels:
        ring = list(p.ring) + [p.ring[0]]
        lonlat = enu.inverse_ring(ring)
        features.append({
            "type": "Feature",
            "properties": {
                "parcel_id": p.id,
                "owner": p.owner,
                "land_use": p.land_use,
                "survey_no": p.survey_no,
                "area_m2": round(Polygon(p.ring).area, 2),
            },
            "geometry": {"type": "Polygon", "coordinates": [lonlat]},
        })
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features}


def simulate_corridor_records(scene: Scene) -> list[dict]:
    """Utility-authority records for the subsurface corridors."""
    return [
        {
            "id": c.id, "name": c.name, "kind": c.kind, "owner": c.owner,
            "alignment": [[round(x, 3), round(y, 3)] for x, y in c.alignment],
            "width_m": c.width, "height_m": c.height,
            "invert_levels_m": [round(v, 3) for v in c.invert_levels],
            "length_m": round(LineString(c.alignment).length, 2),
        }
        for c in scene.corridors
    ]

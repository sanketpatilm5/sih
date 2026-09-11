"""
Volumetric geometry kernel.

A 2D cadastre stores polygons. A 3D cadastre has to store *solids*, and the
overwhelming majority of legal solids in a built environment are **prisms**: a
horizontal footprint swept between two elevations. A flat, a floor slab, a
basement, a parking bay, a stretch of pipeline, a segment of metro tunnel - all
of them are, to the accuracy the law cares about, a footprint plus a z-range.

Restricting the primitive to a prism is what makes exact, fast answers possible:
the intersection of two prisms is the (2D) intersection of their footprints
swept over the overlap of their z-ranges, so volumes and clearances come out in
closed form instead of needing a mesh boolean library.

Genuinely non-prismatic solids (a sloping tunnel, a stepped podium) are modelled
as a :class:`CompositeSolid` - an ordered chain of prisms - which keeps every
operation exact while covering the awkward cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

# A tolerance of 1 mm: finer than any cadastral survey, coarse enough to absorb
# floating-point noise from the extraction pipeline.
EPS = 1e-3


# --- helpers ------------------------------------------------------------------
def clean_polygon(poly: BaseGeometry, *, min_area: float = 1e-6) -> Optional[Polygon]:
    """
    Repair a polygon into something a cadastre can store.

    Extraction pipelines routinely emit bow-ties and zero-width spikes; storing
    those would make every downstream area and overlap computation meaningless,
    so we repair first and keep the largest resulting ring.
    """
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.geom_type == "GeometryCollection":
        parts = [g for g in poly.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not parts:
            return None
        poly = unary_union(parts)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.area < min_area:
        return None
    return orient_ccw(poly)


def orient_ccw(poly: Polygon) -> Polygon:
    """Normalise ring winding: exterior counter-clockwise, holes clockwise."""
    from shapely.geometry.polygon import orient
    return orient(poly, sign=1.0)


def polygon_from_rings(rings: Sequence[Sequence[Sequence[float]]]) -> Polygon:
    """Build a polygon from [exterior, *holes] coordinate rings."""
    if not rings:
        raise ValueError("at least an exterior ring is required")
    return orient_ccw(Polygon(rings[0], rings[1:]))


def rings_from_polygon(poly: Polygon) -> list[list[list[float]]]:
    """Serialise a polygon to [exterior, *holes], each a closed coordinate ring."""
    out = [[[round(x, 4), round(y, 4)] for x, y in poly.exterior.coords]]
    for hole in poly.interiors:
        out.append([[round(x, 4), round(y, 4)] for x, y in hole.coords])
    return out


# --- the prism ----------------------------------------------------------------
@dataclass
class Prism:
    """
    A right prism: `footprint` swept from `z_min` to `z_max`.

    All coordinates are metres in the project-local ENU frame, so `z` is height
    above the project datum - negative underground, which is exactly how a
    basement or a tunnel wants to be described.
    """

    footprint: Polygon
    z_min: float
    z_max: float
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.z_max < self.z_min:
            self.z_min, self.z_max = self.z_max, self.z_min
        cleaned = clean_polygon(self.footprint)
        if cleaned is None:
            raise ValueError("prism footprint is empty or degenerate")
        self.footprint = cleaned

    # -- measures ------------------------------------------------------------
    @property
    def height(self) -> float:
        return self.z_max - self.z_min

    @property
    def area(self) -> float:
        """Plan area (the 'carpet area' in cadastral terms), m2."""
        return self.footprint.area

    @property
    def volume(self) -> float:
        return self.footprint.area * self.height

    @property
    def lateral_area(self) -> float:
        """Area of the vertical faces, m2 - the party-wall surface."""
        return self.footprint.length * self.height

    @property
    def centroid(self) -> tuple[float, float, float]:
        c = self.footprint.centroid
        return c.x, c.y, (self.z_min + self.z_max) / 2

    @property
    def bbox(self) -> tuple[float, float, float, float, float, float]:
        x0, y0, x1, y1 = self.footprint.bounds
        return x0, y0, self.z_min, x1, y1, self.z_max

    # -- predicates ----------------------------------------------------------
    def z_overlap(self, other: "Prism") -> float:
        """Length of the shared z-interval, 0 if the prisms are stacked clear."""
        return max(0.0, min(self.z_max, other.z_max) - max(self.z_min, other.z_min))

    def bbox_disjoint(self, other: "Prism", tol: float = 0.0) -> bool:
        """Cheap rejection test used to keep pairwise validation near-linear."""
        a, b = self.bbox, other.bbox
        return (a[3] < b[0] - tol or b[3] < a[0] - tol or
                a[4] < b[1] - tol or b[4] < a[1] - tol or
                a[5] < b[2] - tol or b[5] < a[2] - tol)

    def intersects(self, other: "Prism", tol: float = EPS) -> bool:
        if self.bbox_disjoint(other, tol):
            return False
        if self.z_overlap(other) <= tol:
            return False
        return self.footprint.intersects(other.footprint)

    def contains_point(self, x: float, y: float, z: float, tol: float = EPS) -> bool:
        if not (self.z_min - tol <= z <= self.z_max + tol):
            return False
        return self.footprint.covers(Point(x, y))

    # -- boolean ops ---------------------------------------------------------
    def intersection(self, other: "Prism") -> Optional["Prism"]:
        """The shared solid, or None. Exact: prism ^ prism is a prism."""
        dz = self.z_overlap(other)
        if dz <= 0:
            return None
        inter = self.footprint.intersection(other.footprint)
        fp = clean_polygon(inter)
        if fp is None:
            return None
        return Prism(fp, max(self.z_min, other.z_min), min(self.z_max, other.z_max))

    def intersection_volume(self, other: "Prism") -> float:
        """Volume of overlap in m3 - zero when the objects are legally disjoint."""
        dz = self.z_overlap(other)
        if dz <= 0:
            return 0.0
        try:
            a = self.footprint.intersection(other.footprint).area
        except Exception:
            a = make_valid(self.footprint).intersection(
                make_valid(other.footprint)).area
        return a * dz

    def clearance(self, other: "Prism") -> float:
        """
        Shortest 3D distance between the two solids, in metres. 0 if they touch
        or interpenetrate.

        Exact for prisms: the horizontal and vertical separations are
        independent, so the minimum distance is their Pythagorean combination.
        This is the number that matters for "is this foundation too close to the
        metro tunnel?".
        """
        dxy = self.footprint.distance(other.footprint)
        dz = max(0.0, max(self.z_min, other.z_min) - min(self.z_max, other.z_max))
        return math.hypot(dxy, dz)

    def buffered(self, distance: float) -> "Prism":
        """Grow the solid by `distance` in every direction - the safety envelope."""
        return Prism(self.footprint.buffer(distance),
                     self.z_min - distance, self.z_max + distance,
                     dict(self.attributes))

    # -- serialisation -------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "type": "Prism",
            "rings": rings_from_polygon(self.footprint),
            "z_min": round(self.z_min, 4),
            "z_max": round(self.z_max, 4),
            "area_m2": round(self.area, 3),
            "volume_m3": round(self.volume, 3),
            "height_m": round(self.height, 3),
            "bbox": [round(v, 4) for v in self.bbox],
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prism":
        return cls(polygon_from_rings(d["rings"]), float(d["z_min"]),
                   float(d["z_max"]), dict(d.get("attributes") or {}))

    @classmethod
    def from_box(cls, x0: float, y0: float, x1: float, y1: float,
                 z_min: float, z_max: float, **attrs) -> "Prism":
        return cls(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
                   z_min, z_max, attrs)


# --- composite solids ---------------------------------------------------------
@dataclass
class CompositeSolid:
    """
    An ordered chain of prisms forming one legal object.

    This is how a sloping metro tunnel or a stepped podium is represented: each
    segment is a prism, and the union of segments is the object. Every measure
    stays exact because the segments are disjoint by construction.
    """

    parts: list[Prism]
    attributes: dict = field(default_factory=dict)

    @property
    def volume(self) -> float:
        return sum(p.volume for p in self.parts)

    @property
    def bbox(self) -> tuple[float, ...]:
        boxes = np.array([p.bbox for p in self.parts])
        return (*boxes[:, :3].min(axis=0), *boxes[:, 3:].max(axis=0))

    @property
    def footprint(self) -> BaseGeometry:
        return unary_union([p.footprint for p in self.parts])

    @property
    def z_min(self) -> float:
        return min(p.z_min for p in self.parts)

    @property
    def z_max(self) -> float:
        return max(p.z_max for p in self.parts)

    @property
    def centroid(self) -> tuple[float, float, float]:
        """Volume-weighted centroid - the legally meaningful centre of the solid."""
        tot = self.volume
        if tot <= 0:
            c = self.footprint.centroid
            return c.x, c.y, (self.z_min + self.z_max) / 2
        acc = np.zeros(3)
        for p in self.parts:
            acc += np.array(p.centroid) * p.volume
        return tuple(acc / tot)

    def intersection_volume(self, other) -> float:
        others = other.parts if isinstance(other, CompositeSolid) else [other]
        return sum(a.intersection_volume(b) for a in self.parts for b in others)

    def clearance(self, other) -> float:
        others = other.parts if isinstance(other, CompositeSolid) else [other]
        return min(a.clearance(b) for a in self.parts for b in others)

    def as_dict(self) -> dict:
        return {
            "type": "CompositeSolid",
            "parts": [p.as_dict() for p in self.parts],
            "volume_m3": round(self.volume, 3),
            "bbox": [round(v, 4) for v in self.bbox],
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeSolid":
        return cls([Prism.from_dict(p) for p in d["parts"]],
                   dict(d.get("attributes") or {}))


def sweep_corridor(alignment: Sequence[Sequence[float]], *, width: float,
                   height: float, invert_levels: Sequence[float],
                   cap_style: int = 2, **attrs) -> CompositeSolid:
    """
    Build a linear underground corridor - pipeline, culvert, metro tunnel.

    `alignment` is a chain of (x, y) points in the local frame; `invert_levels`
    gives the z of the corridor floor at each of those points, so the corridor
    can descend. Each span between consecutive points becomes one prism whose
    z-range covers the span's own gradient, which keeps the solid conservative
    (it fully contains the true sloping tube) without needing a mesh boolean.
    """
    pts = np.asarray(alignment, dtype=float)
    if len(pts) < 2:
        raise ValueError("a corridor alignment needs at least two points")
    inv = np.asarray(invert_levels, dtype=float)
    if len(inv) != len(pts):
        raise ValueError("invert_levels must have one entry per alignment point")

    parts: list[Prism] = []
    for i in range(len(pts) - 1):
        seg = LineString([pts[i], pts[i + 1]])
        fp = seg.buffer(width / 2.0, cap_style=cap_style, join_style=2)
        z0, z1 = float(inv[i]), float(inv[i + 1])
        parts.append(Prism(fp, min(z0, z1), max(z0, z1) + height,
                           {"segment": i}))
    return CompositeSolid(parts, dict(attrs))


# --- footprint regularisation -------------------------------------------------
def dominant_orientation(poly: Polygon, *, weight_by_length: bool = True) -> float:
    """
    Estimate a building's principal wall direction, in radians within [0, pi/2).

    Real buildings are overwhelmingly rectilinear, but a footprint traced from a
    point cloud is not. Recovering the dominant axis lets us square the outline
    up, which is both visually and legally much closer to the truth than a ragged
    traced edge. Edges are binned by direction modulo 90 degrees and weighted by
    length, so a few short noisy edges cannot outvote the main walls.
    """
    coords = np.asarray(poly.exterior.coords)
    d = np.diff(coords, axis=0)
    lengths = np.hypot(d[:, 0], d[:, 1])
    keep = lengths > 0.3  # ignore sub-decimetre traced noise
    if not keep.any():
        return 0.0
    ang = np.arctan2(d[keep, 1], d[keep, 0]) % (math.pi / 2)
    w = lengths[keep] if weight_by_length else np.ones(keep.sum())
    # circular mean over the quarter-turn-periodic angle
    z = (w * np.exp(4j * ang)).sum()
    return (np.angle(z) / 4.0) % (math.pi / 2)


def regularize_footprint(poly: Polygon, *, angle_tol_deg: float = 22.5,
                         simplify_tol: float = 0.35,
                         min_edge: float = 0.6) -> Polygon:
    """
    Square a traced footprint up to its dominant axis.

    The outline is rotated onto the dominant orientation, simplified, then each
    edge that is within `angle_tol_deg` of an axis is snapped to it; edges that
    are genuinely oblique (a splayed corner plot, a curved facade) are left
    alone. Finally the polygon is rotated back. This is the standard
    "rectilinear regularisation" step in an automated cadastral pipeline.
    """
    poly = clean_polygon(poly)
    if poly is None:
        raise ValueError("cannot regularise an empty footprint")
    theta = dominant_orientation(poly)
    rot = affinity.rotate(poly, -math.degrees(theta), origin="centroid")
    rot = rot.simplify(simplify_tol, preserve_topology=True)

    coords = list(rot.exterior.coords)[:-1]
    n = len(coords)
    if n < 4:
        return affinity.rotate(rot, math.degrees(theta), origin="centroid")

    pts = np.array(coords, dtype=float)
    snapped = pts.copy()
    tol = math.radians(angle_tol_deg)
    for i in range(n):
        j = (i + 1) % n
        dx, dy = pts[j] - pts[i]
        if math.hypot(dx, dy) < min_edge:
            continue
        a = math.atan2(dy, dx)
        # distance to the nearest axis direction
        if min(abs(a), abs(abs(a) - math.pi)) < tol:          # near-horizontal
            mid = (snapped[i, 1] + snapped[j, 1]) / 2
            snapped[i, 1] = snapped[j, 1] = mid
        elif abs(abs(a) - math.pi / 2) < tol:                 # near-vertical
            mid = (snapped[i, 0] + snapped[j, 0]) / 2
            snapped[i, 0] = snapped[j, 0] = mid

    out = clean_polygon(Polygon(snapped))
    if out is None:
        out = rot
    result = affinity.rotate(out, math.degrees(theta), origin="centroid")
    # a regularisation that loses or invents more than a quarter of the area has
    # gone wrong; fall back to the honest traced outline rather than a fiction
    if abs(result.area - poly.area) > 0.25 * poly.area:
        return poly
    return clean_polygon(result) or poly


def alpha_shape(points: np.ndarray, alpha: float) -> Optional[Polygon]:
    """
    Concave hull of a 2D point set via the alpha-shape of its Delaunay mesh.

    A convex hull would bridge across courtyards and L-shaped wings, inflating
    every area it touches. The alpha shape keeps only triangles whose
    circumradius is below `alpha`, which traces the real outline of a building
    footprint recovered from LiDAR.
    """
    from scipy.spatial import Delaunay

    pts = np.asarray(points, dtype=float)[:, :2]
    if len(pts) < 4:
        if len(pts) < 3:
            return None
        return clean_polygon(Polygon(pts))

    try:
        tri = Delaunay(pts)
    except Exception:
        return None

    a = pts[tri.simplices[:, 0]]
    b = pts[tri.simplices[:, 1]]
    c = pts[tri.simplices[:, 2]]
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    s = (la + lb + lc) / 2.0
    area = np.sqrt(np.maximum(s * (s - la) * (s - lb) * (s - lc), 1e-12))
    circum_r = (la * lb * lc) / (4.0 * area)

    keep = circum_r < alpha
    if not keep.any():
        return clean_polygon(Polygon(pts).convex_hull)

    triangles = [Polygon(pts[simplex]) for simplex in tri.simplices[keep]]
    merged = unary_union(triangles)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda g: g.area)
    return clean_polygon(merged)


# --- packing for the viewer ---------------------------------------------------
def prism_to_mesh_payload(prism: Prism) -> dict:
    """
    Compact description the browser can extrude directly.

    We deliberately ship rings plus a z-range rather than a triangulated mesh:
    it is an order of magnitude less data over the wire, and the viewer's
    extrusion step is hardware-accelerated anyway.
    """
    return {
        "rings": rings_from_polygon(prism.footprint),
        "z_min": round(prism.z_min, 4),
        "z_max": round(prism.z_max, 4),
    }


def bbox_union(boxes: Iterable[Sequence[float]]) -> Optional[list[float]]:
    arr = np.array([list(b) for b in boxes], dtype=float)
    if len(arr) == 0:
        return None
    return [*arr[:, :3].min(axis=0).tolist(), *arr[:, 3:].max(axis=0).tolist()]

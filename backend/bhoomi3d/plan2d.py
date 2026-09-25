"""
2D plan input: draw it flat, get it back as a 3D register.

This is the shortest path through the whole platform, and for most Indian
municipal offices it is the *only* path they can walk today: they already hold
approved 2D building plans, and almost none of them hold LiDAR.

The idea is simple enough to state in one line:

    a 2D polygon + a floor level + a floor height  =  a 3D property volume

Everything else - the identifier, the ownership hierarchy, the validation, the
viewer - works exactly the same whether the polygon came from a surveyor's
drawing or from an automatically extracted point cloud. So this module gives
the drawing a way in.

Two entry points into the same register
---------------------------------------
``plan2d.build_register()``   2D plans only. No drone, no LiDAR, no GPU.
                              Produces the *sanctioned* state of the site.
``pipeline.run_pipeline()``   Adds a drone survey on top, which measures the
                              *as-built* state and can contradict the plan.

The second is what finds unauthorised floors. The first is what a tehsil office
can run this afternoon.

The input format is deliberately hand-writable: metres, plain nested JSON, no
CRS boilerplate, and a ``repeat_to`` shorthand so a 12-storey tower with
identical floors is a dozen lines rather than a dozen copies.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .core.cadastre import (Cadastre, CadastralObject, ObjectKind, Provenance)
from .core.crs import GeodeticOrigin, LocalENU
from .core.geometry3d import Prism, clean_polygon
from .core.topology import Severity, validate
from .core.ulpin import JurisdictionCode, Stratum

# A structure's development-rights column. Policy, not measurement - carried as
# an auditable attribute rather than hard-coded into the geometry.
AIR_RIGHTS_HEIGHT_M = 30.0

# Wall thickness used to grow a building's envelope out from its unit polygons
# when no explicit outline is given. Also closes the gaps *between* units, so
# the envelope comes out as one solid rather than a scatter of rooms.
DEFAULT_WALL_M = 0.35


class PlanError(ValueError):
    """A plan that cannot be turned into a register, with a fixable reason."""


# --- the format ----------------------------------------------------------------
@dataclass
class Plan2D:
    """A parsed and validated 2D plan."""

    name: str
    origin: dict
    jurisdiction: dict
    parcels: list[dict]
    buildings: list[dict]
    infrastructure: list[dict] = field(default_factory=list)
    source: str = "uploaded plan"

    @property
    def n_units(self) -> int:
        return sum(len(f["units"]) for b in self.buildings
                   for f in expand_floors(b))

    def summary(self) -> dict:
        return {
            "site": self.name,
            "parcels": len(self.parcels),
            "buildings": len(self.buildings),
            "storeys": sum(len(expand_floors(b)) for b in self.buildings),
            "units": self.n_units,
            "infrastructure": len(self.infrastructure),
        }


def _ring(raw: Any, where: str) -> list[list[float]]:
    """Validate one coordinate ring and return it as a list of [x, y] pairs."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise PlanError(
            f"{where}: a boundary needs at least 3 corner points, got "
            f"{0 if raw is None else len(raw)}")
    out = []
    for i, pt in enumerate(raw):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise PlanError(
                f"{where}: corner {i} should be a pair like [12.5, 30.0], "
                f"got {pt!r}")
        try:
            out.append([float(pt[0]), float(pt[1])])
        except (TypeError, ValueError):
            raise PlanError(
                f"{where}: corner {i} has a non-numeric coordinate: {pt!r}")
    return out


def _polygon(raw: Any, where: str) -> Polygon:
    poly = clean_polygon(Polygon(_ring(raw, where)))
    if poly is None:
        raise PlanError(
            f"{where}: the corner points do not enclose an area. Check that "
            f"they trace the outline in order (clockwise or anti-clockwise), "
            f"rather than jumping across the shape.")
    return poly


def parse_plan(doc: dict, *, source: str = "uploaded plan") -> Plan2D:
    """
    Parse and validate a 2D plan document.

    Every error message names the exact building, floor or unit at fault and
    what a correct value looks like. Someone hand-writing a plan file will hit
    these, so they have to be usable without reading this source.
    """
    if not isinstance(doc, dict):
        raise PlanError("the plan should be a JSON object with a 'site' key")

    site = doc.get("site") or {}
    origin = site.get("origin") or {}
    if "lat" not in origin or "lon" not in origin:
        raise PlanError(
            "site.origin needs 'lat' and 'lon' - the real-world position that "
            "local coordinate (0, 0) sits at. Example: "
            '"origin": {"lat": 18.520430, "lon": 73.856744}')
    try:
        lat, lon = float(origin["lat"]), float(origin["lon"])
    except (TypeError, ValueError):
        raise PlanError("site.origin lat/lon must be numbers in degrees")
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise PlanError(
            f"site.origin is not on Earth: lat {lat}, lon {lon}. Latitude runs "
            f"-90..90 and longitude -180..180.")

    juris = dict(site.get("jurisdiction") or {})
    for key, default in (("state", "27"), ("district", "000"), ("tehsil", "000")):
        juris.setdefault(key, default)

    parcels = []
    for i, p in enumerate(doc.get("parcels") or []):
        pid = p.get("id") or f"P-{i + 1:02d}"
        poly = _polygon(p.get("boundary"), f"parcel {pid}")
        parcels.append({**p, "id": pid, "_poly": poly})

    buildings = []
    for i, b in enumerate(doc.get("buildings") or []):
        bid = b.get("id") or f"B-{i + 1:02d}"
        floors = b.get("floors") or []
        if not floors:
            raise PlanError(f"building {bid}: needs at least one entry in 'floors'")
        buildings.append({**b, "id": bid})

    if not buildings and not parcels:
        raise PlanError(
            "the plan has neither parcels nor buildings - nothing to register")

    infra = []
    for i, c in enumerate(doc.get("infrastructure") or []):
        cid = c.get("id") or f"INF-{i + 1:02d}"
        align = c.get("alignment")
        if not isinstance(align, (list, tuple)) or len(align) < 2:
            raise PlanError(
                f"infrastructure {cid}: 'alignment' needs at least 2 points "
                f"tracing the route, e.g. [[0, 15], [60, 15]]")
        infra.append({**c, "id": cid})

    plan = Plan2D(
        name=site.get("name") or "untitled site",
        origin={"lat": lat, "lon": lon,
                "height": float(origin.get("height", 0.0)),
                "label": site.get("name", "plan origin")},
        jurisdiction=juris, parcels=parcels, buildings=buildings,
        infrastructure=infra, source=source,
    )

    # Resolve floors and outlines here rather than during the build, so that
    # every structural problem in the document surfaces at parse time. The
    # preview endpoint only parses, so this is what lets someone see "this
    # building has no shape" before committing to a build.
    for b in plan.buildings:
        expand_floors(b)
        building_outline(b)
    return plan


def load_plan(path) -> Plan2D:
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(
            f"{path.name} is not valid JSON - {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}. A missing or trailing comma is the usual cause.")
    return parse_plan(doc, source=str(path))


# --- floor expansion -----------------------------------------------------------
def expand_floors(building: dict) -> list[dict]:
    """
    Expand ``repeat_to`` shorthand into one concrete entry per storey.

    A 12-storey tower with identical floors should be a few lines of plan, not
    twelve copies of the same polygons. ``{n}`` in a unit name is replaced by
    the floor number, so ``FLAT-{n}02`` becomes FLAT-102, FLAT-202 and so on -
    which is how flats are actually numbered.
    """
    cached = building.get("_floors")
    if cached is not None:
        return cached

    bid = building.get("id", "?")
    out: list[dict] = []
    seen: set[int] = set()

    for entry in building["floors"]:
        if "level" not in entry:
            raise PlanError(
                f"building {bid}: every floor needs a 'level' "
                f"(0 = ground floor, 1 = first floor, -1 = first basement)")
        try:
            start = int(entry["level"])
            stop = int(entry.get("repeat_to", start))
        except (TypeError, ValueError):
            raise PlanError(f"building {bid}: 'level' and 'repeat_to' must be whole numbers")
        if stop < start:
            raise PlanError(
                f"building {bid}, level {start}: 'repeat_to' ({stop}) cannot be "
                f"below 'level' ({start})")

        height = float(entry.get("height", 3.0))
        if not 1.8 <= height <= 12.0:
            raise PlanError(
                f"building {bid}, level {start}: floor height {height} m is "
                f"outside the plausible range 1.8-12 m. Heights are in metres.")

        for level in range(start, stop + 1):
            if level in seen:
                raise PlanError(
                    f"building {bid}: level {level} is defined twice. Check the "
                    f"'repeat_to' ranges do not overlap.")
            seen.add(level)

            units = []
            for j, u in enumerate(entry.get("units") or []):
                name = str(u.get("name") or f"UNIT-{j + 1}")
                name = name.replace("{n}", str(level))
                owner = str(u.get("owner") or "").replace("{n}", str(level))
                poly = _polygon(u.get("boundary"),
                                f"building {bid}, level {level}, unit {name}")
                units.append({
                    "name": name, "owner": owner,
                    "kind": u.get("kind") or entry.get("use") or "residential",
                    "preferred_unit_no": u.get("unit_no",
                                               abs(level) * 100 + j + 1),
                    "_poly": poly,
                })

            out.append({
                "level": level, "height": height,
                "use": entry.get("use") or ("parking" if level < 0 else "residential"),
                "units": units,
            })

    out.sort(key=lambda f: f["level"])
    building["_floors"] = out
    return out


def storey_levels(building: dict, ground_level: float) -> dict[int, tuple[float, float]]:
    """
    Turn floor heights into absolute z-ranges, stacked from the ground level.

    Storeys must meet exactly - a gap between two floors is space nobody owns,
    and an overlap means two floors claim the same slab. Stacking cumulatively
    from level 0 guarantees both, so the validator never has to report a defect
    the input format made inevitable.
    """
    floors = expand_floors(building)
    spans: dict[int, tuple[float, float]] = {}

    z = ground_level
    for f in [x for x in floors if x["level"] >= 0]:
        spans[f["level"]] = (z, z + f["height"])
        z += f["height"]

    z = ground_level
    for f in sorted([x for x in floors if x["level"] < 0],
                    key=lambda x: -x["level"]):
        spans[f["level"]] = (z - f["height"], z)
        z -= f["height"]

    return spans


def building_outline(building: dict, *, wall: float = DEFAULT_WALL_M) -> Polygon:
    """
    The building envelope: an explicit outline, or one derived from the units.

    Deriving it grows each unit by the wall thickness and unions the result, so
    flats separated by a party wall merge into a single envelope instead of
    staying a scatter of disconnected rooms.
    """
    cached = building.get("_outline")
    if cached is not None:
        return cached

    if building.get("outline"):
        outline = _polygon(building["outline"],
                           f"building {building.get('id','?')} outline")
    else:
        polys = [u["_poly"] for f in expand_floors(building) for u in f["units"]]
        if not polys:
            raise PlanError(
                f"building {building.get('id','?')}: no units and no 'outline', so "
                f"there is no shape to build. Give the building an 'outline', or "
                f"give at least one floor some units.")
        merged = unary_union([p.buffer(wall) for p in polys])
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        outline = clean_polygon(merged.simplify(0.05)) or merged

    building["_outline"] = outline
    return outline


# --- 2D -> 3D ------------------------------------------------------------------
def build_register(plan: Plan2D, *, air_rights_m: float = AIR_RIGHTS_HEIGHT_M,
                   progress=None):
    """
    Turn a 2D plan into a validated 3D register.

    This is the whole 2D-to-3D conversion, and it is deliberately unglamorous:
    for every unit polygon on every floor, look up that floor's z-range and
    sweep the polygon through it. The result is a solid with a volume, an
    owner, a place in the ownership hierarchy and its own 3D-ULPIN.

    Returns the same :class:`~bhoomi3d.pipeline.PipelineResult` the LiDAR
    pipeline returns, so the API, the viewer and the validator cannot tell the
    two apart.
    """
    from .pipeline import PipelineResult, StageLog, _add_corridors, _storey_name

    stages: list[StageLog] = []
    enu = LocalENU(GeodeticOrigin(**plan.origin))
    cad = Cadastre(JurisdictionCode(**plan.jurisdiction), plan.origin, plan.name)

    def tick(name, **detail):
        stages.append(StageLog(name, 0.0, detail))
        if progress:
            progress(name, "done", detail)

    # --- parcels ----------------------------------------------------------
    parcel_objs = []
    for p in plan.parcels:
        obj = CadastralObject(
            object_id=p["id"], kind=ObjectKind.PARCEL,
            solid=Prism(p["_poly"], -6.0, 0.001),
            name=p.get("survey_no", ""), owner=p.get("owner", ""),
            use=p.get("land_use", ""), level=0, stratum=Stratum.SURFACE,
            provenance=Provenance.EXISTING_RECORD, confidence=1.0,
            attributes={
                "survey_no": p.get("survey_no", ""),
                "recorded_area_m2": p.get("area_m2"),
                "source": "2D cadastral plan",
                "vertical_extent_note":
                    "subsurface limit is a nominal 6 m; the plan boundary is "
                    "the authoritative horizontal extent",
            })
        cad.add(obj, enu=enu)
        parcel_objs.append(obj)
    tick("parcels", parcels=len(parcel_objs))

    # --- buildings, storeys, units ----------------------------------------
    counts = {"buildings": 0, "storeys": 0, "units": 0, "air_rights": 0}
    for b in plan.buildings:
        bid = b["id"]
        ground = float(b.get("ground_level", 0.0))
        outline = building_outline(b)
        spans = storey_levels(b, ground)
        floors = expand_floors(b)

        parent = b.get("parcel")
        if parent and parent not in cad:
            raise PlanError(
                f"building {bid}: 'parcel' refers to {parent!r}, which is not "
                f"in the parcels list. Parcel ids present: "
                f"{[p['id'] for p in plan.parcels] or 'none'}")
        if not parent:
            best, best_area = None, 0.0
            for p in parcel_objs:
                a = outline.intersection(p.footprint).area
                if a > best_area:
                    best, best_area = p.object_id, a
            parent = best

        z_bottom = min(s[0] for s in spans.values()) if spans else ground
        z_top = max(s[1] for s in spans.values()) if spans else ground
        above = [f for f in floors if f["level"] >= 0]

        cad.add(CadastralObject(
            object_id=bid, kind=ObjectKind.BUILDING,
            solid=Prism(outline, z_bottom, z_top),
            parent_id=parent, name=b.get("name", bid),
            owner=b.get("owner", ""), level=0, stratum=Stratum.SURFACE,
            provenance=Provenance.APPROVED_PLAN, confidence=1.0,
            attributes={
                "ground_level_m": round(ground, 3),
                "eave_level_m": round(z_top, 3),
                "roof_level_m": round(z_top, 3),
                "parapet_level_m": round(z_top, 3),
                "habitable_height_m": round(z_top - ground, 3),
                "structural_height_m": round(z_top - ground, 3),
                "storeys_above_ground": len(above),
                "storeys_below_ground": len(floors) - len(above),
                "storey_source": "2D plan",
                "floor_height_m": round(
                    sum(f["height"] for f in above) / len(above), 3) if above else 0.0,
                "source": "2D plan upload",
                # No scan, so nothing can contradict the plan. Recording the
                # scan estimate as equal to the sanctioned count keeps the
                # unauthorised-construction rule from firing on an absence of
                # evidence, which would be a false accusation.
                "storey_estimate_from_scan": len(above),
                "facade_periodicity": 0.0,
            }), enu=enu)
        counts["buildings"] += 1

        for f in floors:
            level = f["level"]
            z0, z1 = spans[level]
            sid = f"{bid}/L{level:+03d}"
            below = level < 0
            cad.add(CadastralObject(
                object_id=sid, kind=ObjectKind.STOREY,
                solid=Prism(outline, z0, z1), parent_id=bid,
                name=_storey_name(level), level=level, use=f["use"],
                stratum=Stratum.UNDERGROUND if below else Stratum.BUILDING,
                provenance=Provenance.APPROVED_PLAN, confidence=1.0,
                attributes={"floor_height_m": round(f["height"], 3),
                            "level_source": "2D plan"}), enu=enu)
            counts["storeys"] += 1

            for u in f["units"]:
                # ---- this line is the 2D-to-3D conversion ----
                solid = Prism(u["_poly"], z0, z1)
                cad.add(CadastralObject(
                    object_id=f"{sid}/{u['name']}", kind=ObjectKind.UNIT,
                    solid=solid, parent_id=sid, name=u["name"],
                    owner=u["owner"], use=u["kind"], level=level,
                    stratum=Stratum.UNDERGROUND if below else Stratum.BUILDING,
                    provenance=Provenance.APPROVED_PLAN, confidence=1.0,
                    attributes={"carpet_area_m2": round(u["_poly"].area, 3),
                                "source": "2D plan"}),
                    enu=enu, preferred_unit=u["preferred_unit_no"])
                counts["units"] += 1

        cad.add(CadastralObject(
            object_id=f"{bid}/AIR", kind=ObjectKind.AIR_RIGHTS,
            solid=Prism(outline, z_top, z_top + air_rights_m),
            parent_id=parent, name=f"Air rights above {b.get('name', bid)}",
            owner=b.get("owner", ""), level=0, stratum=Stratum.AIR_RIGHTS,
            provenance=Provenance.DERIVED, confidence=0.9,
            attributes={
                "basis": "development-rights column above the built envelope",
                "column_height_m": air_rights_m,
                "note": "extent follows the permissible height limit and is a "
                        "policy parameter, not a measurement",
            }), enu=enu)
        counts["air_rights"] += 1

    tick("volumetric_delineation", **counts)

    # --- infrastructure ----------------------------------------------------
    corridors = []
    for c in plan.infrastructure:
        depth = float(c.get("depth", 2.5))
        height = float(c.get("height", 1.5))
        invert = c.get("invert_levels_m")
        if not invert:
            # a single 'depth' means depth to the crown, which is how a utility
            # authority records cover; the invert sits one section-height below
            invert = [-(depth + height)] * len(c["alignment"])
        corridors.append({
            "id": c["id"], "name": c.get("name", c["id"]),
            "kind": c.get("kind", "water"), "owner": c.get("owner", ""),
            "alignment": c["alignment"],
            "width_m": float(c.get("width", 1.0)), "height_m": height,
            "invert_levels_m": invert,
            "length_m": round(sum(
                math.dist(c["alignment"][i], c["alignment"][i + 1])
                for i in range(len(c["alignment"]) - 1)), 2),
        })
    n_infra = _add_corridors(cad, enu, corridors)
    tick("infrastructure", corridors=n_infra)

    # --- validation --------------------------------------------------------
    report = validate(cad)
    tick("validation",
         findings=len(report.findings),
         errors=len(report.by_severity(Severity.ERROR)),
         warnings=len(report.by_severity(Severity.WARNING)))

    return PipelineResult(
        cadastre=cad, report=report, enu=enu, stages=stages,
        metrics={"input_mode": "2D plan only",
                 "note": "no survey supplied, so the register reflects the "
                         "sanctioned plan. As-built verification needs a drone "
                         "or LiDAR survey through pipeline.run_pipeline().",
                 "plan_summary": plan.summary()},
        terrain=None, extraction=None)


# --- drawing the plan ----------------------------------------------------------
def render_svg(plan: Plan2D, level: int = 0, *, width: int = 720) -> str:
    """
    Draw one floor of the plan as an SVG.

    Being able to *see* the input as a drawing matters more than it sounds: a
    plan file is a list of coordinates, and a wrong sign or a transposed pair
    is invisible in JSON and obvious in a picture. This is the check before
    anyone waits on a 3D build.
    """
    shapes: list[tuple[Polygon, str, str]] = []
    for p in plan.parcels:
        shapes.append((p["_poly"], "parcel", p.get("id", "")))
    for b in plan.buildings:
        shapes.append((building_outline(b), "outline", b.get("name", b["id"])))
        for f in expand_floors(b):
            if f["level"] != level:
                continue
            for u in f["units"]:
                shapes.append((u["_poly"], "unit", u["name"]))
    for c in plan.infrastructure:
        from shapely.geometry import LineString
        line = LineString(c["alignment"]).buffer(float(c.get("width", 1.0)) / 2)
        shapes.append((line, "infra", c.get("name", c["id"])))

    if not shapes:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'

    xs0, ys0, xs1, ys1 = unary_union([s[0] for s in shapes]).bounds
    pad = max((xs1 - xs0), (ys1 - ys0)) * 0.06 + 2
    xs0, ys0, xs1, ys1 = xs0 - pad, ys0 - pad, xs1 + pad, ys1 + pad
    span_x, span_y = xs1 - xs0, ys1 - ys0
    scale = width / span_x
    height = int(span_y * scale)

    # SVG y grows downward; site y grows north, so flip
    def pt(x, y):
        return f"{(x - xs0) * scale:.1f},{(ys1 - y) * scale:.1f}"

    style = {
        "parcel":  ("none", "#2F6B3A", "2.2", "7 4"),
        "outline": ("#D8D2C4", "#2A3038", "2.4", ""),
        "unit":    ("#F0E2B8", "#8A5A12", "1.4", ""),
        "infra":   ("#ED545928", "#C62828", "1.4", "5 3"),
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="ui-monospace, monospace">',
        # Paper-like sheet with a faint blueprint tint
        f'<defs>'
        f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)">'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="#C9C2B2" stroke-width="1"/>'
        f'</pattern>'
        f'</defs>',
        f'<rect width="{width}" height="{height}" fill="#F3EFE6"/>',
    ]

    # a 5 m minor / 10 m major grid — reads as a survey sheet
    for step, stroke, sw in ((5.0, "#E4DDD0", "0.6"), (10.0, "#D0C8B8", "1")):
        gx = math.ceil(xs0 / step) * step
        while gx < xs1:
            x = (gx - xs0) * scale
            parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" '
                         f'stroke="{stroke}" stroke-width="{sw}"/>')
            gx += step
        gy = math.ceil(ys0 / step) * step
        while gy < ys1:
            y = (ys1 - gy) * scale
            parts.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" '
                         f'stroke="{stroke}" stroke-width="{sw}"/>')
            gy += step

    for poly, kind, label in shapes:
        fill, stroke, sw, dash = style[kind]
        geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
        for g in geoms:
            pts = " ".join(pt(x, y) for x, y in g.exterior.coords)
            d = f' stroke-dasharray="{dash}"' if dash else ""
            # Outer building shell gets a wall hatch so it reads as masonry
            if kind == "outline":
                parts.append(f'<polygon points="{pts}" fill="url(#hatch)" '
                             f'stroke="none"/>')
            parts.append(f'<polygon points="{pts}" fill="{fill}" '
                         f'stroke="{stroke}" stroke-width="{sw}"{d} '
                         f'stroke-linejoin="round"/>')
            # Inner wall offset for units — a second stroke sells "wall thickness"
            if kind == "unit" and g.area > 1.0:
                try:
                    inset = g.buffer(-0.35)
                    if not inset.is_empty:
                        igs = inset.geoms if inset.geom_type == "MultiPolygon" else [inset]
                        for ig in igs:
                            ipts = " ".join(pt(x, y) for x, y in ig.exterior.coords)
                            parts.append(
                                f'<polygon points="{ipts}" fill="none" '
                                f'stroke="#C9A66A" stroke-width="0.7" opacity="0.7"/>')
                except Exception:
                    pass
        if kind in ("unit", "parcel") and label:
            c = poly.representative_point()
            parts.append(
                f'<text x="{(c.x - xs0) * scale:.1f}" y="{(ys1 - c.y) * scale:.1f}" '
                f'font-size="10" fill="#3B454A" text-anchor="middle" '
                f'font-weight="600">{label}</text>')

    # North arrow
    parts.append(
        f'<g transform="translate(22,28)">'
        f'<polygon points="0,-14 5,6 -5,6" fill="#2A3038"/>'
        f'<text x="0" y="18" font-size="10" fill="#3B454A" text-anchor="middle" '
        f'font-weight="700">N</text></g>')

    # scale bar
    bar = 10 * scale
    parts.append(
        f'<g transform="translate(12,{height - 18})">'
        f'<line x1="0" y1="0" x2="{bar:.1f}" y2="0" stroke="#191E22" stroke-width="2"/>'
        f'<line x1="0" y1="-4" x2="0" y2="4" stroke="#191E22" stroke-width="2"/>'
        f'<line x1="{bar:.1f}" y1="-4" x2="{bar:.1f}" y2="4" stroke="#191E22" stroke-width="2"/>'
        f'<text x="{bar / 2:.1f}" y="-7" font-size="10" fill="#3B454A" '
        f'text-anchor="middle">10 m</text></g>')
    parts.append(
        f'<text x="{width - 12}" y="18" font-size="11" fill="#5F6B66" '
        f'text-anchor="end">{plan.name} &#183; '
        f'{"basement " + str(abs(level)) if level < 0 else "ground floor" if level == 0 else "floor " + str(level)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --- a worked example ----------------------------------------------------------
def sample_plan(kind: str = "simple") -> dict:
    """
    A ready-to-edit plan document.

    ``simple``   one parcel, one four-storey building, a water main. Small
                 enough to read end to end and change by hand.
    ``society``  a realistic housing society: two basements, shops at grade,
                 ten identical residential floors, and a metro tunnel below.

    Returns a deep copy every time. Callers are *expected* to edit what they
    get back, and :func:`parse_plan` also caches parsed geometry onto the
    document - handing out the shared template would let one caller corrupt it
    for everyone afterwards.
    """
    return copy.deepcopy(_SAMPLE_SOCIETY if kind == "society" else _SAMPLE_SIMPLE)


_SAMPLE_SIMPLE: dict = {
    "site": {
        "name": "Demo Plot - Shivajinagar",
        "origin": {"lat": 18.520430, "lon": 73.856744},
        "jurisdiction": {
            "state": "27", "district": "025", "tehsil": "004",
            "state_name": "Maharashtra", "district_name": "Pune",
            "tehsil_name": "Pune City",
        },
    },
    "parcels": [
        {
            "id": "P-01",
            "owner": "Sharda Co-op Housing Society",
            "survey_no": "SN-118/2",
            "land_use": "residential",
            "boundary": [[0, 0], [44, 0], [44, 32], [0, 32]],
        }
    ],
    "buildings": [
        {
            "id": "B-01",
            "name": "Sharda Apartments",
            "parcel": "P-01",
            "ground_level": 0.0,
            "floors": [
                {
                    "level": -1, "height": 3.0, "use": "parking",
                    "units": [
                        {"name": "PARK-B1", "kind": "parking",
                         "owner": "Society common",
                         "boundary": [[8, 6], [36, 6], [36, 26], [8, 26]]}
                    ],
                },
                {
                    "level": 0, "height": 3.6, "use": "commercial",
                    "units": [
                        {"name": "SHOP-01", "kind": "shop", "owner": "Kirana Stores",
                         "boundary": [[8, 6], [21, 6], [21, 26], [8, 26]]},
                        {"name": "SHOP-02", "kind": "shop", "owner": "Anand Medical",
                         "boundary": [[23, 6], [36, 6], [36, 26], [23, 26]]}
                    ],
                },
                {
                    "level": 1, "repeat_to": 3, "height": 3.0, "use": "residential",
                    "units": [
                        {"name": "FLAT-{n}01", "owner": "Owner of flat {n}01",
                         "boundary": [[8, 6], [21, 6], [21, 26], [8, 26]]},
                        {"name": "FLAT-{n}02", "owner": "Owner of flat {n}02",
                         "boundary": [[23, 6], [36, 6], [36, 26], [23, 26]]}
                    ],
                },
            ],
        }
    ],
    "infrastructure": [
        {
            "id": "INF-WATER-01",
            "name": "300 mm water main",
            "kind": "water",
            "owner": "Pune Municipal Corporation - Water Supply",
            "alignment": [[-6, 29], [50, 29]],
            "width": 0.9, "height": 0.9, "depth": 1.8,
        }
    ],
}


_SAMPLE_SOCIETY: dict = {
    "site": {
        "name": "Ganga Residency - Shivajinagar",
        "origin": {"lat": 18.520430, "lon": 73.856744},
        "jurisdiction": {
            "state": "27", "district": "025", "tehsil": "004",
            "state_name": "Maharashtra", "district_name": "Pune",
            "tehsil_name": "Pune City",
        },
    },
    "parcels": [
        {"id": "P-01", "owner": "Ganga Co-op Housing Society",
         "survey_no": "SN-114/3", "land_use": "residential",
         "boundary": [[0, 0], [58, 0], [58, 40], [0, 40]]}
    ],
    "buildings": [
        {
            "id": "B-01", "name": "Ganga Residency", "parcel": "P-01",
            "owner": "Ganga Co-op Housing Society", "ground_level": 0.0,
            "floors": [
                {"level": -2, "height": 3.0, "use": "parking", "units": [
                    {"name": "PARK-B2", "kind": "parking", "owner": "Society common",
                     "boundary": [[6, 6], [52, 6], [52, 34], [6, 34]]}]},
                {"level": -1, "height": 3.0, "use": "parking", "units": [
                    {"name": "PARK-B1", "kind": "parking", "owner": "Society common",
                     "boundary": [[6, 6], [52, 6], [52, 34], [6, 34]]}]},
                {"level": 0, "height": 3.8, "use": "commercial", "units": [
                    {"name": "SHOP-01", "kind": "shop", "owner": "Deccan Traders",
                     "boundary": [[6, 6], [20, 6], [20, 34], [6, 34]]},
                    {"name": "SHOP-02", "kind": "shop", "owner": "Sai Electronics",
                     "boundary": [[22, 6], [36, 6], [36, 34], [22, 34]]},
                    {"name": "SHOP-03", "kind": "shop", "owner": "Pune Bakers",
                     "boundary": [[38, 6], [52, 6], [52, 34], [38, 34]]}]},
                {"level": 1, "repeat_to": 10, "height": 3.05, "use": "residential",
                 "units": [
                     {"name": "FLAT-{n}01", "owner": "Resident {n}01",
                      "boundary": [[6, 6], [20, 6], [20, 19], [6, 19]]},
                     {"name": "FLAT-{n}02", "owner": "Resident {n}02",
                      "boundary": [[22, 6], [36, 6], [36, 19], [22, 19]]},
                     {"name": "FLAT-{n}03", "owner": "Resident {n}03",
                      "boundary": [[38, 6], [52, 6], [52, 19], [38, 19]]},
                     {"name": "FLAT-{n}04", "owner": "Resident {n}04",
                      "boundary": [[6, 21], [20, 21], [20, 34], [6, 34]]},
                     {"name": "FLAT-{n}05", "owner": "Resident {n}05",
                      "boundary": [[22, 21], [36, 21], [36, 34], [22, 34]]},
                     {"name": "FLAT-{n}06", "owner": "Resident {n}06",
                      "boundary": [[38, 21], [52, 21], [52, 34], [38, 34]]}]},
            ],
        }
    ],
    "infrastructure": [
        {"id": "INF-METRO-01", "name": "Metro Line 3 - up tunnel", "kind": "metro",
         "owner": "Maharashtra Metro Rail Corporation",
         "alignment": [[-10, 20], [70, 24]],
         "width": 6.4, "height": 6.4, "depth": 9.0},
        {"id": "INF-WATER-01", "name": "600 mm water main", "kind": "water",
         "owner": "Pune Municipal Corporation - Water Supply",
         "alignment": [[-6, 37], [64, 37]],
         "width": 1.4, "height": 1.4, "depth": 2.0},
    ],
}


def write_samples(directory) -> list[Path]:
    """Write the sample plans and their rendered drawings to a directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for kind in ("simple", "society"):
        doc = sample_plan(kind)
        path = directory / f"plan-{kind}.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        written.append(path)

        plan = parse_plan(doc, source=str(path))
        for level in sorted({f["level"] for b in plan.buildings
                             for f in expand_floors(b)}):
            tag = (f"b{abs(level)}" if level < 0 else
                   "ground" if level == 0 else f"f{level}")
            svg = directory / f"plan-{kind}-{tag}.svg"
            svg.write_text(render_svg(plan, level), encoding="utf-8")
            written.append(svg)
    return written

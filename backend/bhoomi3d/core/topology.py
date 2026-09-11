"""
Topological validation of the 3D register.

A cadastre is only as good as its consistency. In 2D the classic checks are
"do parcels overlap?" and "is there unclaimed space between them?". In 3D both
questions get a vertical dimension, and several new ones appear that have no 2D
analogue at all - a flat that spills through its own ceiling, a foundation
driven too close to a metro tunnel, a basement that extends beyond the parcel
it belongs to.

Every rule here returns a machine-readable finding with the geometry of the
problem attached, so the viewer can fly to it and draw it rather than just
printing a message. That is the difference between a validation report a
surveyor can act on and one they ignore.

Severities
----------
ERROR    a legal defect - the register must not be published in this state
WARNING  probably wrong, or right but unusual; needs a human decision
INFO     worth recording; no action implied

Tolerances are deliberately explicit and generous by survey standards. Cadastral
geometry carries real measurement error, and a validator that flags every
millimetre of float noise trains its users to ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .cadastre import Cadastre, CadastralObject, ObjectKind
from .geometry3d import EPS, rings_from_polygon


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """One validation result."""

    rule: str
    severity: Severity
    title: str
    detail: str
    objects: list[str] = field(default_factory=list)
    measure: dict = field(default_factory=dict)
    geometry: Optional[dict] = None      # where to look, for the viewer

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity.value,
            "title": self.title,
            "detail": self.detail,
            "objects": self.objects,
            "measure": self.measure,
            "geometry": self.geometry,
        }


@dataclass
class ValidationReport:
    findings: list[Finding]
    checked: dict
    tolerances: dict

    def by_severity(self, sev: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == sev]

    @property
    def is_valid(self) -> bool:
        return not self.by_severity(Severity.ERROR)

    def as_dict(self) -> dict:
        counts = {s.value: len(self.by_severity(s)) for s in Severity}
        return {
            "valid": self.is_valid,
            "counts": counts,
            "total_findings": len(self.findings),
            "checked": self.checked,
            "tolerances": self.tolerances,
            "findings": [f.as_dict() for f in self.findings],
        }


# --- default tolerances --------------------------------------------------------
DEFAULT_TOLERANCES = {
    # volumetric overlap below this is measurement noise, not a boundary dispute
    "overlap_volume_m3": 0.05,
    # a unit may poke this far outside its storey before it is a defect
    "containment_slack_m2": 0.25,
    # unassigned floor area below this is not worth reporting
    "gap_area_m2": 1.0,
    # common areas (stairs, lifts, corridors) legitimately occupy this share of
    # a storey without being anyone's private property
    "common_area_fraction": 0.42,
    # minimum clearance between a structure and a utility or transport corridor
    "infra_clearance_m": {"metro": 3.0, "water": 1.0, "storm": 1.0,
                          "power": 1.5, "default": 1.0},
    # consecutive storeys should meet; more than this is a gap or an overlap
    "storey_join_m": 0.15,
    # recorded vs computed area
    "area_mismatch_pct": 5.0,
}


def _bbox_index(objects: list[CadastralObject]):
    """
    A simple sort-and-sweep index over the x axis.

    Pairwise validation is O(n^2) if done naively. Sweeping a sorted x-interval
    list drops it to near-linear for the spatially separated objects a real site
    consists of, which is what makes whole-city validation tractable without a
    spatial database.
    """
    entries = []
    for o in objects:
        x0, y0, z0, x1, y1, z1 = o.bbox
        entries.append((x0, x1, o))
    entries.sort(key=lambda e: e[0])
    return entries


def _candidate_pairs(objects: list[CadastralObject]):
    """Yield object pairs whose bounding boxes could possibly interact."""
    entries = _bbox_index(objects)
    n = len(entries)
    for i in range(n):
        x0_i, x1_i, oi = entries[i]
        for j in range(i + 1, n):
            x0_j, _, oj = entries[j]
            if x0_j > x1_i:
                break        # sorted by x0, so nothing further can overlap
            yield oi, oj


# --- individual rules ----------------------------------------------------------
def check_geometry_validity(cad: Cadastre) -> list[Finding]:
    """Every solid must be a well-formed, positively-oriented, non-degenerate volume."""
    out = []
    for o in cad.objects:
        fp = o.footprint
        if fp.is_empty:
            out.append(Finding("GEOM_EMPTY", Severity.ERROR,
                               "Empty footprint",
                               f"{o.object_id} has no plan geometry.", [o.object_id]))
            continue
        if isinstance(fp, Polygon) and not fp.is_valid:
            out.append(Finding("GEOM_INVALID", Severity.ERROR,
                               "Invalid polygon",
                               f"{o.object_id} has a self-intersecting or "
                               f"otherwise invalid footprint.", [o.object_id]))
        if o.z_max - o.z_min <= EPS:
            out.append(Finding("GEOM_ZERO_HEIGHT", Severity.ERROR,
                               "Zero-height solid",
                               f"{o.object_id} spans no vertical extent "
                               f"(z_min == z_max == {o.z_min:.3f} m); it is a "
                               f"surface, not a volume.", [o.object_id],
                               {"z_min": o.z_min, "z_max": o.z_max}))
        if o.volume <= 0:
            out.append(Finding("GEOM_ZERO_VOLUME", Severity.ERROR,
                               "Zero volume", f"{o.object_id} encloses no space.",
                               [o.object_id]))
    return out


def check_volumetric_overlap(cad: Cadastre, tol: dict) -> list[Finding]:
    """
    No two privately-held volumes may intersect.

    This is the headline 3D rule. Two flats can share a footprint if they are on
    different floors - that is normal and must *not* be flagged - but if their
    z-ranges overlap as well, two people own the same cubic metres and there is
    a real dispute to resolve.
    """
    out = []
    exclusive = [o for o in cad.objects
                 if o.kind in (ObjectKind.UNIT, ObjectKind.INFRASTRUCTURE)]
    min_vol = tol["overlap_volume_m3"]

    for a, b in _candidate_pairs(exclusive):
        # objects in a parent/child line are meant to nest, not to compete
        if a.parent_id == b.object_id or b.parent_id == a.object_id:
            continue
        vol = a.intersection_volume(b)
        if vol <= min_vol:
            continue
        inter_fp = a.footprint.intersection(b.footprint)
        dz = min(a.z_max, b.z_max) - max(a.z_min, b.z_min)
        out.append(Finding(
            "VERT_OVERLAP", Severity.ERROR,
            "Overlapping ownership volumes",
            f"{a.object_id} and {b.object_id} share {vol:.2f} m3 of space. "
            f"Their footprints overlap by {inter_fp.area:.2f} m2 and their "
            f"vertical extents overlap by {dz:.2f} m.",
            [a.object_id, b.object_id],
            {"overlap_volume_m3": round(vol, 3),
             "overlap_area_m2": round(inter_fp.area, 3),
             "z_overlap_m": round(dz, 3)},
            geometry={"rings": rings_from_polygon(inter_fp)
                      if isinstance(inter_fp, Polygon) and not inter_fp.is_empty else [],
                      "z_min": round(max(a.z_min, b.z_min), 3),
                      "z_max": round(min(a.z_max, b.z_max), 3)},
        ))
    return out


def check_containment(cad: Cadastre, tol: dict) -> list[Finding]:
    """
    A child object must lie within its parent, in plan and in elevation.

    A flat outside its storey, or a storey outside its building, means the
    hierarchy is lying about what contains what - and any query that walks the
    tree (like "what do I own in this tower?") will return the wrong answer.
    """
    out = []
    slack = tol["containment_slack_m2"]
    for o in cad.objects:
        if not o.parent_id:
            continue
        parent = cad.get(o.parent_id)
        if parent is None:
            out.append(Finding("HIER_ORPHAN", Severity.ERROR, "Missing parent",
                               f"{o.object_id} references parent "
                               f"{o.parent_id!r}, which is not in the register.",
                               [o.object_id]))
            continue
        if o.kind == ObjectKind.AIR_RIGHTS:
            continue   # air rights sit above their parcel, not inside a solid

        outside = o.footprint.difference(parent.footprint.buffer(0.05))
        if outside.area > slack:
            out.append(Finding(
                "PLAN_CONTAINMENT", Severity.ERROR,
                "Child extends beyond its parent in plan",
                f"{o.object_id} projects {outside.area:.2f} m2 outside "
                f"{parent.object_id}.",
                [o.object_id, parent.object_id],
                {"outside_area_m2": round(outside.area, 3)},
                geometry={"rings": rings_from_polygon(
                    max(outside.geoms, key=lambda g: g.area)
                    if outside.geom_type == "MultiPolygon" else outside),
                    "z_min": round(o.z_min, 3), "z_max": round(o.z_max, 3)},
            ))

        # A surface parcel bounds its buildings in plan but not in elevation.
        # That is the whole premise of a 3D cadastre: the parcel is a footprint
        # on the ground with a nominal subsurface extent, and the column above
        # it - the building, and the air rights above that - rises out of it by
        # design. Only containers that genuinely bound a volume vertically
        # (a storey inside a building, a flat inside a storey) are checked here.
        if parent.kind is ObjectKind.PARCEL:
            continue

        if o.z_min < parent.z_min - 0.05 or o.z_max > parent.z_max + 0.05:
            out.append(Finding(
                "VERT_CONTAINMENT", Severity.ERROR,
                "Child extends beyond its parent vertically",
                f"{o.object_id} spans {o.z_min:.2f}..{o.z_max:.2f} m but "
                f"{parent.object_id} only spans {parent.z_min:.2f}.."
                f"{parent.z_max:.2f} m.",
                [o.object_id, parent.object_id],
                {"child_z": [round(o.z_min, 3), round(o.z_max, 3)],
                 "parent_z": [round(parent.z_min, 3), round(parent.z_max, 3)]},
            ))
    return out


def check_storey_continuity(cad: Cadastre, tol: dict) -> list[Finding]:
    """
    Consecutive storeys must meet: no gaps, no overlaps.

    A gap between two storeys is space nobody owns, sitting inside a building.
    An overlap means two floors claim the same slab. Both are usually a sign
    that a floor plan's levels were entered against the wrong datum.
    """
    out = []
    join_tol = tol["storey_join_m"]
    for building in cad.of_kind(ObjectKind.BUILDING):
        storeys = sorted([c for c in cad.children(building.object_id)
                          if c.kind == ObjectKind.STOREY],
                         key=lambda s: s.level)
        for lower, upper in zip(storeys, storeys[1:]):
            gap = upper.z_min - lower.z_max
            if abs(gap) <= join_tol:
                continue
            if gap > 0:
                out.append(Finding(
                    "STOREY_GAP", Severity.WARNING,
                    "Unassigned space between storeys",
                    f"A {gap:.2f} m gap sits between {lower.object_id} "
                    f"(top {lower.z_max:.2f} m) and {upper.object_id} "
                    f"(base {upper.z_min:.2f} m). No object owns it.",
                    [lower.object_id, upper.object_id],
                    {"gap_m": round(gap, 3)}))
            else:
                out.append(Finding(
                    "STOREY_OVERLAP", Severity.ERROR,
                    "Storeys overlap vertically",
                    f"{lower.object_id} and {upper.object_id} both claim "
                    f"{-gap:.2f} m of vertical extent.",
                    [lower.object_id, upper.object_id],
                    {"overlap_m": round(-gap, 3)}))
    return out


def check_unassigned_floor_area(cad: Cadastre, tol: dict) -> list[Finding]:
    """
    Report floor area inside a storey that no unit claims.

    Some unassigned area is correct and expected - stairs, lifts, corridors and
    shafts are common property, not anybody's flat. So this reports the residue
    and only escalates when it is implausibly large, which usually means a unit
    is missing from the record entirely.
    """
    out = []
    min_gap = tol["gap_area_m2"]
    max_common = tol["common_area_fraction"]

    for storey in cad.of_kind(ObjectKind.STOREY):
        units = [c for c in cad.children(storey.object_id)
                 if c.kind == ObjectKind.UNIT]
        if not units:
            continue
        claimed = unary_union([u.footprint for u in units])
        residue = storey.footprint.difference(claimed)
        if residue.is_empty or residue.area < min_gap:
            continue
        frac = residue.area / max(storey.footprint.area, 1e-9)
        sev = Severity.WARNING if frac > max_common else Severity.INFO
        biggest = (max(residue.geoms, key=lambda g: g.area)
                   if residue.geom_type == "MultiPolygon" else residue)
        out.append(Finding(
            "UNASSIGNED_AREA", sev,
            "Unassigned floor area",
            f"{residue.area:.2f} m2 ({frac:.0%}) of {storey.object_id} is not "
            f"claimed by any unit." + (
                " That is more than common areas would normally account for, so "
                "a unit may be missing from the record."
                if sev is Severity.WARNING else
                " This is consistent with stairs, lifts and corridors."),
            [storey.object_id],
            {"unassigned_area_m2": round(residue.area, 3),
             "storey_area_m2": round(storey.footprint.area, 3),
             "fraction": round(frac, 4)},
            geometry={"rings": rings_from_polygon(biggest),
                      "z_min": round(storey.z_min, 3),
                      "z_max": round(storey.z_max, 3)}
            if isinstance(biggest, Polygon) else None,
        ))
    return out


def check_infrastructure_clearance(cad: Cadastre, tol: dict) -> list[Finding]:
    """
    Structures must keep clear of underground corridors.

    This is the check that motivates the whole project. A 2D cadastre cannot
    even express the question: the tunnel and the tower occupy the same plan
    coordinates, so on a flat map they are simply "the same place". In 3D the
    answer is a number - the shortest distance between two solids - and it is
    either above the safe-distance rule or it is not.
    """
    out = []
    clearances = tol["infra_clearance_m"]
    infra = cad.of_kind(ObjectKind.INFRASTRUCTURE)
    if not infra:
        return out
    structures = cad.of_kind(ObjectKind.BUILDING, ObjectKind.UNIT)

    for corridor in infra:
        kind = corridor.attributes.get("kind", "default")
        required = clearances.get(kind, clearances["default"])
        for s in structures:
            # cheap plan rejection first: if they are far apart horizontally,
            # no vertical arrangement can bring them within the limit
            if s.footprint.distance(corridor.footprint) > required:
                continue
            d = s.clearance(corridor)
            if d >= required:
                continue
            overlapping = d <= EPS
            out.append(Finding(
                "INFRA_CLEARANCE",
                Severity.ERROR if overlapping else Severity.WARNING,
                "Structure encroaches on an underground corridor"
                if overlapping else "Insufficient clearance to underground corridor",
                f"{s.object_id} is {d:.2f} m from {corridor.object_id} "
                f"({corridor.name}); the rule for a {kind} corridor requires "
                f"{required:.2f} m." + (
                    " The volumes actually intersect."
                    if overlapping else ""),
                [s.object_id, corridor.object_id],
                {"clearance_m": round(d, 3), "required_m": required,
                 "shortfall_m": round(required - d, 3),
                 "corridor_kind": kind},
            ))
    return out


def check_ulpin_integrity(cad: Cadastre) -> list[Finding]:
    """Every object must carry exactly one well-formed, unique identifier."""
    out = []
    seen: dict[str, str] = {}
    for o in cad.objects:
        if o.ulpin is None:
            out.append(Finding("ULPIN_MISSING", Severity.ERROR,
                               "No identifier issued",
                               f"{o.object_id} has no 3D-ULPIN.", [o.object_id]))
            continue
        key = o.ulpin.compact
        if key in seen:
            out.append(Finding(
                "ULPIN_DUPLICATE", Severity.ERROR, "Duplicate identifier",
                f"{o.object_id} and {seen[key]} both carry {o.ulpin}.",
                [o.object_id, seen[key]], {"ulpin": str(o.ulpin)}))
        seen[key] = o.object_id
    return out


def check_area_consistency(cad: Cadastre, tol: dict) -> list[Finding]:
    """
    Compare the area recorded against the area the geometry actually encloses.

    A mismatch between the two is the most common defect in a legacy land
    record, and it is exactly what a machine should catch.
    """
    out = []
    limit = tol["area_mismatch_pct"]
    for o in cad.objects:
        recorded = o.attributes.get("recorded_area_m2")
        if recorded in (None, 0):
            continue
        computed = o.area
        err = 100.0 * abs(computed - recorded) / recorded
        if err <= limit:
            continue
        out.append(Finding(
            "AREA_MISMATCH", Severity.WARNING, "Recorded area disagrees with geometry",
            f"{o.object_id} is recorded as {recorded:.2f} m2 but its geometry "
            f"encloses {computed:.2f} m2, a difference of {err:.1f}%.",
            [o.object_id],
            {"recorded_area_m2": round(recorded, 3),
             "computed_area_m2": round(computed, 3),
             "error_pct": round(err, 2)}))
    return out


def check_unauthorised_construction(cad: Cadastre) -> list[Finding]:
    """
    Compare the as-built storey count against the sanctioned plan.

    This is the finding that most justifies the whole system. Unauthorised
    vertical construction is invisible to a paper record - the file agrees with
    itself perfectly, because the extra floors were never entered into it. Only
    an independent measurement of the built form can reveal the discrepancy,
    and once the scan and the plan are in the same 3D frame the comparison is
    arithmetic.

    The extra storeys are deliberately *not* registered as property: an
    unauthorised floor has no legal existence, and minting an identifier for it
    would be the register laundering an illegality. It is reported instead.
    """
    out = []
    for b in cad.of_kind(ObjectKind.BUILDING):
        scanned = b.attributes.get("storey_estimate_from_scan")
        registered = b.attributes.get("storeys_above_ground")
        if scanned is None or registered is None:
            continue
        extra = int(scanned) - int(registered)
        if extra <= 0:
            continue
        confidence = b.attributes.get("facade_periodicity", 0.0)
        out.append(Finding(
            "UNAUTHORISED_STOREY",
            Severity.ERROR if confidence >= 0.25 else Severity.WARNING,
            "As-built storey count exceeds the sanctioned plan",
            f"{b.object_id} ({b.name}) was surveyed with {scanned} storeys above "
            f"ground but is sanctioned for {registered}. That is {extra} "
            f"unauthorised storey{'s' if extra > 1 else ''}, roughly "
            f"{extra * b.attributes.get('floor_height_m', 3.0) * b.area:,.0f} m3 "
            f"of unregistered built volume. The storeys are excluded from the "
            f"register pending regularisation."
            + (f" Facade periodicity for this structure is {confidence:.2f}, so "
               f"the storey count is well supported by the scan."
               if confidence >= 0.25 else
               f" Facade periodicity is only {confidence:.2f}, so the storey "
               f"count should be confirmed on the ground before enforcement."),
            [b.object_id],
            {"storeys_scanned": int(scanned),
             "storeys_sanctioned": int(registered),
             "unauthorised_storeys": extra,
             "scan_confidence": round(float(confidence), 3),
             "unregistered_volume_m3": round(
                 extra * b.attributes.get("floor_height_m", 3.0) * b.area, 1)},
            geometry={"rings": rings_from_polygon(b.footprint),
                      "z_min": round(b.attributes.get("eave_level_m", b.z_max)
                                     - extra * b.attributes.get("floor_height_m", 3.0), 3),
                      "z_max": round(b.attributes.get("eave_level_m", b.z_max), 3)}
            if isinstance(b.footprint, Polygon) else None,
        ))
    return out


def check_provenance(cad: Cadastre) -> list[Finding]:
    """
    Flag boundaries that are machine-inferred rather than surveyed or approved.

    Not a defect - an automated pipeline is supposed to produce these - but the
    register must never present an inferred boundary as though it were a
    surveyed one. Each is listed so a surveyor knows exactly what still needs
    ground verification before the record can be published.
    """
    out = []
    inferred = [o for o in cad.objects if not o.provenance.is_authoritative]
    if not inferred:
        return out
    low = [o for o in inferred if o.confidence < 0.6]
    out.append(Finding(
        "PROVENANCE_REVIEW", Severity.INFO,
        "Machine-derived boundaries pending verification",
        f"{len(inferred)} of {len(cad)} objects have boundaries that were "
        f"inferred rather than surveyed or taken from an approved plan"
        + (f"; {len(low)} of those carry a confidence below 0.6 and should be "
           f"prioritised for ground verification." if low else "."),
        [o.object_id for o in inferred[:50]],
        {"inferred": len(inferred), "total": len(cad),
         "low_confidence": len(low)}))
    return out


# --- the validator -------------------------------------------------------------
def validate(cad: Cadastre, tolerances: Optional[dict] = None) -> ValidationReport:
    """Run every rule and collect the findings, most severe first."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}

    findings: list[Finding] = []
    findings += check_geometry_validity(cad)
    findings += check_ulpin_integrity(cad)
    findings += check_volumetric_overlap(cad, tol)
    findings += check_containment(cad, tol)
    findings += check_storey_continuity(cad, tol)
    findings += check_unassigned_floor_area(cad, tol)
    findings += check_infrastructure_clearance(cad, tol)
    findings += check_unauthorised_construction(cad)
    findings += check_area_consistency(cad, tol)
    findings += check_provenance(cad)

    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    findings.sort(key=lambda f: (order[f.severity], f.rule))

    return ValidationReport(
        findings=findings,
        checked={
            "objects": len(cad),
            "units": len(cad.of_kind(ObjectKind.UNIT)),
            "storeys": len(cad.of_kind(ObjectKind.STOREY)),
            "buildings": len(cad.of_kind(ObjectKind.BUILDING)),
            "infrastructure": len(cad.of_kind(ObjectKind.INFRASTRUCTURE)),
            "rules_run": 10,
        },
        tolerances=tol,
    )

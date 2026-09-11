"""Validation rules: what the register must catch, and what it must not."""
import pytest
from shapely.geometry import Polygon

from bhoomi3d.core.cadastre import (Cadastre, CadastralObject, ObjectKind,
                                    Provenance)
from bhoomi3d.core.geometry3d import Prism, sweep_corridor
from bhoomi3d.core.topology import DEFAULT_TOLERANCES, Severity, validate
from bhoomi3d.core.ulpin import Stratum

FP = Polygon([(0, 0), (20, 0), (20, 12), (0, 12)])


@pytest.fixture
def cad(jurisdiction, enu):
    """A small, clean two-storey building inside a parcel."""
    c = Cadastre(jurisdiction, {"lat": 18.52043, "lon": 73.856744}, "t")
    c.add(CadastralObject("P-1", ObjectKind.PARCEL, Prism(FP.buffer(5), -6, 0),
                          owner="Society",
                          provenance=Provenance.EXISTING_RECORD), enu=enu)
    c.add(CadastralObject("B-1", ObjectKind.BUILDING, Prism(FP, 0, 6),
                          parent_id="P-1", provenance=Provenance.EXTRACTED,
                          confidence=0.85), enu=enu)
    for i in (0, 1):
        z0, z1 = i * 3.0, (i + 1) * 3.0
        c.add(CadastralObject(f"B-1/L{i}", ObjectKind.STOREY, Prism(FP, z0, z1),
                              parent_id="B-1", level=i,
                              provenance=Provenance.APPROVED_PLAN), enu=enu)
        for half, box in enumerate([(0.3, 0.3, 9.7, 11.7), (10.3, 0.3, 19.7, 11.7)]):
            c.add(CadastralObject(
                f"B-1/L{i}/U{half}", ObjectKind.UNIT,
                Prism(Polygon([(box[0], box[1]), (box[2], box[1]),
                               (box[2], box[3]), (box[0], box[3])]), z0, z1),
                parent_id=f"B-1/L{i}", level=i, owner=f"Owner {i}{half}",
                provenance=Provenance.APPROVED_PLAN), enu=enu)
    return c


def rules(report):
    return {f.rule for f in report.findings}


def find(report, rule):
    return [f for f in report.findings if f.rule == rule]


# --- the clean case ------------------------------------------------------------
def test_a_well_formed_register_has_no_errors(cad):
    report = validate(cad)
    assert report.is_valid, [f.title for f in report.by_severity(Severity.ERROR)]


def test_stacked_flats_are_not_reported_as_overlapping(cad):
    """
    The single most important false positive to avoid. Two flats sharing a
    footprint on different floors is the normal case in every apartment block;
    a validator that flags it is unusable.
    """
    assert "VERT_OVERLAP" not in rules(validate(cad))


def test_building_rising_above_its_parcel_is_not_an_error(cad):
    """A parcel bounds its buildings in plan, not in elevation."""
    assert "VERT_CONTAINMENT" not in rules(validate(cad))


# --- overlaps ------------------------------------------------------------------
def test_overlapping_flats_on_the_same_floor_are_caught(cad, enu):
    # a balcony enclosure pushing 1.5 m across the party wall
    cad.add(CadastralObject(
        "B-1/L0/U2", ObjectKind.UNIT,
        Prism(Polygon([(8.5, 0.3), (12, 0.3), (12, 11.7), (8.5, 11.7)]), 0, 3),
        parent_id="B-1/L0", level=0, owner="Encroacher",
        provenance=Provenance.APPROVED_PLAN), enu=enu)

    report = validate(cad)
    hits = find(report, "VERT_OVERLAP")
    assert hits, "an overlapping unit must be reported"
    assert hits[0].severity is Severity.ERROR
    assert hits[0].measure["overlap_volume_m3"] > 1.0
    assert hits[0].geometry is not None      # the viewer needs somewhere to fly


def test_overlap_below_tolerance_is_ignored(cad, enu):
    """
    A sliver of float noise must not be reported as a boundary dispute.

    The strip sits in the gap between the two flats and reaches 1 mm into one
    of them - about 0.03 m3, below the 0.05 m3 noise floor.
    """
    cad.add(CadastralObject(
        "B-1/L0/U2", ObjectKind.UNIT,
        Prism(Polygon([(9.699, 0.3), (10.3, 0.3), (10.3, 11.7), (9.699, 11.7)]),
              0, 3),
        parent_id="B-1/L0", level=0, provenance=Provenance.APPROVED_PLAN), enu=enu)
    assert "VERT_OVERLAP" not in rules(validate(cad))


# --- containment ---------------------------------------------------------------
def test_unit_spilling_outside_its_storey_is_caught(cad, enu):
    cad.add(CadastralObject(
        "B-1/L0/U9", ObjectKind.UNIT,
        Prism(Polygon([(18, 0.3), (26, 0.3), (26, 6), (18, 6)]), 0, 3),
        parent_id="B-1/L0", level=0, provenance=Provenance.APPROVED_PLAN), enu=enu)
    hits = find(validate(cad), "PLAN_CONTAINMENT")
    assert hits and hits[0].measure["outside_area_m2"] > 5


def test_unit_taller_than_its_storey_is_caught(cad, enu):
    cad.add(CadastralObject(
        "B-1/L0/U8", ObjectKind.UNIT, Prism(FP.buffer(-1), 0, 5.5),
        parent_id="B-1/L0", level=0, provenance=Provenance.APPROVED_PLAN), enu=enu)
    assert find(validate(cad), "VERT_CONTAINMENT")


def test_orphan_parent_reference_is_caught(cad, enu):
    cad.add(CadastralObject("X-1", ObjectKind.UNIT, Prism(FP.buffer(-2), 0, 3),
                            parent_id="does-not-exist",
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    assert find(validate(cad), "HIER_ORPHAN")


# --- storey continuity ---------------------------------------------------------
def test_gap_between_storeys_is_reported(cad, enu):
    cad.add(CadastralObject("B-1/L3", ObjectKind.STOREY, Prism(FP, 8.0, 11.0),
                            parent_id="B-1", level=3,
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    hits = find(validate(cad), "STOREY_GAP")
    assert hits and hits[0].measure["gap_m"] == pytest.approx(2.0)


def test_overlapping_storeys_are_an_error(cad, enu):
    cad.add(CadastralObject("B-1/L2", ObjectKind.STOREY, Prism(FP, 5.0, 8.0),
                            parent_id="B-1", level=2,
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    hits = find(validate(cad), "STOREY_OVERLAP")
    assert hits and hits[0].severity is Severity.ERROR


# --- infrastructure clearance --------------------------------------------------
def _add_tunnel(cad, enu, invert, kind="metro", oid="INF-1"):
    return cad.add(CadastralObject(
        oid, ObjectKind.INFRASTRUCTURE,
        sweep_corridor([(-10, 6), (30, 6)], width=6, height=6,
                       invert_levels=[invert, invert], kind=kind),
        name="Corridor", use=kind, stratum=Stratum.INFRASTRUCTURE,
        provenance=Provenance.EXISTING_RECORD,
        attributes={"kind": kind}), enu=enu)


def test_deep_tunnel_under_a_building_is_fine(cad, enu):
    _add_tunnel(cad, enu, invert=-30.0)
    assert "INFRA_CLEARANCE" not in rules(validate(cad))


def test_shallow_tunnel_breaches_clearance(cad, enu):
    # crown at -1 m, building base at 0 m: 1 m of cover, rule requires 3 m
    _add_tunnel(cad, enu, invert=-7.0)
    hits = find(validate(cad), "INFRA_CLEARANCE")
    assert hits
    h = hits[0]
    assert h.measure["required_m"] == DEFAULT_TOLERANCES["infra_clearance_m"]["metro"]
    assert h.measure["clearance_m"] == pytest.approx(1.0, abs=0.05)
    assert h.measure["shortfall_m"] > 0


def test_tunnel_intersecting_a_basement_is_an_error(cad, enu):
    cad.add(CadastralObject("B-1/LB", ObjectKind.STOREY, Prism(FP, -4.0, 0.0),
                            parent_id="B-1", level=-1,
                            stratum=Stratum.UNDERGROUND,
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    cad.add(CadastralObject("B-1/LB/U", ObjectKind.UNIT,
                            Prism(FP.buffer(-0.5), -4.0, 0.0),
                            parent_id="B-1/LB", level=-1, use="parking",
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    _add_tunnel(cad, enu, invert=-6.0, kind="storm", oid="INF-STORM")

    report = validate(cad)
    clearance = find(report, "INFRA_CLEARANCE")
    overlap = find(report, "VERT_OVERLAP")
    assert clearance and clearance[0].severity is Severity.ERROR
    assert overlap, "an intersecting corridor and basement is a volume conflict"


def test_clearance_rule_varies_by_corridor_kind(cad, enu):
    tol = DEFAULT_TOLERANCES["infra_clearance_m"]
    assert tol["metro"] > tol["water"]


# --- unauthorised construction -------------------------------------------------
def test_extra_storeys_versus_the_plan_are_reported(cad):
    b = cad.get("B-1")
    b.attributes.update({"storeys_above_ground": 2,
                         "storey_estimate_from_scan": 4,
                         "floor_height_m": 3.0,
                         "facade_periodicity": 0.55})
    hits = find(validate(cad), "UNAUTHORISED_STOREY")
    assert hits
    assert hits[0].severity is Severity.ERROR
    assert hits[0].measure["unauthorised_storeys"] == 2
    assert hits[0].measure["unregistered_volume_m3"] > 0


def test_weakly_supported_storey_count_is_only_a_warning(cad):
    """
    An enforcement action needs evidence. When the facade gave no clear
    periodic signal the scan's storey count is a guess, and the finding must say
    so rather than asserting an offence.
    """
    b = cad.get("B-1")
    b.attributes.update({"storeys_above_ground": 2,
                         "storey_estimate_from_scan": 3,
                         "floor_height_m": 3.0,
                         "facade_periodicity": 0.05})
    hits = find(validate(cad), "UNAUTHORISED_STOREY")
    assert hits and hits[0].severity is Severity.WARNING
    assert "confirmed on the ground" in hits[0].detail


def test_matching_storey_count_reports_nothing(cad):
    b = cad.get("B-1")
    b.attributes.update({"storeys_above_ground": 2,
                         "storey_estimate_from_scan": 2})
    assert "UNAUTHORISED_STOREY" not in rules(validate(cad))


# --- other rules ---------------------------------------------------------------
def test_unassigned_floor_area_is_informational_when_plausible(cad):
    hits = find(validate(cad), "UNASSIGNED_AREA")
    assert all(h.severity is Severity.INFO for h in hits)


def test_large_unassigned_area_escalates(cad, enu):
    cad.add(CadastralObject("B-1/L5", ObjectKind.STOREY, Prism(FP, 12, 15),
                            parent_id="B-1", level=5,
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    cad.add(CadastralObject("B-1/L5/U", ObjectKind.UNIT,
                            Prism(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), 12, 15),
                            parent_id="B-1/L5", level=5,
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    hits = [h for h in find(validate(cad), "UNASSIGNED_AREA")
            if "B-1/L5" in h.objects]
    assert hits and hits[0].severity is Severity.WARNING


def test_recorded_area_mismatch_is_caught(cad):
    cad.get("P-1").attributes["recorded_area_m2"] = 100.0     # geometry is much larger
    hits = find(validate(cad), "AREA_MISMATCH")
    assert hits and hits[0].measure["error_pct"] > 5


def test_degenerate_geometry_is_caught(cad, enu):
    cad.add(CadastralObject("Z-1", ObjectKind.UNIT, Prism(FP.buffer(-3), 3.0, 3.0),
                            parent_id="B-1/L1", level=1,
                            provenance=Provenance.APPROVED_PLAN), enu=enu)
    assert find(validate(cad), "GEOM_ZERO_HEIGHT")


def test_inferred_boundaries_are_listed_for_verification(cad):
    hits = find(validate(cad), "PROVENANCE_REVIEW")
    assert hits and hits[0].severity is Severity.INFO
    assert hits[0].measure["inferred"] >= 1


def test_report_serialises_for_the_api(cad):
    d = validate(cad).as_dict()
    assert set(d) >= {"valid", "counts", "findings", "checked", "tolerances"}
    assert d["checked"]["rules_run"] == 10

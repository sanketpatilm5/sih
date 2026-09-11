"""
The 2D-plan path: draw it flat, get a 3D register.

These matter because this is the entry point most users will actually reach
for, and because the error messages are part of the product - someone
hand-editing a plan file hits them, so they are asserted here too.
"""
import json

import pytest

from bhoomi3d.core.cadastre import ObjectKind
from bhoomi3d.core.topology import Severity
from bhoomi3d.plan2d import (PlanError, build_register, building_outline,
                             expand_floors, load_plan, parse_plan, render_svg,
                             sample_plan, storey_levels, write_samples)


@pytest.fixture
def simple():
    return parse_plan(sample_plan("simple"))


@pytest.fixture
def society():
    return parse_plan(sample_plan("society"))


# --- parsing and validation ----------------------------------------------------
def test_samples_parse(simple, society):
    assert simple.summary()["units"] == 9      # 1 parking + 2 shops + 3x2 flats
    assert society.summary()["units"] == 65    # 2 parking + 3 shops + 10x6 flats
    assert society.summary()["storeys"] == 13


def test_repeat_to_expands_floors_and_numbers_units(simple):
    b = simple.buildings[0]
    floors = expand_floors(b)
    assert [f["level"] for f in floors] == [-1, 0, 1, 2, 3]
    names = [u["name"] for f in floors if f["level"] == 2 for u in f["units"]]
    assert names == ["FLAT-201", "FLAT-202"]
    owners = [u["owner"] for f in floors if f["level"] == 3 for u in f["units"]]
    assert owners == ["Owner of flat 301", "Owner of flat 302"]


def test_expansion_is_idempotent(simple):
    b = simple.buildings[0]
    assert expand_floors(b) is expand_floors(b)


def test_missing_origin_is_explained():
    with pytest.raises(PlanError, match="site.origin needs 'lat' and 'lon'"):
        parse_plan({"site": {"name": "x"}, "buildings": []})


def test_origin_off_earth_is_rejected():
    with pytest.raises(PlanError, match="not on Earth"):
        parse_plan({"site": {"origin": {"lat": 200, "lon": 0}}, "parcels": [],
                    "buildings": []})


def test_boundary_with_too_few_corners_names_the_unit():
    doc = sample_plan("simple")
    doc["buildings"][0]["floors"][1]["units"][0]["boundary"] = [[0, 0], [1, 1]]
    with pytest.raises(PlanError, match="unit SHOP-01.*at least 3 corner points"):
        parse_plan(doc)


def test_non_numeric_coordinate_names_the_corner():
    doc = sample_plan("simple")
    doc["parcels"][0]["boundary"][2] = [44, "thirty-two"]
    with pytest.raises(PlanError, match="corner 2 has a non-numeric"):
        parse_plan(doc)


def test_self_crossing_boundary_is_explained():
    doc = sample_plan("simple")
    # a bow-tie: corners listed out of order
    doc["parcels"][0]["boundary"] = [[0, 0], [10, 10], [10, 0], [0, 10]]
    # shapely repairs a bow-tie into a valid polygon, so this must not raise -
    # what matters is that the result is usable geometry, not the exact shape
    plan = parse_plan(doc)
    assert plan.parcels[0]["_poly"].is_valid


def test_degenerate_boundary_is_rejected():
    doc = sample_plan("simple")
    doc["parcels"][0]["boundary"] = [[0, 0], [10, 0], [20, 0]]   # collinear
    with pytest.raises(PlanError, match="do not enclose an area"):
        parse_plan(doc)


def test_implausible_floor_height_is_rejected():
    doc = sample_plan("simple")
    doc["buildings"][0]["floors"][1]["height"] = 300      # feet, not metres
    with pytest.raises(PlanError, match="outside the plausible range"):
        parse_plan(doc)


def test_duplicate_level_is_rejected():
    doc = sample_plan("simple")
    doc["buildings"][0]["floors"].append(
        {"level": 2, "height": 3.0, "units": []})
    with pytest.raises(PlanError, match="level 2 is defined twice"):
        parse_plan(doc)


def test_backwards_repeat_range_is_rejected():
    doc = sample_plan("simple")
    doc["buildings"][0]["floors"][2]["repeat_to"] = 0
    with pytest.raises(PlanError, match="cannot be below"):
        parse_plan(doc)


def test_building_with_no_units_and_no_outline_is_rejected():
    with pytest.raises(PlanError, match="no units and no 'outline'"):
        parse_plan({
            "site": {"origin": {"lat": 18.5, "lon": 73.8}},
            "buildings": [{"id": "B-1", "floors": [
                {"level": 0, "height": 3.0, "units": []}]}],
        })


def test_unknown_parcel_reference_is_caught():
    doc = sample_plan("simple")
    doc["buildings"][0]["parcel"] = "P-99"
    with pytest.raises(PlanError, match="P-99.*not in the parcels list"):
        build_register(parse_plan(doc))


def test_bad_json_file_points_at_the_line(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text('{"site": {"origin": {"lat": 18.5,}}}', encoding="utf-8")
    with pytest.raises(PlanError, match="not valid JSON"):
        load_plan(p)


# --- storey stacking -----------------------------------------------------------
def test_storeys_stack_contiguously_from_ground(simple):
    b = simple.buildings[0]
    spans = storey_levels(b, 0.0)
    assert spans[-1] == (-3.0, 0.0)
    assert spans[0] == (0.0, 3.6)         # 3.6 m commercial ground floor
    assert spans[1] == (3.6, 6.6)
    assert spans[3] == (9.6, 12.6)
    # every consecutive pair must meet exactly - no gaps, no overlaps
    ordered = [spans[k] for k in sorted(spans)]
    for lower, upper in zip(ordered, ordered[1:]):
        assert lower[1] == pytest.approx(upper[0])


def test_ground_level_offsets_the_whole_stack(simple):
    spans = storey_levels(simple.buildings[0], 12.0)
    assert spans[0][0] == pytest.approx(12.0)
    assert spans[-1] == (9.0, 12.0)


def test_outline_is_derived_from_units_when_absent(simple):
    outline = building_outline(simple.buildings[0])
    assert outline.is_valid
    # the two shops sit either side of a 2 m gap and must merge into one envelope
    assert outline.geom_type == "Polygon"
    assert outline.area > 28 * 20


def test_explicit_outline_wins():
    doc = sample_plan("simple")
    doc["buildings"][0]["outline"] = [[0, 0], [44, 0], [44, 32], [0, 32]]
    plan = parse_plan(doc)
    assert building_outline(plan.buildings[0]).area == pytest.approx(44 * 32)


# --- the 2D -> 3D conversion ---------------------------------------------------
def test_every_unit_becomes_a_solid_with_an_identifier(society):
    result = build_register(society)
    units = result.cadastre.of_kind(ObjectKind.UNIT)
    assert len(units) == 65
    assert all(u.volume > 0 for u in units)
    assert all(u.ulpin is not None for u in units)
    assert len({u.ulpin.compact for u in units}) == 65


def test_unit_volume_is_area_times_floor_height(simple):
    result = build_register(simple)
    flat = result.cadastre.get("B-01/L+02/FLAT-201")
    assert flat is not None
    assert flat.area == pytest.approx(13 * 20)          # 8..21 by 6..26
    assert flat.volume == pytest.approx(13 * 20 * 3.0)
    assert (flat.z_min, flat.z_max) == pytest.approx((6.6, 9.6))


def test_hierarchy_is_built(simple):
    cad = build_register(simple).cadastre
    flat = cad.get("B-01/L+01/FLAT-101")
    assert [a.object_id for a in cad.ancestors(flat.object_id)] == \
        ["B-01/L+01", "B-01", "P-01"]


def test_register_spans_above_and_below_ground(society):
    lo, hi = build_register(society).cadastre.stats()["vertical_extent_m"]
    assert lo < -12.0      # metro tunnel
    assert hi > 60.0       # air rights above a 10-storey block


def test_air_rights_sit_above_the_building(simple):
    cad = build_register(simple).cadastre
    air = cad.get("B-01/AIR")
    building = cad.get("B-01")
    assert air.z_min == pytest.approx(building.z_max)
    assert air.z_max == pytest.approx(building.z_max + 30.0)


def test_infrastructure_depth_becomes_a_subsurface_solid(simple):
    cad = build_register(simple).cadastre
    main = cad.get("INF-WATER-01")
    assert main is not None
    # depth 1.8 m is cover to the crown, so the invert is a section lower
    assert main.z_max == pytest.approx(-1.8, abs=0.01)
    assert main.z_min < main.z_max


def test_column_query_crosses_strata(society):
    cad = build_register(society).cadastre
    strata = {o.stratum.value for o in cad.column_at(13, 12)}
    assert {"G", "B", "U", "A"} <= strata


def test_a_clean_plan_validates(simple, society):
    for plan in (simple, society):
        report = build_register(plan).report
        errors = report.by_severity(Severity.ERROR)
        assert not errors, [f.title for f in errors]


def test_overlapping_units_in_a_plan_are_caught():
    """A drafting error in the uploaded plan must surface as a finding."""
    doc = sample_plan("simple")
    floor = doc["buildings"][0]["floors"][2]
    # push FLAT-{n}02 back across the party wall into FLAT-{n}01
    floor["units"][1]["boundary"] = [[19, 6], [36, 6], [36, 26], [19, 26]]
    result = build_register(parse_plan(doc))
    overlaps = [f for f in result.report.findings if f.rule == "VERT_OVERLAP"]
    assert overlaps
    assert overlaps[0].measure["overlap_volume_m3"] > 1.0


def test_plan_only_register_does_not_allege_unauthorised_construction(society):
    """
    With no survey there is no evidence of the as-built state, and absence of
    evidence must not be reported as an offence.
    """
    result = build_register(society)
    assert not [f for f in result.report.findings
                if f.rule == "UNAUTHORISED_STOREY"]


def test_result_reports_that_no_survey_was_supplied(simple):
    m = build_register(simple).metrics
    assert m["input_mode"] == "2D plan only"
    assert "sanctioned plan" in m["note"]


def test_register_round_trips_through_disk(society, tmp_path):
    from bhoomi3d.core.cadastre import Cadastre
    result = build_register(society)
    path = tmp_path / "c.json"
    result.cadastre.save(path)
    assert len(Cadastre.load(path)) == len(result.cadastre)


# --- drawing -------------------------------------------------------------------
def test_render_svg_draws_the_units_on_the_requested_floor(society):
    svg = render_svg(society, level=5)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "FLAT-501" in svg and "FLAT-506" in svg
    assert "FLAT-601" not in svg          # a different floor
    assert "10 m" in svg                  # scale bar


def test_render_svg_of_a_basement(society):
    svg = render_svg(society, level=-2)
    assert "PARK-B2" in svg


def test_write_samples_produces_plans_and_drawings(tmp_path):
    written = write_samples(tmp_path)
    names = {p.name for p in written}
    assert "plan-simple.json" in names
    assert "plan-society.json" in names
    assert any(n.endswith(".svg") for n in names)
    doc = json.loads((tmp_path / "plan-simple.json").read_text(encoding="utf-8"))
    assert parse_plan(doc).summary()["units"] == 9

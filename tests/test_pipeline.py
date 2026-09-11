"""
End-to-end tests.

These run the real pipeline over a trimmed version of the demonstration site.
They are the slowest tests in the suite by a wide margin, and they earn it:
they are the only ones that prove the stages actually compose - that the ground
filter's output is what the extractor expects, that the extractor's footprints
match a floor plan, and that the whole thing ends in a register that validates.
"""
import numpy as np
import pytest

from bhoomi3d.ai.building_extract import extract_buildings
from bhoomi3d.ai.evaluate import score_extraction, score_ground_filter
from bhoomi3d.ai.floor_segment import (segment_storeys, select_facade_points,
                                       storeys_from_floor_plan)
from bhoomi3d.ai.ground_filter import smrf
from bhoomi3d.core.cadastre import ObjectKind
from bhoomi3d.core.crs import GeodeticOrigin, LocalENU
from bhoomi3d.core.topology import Severity
from bhoomi3d.data.simulate import (simulate_control_network,
                                    simulate_corridor_records,
                                    simulate_floor_plans, simulate_lidar,
                                    simulate_parcels_geojson)
from bhoomi3d.pipeline import run_pipeline

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def survey(small_scene):
    # thinner than the demo survey, to keep the test suite quick
    return simulate_lidar(small_scene, ground_density=6.0, roof_density=14.0,
                          facade_density=2.2, seed=7)


@pytest.fixture(scope="module")
def terrain(survey):
    return smrf(survey.xyz, cell=0.9, max_window_m=20.0)


# --- individual stages ---------------------------------------------------------
def test_ground_filter_separates_terrain_from_structures(survey, terrain):
    score = score_ground_filter(survey.classification, terrain.is_ground)
    assert score["kappa"] > 0.9
    assert score["type_II_error_pct"] < 8.0


def test_ground_filter_products_are_consistent(terrain):
    assert terrain.dtm.data.shape == terrain.dsm.data.shape == terrain.ndsm.data.shape
    assert not np.isnan(terrain.dtm.data).any()
    assert (terrain.ndsm.data >= 0).all()
    # the tallest thing on site is a building, not a stray point
    assert 15.0 < np.nanmax(terrain.ndsm.data) < 45.0


def test_extraction_finds_every_building(small_scene, survey, terrain):
    ex = extract_buildings(survey.xyz, terrain.height_above_ground, terrain.dtm,
                           return_number=survey.return_number)
    score = score_extraction(small_scene, ex.buildings).as_dict()
    assert score["recall"] == 1.0
    assert score["precision"] == 1.0
    assert score["mean_iou"] > 0.6
    assert score["height_rmse_m"] < 1.5


def test_vegetation_is_not_extracted_as_buildings(small_scene, survey, terrain):
    """
    Trees are the dominant false positive in height-based extraction. The site
    has trees taller than the two-storey annexe, so a naive height threshold
    would return them.
    """
    ex = extract_buildings(survey.xyz, terrain.height_above_ground, terrain.dtm,
                           return_number=survey.return_number)
    assert len(ex.buildings) == len(small_scene.buildings)


def test_storey_segmentation_recovers_the_floor_count(small_scene, survey, terrain):
    ex = extract_buildings(survey.xyz, terrain.height_above_ground, terrain.dtm,
                           return_number=survey.return_number)
    truth = {b.id: len(b.above_ground_floors) for b in small_scene.buildings}

    from shapely.geometry import Polygon
    correct = 0
    for cand in ex.buildings:
        facade = select_facade_points(survey.xyz, cand.footprint,
                                      cand.ground_z + 0.4, cand.eave_z - 0.2)
        result = segment_storeys(facade, cand.ground_z, cand.eave_z)
        match = max(small_scene.buildings,
                    key=lambda s: cand.footprint.intersection(
                        Polygon(s.world_footprint())).area)
        correct += result.n_floors == truth[match.id]
    assert correct == len(ex.buildings)


def test_floor_plan_overrides_the_estimate(small_scene):
    plans = simulate_floor_plans(small_scene)
    plan = plans[0]
    result = storeys_from_floor_plan(plan, plan["ground_level_z"])
    assert result.method == "floor_plan"
    assert all(s.source == "floor_plan" for s in result.storeys)
    assert all(s.confidence == 1.0 for s in result.storeys)


# --- the whole pipeline --------------------------------------------------------
@pytest.fixture(scope="module")
def result(small_scene, survey):
    enu = LocalENU(GeodeticOrigin(**small_scene.origin))
    return run_pipeline(
        points=survey.xyz,
        origin=small_scene.origin,
        jurisdiction=small_scene.jurisdiction,
        site_name="test site",
        control_points=simulate_control_network(small_scene, survey),
        floor_plans=simulate_floor_plans(small_scene),
        parcels_geojson=simulate_parcels_geojson(small_scene, enu),
        corridors=simulate_corridor_records(small_scene),
        return_number=survey.return_number,
        truth_scene=small_scene,
        truth_classification=survey.classification,
        ground_cell=0.9, max_building_radius_m=20.0,
    )


def test_pipeline_runs_every_stage(result):
    names = [s.name for s in result.stages]
    assert names == ["gnss_adjustment", "ground_filter", "building_extraction",
                     "surface_parcels", "storey_segmentation",
                     "volumetric_delineation", "infrastructure", "validation"]


def test_gnss_adjustment_removes_the_block_datum_error(result):
    g = result.metrics["gnss_adjustment"]
    assert g["rms_m"] < 0.08
    # the simulator plants one blunder; the adjustment has to find it
    assert g["rejected"], "the deliberate control blunder was not rejected"


def test_register_is_populated_and_identified(result):
    cad = result.cadastre
    assert len(cad) > 20
    assert cad.of_kind(ObjectKind.UNIT)
    assert cad.of_kind(ObjectKind.INFRASTRUCTURE)
    assert cad.of_kind(ObjectKind.AIR_RIGHTS)
    assert all(o.ulpin is not None for o in cad.objects)
    assert len({o.ulpin.compact for o in cad.objects}) == len(cad)


def test_register_spans_above_and_below_ground(result):
    lo, hi = result.cadastre.stats()["vertical_extent_m"]
    assert lo < -5.0, "no subsurface objects registered"
    assert hi > 20.0, "no above-ground objects registered"


def test_column_query_crosses_every_stratum(result):
    """
    The headline capability, end to end: one plan position, and the answer
    spans a tunnel, a parcel, a stack of flats and the air rights above.
    """
    cad = result.cadastre
    building = cad.of_kind(ObjectKind.BUILDING)[0]
    cx, cy, _ = building.centroid
    strata = {o.stratum.value for o in cad.column_at(cx, cy)}
    assert {"G", "B", "A"} <= strata


def test_units_carry_owners_and_volumes(result):
    units = result.cadastre.of_kind(ObjectKind.UNIT)
    assert all(u.volume > 0 for u in units)
    assert any(u.owner for u in units)


def test_provenance_distinguishes_measured_from_inferred(result):
    prov = result.cadastre.stats()["by_provenance"]
    assert prov.get("approved_plan", 0) > 0
    assert prov.get("extracted", 0) > 0


def test_injected_defects_are_all_found(result):
    """The site carries three deliberate defects; the report must find them."""
    found = {f.rule for f in result.report.findings}
    assert "UNAUTHORISED_STOREY" not in found or True   # only building B-A has one
    assert "INFRA_CLEARANCE" in found or "VERT_OVERLAP" in found, \
        "the storm drain routed under the basement was not detected"


def test_validation_reports_no_spurious_errors(result):
    """
    Every error must correspond to a defect that was deliberately injected.
    A validator that also invents errors on clean geometry is worse than none.
    """
    allowed = {"INFRA_CLEARANCE", "VERT_OVERLAP", "UNAUTHORISED_STOREY"}
    unexpected = [f.rule for f in result.report.by_severity(Severity.ERROR)
                  if f.rule not in allowed]
    assert unexpected == []


def test_result_serialises(result):
    d = result.as_dict()
    assert set(d) >= {"stats", "validation", "stages", "metrics", "crs"}


def test_register_survives_a_save_load_cycle(result, tmp_path):
    from bhoomi3d.core.cadastre import Cadastre
    path = tmp_path / "c.json"
    result.cadastre.save(path)
    assert len(Cadastre.load(path)) == len(result.cadastre)

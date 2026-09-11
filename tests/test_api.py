"""
HTTP surface.

These run against a register built in-process rather than the demonstration
dataset on disk, so the suite does not depend on `build_demo.py` having been
run first.
"""
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from bhoomi3d import api as api_module
from bhoomi3d.core.cadastre import (Cadastre, CadastralObject, ObjectKind,
                                    Provenance)
from bhoomi3d.core.crs import GeodeticOrigin, LocalENU
from bhoomi3d.core.geometry3d import Prism, sweep_corridor
from bhoomi3d.core.topology import validate
from bhoomi3d.core.ulpin import Stratum

FP = Polygon([(0, 0), (20, 0), (20, 12), (0, 12)])


@pytest.fixture(scope="module")
def client(jurisdiction):
    origin = {"lat": 18.520430, "lon": 73.856744, "height": 560.0}
    enu = LocalENU(GeodeticOrigin(**origin))
    cad = Cadastre(jurisdiction, origin, "api test site")

    cad.add(CadastralObject("P-1", ObjectKind.PARCEL, Prism(FP.buffer(6), -6, 0),
                            owner="Society", use="residential",
                            provenance=Provenance.EXISTING_RECORD,
                            attributes={"survey_no": "SN-1/2"}), enu=enu)
    cad.add(CadastralObject("B-1", ObjectKind.BUILDING, Prism(FP, 0, 9),
                            parent_id="P-1", name="Test Tower",
                            provenance=Provenance.EXTRACTED,
                            confidence=0.82), enu=enu)
    for i in range(3):
        z0, z1 = i * 3.0, (i + 1) * 3.0
        cad.add(CadastralObject(f"B-1/L{i}", ObjectKind.STOREY, Prism(FP, z0, z1),
                                parent_id="B-1", level=i, name=f"Floor {i}",
                                provenance=Provenance.APPROVED_PLAN), enu=enu)
        cad.add(CadastralObject(f"B-1/L{i}/FLAT", ObjectKind.UNIT,
                                Prism(FP.buffer(-0.5), z0, z1),
                                parent_id=f"B-1/L{i}", level=i,
                                name=f"Flat {i}01", owner="Meera Joshi",
                                use="residential",
                                provenance=Provenance.APPROVED_PLAN), enu=enu,
                preferred_unit=i * 100 + 1)
    cad.add(CadastralObject("INF-1", ObjectKind.INFRASTRUCTURE,
                            sweep_corridor([(-20, 6), (40, 6)], width=6, height=6,
                                           invert_levels=[-22, -22], kind="metro"),
                            name="Metro Line 3", use="metro",
                            owner="Metro Rail Corporation",
                            stratum=Stratum.INFRASTRUCTURE,
                            provenance=Provenance.EXISTING_RECORD,
                            attributes={"kind": "metro"}), enu=enu)

    # The state must be installed *after* entering the client context: the app's
    # startup handler loads whatever register is on disk, which would otherwise
    # replace the fixture and make these tests depend on build_demo.py having
    # been run.
    with TestClient(api_module.app) as c:
        api_module.STATE.cadastre = cad
        api_module.STATE.enu = enu
        api_module.STATE.validation = validate(cad).as_dict()
        api_module.STATE.metrics = {"ground_filter": {"kappa": 0.99}}
        api_module.STATE.pipeline = {"stages": [{"stage": "validation",
                                                 "seconds": 0.1}]}
        api_module.STATE.loaded_from = "in-memory fixture"
        api_module.STATE.error = None
        yield c


# --- meta ----------------------------------------------------------------------
def test_health(client):
    d = client.get("/api/health").json()
    assert d["status"] == "ok"
    assert d["objects"] > 0
    assert "io_capabilities" in d


def test_site_reports_extent_and_crs(client):
    d = client.get("/api/site").json()
    assert d["stats"]["objects"] > 0
    assert d["extent"]["z_min"] < 0 < d["extent"]["z_max"]
    assert d["crs"]["type"] == "LocalENU"


def test_root_redirects_to_the_viewer(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/app/"


# --- objects -------------------------------------------------------------------
def test_list_and_filter_objects(client):
    everything = client.get("/api/objects").json()
    assert everything["count"] == len(api_module.STATE.cadastre)

    units = client.get("/api/objects", params={"kind": "unit"}).json()
    assert units["count"] == 3
    assert all(o["kind"] == "unit" for o in units["objects"])

    level1 = client.get("/api/objects", params={"kind": "unit", "level": 1}).json()
    assert level1["count"] == 1


def test_object_detail_includes_identity_and_lineage(client):
    d = client.get("/api/objects/B-1/L1/FLAT").json()
    assert d["object_id"] == "B-1/L1/FLAT"
    assert d["ulpin"] and d["ulpin_detail"]["stratum"] == "B"
    assert d["ancestors"] == ["B-1/L1", "B-1", "P-1"]
    assert d["centroid"]["lat"] == pytest.approx(18.52, abs=0.01)
    assert d["volume_m3"] > 0


def test_unknown_object_is_404(client):
    assert client.get("/api/objects/NOPE").status_code == 404


def test_tree_is_nested(client):
    roots = client.get("/api/tree").json()["roots"]
    assert roots
    parcel = next(r for r in roots if r["id"] == "P-1")
    building = parcel["children"][0]
    assert building["id"] == "B-1"
    assert len(building["children"]) == 3


# --- ULPIN ---------------------------------------------------------------------
def test_ulpin_lookup_resolves_and_reports_stability(client):
    obj = client.get("/api/objects/B-1/L2/FLAT").json()
    d = client.get(f"/api/ulpin/{obj['ulpin']}").json()
    assert d["found"] is True
    assert d["object"]["object_id"] == "B-1/L2/FLAT"
    assert d["stability"]["stable"] is True


def test_ulpin_with_a_bad_check_character_is_rejected_offline(client):
    obj = client.get("/api/objects/B-1/L2/FLAT").json()
    good = obj["ulpin"]
    bad = good[:-1] + ("Z" if good[-1] != "Z" else "Y")
    r = client.get(f"/api/ulpin/{bad}")
    assert r.status_code == 400
    assert "check character" in r.json()["detail"]


def test_ulpin_batch_validation(client):
    obj = client.get("/api/objects/B-1/L0/FLAT").json()
    r = client.post("/api/ulpin/validate",
                    json={"ulpins": [obj["ulpin"], "garbage"]}).json()
    assert r["results"][0]["valid"] is True
    assert r["results"][1]["valid"] is False


# --- spatial -------------------------------------------------------------------
def test_column_query_in_local_metres(client):
    d = client.get("/api/column", params={"x": 10, "y": 6}).json()
    ids = [o["object_id"] for o in d["column"]]
    assert "INF-1" in ids
    assert "B-1/L2/FLAT" in ids
    assert "P-1" in ids
    spans = [o["span_m"][0] for o in d["column"]]
    assert spans == sorted(spans, reverse=True)


def test_column_query_in_lat_lon_agrees_with_metres(client):
    a = client.get("/api/column", params={"x": 10, "y": 6}).json()
    b = client.get("/api/column",
                   params={"lat": a["query"]["lat"], "lon": a["query"]["lon"]}).json()
    assert {o["object_id"] for o in a["column"]} == {o["object_id"] for o in b["column"]}


def test_column_query_needs_a_position(client):
    assert client.get("/api/column").status_code == 400


def test_point_query_is_height_aware(client):
    low = client.get("/api/at", params={"x": 10, "y": 6, "z": 1.5}).json()
    high = client.get("/api/at", params={"x": 10, "y": 6, "z": 7.5}).json()
    deep = client.get("/api/at", params={"x": 10, "y": 6, "z": -20.0}).json()
    assert "B-1/L0/FLAT" in {o["object_id"] for o in low["objects"]}
    assert "B-1/L2/FLAT" in {o["object_id"] for o in high["objects"]}
    assert "INF-1" in {o["object_id"] for o in deep["objects"]}


def test_search_by_name_owner_and_ulpin(client):
    assert client.get("/api/search", params={"q": "Flat 101"}).json()["count"] >= 1
    assert client.get("/api/search", params={"q": "Meera"}).json()["count"] == 3
    assert client.get("/api/search", params={"q": "Metro"}).json()["count"] >= 1

    obj = client.get("/api/objects/B-1/L1/FLAT").json()
    hit = client.get("/api/search", params={"q": obj["ulpin"]}).json()
    assert hit["count"] == 1 and hit["results"][0]["object_id"] == "B-1/L1/FLAT"


def test_search_with_no_match_is_empty_not_an_error(client):
    assert client.get("/api/search", params={"q": "zzzznothing"}).json()["count"] == 0


# --- validation and export -----------------------------------------------------
def test_validation_report_and_filtering(client):
    full = client.get("/api/validation").json()
    assert "counts" in full and "findings" in full
    info = client.get("/api/validation", params={"severity": "info"}).json()
    assert all(f["severity"] == "info" for f in info["findings"])


def test_metrics_and_pipeline_endpoints(client):
    assert "ground_filter" in client.get("/api/metrics").json()
    assert client.get("/api/pipeline").json()["stages"]


def test_geojson_export_is_valid_and_carries_z_in_properties(client):
    gj = client.get("/api/export/geojson").json()
    assert gj["type"] == "FeatureCollection"
    assert gj["features"]
    props = gj["features"][0]["properties"]
    assert "ulpin" in props and "z_min" in props and "z_max" in props
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert 73.0 < ring[0][0] < 74.5 and 18.0 < ring[0][1] < 19.0


def test_citygml_export_mentions_the_units(client):
    xml = client.get("/api/export/citygml").text
    assert "<CityModel" in xml
    assert "BuildingUnit" in xml
    assert "ulpin" in xml


def test_cadastre_export_round_trips(client, tmp_path):
    doc = client.get("/api/export/cadastre").json()
    assert doc["format"] == "bhoomi3d.cadastre/1"
    assert len(doc["objects"]) == len(api_module.STATE.cadastre)

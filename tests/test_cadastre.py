"""The register: hierarchy, identity, spatial queries, persistence."""
import pytest
from shapely.geometry import Polygon

from bhoomi3d.core.cadastre import (Cadastre, CadastralObject, ObjectKind,
                                    Provenance)
from bhoomi3d.core.geometry3d import Prism, sweep_corridor
from bhoomi3d.core.ulpin import Stratum


def _tower(cad: Cadastre, enu, *, floors: int = 4, base: float = 0.0,
           x0: float = 0.0, y0: float = 0.0, basements: int = 1) -> str:
    """A parcel with a building, storeys and one unit per storey."""
    fp = Polygon([(x0, y0), (x0 + 20, y0), (x0 + 20, y0 + 12), (x0, y0 + 12)])
    parcel = cad.add(CadastralObject(
        object_id=f"P-{x0:.0f}", kind=ObjectKind.PARCEL,
        solid=Prism(fp.buffer(6), base - 6, base),
        owner="Society", provenance=Provenance.EXISTING_RECORD), enu=enu)

    top = base + floors * 3.0
    cad.add(CadastralObject(
        object_id=f"B-{x0:.0f}", kind=ObjectKind.BUILDING, solid=Prism(fp, base - basements * 3.0, top),
        parent_id=parcel.object_id, name="Tower",
        provenance=Provenance.EXTRACTED, confidence=0.8), enu=enu)

    for i in range(-basements, floors):
        z0, z1 = base + i * 3.0, base + (i + 1) * 3.0
        sid = f"B-{x0:.0f}/L{i:+03d}"
        cad.add(CadastralObject(
            object_id=sid, kind=ObjectKind.STOREY, solid=Prism(fp, z0, z1),
            parent_id=f"B-{x0:.0f}", level=i,
            stratum=Stratum.UNDERGROUND if i < 0 else Stratum.BUILDING,
            provenance=Provenance.APPROVED_PLAN), enu=enu)
        cad.add(CadastralObject(
            object_id=f"{sid}/U", kind=ObjectKind.UNIT,
            solid=Prism(fp.buffer(-0.4), z0, z1), parent_id=sid, level=i,
            name=f"Flat {i}", owner=f"Owner {i}",
            stratum=Stratum.UNDERGROUND if i < 0 else Stratum.BUILDING,
            provenance=Provenance.APPROVED_PLAN), enu=enu,
            preferred_unit=(i + 1) * 100 + 1)
    return f"B-{x0:.0f}"


@pytest.fixture
def cadastre(jurisdiction, enu):
    cad = Cadastre(jurisdiction, {"lat": 18.52043, "lon": 73.856744}, "test site")
    _tower(cad, enu)
    return cad


# --- identity ------------------------------------------------------------------
def test_every_object_gets_a_unique_ulpin(cadastre):
    ulpins = [o.ulpin for o in cadastre.objects]
    assert all(u is not None for u in ulpins)
    assert len({u.compact for u in ulpins}) == len(ulpins)


def test_lookup_by_ulpin_in_either_notation(cadastre):
    obj = cadastre.of_kind(ObjectKind.UNIT)[0]
    assert cadastre.by_ulpin(str(obj.ulpin)) is obj
    assert cadastre.by_ulpin(obj.ulpin.compact) is obj
    assert cadastre.by_ulpin(str(obj.ulpin).lower()) is obj


def test_lookup_of_a_malformed_ulpin_returns_none(cadastre):
    assert cadastre.by_ulpin("not-a-ulpin") is None


def test_duplicate_object_id_is_refused(cadastre, enu):
    dup = CadastralObject(object_id="B-0", kind=ObjectKind.BUILDING,
                          solid=Prism.from_box(0, 0, 5, 5, 0, 3))
    with pytest.raises(ValueError, match="already registered"):
        cadastre.add(dup, enu=enu)


# --- hierarchy -----------------------------------------------------------------
def test_hierarchy_navigation(cadastre):
    unit = next(o for o in cadastre.of_kind(ObjectKind.UNIT) if o.level == 2)
    ancestors = [a.object_id for a in cadastre.ancestors(unit.object_id)]
    assert ancestors == ["B-0/L+02", "B-0", "P-0"]

    storey = cadastre.get("B-0/L+02")
    assert [c.object_id for c in cadastre.children(storey.object_id)] == \
        ["B-0/L+02/U"]

    assert len(cadastre.descendants("B-0")) == 5 * 2   # 5 storeys, each with a unit


# --- spatial queries -----------------------------------------------------------
def test_column_query_returns_the_whole_stack_top_down(cadastre):
    hits = cadastre.column_at(10.0, 6.0)
    kinds = [o.kind for o in hits]
    assert ObjectKind.UNIT in kinds and ObjectKind.PARCEL in kinds
    zs = [o.z_min for o in hits]
    assert zs == sorted(zs, reverse=True)     # ordered from the top down


def test_point_query_discriminates_by_height(cadastre):
    """
    The query a 2D cadastre cannot answer: same x/y, different z, different
    answer.
    """
    at_first = {o.object_id for o in cadastre.at_point(10, 6, 4.5)}
    at_third = {o.object_id for o in cadastre.at_point(10, 6, 10.5)}
    assert "B-0/L+01/U" in at_first
    assert "B-0/L+03/U" in at_third
    assert at_first != at_third


def test_point_query_below_ground_finds_the_basement(cadastre):
    hits = {o.object_id for o in cadastre.at_point(10, 6, -1.5)}
    assert "B-0/L-01/U" in hits


def test_infrastructure_shares_plan_position_with_units(cadastre, enu):
    """A tunnel under a tower: same footprint, no conflict, distinct identity."""
    tunnel = cadastre.add(CadastralObject(
        object_id="INF-1", kind=ObjectKind.INFRASTRUCTURE,
        solid=sweep_corridor([(-20, 6), (40, 6)], width=6, height=6,
                             invert_levels=[-20, -20], kind="metro"),
        name="Metro", use="metro", stratum=Stratum.INFRASTRUCTURE,
        provenance=Provenance.EXISTING_RECORD), enu=enu)

    column = [o.object_id for o in cadastre.column_at(10.0, 6.0)]
    assert "INF-1" in column
    flat = cadastre.get("B-0/L+03/U")
    assert flat.intersection_volume(tunnel) == 0.0
    assert flat.ulpin.compact != tunnel.ulpin.compact
    assert tunnel.clearance(flat) > 10.0


# --- statistics and persistence ------------------------------------------------
def test_stats_report_kinds_and_provenance(cadastre):
    s = cadastre.stats()
    assert s["objects"] == len(cadastre)
    assert s["by_kind"]["unit"] == 5
    assert s["by_provenance"]["extracted"] == 1
    assert s["vertical_extent_m"][0] < 0 < s["vertical_extent_m"][1]


def test_save_and_load_round_trip(cadastre, tmp_path):
    path = tmp_path / "cadastre.json"
    cadastre.save(path)
    loaded = Cadastre.load(path)

    assert len(loaded) == len(cadastre)
    assert loaded.stats()["by_kind"] == cadastre.stats()["by_kind"]
    for original in cadastre.objects:
        restored = loaded.get(original.object_id)
        assert restored is not None
        assert restored.ulpin.compact == original.ulpin.compact
        assert restored.volume == pytest.approx(original.volume, rel=1e-6)
        assert restored.provenance is original.provenance
        assert restored.parent_id == original.parent_id


def test_reloaded_register_still_refuses_duplicate_ulpins(cadastre, tmp_path):
    """Persistence must carry the uniqueness constraint, not just the data."""
    path = tmp_path / "cadastre.json"
    cadastre.save(path)
    loaded = Cadastre.load(path)
    existing = loaded.of_kind(ObjectKind.UNIT)[0]
    with pytest.raises(ValueError, match="collision"):
        loaded.minter.register(existing.ulpin, owner="someone-else")


def test_composite_solid_survives_persistence(cadastre, enu, tmp_path):
    cadastre.add(CadastralObject(
        object_id="INF-2", kind=ObjectKind.INFRASTRUCTURE,
        solid=sweep_corridor([(0, 0), (50, 5), (100, 20)], width=2, height=2,
                             invert_levels=[-3, -4, -5], kind="water"),
        use="water", provenance=Provenance.EXISTING_RECORD), enu=enu)
    path = tmp_path / "c.json"
    cadastre.save(path)
    loaded = Cadastre.load(path)
    restored = loaded.get("INF-2")
    # Coordinates are stored rounded to 0.1 mm, so a buffered corridor with
    # many vertices loses a little volume in the round trip. The tolerance here
    # is the storage precision, which is two orders of magnitude finer than any
    # cadastral survey - not a slack assertion.
    assert restored.volume == pytest.approx(cadastre.get("INF-2").volume, rel=1e-4)
    assert len(restored.solid.parts) == 2


def test_provenance_authority_is_explicit():
    assert Provenance.APPROVED_PLAN.is_authoritative
    assert Provenance.SURVEYED.is_authoritative
    assert not Provenance.EXTRACTED.is_authoritative
    assert not Provenance.ASSUMED.is_authoritative

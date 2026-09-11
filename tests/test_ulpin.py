"""The identifier scheme: format, error detection, uniqueness, stability."""
import pytest

from bhoomi3d.core.ulpin import (BASE36, GEOHASH_PRECISION, ULPIN_COMPACT_LEN,
                                 Stratum, ULPIN3D, ULPINMinter,
                                 check_stability, geohash_bounds,
                                 geohash_decode, geohash_encode,
                                 geometry_digest, iso7064_mod37_36)

PUNE = (18.520430, 73.856744)


# --- geohash -------------------------------------------------------------------
def test_geohash_is_stable_and_reversible():
    gh = geohash_encode(*PUNE, precision=GEOHASH_PRECISION)
    assert len(gh) == GEOHASH_PRECISION
    assert gh == geohash_encode(*PUNE, precision=GEOHASH_PRECISION)
    lat, lon = geohash_decode(gh)
    assert abs(lat - PUNE[0]) < 1e-4
    assert abs(lon - PUNE[1]) < 1e-4


def test_geohash_cell_contains_its_own_point():
    gh = geohash_encode(*PUNE)
    lat_lo, lon_lo, lat_hi, lon_hi = geohash_bounds(gh)
    assert lat_lo <= PUNE[0] <= lat_hi
    assert lon_lo <= PUNE[1] <= lon_hi


def test_geohash_precision_is_sub_metre():
    """A 10-character cell has to be small enough to separate adjacent flats."""
    u = ULPIN3D(state="27", district="025", tehsil="004",
                geohash=geohash_encode(*PUNE), stratum=Stratum.BUILDING,
                level=5, unit=502)
    dx, dy = u.cell_size_m
    assert dx < 2.0 and dy < 1.0


def test_geohash_rejects_out_of_range():
    with pytest.raises(ValueError):
        geohash_encode(95.0, 73.0)
    with pytest.raises(ValueError):
        geohash_encode(18.0, 200.0)


# --- check character -----------------------------------------------------------
def test_iso7064_rejects_non_base36():
    with pytest.raises(ValueError):
        iso7064_mod37_36("ABC-123")


def _sample_ulpin() -> ULPIN3D:
    return ULPIN3D(state="27", district="025", tehsil="004",
                   geohash=geohash_encode(*PUNE), stratum=Stratum.BUILDING,
                   level=5, unit=502)


def test_check_character_catches_every_single_substitution():
    """
    The whole point of a check character is offline detection of keying errors,
    so this asserts the property exhaustively rather than on a sample.
    """
    good = _sample_ulpin().compact
    slipped = []
    for i, ch in enumerate(good):
        for repl in BASE36:
            if repl == ch:
                continue
            if ULPIN3D.is_valid(good[:i] + repl + good[i + 1:]):
                slipped.append(good[:i] + repl + good[i + 1:])
    assert slipped == []


def test_check_character_catches_adjacent_transpositions():
    good = _sample_ulpin().compact
    slipped = [good[:i] + good[i + 1] + good[i] + good[i + 2:]
               for i in range(len(good) - 1)
               if good[i] != good[i + 1]
               and ULPIN3D.is_valid(good[:i] + good[i + 1] + good[i] + good[i + 2:])]
    assert slipped == []


def test_wrong_check_character_is_rejected_on_construction():
    u = _sample_ulpin()
    wrong = "Z" if u.check != "Z" else "Y"
    with pytest.raises(ValueError, match="check character"):
        ULPIN3D(state=u.state, district=u.district, tehsil=u.tehsil,
                geohash=u.geohash, stratum=u.stratum, level=u.level,
                unit=u.unit, check=wrong)


# --- formatting and parsing ----------------------------------------------------
def test_round_trip_formatted_and_compact():
    u = _sample_ulpin()
    assert ULPIN3D.parse(str(u)) == u
    assert ULPIN3D.parse(u.compact) == u
    assert ULPIN3D.parse(u.compact.lower()) == u
    assert ULPIN3D.parse("  " + str(u).lower() + "  ") == u


def test_compact_length_is_fixed():
    assert len(_sample_ulpin().compact) == ULPIN_COMPACT_LEN


def test_malformed_input_raises():
    for bad in ("", "hello", "27-025-004", "1" * 26):
        with pytest.raises(ValueError):
            ULPIN3D.parse(bad)


def test_stratum_and_level_are_readable_without_a_lookup():
    u = _sample_ulpin()
    assert u.stratum is Stratum.BUILDING
    assert u.signed_level == 5
    assert "level 5 above ground" in u.describe()

    basement = ULPIN3D(state="27", district="025", tehsil="004",
                       geohash=geohash_encode(*PUNE),
                       stratum=Stratum.UNDERGROUND, level=2, unit=901)
    assert basement.signed_level == -2
    assert basement.stratum.is_subsurface


# --- minting -------------------------------------------------------------------
def test_minting_is_unique_within_a_cell(jurisdiction):
    minter = ULPINMinter(jurisdiction)
    issued = {minter.mint(lat=PUNE[0], lon=PUNE[1], stratum=Stratum.BUILDING,
                          level=3, owner=f"unit-{i}").compact
              for i in range(50)}
    assert len(issued) == 50


def test_preferred_unit_number_is_honoured_then_falls_back(jurisdiction):
    minter = ULPINMinter(jurisdiction)
    a = minter.mint(lat=PUNE[0], lon=PUNE[1], stratum=Stratum.BUILDING,
                    level=5, owner="a", preferred_unit=502)
    assert a.unit == 502
    # the same slot is taken, so the next mint must not collide
    b = minter.mint(lat=PUNE[0], lon=PUNE[1], stratum=Stratum.BUILDING,
                    level=5, owner="b", preferred_unit=502)
    assert b.unit != 502
    assert a.compact != b.compact


def test_same_footprint_different_levels_do_not_collide(jurisdiction):
    """The core 3D requirement: identical x/y, different z, distinct identity."""
    minter = ULPINMinter(jurisdiction)
    ids = {minter.mint(lat=PUNE[0], lon=PUNE[1], stratum=Stratum.BUILDING,
                       level=lvl, owner=f"flat-{lvl}", preferred_unit=1).compact
           for lvl in range(12)}
    assert len(ids) == 12


def test_tunnel_under_a_flat_gets_a_distinct_identifier(jurisdiction):
    minter = ULPINMinter(jurisdiction)
    flat = minter.mint(lat=PUNE[0], lon=PUNE[1], stratum=Stratum.BUILDING,
                       level=10, owner="flat")
    tunnel = minter.mint(lat=PUNE[0], lon=PUNE[1],
                         stratum=Stratum.INFRASTRUCTURE, level=4, owner="metro")
    assert flat.compact != tunnel.compact
    assert flat.geohash == tunnel.geohash      # same ground position
    assert tunnel.stratum.is_subsurface


def test_re_registering_to_another_owner_is_refused(jurisdiction):
    minter = ULPINMinter(jurisdiction)
    u = minter.mint(lat=PUNE[0], lon=PUNE[1], stratum=Stratum.SURFACE,
                    level=0, owner="parcel-1")
    with pytest.raises(ValueError, match="collision"):
        minter.register(u, owner="parcel-2")


# --- stability -----------------------------------------------------------------
def test_small_resurvey_drift_keeps_the_identifier_stable():
    u = _sample_ulpin()
    lat, lon = u.centroid
    report = check_stability(u, lat + 1e-7, lon + 1e-7)
    assert report["stable"] is True
    assert report["advisory"] is None


def test_large_drift_raises_an_advisory_without_changing_the_identifier():
    u = _sample_ulpin()
    lat, lon = u.centroid
    report = check_stability(u, lat + 0.001, lon + 0.001)
    assert report["stable"] is False
    assert report["drift_m"] > 10
    assert "remains valid and unchanged" in report["advisory"]


def test_geometry_digest_ignores_float_noise_but_not_real_change():
    ring = [(0, 0), (10, 0), (10, 8), (0, 8)]
    base = geometry_digest(ring, 3.0, 6.0)
    noisy = geometry_digest([(x + 1e-7, y - 1e-7) for x, y in ring], 3.0, 6.0)
    moved = geometry_digest([(x + 0.05, y) for x, y in ring], 3.0, 6.0)
    assert base == noisy
    assert base != moved

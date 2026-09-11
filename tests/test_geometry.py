"""Geodesy, the volumetric kernel, and the raster grid."""
import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from bhoomi3d.core.crs import (UTM, ControlPoint, ecef_to_geodetic,
                               fit_helmert, geodetic_to_ecef, utm_epsg,
                               utm_zone_for)
from bhoomi3d.core.geometry3d import (CompositeSolid, Prism, alpha_shape,
                                      clean_polygon, dominant_orientation,
                                      polygon_from_rings, regularize_footprint,
                                      rings_from_polygon, sweep_corridor)
from bhoomi3d.core.grid import Grid


# --- geodesy -------------------------------------------------------------------
def test_ecef_round_trip_is_sub_millimetre():
    for lat, lon, h in [(18.52, 73.86, 560.0), (0.0, 0.0, 0.0),
                        (-33.87, 151.21, 40.0), (78.2, 15.6, 12.0)]:
        x, y, z = geodetic_to_ecef(lat, lon, h)
        lat2, lon2, h2 = ecef_to_geodetic(x, y, z)
        assert abs(lat2[0] - lat) < 1e-9
        assert abs(lon2[0] - lon) < 1e-9
        assert abs(h2[0] - h) < 1e-3


def test_enu_round_trip(enu):
    for e, n, u in [(0, 0, 0), (150.0, -80.0, 25.0), (-2000.0, 3000.0, -18.0)]:
        lat, lon, h = enu.inverse(e, n, u)
        e2, n2, u2 = enu.forward(lat, lon, h)
        assert abs(e2[0] - e) < 1e-4
        assert abs(n2[0] - n) < 1e-4
        assert abs(u2[0] - u) < 1e-4


def test_enu_origin_maps_to_zero(enu):
    o = enu.origin
    e, n, u = enu.forward(o.lat, o.lon, o.height)
    assert abs(e[0]) < 1e-6 and abs(n[0]) < 1e-6 and abs(u[0]) < 1e-6


def test_enu_distances_are_metric(enu):
    """100 m east in the local frame must be ~100 m of ground distance."""
    lat, lon, h = enu.inverse(100.0, 0.0, 0.0)
    e, n, u = enu.forward(lat, lon, h)
    assert abs(e[0] - 100.0) < 1e-3
    assert abs(n[0]) < 1e-3
    # and the point really is ~100 m away along the ellipsoid
    o = enu.origin
    dlon = math.radians(lon[0] - o.lon)
    ground = dlon * 6378137.0 * math.cos(math.radians(o.lat))
    assert abs(ground - 100.0) < 0.5


def test_enu_accounts_for_curvature_not_a_flat_plane(enu):
    """
    At several kilometres the tangent plane departs from the ellipsoid by
    metres. A point lying *on* the plane 5 km out therefore sits ~d^2/2R above
    the ellipsoid, because the earth curves away beneath it. Asserting that
    offset is what distinguishes a real ENU frame from an equirectangular
    shortcut, which would report a height of exactly zero.
    """
    d = 5000.0
    lat, lon, h = enu.inverse(d, 0.0, 0.0)
    rise = h[0] - enu.origin.height
    expected = d ** 2 / (2 * 6378137.0)      # ~1.96 m
    assert rise == pytest.approx(expected, rel=0.05)


def test_utm_round_trip():
    lat, lon = 18.520430, 73.856744
    zone, hemi = utm_zone_for(lon, lat)
    assert zone == 43 and hemi == "N"
    assert utm_epsg(lon, lat) == 32643
    proj = UTM(zone, hemi)
    e, n = proj.forward(lat, lon)
    lat2, lon2 = proj.inverse(e, n)
    assert abs(lat2[0] - lat) < 1e-8
    assert abs(lon2[0] - lon) < 1e-8


# --- GNSS / CORS adjustment ----------------------------------------------------
def _control_network(shift, yaw_deg, scale_ppm, blunder=None, seed=3):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-100, 100, (8, 3))
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    k = 1 + scale_ppm * 1e-6
    observed = k * (truth @ R.T) + np.asarray(shift)
    pts = []
    for i in range(len(truth)):
        ref = truth[i].copy()
        if blunder is not None and i == blunder:
            ref = ref + np.array([0.0, 0.75, 0.0])
        pts.append(ControlPoint(f"GCP-{i:02d}", tuple(observed[i]), tuple(ref)))
    return pts, truth, observed


def test_helmert_recovers_a_known_transform():
    pts, truth, observed = _control_network([12.0, -8.0, 3.0], 0.4, 45.0)
    fit = fit_helmert(pts)
    assert fit.rms < 1e-6
    assert np.allclose(fit.apply(observed), truth, atol=1e-6)


def test_helmert_rejects_a_blunder():
    """
    A single bad control point must be identified and dropped, not averaged in.
    This is the difference between a usable adjustment and a block pulled off
    its datum by one transposed field measurement.
    """
    pts, truth, observed = _control_network([12.0, -8.0, 3.0], 0.4, 45.0,
                                            blunder=5)
    fit = fit_helmert(pts)
    assert "GCP-05" in fit.rejected
    assert fit.n_points == 7
    assert fit.rms < 1e-3


def test_helmert_needs_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        fit_helmert([ControlPoint("a", (0, 0, 0), (1, 1, 1)),
                     ControlPoint("b", (1, 0, 0), (2, 1, 1))])


# --- prisms --------------------------------------------------------------------
def test_prism_measures():
    p = Prism.from_box(0, 0, 10, 20, 15, 18)
    assert p.area == pytest.approx(200.0)
    assert p.height == pytest.approx(3.0)
    assert p.volume == pytest.approx(600.0)
    cx, cy, cz = p.centroid
    assert (cx, cy, cz) == pytest.approx((5.0, 10.0, 16.5))


def test_prism_normalises_inverted_z():
    p = Prism.from_box(0, 0, 4, 4, 9.0, 2.0)
    assert p.z_min == 2.0 and p.z_max == 9.0


def test_stacked_prisms_do_not_overlap():
    """Two flats on different floors share a footprint. That is not a defect."""
    lower = Prism.from_box(0, 0, 10, 20, 12, 15)
    upper = Prism.from_box(0, 0, 10, 20, 15, 18)
    assert lower.intersection_volume(upper) == 0.0
    assert not lower.intersects(upper)


def test_interpenetrating_prisms_report_exact_volume():
    a = Prism.from_box(0, 0, 10, 20, 15, 18)
    b = Prism.from_box(8, 0, 18, 20, 16, 18)
    # shared plan area 2 x 20, shared z-range 2 m
    assert a.intersection_volume(b) == pytest.approx(80.0)
    assert a.intersects(b)


def test_clearance_combines_horizontal_and_vertical_separation():
    a = Prism.from_box(0, 0, 10, 10, 0, 5)
    # directly below, 4 m of vertical gap
    below = Prism.from_box(0, 0, 10, 10, -10, -4)
    assert a.clearance(below) == pytest.approx(4.0)
    # offset in plan only
    beside = Prism.from_box(13, 0, 20, 10, 0, 5)
    assert a.clearance(beside) == pytest.approx(3.0)
    # both, so the distance is the hypotenuse
    diag = Prism.from_box(13, 0, 20, 10, -10, -4)
    assert a.clearance(diag) == pytest.approx(5.0)


def test_contains_point():
    p = Prism.from_box(0, 0, 10, 10, 3, 6)
    assert p.contains_point(5, 5, 4.5)
    assert not p.contains_point(5, 5, 7.0)
    assert not p.contains_point(15, 5, 4.5)


def test_prism_serialises_round_trip():
    p = Prism.from_box(1, 2, 11, 22, 3, 9, use="flat")
    q = Prism.from_dict(p.as_dict())
    assert q.volume == pytest.approx(p.volume)
    assert q.z_min == pytest.approx(p.z_min)


def test_polygon_with_hole_excludes_the_courtyard():
    outer = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
    poly = polygon_from_rings([outer, hole])
    p = Prism(poly, 0, 3)
    assert p.area == pytest.approx(400 - 16)
    assert len(rings_from_polygon(p.footprint)) == 2


# --- composite solids ----------------------------------------------------------
def test_sweep_corridor_descends_and_measures():
    solid = sweep_corridor([(0, 0), (100, 0), (200, 0)], width=6.0, height=6.0,
                           invert_levels=[-10.0, -12.0, -14.0], kind="metro")
    assert isinstance(solid, CompositeSolid)
    assert len(solid.parts) == 2
    assert solid.z_min == pytest.approx(-14.0)
    assert solid.volume > 0
    cx, cy, cz = solid.centroid
    assert -14.0 < cz < 0.0


def test_corridor_clearance_to_a_basement():
    tunnel = sweep_corridor([(0, 0), (100, 0)], width=6.0, height=6.0,
                            invert_levels=[-20.0, -20.0])
    basement = Prism.from_box(40, -5, 60, 5, -6.0, 0.0)
    # tunnel crown at -14, basement raft at -6, so 8 m of cover
    assert basement.clearance(tunnel) == pytest.approx(8.0, abs=1e-6)


def test_corridor_alignment_must_have_matching_levels():
    with pytest.raises(ValueError):
        sweep_corridor([(0, 0), (10, 0)], width=2, height=2,
                       invert_levels=[-3.0])


# --- footprint processing ------------------------------------------------------
def test_clean_polygon_repairs_a_bowtie():
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid
    fixed = clean_polygon(bowtie)
    assert fixed is not None and fixed.is_valid


def test_dominant_orientation_finds_the_wall_direction():
    theta = math.radians(31.0)
    base = np.array([[0, 0], [24, 0], [24, 12], [0, 12]], dtype=float)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    poly = Polygon(base @ R.T)
    assert math.degrees(dominant_orientation(poly)) == pytest.approx(31.0, abs=1.0)


def test_regularisation_squares_a_noisy_outline():
    rng = np.random.default_rng(5)
    base = np.array([[0, 0], [24, 0], [24, 12], [14, 12], [14, 20], [0, 20]],
                    dtype=float)
    noisy = Polygon(base + rng.normal(0, 0.15, base.shape))
    reg = regularize_footprint(noisy)
    assert reg.is_valid
    assert reg.area == pytest.approx(Polygon(base).area, rel=0.06)


def test_regularisation_keeps_the_traced_shape_when_it_would_distort_it():
    """A genuinely round footprint must not be forced into a rectangle."""
    circle = Polygon([(20 * math.cos(t), 20 * math.sin(t))
                      for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)])
    reg = regularize_footprint(circle)
    assert reg.area == pytest.approx(circle.area, rel=0.25)


def test_alpha_shape_traces_a_concavity_a_convex_hull_would_miss():
    base = Polygon([(0, 0), (24, 0), (24, 12), (14, 12), (14, 20), (0, 20)])
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 2500:
        q = rng.uniform([0, 0], [24, 20], 2)
        if base.contains(Polygon([(q[0], q[1]), (q[0] + .01, q[1]),
                                  (q[0], q[1] + .01)]).centroid):
            pts.append(q)
    shape = alpha_shape(np.array(pts), alpha=3.0)
    assert shape.area == pytest.approx(base.area, rel=0.05)
    assert shape.area < Polygon(np.array(pts)).convex_hull.area * 0.97


# --- grids ---------------------------------------------------------------------
def test_grid_from_points_min_and_max():
    pts = np.array([[0.1, 0.1, 5.0], [0.2, 0.2, 9.0], [3.1, 0.1, 2.0]])
    gmin = Grid.from_points(pts, cell=1.0, stat="min")
    gmax = Grid.from_points(pts, cell=1.0, stat="max")
    assert np.nanmin(gmin.data) == pytest.approx(2.0)
    assert np.nanmax(gmax.data) == pytest.approx(9.0)


def test_grid_bilinear_sampling_interpolates():
    g = Grid(np.array([[0.0, 10.0], [0.0, 10.0]]), x0=0.0, y0=0.0, cell=1.0)
    assert g.sample(0.5, 0.0)[0] == pytest.approx(5.0)
    assert g.sample(0.0, 0.0)[0] == pytest.approx(0.0)
    assert g.sample(1.0, 0.0)[0] == pytest.approx(10.0)


def test_grid_fills_voids():
    data = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
    g = Grid(data, 0, 0, 1.0)
    assert g.nodata_mask().sum() == 1
    assert not np.isnan(g.filled().data).any()


def test_grid_subtraction_requires_matching_lattice():
    a = Grid(np.zeros((4, 4)), 0, 0, 1.0)
    b = Grid(np.zeros((3, 3)), 0, 0, 1.0)
    with pytest.raises(ValueError):
        a - b

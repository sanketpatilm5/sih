"""
Coordinate reference systems for the 3D cadastre.

Every geometry in the platform is stored in a project-local **ENU** frame
(East / North / Up, metres) anchored on a geodetic origin. That gives us:

  * metric geometry - areas, volumes, buffers and clearances are all in m/m2/m3
  * an exact, reversible mapping back to WGS-84 lat/lon/ellipsoidal height
  * a single place to apply the GNSS/CORS datum adjustment

Nothing here depends on PROJ/pyproj - the ellipsoidal maths is implemented
directly so the platform runs on a bare Python install.

References
----------
* WGS-84 defining parameters: NIMA TR8350.2
* Bowring's method for ECEF -> geodetic (sub-mm for terrestrial heights)
* Kruger series for Transverse Mercator / UTM
* Umeyama (1991) closed-form least-squares similarity transform
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --- WGS-84 defining constants ------------------------------------------------
WGS84_A = 6378137.0                      # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563            # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)      # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # first eccentricity squared
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)  # second eccentricity squared


def _arr(*values):
    return tuple(np.atleast_1d(np.asarray(v, dtype=float)) for v in values)


def geodetic_to_ecef(lat_deg, lon_deg, h=0.0):
    """Geodetic (deg, deg, m) -> earth-centred earth-fixed XYZ (m). Vectorised."""
    lat, lon, hh = _arr(lat_deg, lon_deg, h)
    if hh.size == 1 and lat.size > 1:
        hh = np.full_like(lat, float(hh[0]))
    slat, clat = np.sin(np.radians(lat)), np.cos(np.radians(lat))
    slon, clon = np.sin(np.radians(lon)), np.cos(np.radians(lon))
    # radius of curvature in the prime vertical
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * slat * slat)
    x = (n + hh) * clat * clon
    y = (n + hh) * clat * slon
    z = (n * (1.0 - WGS84_E2) + hh) * slat
    return x, y, z


def ecef_to_geodetic(x, y, z):
    """ECEF XYZ (m) -> geodetic (deg, deg, m) using Bowring's method. Vectorised."""
    x, y, z = _arr(x, y, z)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # Bowring's parametric-latitude seed makes one pass enough for terrestrial h
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    st, ct = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * st ** 3,
                     p - WGS84_E2 * WGS84_A * ct ** 3)
    slat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * slat * slat)
    # near the poles p -> 0, so switch formulation to stay well conditioned
    near_pole = p < 1.0
    h = np.where(near_pole,
                 np.abs(z) - WGS84_B,
                 p / np.maximum(np.cos(lat), 1e-15) - n)
    return np.degrees(lat), np.degrees(lon), h


@dataclass(frozen=True)
class GeodeticOrigin:
    """Anchor point of a project local ENU frame."""

    lat: float
    lon: float
    height: float = 0.0
    label: str = "project origin"

    def as_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "height": self.height,
                "label": self.label}


class LocalENU:
    """
    Local tangent-plane (East, North, Up) frame anchored at a geodetic origin.

    We do the full ECEF round trip rather than an equirectangular approximation:
    at 5 km from the origin the earth's curvature already accounts for ~2 m of
    height, which matters when the whole point of the system is the Z axis.
    Positions round-trip to sub-millimetre across a whole city.
    """

    def __init__(self, origin: GeodeticOrigin):
        self.origin = origin
        lat0, lon0 = math.radians(origin.lat), math.radians(origin.lon)
        x0, y0, z0 = geodetic_to_ecef(origin.lat, origin.lon, origin.height)
        self._ecef0 = np.array([x0[0], y0[0], z0[0]])
        sla, cla = math.sin(lat0), math.cos(lat0)
        slo, clo = math.sin(lon0), math.cos(lon0)
        # rows = east, north, up unit vectors expressed in ECEF
        self._R = np.array([
            [-slo, clo, 0.0],
            [-sla * clo, -sla * slo, cla],
            [cla * clo, cla * slo, sla],
        ])

    # -- conversions ---------------------------------------------------------
    def forward(self, lat, lon, h=0.0):
        """Geodetic -> local ENU metres."""
        x, y, z = geodetic_to_ecef(lat, lon, h)
        d = np.stack([x, y, z], axis=-1) - self._ecef0
        enu = d @ self._R.T
        return enu[..., 0], enu[..., 1], enu[..., 2]

    def inverse(self, e, n, u=0.0):
        """Local ENU metres -> geodetic."""
        e, n, u = _arr(e, n, u)
        if u.size == 1 and e.size > 1:
            u = np.full_like(e, float(u[0]))
        enu = np.stack([e, n, u], axis=-1)
        ecef = enu @ self._R + self._ecef0
        return ecef_to_geodetic(ecef[..., 0], ecef[..., 1], ecef[..., 2])

    # -- convenience for single points and rings -----------------------------
    def forward_xy(self, lon: float, lat: float) -> tuple[float, float]:
        e, n, _ = self.forward(lat, lon, 0.0)
        return float(e[0]), float(n[0])

    def inverse_xy(self, e: float, n: float) -> tuple[float, float]:
        lat, lon, _ = self.inverse(e, n, 0.0)
        return float(lon[0]), float(lat[0])

    def forward_ring(self, ring_lonlat: Sequence[Sequence[float]]) -> list[list[float]]:
        arr = np.asarray(ring_lonlat, dtype=float)
        e, n, _ = self.forward(arr[:, 1], arr[:, 0], 0.0)
        return np.stack([e, n], axis=-1).tolist()

    def inverse_ring(self, ring_en: Sequence[Sequence[float]]) -> list[list[float]]:
        arr = np.asarray(ring_en, dtype=float)
        lat, lon, _ = self.inverse(arr[:, 0], arr[:, 1], 0.0)
        return np.stack([lon, lat], axis=-1).tolist()

    def as_dict(self) -> dict:
        return {"type": "LocalENU", "origin": self.origin.as_dict(),
                "units": "metre", "datum": "WGS84"}


# --- UTM ----------------------------------------------------------------------
def utm_zone_for(lon_deg: float, lat_deg: float) -> tuple[int, str]:
    """Return (zone number, hemisphere) for a longitude/latitude."""
    zone = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
    return zone, "N" if lat_deg >= 0 else "S"


def utm_epsg(lon_deg: float, lat_deg: float) -> int:
    """EPSG code of the WGS-84 / UTM zone containing the point."""
    zone, hemi = utm_zone_for(lon_deg, lat_deg)
    return (32600 if hemi == "N" else 32700) + zone


class UTM:
    """
    Transverse-Mercator (UTM) projection, Kruger series to third order.

    State land-records departments hand over data in a projected CRS, so we keep
    a real UTM implementation available for import/export round-tripping rather
    than forcing everything through the local ENU frame.
    """

    K0 = 0.9996
    FALSE_EASTING = 500000.0
    FALSE_NORTHING = 10000000.0  # southern hemisphere only

    def __init__(self, zone: int, hemisphere: str = "N"):
        self.zone = int(zone)
        self.hemisphere = hemisphere.upper()
        self.lon0 = math.radians((self.zone - 1) * 6 - 180 + 3)
        n = WGS84_F / (2 - WGS84_F)
        self._n = n
        self._A = WGS84_A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)  # rectifying radius
        self._alpha = [
            n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16,
            13 * n ** 2 / 48 - 3 * n ** 3 / 5,
            61 * n ** 3 / 240,
        ]
        self._beta = [
            n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96,
            n ** 2 / 48 + n ** 3 / 15,
            17 * n ** 3 / 480,
        ]

    @property
    def epsg(self) -> int:
        return (32600 if self.hemisphere == "N" else 32700) + self.zone

    def forward(self, lat_deg, lon_deg):
        lat, lon = _arr(lat_deg, lon_deg)
        phi, lam = np.radians(lat), np.radians(lon) - self.lon0
        e = math.sqrt(WGS84_E2)
        t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
        xi = np.arctan2(t, np.cos(lam))
        eta = np.arctanh(np.sin(lam) / np.hypot(1, t))
        x, y = eta.copy(), xi.copy()
        for j, a in enumerate(self._alpha, start=1):
            x = x + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
            y = y + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        easting = self.K0 * self._A * x + self.FALSE_EASTING
        northing = self.K0 * self._A * y
        if self.hemisphere == "S":
            northing = northing + self.FALSE_NORTHING
        return easting, northing

    def inverse(self, easting, northing):
        e_, n_ = _arr(easting, northing)
        y = n_ - (self.FALSE_NORTHING if self.hemisphere == "S" else 0.0)
        xi = y / (self.K0 * self._A)
        eta = (e_ - self.FALSE_EASTING) / (self.K0 * self._A)
        xi_p, eta_p = xi.copy(), eta.copy()
        for j, b in enumerate(self._beta, start=1):
            xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
            eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
        # invert the conformal latitude by fixed-point iteration
        phi = chi.copy()
        e = math.sqrt(WGS84_E2)
        for _ in range(6):
            phi = np.arcsin(np.tanh(np.arctanh(np.sin(chi))
                                    + e * np.arctanh(e * np.sin(phi))))
        lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p)) + self.lon0
        return np.degrees(phi), np.degrees(lam)


# --- GNSS / CORS datum adjustment ---------------------------------------------
@dataclass
class ControlPoint:
    """A GNSS/CORS-observed control point paired with its position in the survey."""

    name: str
    observed: tuple[float, float, float]   # ENU from the raw survey / drone block
    reference: tuple[float, float, float]  # CORS-derived truth, same frame
    sigma: float = 0.02                    # 1-sigma of the CORS fix [m]


@dataclass
class HelmertResult:
    """Outcome of fitting a similarity transform to the control network."""

    translation: np.ndarray                # (3,)
    rotation: np.ndarray                   # (3,3)
    scale: float
    residuals: np.ndarray                  # (n,3), reference - transformed
    rms: float
    max_residual: float
    n_points: int
    rejected: list[str] = field(default_factory=list)

    def apply(self, pts) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return self.scale * (pts @ self.rotation.T) + self.translation

    def as_dict(self) -> dict:
        return {
            "translation_m": np.round(self.translation, 4).tolist(),
            "rotation": np.round(self.rotation, 9).tolist(),
            "scale": round(self.scale, 9),
            "scale_ppm": round((self.scale - 1.0) * 1e6, 3),
            "rms_m": round(self.rms, 4),
            "max_residual_m": round(self.max_residual, 4),
            "n_points": self.n_points,
            "rejected": self.rejected,
        }


def fit_helmert(points: Iterable[ControlPoint], *, with_scale: bool = True,
                reject_sigma: float = 4.0, max_reject_fraction: float = 0.35,
                max_iter: int = 8) -> HelmertResult:
    """
    Least-squares 7-parameter (Helmert) similarity fit with outlier rejection.

    This is what ties a photogrammetric / LiDAR block to the national reference
    frame. A drone block is internally consistent but floats in an arbitrary
    datum until CORS-observed ground control constrains it, and blunders in
    ground control are common in practice.

    The fit itself is the Umeyama/Kabsch closed form - the exact least-squares
    minimiser - so no iteration is needed for a given point set. The iteration
    is purely for outlier rejection, and it does two things that matter:

    **It tests against the a-priori precision, not the observed spread.** A GNSS
    control point is good to a couple of centimetres and we know that in
    advance. Judging residuals against their own scatter is circular: a single
    blunder inflates that scatter enough to hide itself. This is Baarda's data
    snooping - the standard geodetic treatment.

    **It rejects one point per pass, not all suspects at once.** Least squares
    smears a single blunder across every residual in the network, so after one
    bad point is removed the rest usually collapse to noise. Rejecting the whole
    suspect set in one pass would throw away good control.

    `max_reject_fraction` caps the damage if the a-priori precision was
    optimistic: rejection stops rather than eating a whole network.
    """
    pts = list(points)
    if len(pts) < 3:
        raise ValueError(
            "a Helmert fit needs at least 3 control points, got %d" % len(pts))

    names = [p.name for p in pts]
    src = np.array([p.observed for p in pts], dtype=float)
    dst = np.array([p.reference for p in pts], dtype=float)
    sigmas = np.array([max(p.sigma, 1e-3) for p in pts], dtype=float)
    active = np.ones(len(pts), dtype=bool)
    rejected: list[str] = []
    min_active = max(3, int(np.ceil(len(pts) * (1.0 - max_reject_fraction))))
    R = np.eye(3)
    scale = 1.0
    t = np.zeros(3)

    for _ in range(max_iter):
        s, d = src[active], dst[active]
        if len(s) < 3:
            raise ValueError("too many control points rejected; fewer than 3 remain")
        mu_s, mu_d = s.mean(axis=0), d.mean(axis=0)
        sc, dc = s - mu_s, d - mu_d
        cov = dc.T @ sc / len(s)
        U, S, Vt = np.linalg.svd(cov)
        # guard against fitting a reflection instead of a rotation
        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[2, 2] = -1.0
        R = U @ D @ Vt
        var_s = (sc ** 2).sum() / len(s)
        scale = float((S * np.diag(D)).sum() / var_s) if (with_scale and var_s > 0) else 1.0
        t = mu_d - scale * (R @ mu_s)

        if active.sum() <= min_active:
            break
        norms = np.linalg.norm(dst - (scale * (src @ R.T) + t), axis=1)
        # normalised residual: how many times its own stated precision each
        # control point misses by
        w = np.where(active, norms / sigmas, 0.0)
        worst = int(np.argmax(w))
        if w[worst] <= reject_sigma:
            break
        rejected.append(names[worst])
        active[worst] = False

    resid = dst - (scale * (src @ R.T) + t)
    rms = float(np.sqrt((resid[active] ** 2).sum() / active.sum()))
    return HelmertResult(
        translation=t, rotation=R, scale=scale, residuals=resid, rms=rms,
        max_residual=float(np.linalg.norm(resid[active], axis=1).max()),
        n_points=int(active.sum()), rejected=rejected,
    )

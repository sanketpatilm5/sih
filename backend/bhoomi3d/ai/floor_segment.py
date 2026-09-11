"""
Storey segmentation: finding the floors inside a structure.

Once a building has been extracted we know its outline and its height. The
cadastre needs more than that - it needs to know where each *storey* begins and
ends, because a flat's legal volume is bounded above and below by slab levels.

Doing this from an exterior scan alone looks impossible until you notice that
building facades are strongly **periodic** in the vertical: balcony slabs,
window bands, string courses and floor-level service runs all repeat at exactly
the storey pitch. So the facade's height histogram carries a periodic signal
whose fundamental period *is* the floor-to-floor height.

The method here is therefore signal processing rather than geometry:

  1. isolate facade returns (exclude roof and ground)
  2. build a fine height histogram and remove its slow trend, leaving the
     periodic component
  3. recover the period by autocorrelation, restricted to physically plausible
     storey heights
  4. recover the phase by matched-filtering a comb against the signal
  5. snap each individual slab to its nearest local peak, so a mezzanine or an
     unusually tall ground floor does not force the whole stack out of step

Where an approved floor plan exists it is authoritative and simply overrides
this estimate - the plan is the legal document. The estimate still runs, because
disagreement between the plan and the as-built scan is itself a finding worth
reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Storey:
    """One storey of a building, in absolute local-ENU metres."""

    index: int              # 0 = ground floor, negative = basement
    z_bottom: float
    z_top: float
    source: str = "estimated"   # "estimated" | "floor_plan" | "assumed"
    confidence: float = 0.0

    @property
    def height(self) -> float:
        return self.z_top - self.z_bottom

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "z_bottom": round(self.z_bottom, 3),
            "z_top": round(self.z_top, 3),
            "height_m": round(self.height, 3),
            "source": self.source,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class StoreyResult:
    storeys: list[Storey]
    floor_height: float
    periodicity_strength: float     # 0-1; how clearly periodic the facade is
    method: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_floors(self) -> int:
        return sum(1 for s in self.storeys if s.index >= 0)

    def as_dict(self) -> dict:
        return {
            "n_storeys": len(self.storeys),
            "n_above_ground": self.n_floors,
            "floor_height_m": round(self.floor_height, 3),
            "periodicity_strength": round(self.periodicity_strength, 3),
            "method": self.method,
            "storeys": [s.as_dict() for s in self.storeys],
            "diagnostics": self.diagnostics,
        }


def select_facade_points(all_xyz: np.ndarray, footprint, z_lo: float,
                         z_hi: float, *, outward: float = 1.8,
                         inward: float = 2.5) -> np.ndarray:
    """
    Select the points forming a building's facade shell.

    Two things matter here, and both are easy to get wrong.

    *Use the raw cloud, not the semantically filtered one.* The filter that
    removes vegetation keys on planarity, and balconies - small, isolated,
    geometrically scattered protrusions - score much like foliage. Filtering
    first therefore destroys precisely the periodic signal this stage depends
    on, which is a subtle way to end up with a building whose storeys are all
    invented.

    *Take a ring, not the whole footprint.* Restricting to a band around the
    wall line excludes the roof, whose tens of thousands of coplanar returns
    would otherwise swamp the histogram, and captures balconies that overhang
    the wall.
    """
    import shapely

    ring = footprint.buffer(outward).difference(footprint.buffer(-inward))
    if ring.is_empty:                     # very slender building - use it all
        ring = footprint.buffer(outward)
    z = all_xyz[:, 2]
    band = (z >= z_lo) & (z <= z_hi)
    if not band.any():
        return np.empty((0, 3))
    cand = all_xyz[band]
    inside = shapely.contains_xy(ring, cand[:, 0], cand[:, 1])
    return cand[inside]


def _facade_histogram(points: np.ndarray, z_lo: float, z_hi: float,
                      bin_h: float) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    sel = (z >= z_lo) & (z <= z_hi)
    if sel.sum() < 40:
        return np.empty(0), np.empty(0)
    nbins = max(8, int((z_hi - z_lo) / bin_h))
    counts, edges = np.histogram(z[sel], bins=nbins, range=(z_lo, z_hi))
    centres = (edges[:-1] + edges[1:]) / 2
    return counts.astype(float), centres


def _detrend(signal: np.ndarray, window: int) -> np.ndarray:
    """Remove the slow envelope, keeping the periodic component."""
    if window < 3 or window >= len(signal):
        return signal - signal.mean()
    kernel = np.ones(window) / window
    trend = np.convolve(signal, kernel, mode="same")
    return signal - trend


def estimate_floor_height(points: np.ndarray, z_lo: float, z_hi: float, *,
                          bin_h: float = 0.10,
                          min_floor_h: float = 2.4,
                          max_floor_h: float = 5.2) -> tuple[float, float, dict]:
    """
    Recover the floor-to-floor pitch from the facade's vertical periodicity.

    Returns `(floor_height, strength, diagnostics)`. `strength` is the
    normalised autocorrelation at the winning lag - effectively how confident
    we are that the facade really is periodic. A blank curtain wall has no
    periodic signal, and this correctly reports a low strength rather than
    inventing floors.
    """
    counts, centres = _facade_histogram(points, z_lo, z_hi, bin_h)
    diag: dict = {"bins": int(len(counts)), "bin_h_m": bin_h}
    if len(counts) < 24:
        return 0.0, 0.0, {**diag, "reason": "too few facade returns"}

    # Smooth at a scale well below a storey, then remove the slow envelope so
    # occlusion - which thins the lower floors badly - does not dominate the
    # transform. The detrending window spans about two storeys: narrower and
    # the moving average starts tracking the very periodicity we are trying to
    # measure, cancelling the signal.
    from scipy.ndimage import uniform_filter1d
    sig = uniform_filter1d(counts, size=3)
    sig = _detrend(sig, window=int(round(2.0 * max_floor_h / bin_h)))

    # unbiased autocorrelation
    sig = sig - sig.mean()
    denom = float(np.dot(sig, sig))
    if denom <= 1e-9:
        return 0.0, 0.0, {**diag, "reason": "flat facade signal"}
    full = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    acf = full / denom

    lag_lo = max(2, int(round(min_floor_h / bin_h)))
    lag_hi = min(len(acf) - 1, int(round(max_floor_h / bin_h)))
    if lag_hi <= lag_lo:
        return 0.0, 0.0, {**diag, "reason": "building too short to be periodic"}

    window = acf[lag_lo:lag_hi + 1]
    best = int(np.argmax(window)) + lag_lo
    strength = float(np.clip(acf[best], 0.0, 1.0))

    # parabolic interpolation around the peak gives sub-bin precision, which
    # matters because the error compounds over a dozen storeys
    if 0 < best < len(acf) - 1:
        y0, y1, y2 = acf[best - 1], acf[best], acf[best + 1]
        denom2 = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom2 if abs(denom2) > 1e-12 else 0.0
        shift = float(np.clip(shift, -1.0, 1.0))
    else:
        shift = 0.0

    floor_h = (best + shift) * bin_h
    diag.update({
        "acf_peak_lag_bins": best,
        "sub_bin_shift": round(shift, 3),
        "search_range_m": [min_floor_h, max_floor_h],
    })
    return floor_h, strength, diag


def _best_phase(points: np.ndarray, z_lo: float, z_hi: float,
                floor_h: float, bin_h: float = 0.10) -> float:
    """
    Find where the storey stack starts, by matched-filtering a comb.

    The period alone does not place the slabs - a half-storey error puts every
    boundary in the middle of a flat. The comb is correlated at every candidate
    offset and the best alignment wins.
    """
    counts, centres = _facade_histogram(points, z_lo, z_hi, bin_h)
    if len(counts) < 8:
        return 0.0
    from scipy.ndimage import uniform_filter1d
    sig = uniform_filter1d(counts, size=3)
    sig = sig - sig.mean()

    n_offsets = max(4, int(round(floor_h / bin_h)))
    best_off, best_score = 0.0, -np.inf
    for k in range(n_offsets):
        off = k * bin_h
        comb_z = np.arange(z_lo + off, z_hi, floor_h)
        if len(comb_z) < 1:
            continue
        idx = np.clip(((comb_z - z_lo) / bin_h).astype(int), 0, len(sig) - 1)
        score = float(sig[idx].mean())
        if score > best_score:
            best_score, best_off = score, off
    return best_off


def segment_storeys(points: np.ndarray, ground_z: float, eave_z: float, *,
                    basements: int = 0, basement_height: float = 3.0,
                    bin_h: float = 0.10,
                    min_floor_h: float = 2.4,
                    max_floor_h: float = 5.2,
                    snap_window: float = 0.45,
                    default_floor_h: float = 3.0) -> StoreyResult:
    """
    Segment a building's point cloud into storeys.

    `basements` is taken from the municipal record rather than the scan: an
    aerial survey cannot see below ground, and pretending otherwise would be
    fabricating a legal boundary. Basement storeys are laid out downward at
    `basement_height` and flagged with source "assumed" so the provenance is
    explicit in the output.
    """
    z_lo = ground_z + 0.4        # skip the plinth and ground clutter
    z_hi = eave_z - 0.2          # stop below the roof slab
    height = max(eave_z - ground_z, 0.0)

    floor_h, strength, diag = estimate_floor_height(
        points, z_lo, z_hi, bin_h=bin_h,
        min_floor_h=min_floor_h, max_floor_h=max_floor_h)

    method = "autocorrelation"
    if floor_h <= 0 or strength < 0.12 or height < min_floor_h:
        # No usable periodic signal. Rather than invent storeys, divide the
        # measured height by a typical storey height and say so.
        floor_h = default_floor_h
        method = "uniform_division"
        strength = 0.0

    n_floors = max(1, int(round(height / floor_h)))
    # reconcile: the storeys must exactly fill the measured height, so adjust
    # the pitch rather than leaving a sliver at the top
    fitted_h = height / n_floors
    if not (min_floor_h * 0.8 <= fitted_h <= max_floor_h * 1.2):
        fitted_h = floor_h
        n_floors = max(1, int(round(height / fitted_h)))

    phase = _best_phase(points, z_lo, z_hi, fitted_h, bin_h) if method == \
        "autocorrelation" else 0.0

    # candidate slab levels, then snap each to the nearest strong local peak
    counts, centres = _facade_histogram(points, z_lo, z_hi, bin_h)
    boundaries = [ground_z + i * fitted_h for i in range(n_floors + 1)]
    if method == "autocorrelation" and len(counts) > 8:
        from scipy.ndimage import uniform_filter1d
        sig = uniform_filter1d(counts, size=3)
        snapped = []
        for i, b in enumerate(boundaries):
            if i == 0:
                snapped.append(ground_z)          # the ground floor slab is fixed
                continue
            if i == len(boundaries) - 1:
                snapped.append(eave_z)            # so is the top ceiling
                continue
            near = np.flatnonzero(np.abs(centres - (b + phase)) <= snap_window)
            if len(near):
                snapped.append(float(centres[near[np.argmax(sig[near])]]))
            else:
                snapped.append(b)
        # keep the stack monotonic - a bad snap must not invert two storeys
        for i in range(1, len(snapped)):
            if snapped[i] <= snapped[i - 1] + 0.5:
                snapped[i] = snapped[i - 1] + fitted_h
        boundaries = snapped

    storeys: list[Storey] = []
    for b in range(basements, 0, -1):
        top = ground_z - (b - 1) * basement_height
        storeys.append(Storey(index=-b, z_bottom=top - basement_height,
                              z_top=top, source="assumed", confidence=0.3))
    for i in range(n_floors):
        storeys.append(Storey(
            index=i, z_bottom=boundaries[i], z_top=boundaries[i + 1],
            source="estimated",
            confidence=float(np.clip(strength * 1.4, 0.0, 1.0)),
        ))

    diag.update({
        "measured_height_m": round(height, 3),
        "raw_period_m": round(floor_h, 3),
        "fitted_floor_height_m": round(fitted_h, 3),
        "phase_offset_m": round(phase, 3),
    })
    return StoreyResult(storeys=storeys, floor_height=fitted_h,
                        periodicity_strength=strength, method=method,
                        diagnostics=diag)


def storeys_from_floor_plan(plan: dict, ground_level_z: float) -> StoreyResult:
    """
    Build the storey stack from an approved floor plan.

    The plan is the legal instrument, so where one exists its levels are used
    verbatim; the scan-derived estimate is retained only for comparison.
    """
    storeys = []
    for f in plan.get("floors", []):
        storeys.append(Storey(
            index=int(f["index"]),
            z_bottom=ground_level_z + float(f["z_bottom_local"]),
            z_top=ground_level_z + float(f["z_top_local"]),
            source="floor_plan", confidence=1.0,
        ))
    storeys.sort(key=lambda s: s.index)
    above = [s for s in storeys if s.index >= 0]
    fh = float(np.mean([s.height for s in above])) if above else 0.0
    return StoreyResult(storeys=storeys, floor_height=fh,
                        periodicity_strength=1.0, method="floor_plan",
                        diagnostics={"source": "approved plan"})


def compare_storeys(estimated: StoreyResult, authoritative: StoreyResult) -> dict:
    """
    Compare a scan-derived stack with the approved plan.

    A mismatch is a genuine finding, not a nuisance: extra storeys in the scan
    are the signature of unauthorised vertical construction, and a systematic
    level offset usually means the plan's datum was never tied to the ground.
    """
    est = [s for s in estimated.storeys if s.index >= 0]
    auth = [s for s in authoritative.storeys if s.index >= 0]
    n_est, n_auth = len(est), len(auth)

    pairs = min(n_est, n_auth)
    slab_err = [abs(est[i].z_bottom - auth[i].z_bottom) for i in range(pairs)]
    return {
        "storeys_estimated": n_est,
        "storeys_on_plan": n_auth,
        "count_matches": n_est == n_auth,
        "extra_storeys_detected": max(0, n_est - n_auth),
        "floor_height_estimated_m": round(estimated.floor_height, 3),
        "floor_height_on_plan_m": round(authoritative.floor_height, 3),
        "mean_slab_error_m": round(float(np.mean(slab_err)), 3) if slab_err else None,
        "max_slab_error_m": round(float(np.max(slab_err)), 3) if slab_err else None,
        "periodicity_strength": round(estimated.periodicity_strength, 3),
    }

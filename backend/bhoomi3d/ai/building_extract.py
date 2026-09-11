"""
Automated building extraction from a classified point cloud.

Given a cloud and the terrain products from the ground filter, this stage
answers: *how many buildings are there, where exactly is each one, and how tall
is it?* - without anyone drawing a single polygon.

Pipeline
--------
1. **Semantic scoring.** Height above ground gates the search; local eigenvalue
   features (planarity vs sphericity) separate roofs from tree canopy, which is
   the failure mode that ruins naive height-threshold extraction.
2. **Instance segmentation.** DBSCAN over the surviving points splits the
   "building" class into individual structures. Clustering runs on (x, y, z)
   with z down-weighted, so two towers that nearly touch at ground level but are
   distinct above it still separate, while a single building is not split
   across its own height.
3. **Footprint tracing.** An alpha shape gives the concave outline - a convex
   hull would bridge courtyards and swallow the notch out of an L-shaped block.
4. **Regularisation.** The traced outline is squared to its dominant wall
   direction, because real buildings are rectilinear and a ragged traced edge is
   both wrong and unusable as a legal boundary.
5. **Height estimation.** Robust percentiles of roof height, plus explicit
   detection of the parapet, so the top storey's ceiling is not confused with
   the top of the parapet wall.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from ..core.geometry3d import (alpha_shape, clean_polygon, regularize_footprint,
                               rings_from_polygon)
from ..core.grid import Grid
from .cluster import cluster_sizes, dbscan, ransac_plane
from .pointfeatures import building_score, compute_features


@dataclass
class BuildingCandidate:
    """One extracted structure, before it is reconciled with the legal record."""

    index: int
    footprint: Polygon              # regularised, local ENU metres
    raw_footprint: Polygon          # the traced alpha shape, before squaring
    ground_z: float                 # terrain level under the structure
    roof_z: float                   # top of the structural roof (parapet excluded)
    eave_z: float                   # top of the walls - the habitable ceiling
    parapet_z: float                # highest built point, parapet included
    n_points: int
    point_indices: np.ndarray
    roof_planes: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    metrics: dict = field(default_factory=dict)

    @property
    def height(self) -> float:
        """Habitable height: ground to the top of the walls, excluding the roof
        structure above the eave and any parapet."""
        return max(self.eave_z - self.ground_z, 0.0)

    @property
    def structural_height(self) -> float:
        """Ground to the highest point of the roof structure."""
        return max(self.roof_z - self.ground_z, 0.0)

    @property
    def area(self) -> float:
        return self.footprint.area

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.footprint.centroid
        return c.x, c.y

    def as_dict(self) -> dict:
        cx, cy = self.centroid
        return {
            "index": self.index,
            "rings": rings_from_polygon(self.footprint),
            "centroid": [round(cx, 3), round(cy, 3)],
            "ground_z": round(self.ground_z, 3),
            "roof_z": round(self.roof_z, 3),
            "eave_z": round(self.eave_z, 3),
            "parapet_z": round(self.parapet_z, 3),
            "height_m": round(self.height, 3),
            "structural_height_m": round(self.structural_height, 3),
            "area_m2": round(self.area, 2),
            "n_points": int(self.n_points),
            "confidence": round(self.confidence, 3),
            "roof_planes": self.roof_planes,
            "metrics": self.metrics,
        }


@dataclass
class ExtractionResult:
    buildings: list[BuildingCandidate]
    scores: np.ndarray              # per-point building score
    labels: np.ndarray              # per-point instance label, -1 = not a building
    params: dict

    def summary(self) -> dict:
        return {
            "buildings_found": len(self.buildings),
            "total_footprint_area_m2": round(
                sum(b.area for b in self.buildings), 2),
            "heights_m": [round(b.height, 2) for b in self.buildings],
            "params": self.params,
        }


def extract_buildings(points: np.ndarray, height_above_ground: np.ndarray,
                      dtm: Grid, *,
                      min_height: float = 2.5,
                      score_threshold: float = 0.22,
                      feature_radius: float = 1.1,
                      cluster_eps: float = 2.2,
                      cluster_min_samples: int = 6,
                      cluster_voxel: float = 0.5,
                      z_weight: float = 0.35,
                      min_points: int = 400,
                      min_area: float = 25.0,
                      alpha: float = 3.2,
                      regularize: bool = True,
                      return_number: Optional[np.ndarray] = None,
                      progress=None) -> ExtractionResult:
    """
    Extract building instances from a point cloud.

    Parameters that matter most in practice:

    `cluster_eps`
        Must be larger than the point spacing on a roof but smaller than the
        narrowest gap between two buildings. 2.2 m suits a survey at 10-25
        pts/m2 with normal urban setbacks.
    `cluster_voxel`
        Edge of the voxel the cloud is thinned to before clustering. Must stay
        well below `cluster_eps` or genuinely connected parts of one building
        can end up in different clusters.
    `z_weight`
        Scales the vertical axis before clustering. At 1.0 a tall building can
        fragment into separate storeys; at 0.0 two adjacent buildings merge
        through the ground plane. A low but non-zero value keeps a structure
        whole while still using height to separate neighbours.
    `return_number`
        Optional LiDAR return numbers. First-of-many returns are a strong
        vegetation cue, so when the sensor provides them they sharpen the
        semantic step considerably.
    """
    pts = np.asarray(points, dtype=float)
    hag = np.asarray(height_above_ground, dtype=float)
    if len(pts) != len(hag):
        raise ValueError("points and height_above_ground must be the same length")

    def _tick(stage, **kw):
        if progress:
            progress(stage, **kw)

    # --- 1. semantic scoring ----------------------------------------------
    # Only points that clear the height gate are worth the cost of a
    # neighbourhood eigendecomposition, which is the expensive part.
    tall = np.flatnonzero(hag > min_height * 0.8)
    _tick("semantic", candidates=int(len(tall)))
    scores = np.zeros(len(pts))
    if len(tall) < 3:
        return ExtractionResult([], scores, np.full(len(pts), -1), {})

    tall_pts = pts[tall]
    tree = cKDTree(tall_pts)
    feats = compute_features(tall_pts, radius=feature_radius, tree=tree)
    s = building_score(feats, hag[tall], min_height=min_height)

    if return_number is not None:
        # a beam that produced more than one return passed through something
        # porous; that is foliage, not a roof
        rn = np.asarray(return_number)[tall]
        s = s * np.where(rn > 1, 0.25, 1.0)

    scores[tall] = s
    building_pts_idx = tall[s > score_threshold]
    _tick("semantic_done", building_points=int(len(building_pts_idx)))

    labels = np.full(len(pts), -1, dtype=int)
    if len(building_pts_idx) < min_points:
        return ExtractionResult([], scores, labels,
                                {"note": "no points survived semantic filtering"})

    # --- 2. instance segmentation -----------------------------------------
    # Cluster a voxel-downsampled cloud, then push the labels back out to full
    # resolution. DBSCAN's neighbour graph grows with the square of local
    # density, so on a 26 pts/m2 roof the full cloud produces tens of millions
    # of pairs to no benefit: building separation is a metre-scale question and
    # a decimetre-scale sample answers it identically, an order of magnitude
    # faster.
    bp_scaled = pts[building_pts_idx].copy()
    bp_scaled[:, 2] *= z_weight

    keys = np.floor(bp_scaled / cluster_voxel).astype(np.int64)
    _, rep_idx = np.unique(keys, axis=0, return_index=True)
    rep_pts = bp_scaled[rep_idx]
    _tick("downsampled", from_points=len(bp_scaled), to_points=len(rep_pts))

    rep_labels = dbscan(rep_pts, eps=cluster_eps,
                        min_samples=cluster_min_samples)
    # propagate: every point takes the label of its nearest retained sample
    _, nearest = cKDTree(rep_pts).query(bp_scaled, k=1, workers=-1)
    inst = rep_labels[nearest]

    labels[building_pts_idx] = inst
    sizes = cluster_sizes(inst)
    _tick("clustered", clusters=len(sizes))

    # --- 3-5. per-instance geometry ---------------------------------------
    buildings: list[BuildingCandidate] = []
    for cid, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        if size < min_points:
            continue
        sel = building_pts_idx[inst == cid]
        cluster_pts = pts[sel]

        ground_z = float(np.nanmedian(
            dtm.sample(cluster_pts[:, 0], cluster_pts[:, 1])))
        roof_z, parapet_z, roof_metrics = _estimate_roof_levels(
            cluster_pts, ground_z)

        # Locate the roof in two passes. The first uses a deliberately generous
        # band - wide enough to contain a whole pitched roof - purely to fit the
        # roof planes. The dominant plane's lower edge then *is* the eave, which
        # is a direct measurement of where the walls stop. The second pass
        # re-selects the roof region from that eave, so the footprint gets traced
        # from the roof's full plan extent whatever its form.
        coarse = cluster_pts[cluster_pts[:, 2] >= roof_z - 3.0]
        planes = _fit_roof_planes(coarse)
        pitched = bool(planes) and planes[0]["tilt_deg"] > 8.0

        eave_z = roof_z
        if pitched and planes[0].get("min_z") is not None:
            eave_z = min(float(planes[0]["min_z"]), roof_z)

        roof_pts = cluster_pts[(cluster_pts[:, 2] >= eave_z - 0.6) &
                               (cluster_pts[:, 2] <= parapet_z + 0.3)]

        # Trace the outline from the roof, not the whole cluster: balconies,
        # sunshades and canopies hang out past the wall line, and a footprint
        # traced through them comes out systematically too large - it is the
        # wall line, not the balcony edge, that bounds the legal parcel.
        footprint_raw = alpha_shape(roof_pts[:, :2], alpha=alpha) \
            if len(roof_pts) >= 150 else None
        if footprint_raw is None or footprint_raw.area < min_area:
            # sparse or heavily occluded roof - fall back to the full cluster
            footprint_raw = alpha_shape(cluster_pts[:, :2], alpha=alpha)
            roof_metrics["footprint_source"] = "full_cluster"
        else:
            roof_metrics["footprint_source"] = "roof_region"
        if footprint_raw is None or footprint_raw.area < min_area:
            continue
        footprint_raw = clean_polygon(footprint_raw)
        if footprint_raw is None:
            continue

        footprint = (regularize_footprint(footprint_raw) if regularize
                     else footprint_raw)
        if footprint.area < min_area:
            continue

        roof_metrics["roof_form"] = "pitched" if pitched else "flat"
        roof_metrics["eave_height_above_ground_m"] = round(eave_z - ground_z, 3)
        roof_metrics["roof_region_points"] = int(len(roof_pts))

        # confidence blends how building-like the points were, how much of the
        # footprint the cloud actually covers, and how much regularisation had
        # to move the outline
        mean_score = float(scores[sel].mean())
        coverage = min(1.0, size / max(footprint.area * 8.0, 1.0))
        shape_agreement = (footprint.intersection(footprint_raw).area /
                           max(footprint.union(footprint_raw).area, 1e-9))
        confidence = float(np.clip(
            0.45 * mean_score + 0.25 * coverage + 0.30 * shape_agreement, 0, 1))

        buildings.append(BuildingCandidate(
            index=len(buildings),
            footprint=footprint,
            raw_footprint=footprint_raw,
            ground_z=ground_z,
            roof_z=roof_z,
            eave_z=eave_z,
            parapet_z=parapet_z,
            n_points=int(size),
            point_indices=sel,
            roof_planes=planes,
            confidence=confidence,
            metrics={
                "mean_semantic_score": round(mean_score, 3),
                "point_coverage": round(coverage, 3),
                "regularisation_iou": round(shape_agreement, 3),
                "traced_area_m2": round(footprint_raw.area, 2),
                **roof_metrics,
            },
        ))
    _tick("footprints", buildings=len(buildings))

    return ExtractionResult(
        buildings=buildings, scores=scores, labels=labels,
        params={
            "min_height_m": min_height,
            "score_threshold": score_threshold,
            "feature_radius_m": feature_radius,
            "cluster_eps_m": cluster_eps,
            "cluster_min_samples": cluster_min_samples,
            "cluster_voxel_m": cluster_voxel,
            "z_weight": z_weight,
            "alpha_m": alpha,
            "regularised": regularize,
            "used_return_numbers": return_number is not None,
        },
    )



def _estimate_roof_levels(cluster_pts: np.ndarray,
                          ground_z: float) -> tuple[float, float, dict]:
    """
    Separate the structural roof level from the parapet on top of it.

    This distinction is not cosmetic. The top of the point cloud is the parapet
    coping, typically 0.9-1.2 m above the actual roof slab. Taking it as the
    building height inflates every derived storey height and pushes the top
    flat's ceiling a metre into thin air.

    The roof slab is found as the highest *large* horizontal population of
    points: parapets are a thin ring around the perimeter and so contribute few
    points per unit height, whereas the roof surface contributes many.
    """
    z = cluster_pts[:, 2]
    parapet_z = float(np.percentile(z, 99.5))
    top_band = z[z > np.percentile(z, 60)]
    if len(top_band) < 30:
        return parapet_z, parapet_z, {"parapet_detected": False}

    # histogram the upper part of the cloud at 20 cm resolution and take the
    # highest bin that still holds a substantial share of the points
    lo, hi = top_band.min(), top_band.max()
    if hi - lo < 0.3:
        return float(np.median(top_band)), parapet_z, {"parapet_detected": False}
    nbins = max(4, int((hi - lo) / 0.2))
    counts, edges = np.histogram(top_band, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2
    strong = np.flatnonzero(counts >= 0.35 * counts.max())
    roof_z = float(centres[strong[-1]]) if len(strong) else float(np.median(top_band))

    parapet_h = parapet_z - roof_z
    detected = 0.25 < parapet_h < 2.5
    if not detected:
        roof_z = parapet_z
    return roof_z, parapet_z, {
        "parapet_detected": bool(detected),
        "parapet_height_m": round(float(parapet_h), 3) if detected else 0.0,
        "roof_height_above_ground_m": round(float(roof_z - ground_z), 3),
    }


def _fit_roof_planes(roof_pts: np.ndarray, max_planes: int = 3) -> list[dict]:
    """
    Fit the dominant roof faces by sequential RANSAC, largest first.

    Roof form tells us whether the top storey is habitable, feeds the 3D model
    the viewer draws, and flags structures whose "flat" roof is actually pitched
    - which matters when the topmost unit's ceiling has to be defined. The
    reported `min_z` of the dominant face is where that face meets the walls,
    i.e. the eave.
    """
    planes: list[dict] = []
    if len(roof_pts) < 60:
        return planes

    remaining = roof_pts
    rng = np.random.default_rng(11)
    for _ in range(max_planes):
        if len(remaining) < 60:
            break
        fit = ransac_plane(remaining, threshold=0.18, iterations=160,
                           min_inliers=max(45, int(0.12 * len(roof_pts))), rng=rng)
        if fit is None:
            break
        normal, d, mask = fit
        tilt = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))
        inlier_z = remaining[mask][:, 2]
        planes.append({
            "normal": [round(float(v), 4) for v in normal],
            "d": round(d, 4),
            "inliers": int(mask.sum()),
            "tilt_deg": round(tilt, 2),
            "form": "flat" if tilt < 8 else ("pitched" if tilt < 45 else "steep"),
            "mean_z": round(float(inlier_z.mean()), 3),
            # a low percentile rather than the outright minimum, so one stray
            # point below the eave does not define the whole storey boundary
            "min_z": round(float(np.percentile(inlier_z, 2.0)), 3),
        })
        remaining = remaining[~mask]
    # report the largest face first - that is the one that defines the form
    planes.sort(key=lambda p: -p["inliers"])
    return planes

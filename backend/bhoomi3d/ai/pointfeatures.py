"""
Local geometric features for point-cloud semantic classification.

The decisive question in building extraction is not "is this point high?" - a
tree is high too - but "is this point part of a *planar, vertically bounded,
opaque* surface?". The eigenvalues of the local covariance matrix answer that,
and they are the standard feature set in the LiDAR classification literature
(Weinmann et al., "Semantic point cloud interpretation", ISPRS 2015).

For a neighbourhood with sorted eigenvalues l1 >= l2 >= l3:

    linearity   = (l1 - l2) / l1     wires, poles, kerb lines
    planarity   = (l2 - l3) / l1     roofs, walls, roads  <- what we want
    sphericity  =  l3 / l1           foliage, scatter      <- what we reject
    verticality = 1 - |n . z|        walls (used for storey detection)

Vegetation is the dominant false positive in every automated cadastral
pipeline, and it is separable here: a tree canopy is volumetrically scattered
(high sphericity, low planarity) where a roof is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PointFeatures:
    """Per-point local geometry descriptors, all in [0, 1] unless noted."""

    linearity: np.ndarray
    planarity: np.ndarray
    sphericity: np.ndarray
    verticality: np.ndarray
    normals: np.ndarray          # (n, 3), unit, oriented upwards
    curvature: np.ndarray        # l3 / (l1+l2+l3), surface variation
    n_neighbours: np.ndarray

    def as_matrix(self) -> np.ndarray:
        """Stack into an (n, 5) matrix, ready for a downstream classifier."""
        return np.column_stack([self.linearity, self.planarity, self.sphericity,
                                self.verticality, self.curvature])


def compute_features(points: np.ndarray, *, radius: float = 1.2,
                     max_neighbours: int = 40,
                     tree: cKDTree | None = None) -> PointFeatures:
    """
    Compute eigenvalue features for every point using a fixed-radius neighbourhood.

    A fixed radius (rather than fixed k) is deliberate: it makes the descriptors
    depend on real-world scale, so a roof looks planar whether it was scanned at
    20 or 200 points per square metre. Neighbourhoods are capped at
    `max_neighbours` to bound the cost in the very dense patches.

    Vectorised over the whole cloud - points are grouped by neighbourhood size
    so each group's covariance eigendecomposition is a single batched call.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        empty = np.empty(0)
        return PointFeatures(empty, empty, empty, empty,
                             np.empty((0, 3)), empty, np.empty(0, dtype=int))
    if tree is None:
        tree = cKDTree(pts)

    linearity = np.zeros(n)
    planarity = np.zeros(n)
    sphericity = np.zeros(n)
    curvature = np.zeros(n)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    counts = np.zeros(n, dtype=int)

    # One fixed-width k-NN query capped at `radius` gives a rectangular (n, k)
    # index array, which keeps the whole computation in batched numpy. The
    # alternative - a variable-length radius query - forces a per-point Python
    # loop and costs roughly twenty times as much on a city-block cloud.
    # Chunked so the (chunk, k, 3) neighbour gather stays inside cache-friendly
    # memory rather than allocating hundreds of megabytes at once.
    chunk = 20_000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sl = slice(start, stop)
        dist, idx = tree.query(pts[sl], k=max_neighbours,
                               distance_upper_bound=radius, workers=-1)
        if dist.ndim == 1:                     # k == 1 degenerate case
            dist, idx = dist[:, None], idx[:, None]

        valid = np.isfinite(dist)              # (m, k)
        m_counts = valid.sum(axis=1)
        counts[sl] = m_counts
        # a covariance needs at least four points to be meaningful in 3D
        good = m_counts >= 4
        if not good.any():
            continue

        idx_safe = np.where(valid, np.minimum(idx, n - 1), 0)
        w = valid.astype(float)[..., None]     # (m, k, 1)
        nbr = pts[idx_safe]                    # (m, k, 3)
        cnt = np.maximum(w.sum(axis=1), 1.0)   # (m, 1)
        mean = (nbr * w).sum(axis=1) / cnt
        centred = (nbr - mean[:, None, :]) * w
        # w is 0/1 so w == w**2 and this is the exact masked covariance
        cov = np.einsum("mki,mkj->mij", centred, centred) / cnt[:, None, :]

        evals, evecs = np.linalg.eigh(cov)     # ascending
        evals = np.clip(evals[:, ::-1], 0.0, None)   # -> descending l1, l2, l3
        evecs = evecs[:, :, ::-1]

        l1 = np.maximum(evals[:, 0], 1e-12)
        total = np.maximum(evals.sum(axis=1), 1e-12)
        sel = np.flatnonzero(good) + start
        g = good
        linearity[sel] = ((evals[:, 0] - evals[:, 1]) / l1)[g]
        planarity[sel] = ((evals[:, 1] - evals[:, 2]) / l1)[g]
        sphericity[sel] = (evals[:, 2] / l1)[g]
        curvature[sel] = (evals[:, 2] / total)[g]
        nrm = evecs[:, :, 2]                   # smallest-eigenvalue axis
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        normals[sel] = np.where(nrm[:, 2:3] < 0, -nrm, nrm)[g]

    verticality = 1.0 - np.abs(normals[:, 2])
    return PointFeatures(linearity, planarity, sphericity, verticality,
                         normals, curvature, counts)


def building_score(feat: PointFeatures, height_above_ground: np.ndarray, *,
                   min_height: float = 2.5,
                   planarity_weight: float = 1.0,
                   scatter_penalty: float = 1.4) -> np.ndarray:
    """
    Combine features into a per-point "is this a building surface" score in [0,1].

    This is a deliberately transparent linear scoring rule rather than a learned
    classifier. It needs no labelled training data, its behaviour is auditable
    by a surveyor, and every term corresponds to a physical property they can
    argue about - which matters when the output has legal consequences. The
    interface is the same one a trained model would expose, so swapping in a
    learned classifier is a drop-in change (see `ai/backends.py`).

    Terms:
      * a soft height gate - below `min_height` nothing can be a building
      * reward planarity - roofs and walls are locally flat
      * punish sphericity - foliage scatters in all three directions
    """
    h = np.asarray(height_above_ground, dtype=float)
    height_gate = 1.0 / (1.0 + np.exp(-(h - min_height) / 0.4))
    score = np.clip(planarity_weight * feat.planarity
                    - scatter_penalty * feat.sphericity, 0.0, 1.0)
    return height_gate * score

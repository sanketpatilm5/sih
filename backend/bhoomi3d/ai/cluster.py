"""
Density-based clustering, used for building *instance* segmentation.

Semantic segmentation tells us which points are "building". It does not tell us
where one building stops and the next begins - and a cadastre is about
individual objects, so instance separation is the step that actually matters.

DBSCAN is the right tool: buildings are dense blobs separated by low-density
gaps (streets, setbacks), we do not know how many there are in advance, and it
labels the leftover scatter as noise instead of forcing it into a cluster.

Implemented here directly on a KD-tree rather than pulled from scikit-learn, to
keep the dependency footprint small and because the neighbour graph gets reused
by the feature extractor.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

NOISE = -1


def dbscan(points: np.ndarray, eps: float, min_samples: int,
           *, tree: cKDTree | None = None) -> np.ndarray:
    """
    Cluster `points` (n, d) and return an (n,) array of labels, -1 for noise.

    The standard DBSCAN definition, evaluated in the vectorised order:

      1. count each point's eps-neighbours to find *core* points
      2. take connected components of the core-to-core neighbour graph
      3. attach each *border* point to a core neighbour's cluster
      4. anything left is noise

    That yields exactly the canonical DBSCAN partition, except that a border
    point reachable from two clusters is assigned to its nearest core point
    rather than to whichever cluster happened to be visited first - which makes
    the result independent of point ordering.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return np.empty(0, dtype=int)
    if tree is None:
        tree = cKDTree(pts)

    counts = np.array(tree.query_ball_point(pts, eps, return_length=True))
    is_core = counts >= min_samples
    labels = np.full(n, NOISE, dtype=int)
    if not is_core.any():
        return labels

    core_idx = np.flatnonzero(is_core)
    core_pts = pts[core_idx]
    core_tree = cKDTree(core_pts)

    # sparse core-to-core adjacency, then connected components
    pairs = core_tree.query_pairs(eps, output_type="ndarray")
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    else:
        rows = cols = np.empty(0, dtype=int)
    adj = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                     shape=(len(core_idx), len(core_idx)))
    _, core_labels = connected_components(adj, directed=False)
    labels[core_idx] = core_labels

    # border points: non-core, but within eps of some core point
    border = np.flatnonzero(~is_core)
    if len(border):
        dist, nearest = core_tree.query(pts[border], k=1,
                                        distance_upper_bound=eps)
        reachable = np.isfinite(dist)
        labels[border[reachable]] = core_labels[nearest[reachable]]

    return _compact_labels(labels)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster ids to 0..k-1 by descending size, keeping -1 as noise."""
    out = np.full_like(labels, NOISE)
    valid = labels[labels != NOISE]
    if valid.size == 0:
        return out
    uniq, counts = np.unique(valid, return_counts=True)
    for new, old in enumerate(uniq[np.argsort(-counts)]):
        out[labels == old] = new
    return out


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    valid = labels[labels != NOISE]
    if valid.size == 0:
        return {}
    uniq, counts = np.unique(valid, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def ransac_plane(points: np.ndarray, *, threshold: float = 0.15,
                 iterations: int = 250, min_inliers: int = 25,
                 rng: np.random.Generator | None = None):
    """
    Fit the dominant plane to a point set by RANSAC.

    Used for roof faces and for floor slabs. RANSAC rather than least squares
    because the input is contaminated: a "roof" cluster contains chimneys, water
    tanks, parapets and antennas, and a least-squares plane would be dragged off
    the true roof surface by exactly those outliers.

    Returns `(normal, d, inlier_mask)` for the plane `normal . x + d = 0`, with
    the normal unit length and oriented upwards, or None if no plane is found.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < max(3, min_inliers):
        return None
    rng = rng or np.random.default_rng(42)

    best_mask = None
    best_count = 0
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:          # degenerate (collinear) sample
            continue
        normal = normal / norm
        d = -normal @ p0
        dist = np.abs(pts @ normal + d)
        mask = dist < threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask

    if best_mask is None or best_count < min_inliers:
        return None

    # refit by total least squares on the inliers, which tightens the estimate
    inl = pts[best_mask]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    d = -normal @ centroid
    mask = np.abs(pts @ normal + d) < threshold
    return normal, float(d), mask

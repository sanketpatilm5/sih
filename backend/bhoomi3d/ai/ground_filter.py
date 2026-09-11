"""
Ground filtering: separating bare earth from everything standing on it.

This is the first and most consequential stage of the pipeline. Every height in
the finished cadastre - the floor level of a flat, the cover depth over a
pipeline, the height of a building - is measured from the terrain, so an error
here propagates into every single 3D parcel downstream.

The implementation is the **Simple Morphological Filter** (Pingel, Clarke &
McBride, ISPRS 2013). It was chosen over the alternatives because:

  * unlike a plain slope filter, it does not shave the tops off legitimate
    steep terrain;
  * unlike progressive TIN densification, it is fully raster-based and so runs
    in seconds on a city-block-sized cloud;
  * it has exactly two parameters with physical meaning (a maximum object
    radius and a maximum terrain slope), which a surveyor can actually reason
    about and tune for a site.

Products
--------
DTM  bare-earth terrain (the DEM)
DSM  the top of everything - canopy, roofs, gantries
nDSM DSM - DTM, i.e. height above ground: the input to building extraction
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..core.grid import Grid


@dataclass
class GroundFilterResult:
    """Terrain products plus the per-point ground classification."""

    dtm: Grid
    dsm: Grid
    ndsm: Grid
    is_ground: np.ndarray        # (n,) bool, aligned with the input cloud
    height_above_ground: np.ndarray   # (n,) float, metres
    object_mask: np.ndarray      # (rows, cols) bool - cells the filter rejected
    params: dict

    def summary(self) -> dict:
        n = len(self.is_ground)
        return {
            "points": int(n),
            "ground_points": int(self.is_ground.sum()),
            "ground_fraction": round(float(self.is_ground.mean()), 4) if n else 0.0,
            "dtm_cell_m": self.dtm.cell,
            "dtm_range_m": [round(float(np.nanmin(self.dtm.data)), 3),
                            round(float(np.nanmax(self.dtm.data)), 3)],
            "max_height_above_ground_m": round(
                float(np.nanmax(self.height_above_ground)), 3) if n else 0.0,
            "params": self.params,
        }


def _grey_open(arr: np.ndarray, radius: int) -> np.ndarray:
    """
    Grayscale morphological opening with a square window of the given radius.

    A square rather than a disc: scipy evaluates rectangular min/max filters
    separably, which turns each pass from O(r^2) to O(r) per cell. Over the
    ~10 window sizes in a progressive filter that is the difference between
    half a minute and a fraction of a second, and the shape of the structuring
    element makes no practical difference to which cells get flagged as objects.
    """
    if radius <= 0:
        return arr
    size = 2 * radius + 1
    return ndimage.maximum_filter(
        ndimage.minimum_filter(arr, size=size, mode="nearest"),
        size=size, mode="nearest")


def _window_schedule(max_radius: int) -> list[int]:
    """
    Window radii for the progressive opening.

    Geometric rather than linear growth (as in Zhang et al.'s progressive
    morphological filter): object detection depends on the *ratio* of window to
    object size, so linear stepping wastes most of its passes on large windows
    that flag nothing new.
    """
    radii: list[int] = []
    r = 1
    while r < max_radius:
        radii.append(r)
        r = max(r + 1, int(math.ceil(r * 1.6)))
    radii.append(max_radius)
    return radii


def smrf(points: np.ndarray, *, cell: float = 0.75,
         max_window_m: float = 18.0, slope_threshold: float = 0.20,
         elevation_threshold: float = 0.30, elevation_scaler: float = 1.25,
         bounds=None) -> GroundFilterResult:
    """
    Simple Morphological Filter.

    Parameters
    ----------
    cell
        Grid resolution. Should be near the mean point spacing: too fine and
        the minimum surface is noisy, too coarse and small terrain detail is
        lost.
    max_window_m
        Radius of the largest object to be removed. Must exceed the widest
        building on site, or that building's centre will be mistaken for
        terrain and it will be flattened into the DTM.
    slope_threshold
        Maximum plausible terrain slope (rise/run). Elevation drops steeper
        than this across a window are treated as object edges, not terrain.
    elevation_threshold, elevation_scaler
        Point-to-DTM tolerance for the final classification. The tolerance is
        widened on steep ground by `elevation_scaler * local_slope`, because a
        cell straddling a slope legitimately spans a range of heights.

    Returns
    -------
    GroundFilterResult
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("expected an (n, 3+) array of XYZ points")

    # 1. provisional minimum surface, with voids filled ---------------------
    zmin = Grid.from_points(pts, cell, stat="min", bounds=bounds, name="zmin")
    holes = zmin.nodata_mask()
    surface = zmin.filled().data

    # 2. progressive opening ------------------------------------------------
    # Each pass removes objects up to the current window radius. A cell is
    # flagged as object when opening pulls it down by more than the steepest
    # terrain could plausibly account for over that window.
    max_radius = max(1, int(np.ceil(max_window_m / cell)))
    object_mask = np.zeros(surface.shape, dtype=bool)
    current = surface.copy()
    for radius in _window_schedule(max_radius):
        opened = _grey_open(current, radius)
        # the most terrain could legitimately fall across a window of this size
        threshold = slope_threshold * radius * cell
        object_mask |= (current - opened) > threshold
        current = opened

    # voids were invented, not observed, so never trust them as ground
    object_mask |= holes

    # 3. DTM from the surviving ground cells --------------------------------
    dtm_data = np.where(object_mask, np.nan, surface)
    dtm = Grid(dtm_data, zmin.x0, zmin.y0, cell, "DTM").filled()
    # a light smooth removes the stair-stepping the opening leaves behind,
    # without meaningfully displacing real terrain
    dtm = dtm.smoothed(sigma_cells=1.0)
    dtm.name = "DTM"

    # 4. DSM and nDSM -------------------------------------------------------
    dsm = Grid.from_points(pts, cell, stat="max", bounds=bounds, name="DSM").filled()
    ndsm_data = np.maximum(dsm.data - dtm.data, 0.0)
    ndsm = Grid(ndsm_data, dtm.x0, dtm.y0, cell, "nDSM")

    # 5. classify the original points ---------------------------------------
    ground_z = dtm.sample(pts[:, 0], pts[:, 1])
    hag = pts[:, 2] - ground_z

    gy, gx = np.gradient(dtm.data, cell)
    slope = Grid(np.hypot(gx, gy), dtm.x0, dtm.y0, cell, "slope")
    local_slope = slope.sample(pts[:, 0], pts[:, 1], default=0.0)
    tolerance = elevation_threshold + elevation_scaler * np.nan_to_num(local_slope) * cell
    is_ground = np.abs(hag) <= tolerance

    return GroundFilterResult(
        dtm=dtm, dsm=dsm, ndsm=ndsm,
        is_ground=is_ground,
        height_above_ground=hag,
        object_mask=object_mask,
        params={
            "algorithm": "SMRF (Pingel et al. 2013)",
            "cell_m": cell,
            "max_window_m": max_window_m,
            "slope_threshold": slope_threshold,
            "elevation_threshold_m": elevation_threshold,
            "elevation_scaler": elevation_scaler,
        },
    )


def dtm_from_dem_grid(dem: Grid, dsm: Grid) -> Grid:
    """
    Build an nDSM when the DEM and DSM arrive as supplied rasters rather than
    being derived from a cloud - the common case when a state GIS department
    hands over elevation products instead of raw LiDAR.
    """
    if dem.data.shape != dsm.data.shape or abs(dem.cell - dsm.cell) > 1e-9:
        raise ValueError(
            "DEM and DSM must share a lattice; resample them before differencing")
    return Grid(np.maximum(dsm.data - dem.data, 0.0), dem.x0, dem.y0, dem.cell, "nDSM")

"""
Regular raster grids in the project-local ENU frame.

DEM, DSM, nDSM and the intermediate products of the extraction pipeline are all
the same thing: a 2D array of values on a regular grid. This module gives that
one representation with an explicit georeference, so a height sampled at a
parcel corner means the same thing no matter which stage produced it.

Convention: `data[row, col]`, with **row 0 at the minimum y**. That is upside
down relative to how GeoTIFFs are usually stored, and we flip on import, because
having y increase with the row index removes a whole class of sign errors from
the geometry code that consumes these grids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Grid:
    """A georeferenced 2D array of floats. NaN marks no-data."""

    data: np.ndarray          # (rows, cols), float
    x0: float                 # centre of column 0, metres (local ENU)
    y0: float                 # centre of row 0, metres
    cell: float               # cell size, metres
    name: str = "grid"

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(f"a Grid needs a 2D array, got {self.data.ndim}D")

    # -- shape ---------------------------------------------------------------
    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of cell centres, expanded by half a cell."""
        h = self.cell / 2
        return (self.x0 - h, self.y0 - h,
                self.x0 + (self.cols - 1) * self.cell + h,
                self.y0 + (self.rows - 1) * self.cell + h)

    # -- indexing ------------------------------------------------------------
    def xy_to_rc(self, x, y):
        """World coords -> fractional (row, col)."""
        return (np.asarray(y, float) - self.y0) / self.cell, \
               (np.asarray(x, float) - self.x0) / self.cell

    def rc_to_xy(self, r, c):
        return self.x0 + np.asarray(c, float) * self.cell, \
               self.y0 + np.asarray(r, float) * self.cell

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + np.arange(self.cols) * self.cell
        ys = self.y0 + np.arange(self.rows) * self.cell
        return np.meshgrid(xs, ys)

    # -- sampling ------------------------------------------------------------
    def sample(self, x, y, *, method: str = "bilinear", default: float = np.nan):
        """
        Sample the grid at world coordinates.

        Bilinear sampling matters here: a DEM sampled nearest-neighbour makes a
        building's floor level jump by a whole cell's worth of terrain slope
        depending on which side of a cell boundary its centroid lands, and that
        shows up directly as an error in the parcel's z_min.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        rf, cf = self.xy_to_rc(x, y)

        if method == "nearest":
            r = np.rint(rf).astype(int)
            c = np.rint(cf).astype(int)
            ok = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
            out = np.full(x.shape, default, dtype=float)
            out[ok] = self.data[r[ok], c[ok]]
            return out

        r0 = np.floor(rf).astype(int)
        c0 = np.floor(cf).astype(int)
        dr = rf - r0
        dc = cf - c0
        out = np.full(x.shape, default, dtype=float)
        ok = (r0 >= 0) & (r0 < self.rows - 1) & (c0 >= 0) & (c0 < self.cols - 1)
        if ok.any():
            r0o, c0o, dro, dco = r0[ok], c0[ok], dr[ok], dc[ok]
            v00 = self.data[r0o, c0o]
            v01 = self.data[r0o, c0o + 1]
            v10 = self.data[r0o + 1, c0o]
            v11 = self.data[r0o + 1, c0o + 1]
            out[ok] = (v00 * (1 - dro) * (1 - dco) + v01 * (1 - dro) * dco +
                       v10 * dro * (1 - dco) + v11 * dro * dco)
        # fall back to nearest on the border, where bilinear has no full stencil
        edge = (~ok) & (rf >= -0.5) & (rf <= self.rows - 0.5) & \
               (cf >= -0.5) & (cf <= self.cols - 0.5)
        if edge.any():
            r = np.clip(np.rint(rf[edge]).astype(int), 0, self.rows - 1)
            c = np.clip(np.rint(cf[edge]).astype(int), 0, self.cols - 1)
            out[edge] = self.data[r, c]
        return out

    # -- maintenance ---------------------------------------------------------
    def filled(self, max_iter: int = 200) -> "Grid":
        """
        Fill NaN holes by iterative nearest-valid diffusion.

        Point clouds always have voids - under a dense canopy, on a water body,
        in a scan shadow behind a tower. A DEM with holes propagates NaN into
        every derived height, so gaps are closed before the grid leaves the
        ground filter, and the caller can consult :meth:`nodata_mask` to know
        which values were interpolated rather than observed.
        """
        from scipy import ndimage

        arr = self.data.copy()
        nan = np.isnan(arr)
        if not nan.any():
            return Grid(arr, self.x0, self.y0, self.cell, self.name)
        if nan.all():
            return Grid(np.zeros_like(arr), self.x0, self.y0, self.cell, self.name)
        # distance transform gives, for every hole cell, the nearest valid cell
        idx = ndimage.distance_transform_edt(nan, return_distances=False,
                                             return_indices=True)
        arr[nan] = arr[tuple(i[nan] for i in idx)]
        return Grid(arr, self.x0, self.y0, self.cell, self.name)

    def nodata_mask(self) -> np.ndarray:
        return np.isnan(self.data)

    def smoothed(self, sigma_cells: float = 1.0) -> "Grid":
        from scipy import ndimage
        arr = np.nan_to_num(self.data, nan=float(np.nanmedian(self.data)))
        return Grid(ndimage.gaussian_filter(arr, sigma_cells),
                    self.x0, self.y0, self.cell, self.name)

    def __sub__(self, other: "Grid") -> "Grid":
        if other.data.shape != self.data.shape:
            raise ValueError("grids must be on the same lattice to subtract")
        return Grid(self.data - other.data, self.x0, self.y0, self.cell,
                    f"{self.name}-{other.name}")

    # -- construction from a point cloud -------------------------------------
    @classmethod
    def from_points(cls, xyz: np.ndarray, cell: float, *, stat: str = "min",
                    bounds: Optional[tuple[float, float, float, float]] = None,
                    name: str = "grid") -> "Grid":
        """
        Bin a point cloud onto a regular grid.

        `stat` is "min" for a provisional ground surface, "max" for a surface
        model (DSM), "mean" for a smoothed one, or "count" for point density.
        Empty cells become NaN so the caller can decide how to treat voids.
        """
        pts = np.asarray(xyz, dtype=float)
        if pts.size == 0:
            raise ValueError("cannot grid an empty point cloud")
        if bounds is None:
            x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
            x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        else:
            x_min, y_min, x_max, y_max = bounds

        cols = max(1, int(np.ceil((x_max - x_min) / cell)) + 1)
        rows = max(1, int(np.ceil((y_max - y_min) / cell)) + 1)
        c = np.clip(((pts[:, 0] - x_min) / cell).round().astype(int), 0, cols - 1)
        r = np.clip(((pts[:, 1] - y_min) / cell).round().astype(int), 0, rows - 1)
        flat = r * cols + c
        n = rows * cols

        if stat == "count":
            out = np.bincount(flat, minlength=n).astype(float)
            return cls(out.reshape(rows, cols), x_min, y_min, cell, name)

        if stat == "mean":
            total = np.bincount(flat, weights=pts[:, 2], minlength=n)
            cnt = np.bincount(flat, minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
            return cls(out.reshape(rows, cols), x_min, y_min, cell, name)

        # min / max via sorted scatter, which is far faster than a Python loop.
        # Scattered writes keep the *last* value written to each index, so the
        # winner must be sorted last: ascending z for a maximum surface,
        # descending for a minimum one.
        if stat == "max":
            order = np.argsort(pts[:, 2], kind="stable")
        elif stat == "min":
            order = np.argsort(-pts[:, 2], kind="stable")
        else:
            raise ValueError(f"unknown stat {stat!r}")
        out = np.full(n, np.nan)
        out[flat[order]] = pts[order, 2]
        return cls(out.reshape(rows, cols), x_min, y_min, cell, name)

    # -- serialisation -------------------------------------------------------
    def as_dict(self, *, max_cells: int = 160_000, decimals: int = 2) -> dict:
        """
        Pack for the browser, decimating if the grid is large.

        The viewer only needs enough resolution to draw a convincing terrain
        surface; shipping a full-resolution DEM would dominate the payload.
        """
        step = max(1, int(np.ceil(np.sqrt(self.rows * self.cols / max_cells))))
        sub = self.data[::step, ::step]
        return {
            "name": self.name,
            "rows": int(sub.shape[0]),
            "cols": int(sub.shape[1]),
            "x0": round(self.x0, 4),
            "y0": round(self.y0, 4),
            "cell": round(self.cell * step, 6),
            "z_min": None if np.isnan(sub).all() else round(float(np.nanmin(sub)), 3),
            "z_max": None if np.isnan(sub).all() else round(float(np.nanmax(sub)), 3),
            "values": np.where(np.isnan(sub), None,
                               np.round(sub, decimals)).ravel().tolist(),
        }

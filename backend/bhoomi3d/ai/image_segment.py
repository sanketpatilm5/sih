"""
Building extraction from drone orthophotography, and fusion with LiDAR.

LiDAR and imagery fail in opposite ways, which is exactly why fusing them
works. A point cloud knows height but not material: a dense tree canopy and a
flat roof at the same elevation look similar to a geometric classifier. Imagery
knows material but not height: a rooftop and a concrete forecourt are both grey
and flat in a photograph. Each sensor's blind spot is the other's strength.

So this module produces an independent, image-only building probability and
then combines it with the point cloud's height evidence. Three modes:

``image``   imagery alone - the fallback when only an orthophoto exists
``lidar``   height alone - the fallback when only a cloud exists
``fused``   both, which is what the pipeline uses when both are available

Backend
-------
The default segmenter is a transparent, hand-specified feature model rather
than a learned one. That is a deliberate choice for this stage of the project:
it needs no labelled training data for an Indian urban scene (which does not
exist as an open dataset at cadastral quality), every decision it makes can be
explained to a surveyor, and it establishes the interface a trained model would
implement. :class:`SegmentationBackend` is that interface, and
:class:`OnnxBackend` shows how a U-Net or Mask R-CNN checkpoint drops in
without any other stage changing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from scipy import ndimage

from ..core.geometry3d import alpha_shape, clean_polygon, regularize_footprint
from ..core.grid import Grid


# --- backend interface ---------------------------------------------------------
class SegmentationBackend(Protocol):
    """Anything that turns an RGB orthophoto into a per-pixel building probability."""

    name: str

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        """`rgb` is (H, W, 3) in 0-255; returns (H, W) in [0, 1]."""
        ...


@dataclass
class SpectralBackend:
    """
    A vegetation-and-shadow *rejector*, not a roof detector.

    This framing is the honest one, and worth being explicit about because it
    is easy to get wrong: **RGB imagery alone cannot reliably distinguish a
    roof from a paved forecourt.** Both are grey, flat, weakly saturated and
    smooth. Any model claiming otherwise from three visible bands is keying on
    an incidental property of one dataset and will not transfer.

    What colour *can* do, reliably and without training data, is rule things
    out. Vegetation is unmistakable in the visible spectrum even when it is
    exactly as tall as a roof, and cast shadow is unmistakably not a surface.
    Those are precisely the two confusers that height cannot resolve - so this
    backend rejects them, and leaves the roof-versus-ground decision to the
    nDSM, which answers it trivially.

    The output is therefore read as "probability this pixel is a built,
    non-vegetated, directly-illuminated surface". Three independent rejection
    terms, combined as a product because each is sufficient on its own to
    disqualify a pixel:

    ``vegetation``  Excess-Green (2G - R - B), the standard RGB-only greenness
                    index.
    ``shadow``      dark *and* weakly saturated - which separates true cast
                    shadow from genuinely dark roofing material.
    ``roughness``   local intensity standard deviation; canopy and rubble are
                    rough where roofs and paving are smooth.
    """

    name: str = "spectral-rejector"
    veg_gain: float = 1.0
    shadow_gain: float = 1.0
    roughness_gain: float = 0.6
    texture_scale: float = 3.0

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        img = np.asarray(rgb, dtype=float)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("expected an (H, W, 3) RGB image")
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        total = np.maximum(r + g + b, 1e-6)

        exg = (2 * g - r - b) / total
        vegetation = np.clip((exg - 0.015) / 0.14, 0, 1)

        intensity = total / 3.0
        mx, mn = np.max(img, axis=2), np.min(img, axis=2)
        saturation = (mx - mn) / np.maximum(mx, 1e-6)
        dark = np.clip((78.0 - intensity) / 55.0, 0, 1)
        shadow = dark * np.clip(1.0 - saturation * 2.0, 0, 1)

        win = int(self.texture_scale * 2 + 1)
        mean = ndimage.uniform_filter(intensity, size=win)
        sq = ndimage.uniform_filter(intensity ** 2, size=win)
        roughness = np.clip(np.sqrt(np.maximum(sq - mean ** 2, 0)) / 30.0, 0, 1)

        return np.clip((1.0 - self.veg_gain * vegetation)
                       * (1.0 - self.shadow_gain * shadow)
                       * (1.0 - self.roughness_gain * roughness), 0.0, 1.0)


@dataclass
class OnnxBackend:
    """
    Drop-in slot for a trained segmentation network exported to ONNX.

    Deliberately unimplemented rather than faked. It documents the contract a
    learned model must satisfy - same input, same output range, same interface -
    so swapping one in is a configuration change, not a rewrite. Training such a
    model on labelled Indian urban imagery is the obvious next step for this
    component, and nothing else in the pipeline would need to change.
    """

    model_path: str
    name: str = "onnx"
    input_size: tuple[int, int] = (512, 512)

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        try:
            import onnxruntime            # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "the ONNX backend needs onnxruntime; install it with "
                "'pip install onnxruntime', or use SpectralBackend") from exc
        raise NotImplementedError(
            "no trained checkpoint ships with this project. Implement tiled "
            "inference here against your own model: normalise to the training "
            "statistics, run in overlapping tiles of self.input_size, and "
            "return a float32 (H, W) probability in [0, 1].")


# --- results -------------------------------------------------------------------
@dataclass
class ImageExtractionResult:
    footprints: list                      # shapely Polygons in local ENU metres
    probability: np.ndarray               # (H, W) in [0, 1]
    mask: np.ndarray                      # (H, W) bool, after cleanup
    mode: str
    backend: str
    params: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "footprints": len(self.footprints),
            "total_area_m2": round(sum(f.area for f in self.footprints), 2),
            "building_pixel_fraction": round(float(self.mask.mean()), 4),
            "params": self.params,
        }


def _pixel_to_world(rows, cols, meta) -> tuple[np.ndarray, np.ndarray]:
    """
    Map image row/col to local ENU metres.

    The orthophoto sidecar stores its extent and resolution; row 0 of the array
    is the minimum y, because `io.read_orthophoto` flips the north-up file into
    the same south-up convention the rest of the platform uses.
    """
    res = meta["resolution_m"]
    x0, y0 = meta["extent_local_m"][0], meta["extent_local_m"][1]
    return x0 + (np.asarray(cols) + 0.5) * res, y0 + (np.asarray(rows) + 0.5) * res


def extract_from_image(rgb: np.ndarray, meta: dict, *,
                       ndsm: Optional[Grid] = None,
                       backend: Optional[SegmentationBackend] = None,
                       threshold: float = 0.5,
                       min_height: float = 2.5,
                       min_area_m2: float = 30.0,
                       open_px: int = 2,
                       close_px: int = 4,
                       max_trace_points: int = 3500,
                       regularize: bool = True) -> ImageExtractionResult:
    """
    Extract building footprints from an orthophoto, optionally fused with height.

    Fusion is a product of evidence, not a sum: a pixel must be *both*
    building-like in appearance *and* raised above the terrain. That is
    deliberately conservative - it rejects a grey forecourt (right colour, no
    height) and a tree (right height, wrong colour), which are the two errors
    that matter. A sum would let either one through on its own.
    """
    backend = backend or SpectralBackend()
    prob = backend.probability(rgb)
    mode = "image"

    if ndsm is not None:
        h, w = prob.shape
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        xs, ys = _pixel_to_world(rr, cc, meta)
        height = ndsm.sample(xs.ravel(), ys.ravel(), default=0.0).reshape(h, w)
        height = np.nan_to_num(height, nan=0.0)
        # soft height gate, matching the point-cloud stage's transition width
        height_p = 1.0 / (1.0 + np.exp(-(height - min_height) / 0.5))
        prob = prob * height_p
        mode = "fused"

    mask = prob > threshold

    # Morphological cleanup: opening removes speckle and thin bridges between
    # neighbouring roofs, closing fills the holes punched by rooftop plant,
    # skylights and antenna shadows.
    if open_px > 0:
        mask = ndimage.binary_opening(mask, structure=np.ones((open_px * 2 + 1,) * 2))
    if close_px > 0:
        mask = ndimage.binary_closing(mask, structure=np.ones((close_px * 2 + 1,) * 2))
    mask = ndimage.binary_fill_holes(mask)

    res = meta["resolution_m"]
    min_px = max(8, int(min_area_m2 / (res * res)))
    labels, n = ndimage.label(mask)

    footprints = []
    rng = np.random.default_rng(17)
    flat_labels = labels.ravel()

    for i in range(1, n + 1):
        idx = np.flatnonzero(flat_labels == i)
        if len(idx) < min_px:
            continue
        # Thin dense components before tracing. A 900 m2 roof at 20 cm/px is
        # 22,000 pixels, and the alpha shape's cost is superlinear in the point
        # count - while a 30 cm effective spacing already resolves the outline
        # far finer than the boundary is actually known. Sampling the *filled*
        # component rather than just its rim also keeps the triangulation well
        # conditioned: a one-pixel-wide rim is nearly collinear, and its
        # triangles have huge circumradii that the alpha test then discards.
        if len(idx) > max_trace_points:
            idx = rng.choice(idx, max_trace_points, replace=False)
        rows, cols = np.unravel_index(idx, labels.shape)
        xs, ys = _pixel_to_world(rows, cols, meta)
        # Reuse the point-cloud tracer, so the two sensors' footprints are
        # produced by the same routine and are directly comparable.
        spacing = res * max(1.0, np.sqrt(len(np.flatnonzero(flat_labels == i)) / len(idx)))
        poly = alpha_shape(np.column_stack([xs, ys]), alpha=max(4.0 * spacing, 1.0))
        poly = clean_polygon(poly) if poly is not None else None
        if poly is None or poly.area < min_area_m2:
            continue
        if regularize:
            poly = regularize_footprint(poly)
        footprints.append(poly)

    footprints.sort(key=lambda p: -p.area)
    return ImageExtractionResult(
        footprints=footprints, probability=prob, mask=mask, mode=mode,
        backend=backend.name,
        params={
            "threshold": threshold, "min_height_m": min_height,
            "min_area_m2": min_area_m2, "open_px": open_px,
            "close_px": close_px, "regularised": regularize,
            "max_trace_points": max_trace_points,
            "resolution_m": res, "fused_with_ndsm": ndsm is not None,
        },
    )


def agreement(image_footprints: list, lidar_footprints: list,
              *, iou_threshold: float = 0.5) -> dict:
    """
    Cross-check two independent extractions against each other.

    Two sensors agreeing is real evidence; the same sensor asserting something
    twice is not. Where imagery and LiDAR both find a building, the footprint
    can be published with high confidence. Where only one does, the object is
    flagged for a surveyor - and that flag is far more useful than a single
    detector's own confidence score, which is only ever a statement about its
    own internal consistency.
    """
    matches = []
    used = set()
    for i, a in enumerate(image_footprints):
        best, best_iou = None, 0.0
        for j, b in enumerate(lidar_footprints):
            if j in used:
                continue
            inter = a.intersection(b).area
            if inter <= 0:
                continue
            iou = inter / a.union(b).area
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None and best_iou >= iou_threshold:
            used.add(best)
            matches.append({"image_index": i, "lidar_index": best,
                            "iou": round(best_iou, 4)})

    return {
        "image_footprints": len(image_footprints),
        "lidar_footprints": len(lidar_footprints),
        "agreed": len(matches),
        "image_only": len(image_footprints) - len(matches),
        "lidar_only": len(lidar_footprints) - len(matches),
        "mean_iou": round(float(np.mean([m["iou"] for m in matches])), 4)
        if matches else 0.0,
        "matches": matches,
    }

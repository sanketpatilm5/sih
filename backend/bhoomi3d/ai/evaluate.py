"""
Scoring the extraction against ground truth.

An automated cadastral pipeline that cannot state its own accuracy is not
usable for a legal record - "the buildings look right" is not a standard of
evidence. Because the demonstration scene is synthetic we know the true
footprint, height and storey count of every structure, so every stage can be
given a number.

Metrics follow the ISPRS benchmark convention for building extraction:
detection is scored per object at an IoU threshold, geometry is scored as mean
IoU over matched pairs, and heights are reported as signed error so systematic
bias (e.g. mistaking a parapet for the roof) is visible rather than averaged
away.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon

from ..data.scene import Scene


@dataclass
class MatchResult:
    """One truth/prediction pairing."""

    truth_id: str
    pred_index: int | None
    iou: float
    area_truth: float
    area_pred: float
    height_truth: float
    height_pred: float
    floors_truth: int
    floors_pred: int | None = None

    @property
    def height_error(self) -> float:
        return self.height_pred - self.height_truth

    def as_dict(self) -> dict:
        return {
            "truth_id": self.truth_id,
            "pred_index": self.pred_index,
            "iou": round(self.iou, 4),
            "area_truth_m2": round(self.area_truth, 2),
            "area_pred_m2": round(self.area_pred, 2),
            "area_error_pct": round(
                100 * (self.area_pred - self.area_truth) / max(self.area_truth, 1e-9), 2),
            "height_truth_m": round(self.height_truth, 3),
            "height_pred_m": round(self.height_pred, 3),
            "height_error_m": round(self.height_error, 3),
            "floors_truth": self.floors_truth,
            "floors_pred": self.floors_pred,
            "floors_correct": (self.floors_pred == self.floors_truth
                               if self.floors_pred is not None else None),
        }


@dataclass
class ExtractionScore:
    matches: list[MatchResult]
    n_truth: int
    n_pred: int
    iou_threshold: float
    false_positives: list[int] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def detected(self) -> list[MatchResult]:
        return [m for m in self.matches if m.pred_index is not None
                and m.iou >= self.iou_threshold]

    def as_dict(self) -> dict:
        det = self.detected
        tp = len(det)
        precision = tp / self.n_pred if self.n_pred else 0.0
        recall = tp / self.n_truth if self.n_truth else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        ious = [m.iou for m in det]
        herr = [m.height_error for m in det]
        aerr = [abs(m.area_pred - m.area_truth) / max(m.area_truth, 1e-9) for m in det]
        floors_scored = [m for m in det if m.floors_pred is not None]
        return {
            "n_truth": self.n_truth,
            "n_predicted": self.n_pred,
            "true_positives": tp,
            "false_positives": len(self.false_positives),
            "missed": self.missed,
            "iou_threshold": self.iou_threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "min_iou": round(float(np.min(ious)), 4) if ious else 0.0,
            "mean_abs_area_error_pct": round(float(np.mean(aerr)) * 100, 2) if aerr else 0.0,
            "height_bias_m": round(float(np.mean(herr)), 3) if herr else 0.0,
            "height_rmse_m": round(float(np.sqrt(np.mean(np.square(herr)))), 3) if herr else 0.0,
            "floor_count_accuracy": (
                round(sum(1 for m in floors_scored
                          if m.floors_pred == m.floors_truth) / len(floors_scored), 4)
                if floors_scored else None),
            "per_building": [m.as_dict() for m in self.matches],
        }


def score_extraction(scene: Scene, buildings, *, iou_threshold: float = 0.5,
                     floors_by_pred_index: dict[int, int] | None = None
                     ) -> ExtractionScore:
    """
    Match extracted buildings to the scene's ground truth and score them.

    Matching is greedy by descending IoU, which is the standard evaluation
    protocol: each truth object may claim at most one prediction and vice versa,
    so a pipeline cannot inflate recall by emitting many overlapping guesses.
    """
    truths = []
    for b in scene.buildings:
        poly = Polygon(b.world_footprint())
        ground = float(scene.terrain.height(*b.origin))
        truths.append({
            "id": b.id,
            "poly": poly,
            "height": b.height_above_ground,
            "floors": len(b.above_ground_floors),
            "ground": ground,
        })

    preds = [{"index": i, "poly": b.footprint, "height": b.height}
             for i, b in enumerate(buildings)]

    # all pairwise IoUs, then greedy assignment
    pairs = []
    for ti, t in enumerate(truths):
        for p in preds:
            inter = t["poly"].intersection(p["poly"]).area
            if inter <= 0:
                continue
            union = t["poly"].union(p["poly"]).area
            pairs.append((inter / union, ti, p["index"]))
    pairs.sort(reverse=True)

    used_t: set[int] = set()
    used_p: set[int] = set()
    matches: list[MatchResult] = []
    for iou, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        t = truths[ti]
        p = next(x for x in preds if x["index"] == pi)
        matches.append(MatchResult(
            truth_id=t["id"], pred_index=pi, iou=iou,
            area_truth=t["poly"].area, area_pred=p["poly"].area,
            height_truth=t["height"], height_pred=p["height"],
            floors_truth=t["floors"],
            floors_pred=(floors_by_pred_index or {}).get(pi),
        ))

    missed = []
    for ti, t in enumerate(truths):
        if ti not in used_t:
            missed.append(t["id"])
            matches.append(MatchResult(
                truth_id=t["id"], pred_index=None, iou=0.0,
                area_truth=t["poly"].area, area_pred=0.0,
                height_truth=t["height"], height_pred=0.0,
                floors_truth=t["floors"]))

    false_positives = [p["index"] for p in preds if p["index"] not in used_p]
    # a match below the IoU threshold counts as both a miss and a false positive
    for m in list(matches):
        if m.pred_index is not None and m.iou < iou_threshold:
            missed.append(m.truth_id)
            false_positives.append(m.pred_index)

    return ExtractionScore(
        matches=matches, n_truth=len(truths), n_pred=len(preds),
        iou_threshold=iou_threshold,
        false_positives=sorted(set(false_positives)), missed=sorted(set(missed)),
    )


def score_ground_filter(true_classification: np.ndarray,
                        predicted_ground: np.ndarray) -> dict:
    """
    Kappa and the two error types used in the ISPRS ground-filtering benchmark.

    Type I error (ground rejected as object) thins the terrain model; Type II
    (object accepted as ground) drags buildings and trees down into it. They are
    not symmetric in consequence, so they are reported separately rather than
    collapsed into one accuracy figure.
    """
    truth = np.asarray(true_classification) == 2
    pred = np.asarray(predicted_ground, dtype=bool)
    n = len(truth)
    if n == 0:
        return {}
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())

    accuracy = (tp + tn) / n
    expected = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "total_points": n,
        "type_I_error_pct": round(100 * fn / max(tp + fn, 1), 3),
        "type_II_error_pct": round(100 * fp / max(fp + tn, 1), 3),
        "total_error_pct": round(100 * (fp + fn) / n, 3),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "kappa": round(kappa, 4),
    }

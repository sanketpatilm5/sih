"""
The end-to-end pipeline: sensor data in, validated 3D register out.

    LiDAR / DEM / imagery -----+
    GNSS-CORS control ---------+
    GIS parcel layer ----------+--> [ 8 stages ] --> Cadastre + ValidationReport
    Approved floor plans ------+
    Utility corridor records --+

Each stage is a pure function over the previous stage's output, and every stage
records what it did and how confident it is. Two consequences matter:

* the register can always say *where each boundary came from* - a surveyed
  corner, an approved plan, or an inference - which is what makes the output
  admissible as a land record rather than a visualisation;
* any stage can be replaced (a trained segmentation model for the classical
  one, real LAS files for the simulator) without touching the others.

The fusion rule throughout is that **the legal document wins**. Where an
approved floor plan exists, its unit boundaries and slab levels are used
verbatim and the scan-derived estimate is retained only for comparison. The
scan's job is to find what the paper record does not know: unauthorised
storeys, encroachments, and where the buried infrastructure actually runs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from shapely.geometry import Polygon

from .ai.building_extract import BuildingCandidate, extract_buildings
from .ai.evaluate import score_extraction, score_ground_filter
from .ai.floor_segment import (compare_storeys, segment_storeys,
                               select_facade_points, storeys_from_floor_plan)
from .ai.ground_filter import smrf
from .core.cadastre import Cadastre, CadastralObject, ObjectKind, Provenance
from .core.crs import ControlPoint, GeodeticOrigin, LocalENU, fit_helmert
from .core.geometry3d import Prism, clean_polygon, sweep_corridor
from .core.topology import Severity, validate
from .core.ulpin import JurisdictionCode, Stratum

# Height of the air-rights block granted above a structure. In Indian practice
# this follows from the permissible FSI and the local height limit; 30 m is a
# stand-in for that rule, and is carried as an attribute so it is auditable.
AIR_RIGHTS_HEIGHT_M = 30.0


@dataclass
class StageLog:
    name: str
    seconds: float
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"stage": self.name, "seconds": round(self.seconds, 3),
                **self.detail}


@dataclass
class PipelineResult:
    cadastre: Cadastre
    report: object                      # ValidationReport
    enu: LocalENU
    stages: list[StageLog]
    metrics: dict = field(default_factory=dict)
    terrain: object = None              # GroundFilterResult
    extraction: object = None           # ExtractionResult

    def as_dict(self) -> dict:
        return {
            "stats": self.cadastre.stats(),
            "validation": self.report.as_dict(),
            "stages": [s.as_dict() for s in self.stages],
            "metrics": self.metrics,
            "crs": self.enu.as_dict(),
        }


class _Timer:
    def __init__(self, stages: list[StageLog], name: str, progress=None):
        self.stages, self.name, self.progress = stages, name, progress
        self.detail: dict = {}

    def __enter__(self):
        self.t0 = time.perf_counter()
        if self.progress:
            self.progress(self.name, "start", {})
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        self.stages.append(StageLog(self.name, dt, self.detail))
        if self.progress:
            self.progress(self.name, "done", {"seconds": round(dt, 3), **self.detail})
        return False


def run_pipeline(*,
                 points: np.ndarray,
                 origin: dict,
                 jurisdiction: dict,
                 site_name: str = "site",
                 control_points: Optional[list[dict]] = None,
                 floor_plans: Optional[list[dict]] = None,
                 parcels_geojson: Optional[dict] = None,
                 corridors: Optional[list[dict]] = None,
                 orthophoto: Optional[tuple] = None,
                 return_number: Optional[np.ndarray] = None,
                 truth_scene=None,
                 truth_classification: Optional[np.ndarray] = None,
                 ground_cell: float = 0.75,
                 max_building_radius_m: float = 22.0,
                 progress: Optional[Callable[[str, str, dict], None]] = None
                 ) -> PipelineResult:
    """
    Run every stage and return the finished register.

    `truth_scene` and `truth_classification` are optional. When supplied (they
    are, for the synthetic demonstration site) the pipeline additionally scores
    itself against ground truth, so the output carries measured accuracy rather
    than an assertion of it.
    """
    stages: list[StageLog] = []
    metrics: dict = {}
    pts = np.asarray(points, dtype=float)

    enu = LocalENU(GeodeticOrigin(
        lat=origin["lat"], lon=origin["lon"],
        height=origin.get("height", 0.0), label=origin.get("label", "origin")))
    juris = JurisdictionCode(**jurisdiction)
    cad = Cadastre(juris, origin, site_name)

    # --- stage 1: GNSS / CORS datum adjustment ----------------------------
    with _Timer(stages, "gnss_adjustment", progress) as t:
        if control_points:
            cps = [ControlPoint(name=c["name"], observed=tuple(c["observed"]),
                                reference=tuple(c["reference"]),
                                sigma=c.get("sigma", 0.02))
                   for c in control_points]
            fit = fit_helmert(cps)
            pts = fit.apply(pts)
            metrics["gnss_adjustment"] = fit.as_dict()
            t.detail = {"control_points": len(cps), "rms_m": round(fit.rms, 4),
                        "rejected": fit.rejected}
        else:
            t.detail = {"control_points": 0,
                        "note": "no ground control supplied; the block keeps its "
                                "delivered datum and absolute accuracy is unverified"}

    # --- stage 2: ground filtering ----------------------------------------
    with _Timer(stages, "ground_filter", progress) as t:
        ground = smrf(pts, cell=ground_cell, max_window_m=max_building_radius_m)
        t.detail = {"ground_points": int(ground.is_ground.sum()),
                    "total_points": int(len(pts))}
        if truth_classification is not None:
            metrics["ground_filter"] = score_ground_filter(
                truth_classification, ground.is_ground)

    # --- stage 3: building extraction -------------------------------------
    with _Timer(stages, "building_extraction", progress) as t:
        extraction = extract_buildings(
            pts, ground.height_above_ground, ground.dtm,
            return_number=return_number)
        t.detail = {"buildings": len(extraction.buildings)}

    # --- stage 3b: independent extraction from imagery, as a cross-check --
    # Two sensors agreeing is evidence; one sensor repeating itself is not. The
    # image path is fused with the nDSM (imagery alone cannot tell a roof from
    # a forecourt - see ai.image_segment) and its footprints are then matched
    # against the LiDAR ones, so any structure found by only one sensor is
    # surfaced for a surveyor rather than silently trusted.
    if orthophoto is not None:
        with _Timer(stages, "image_extraction", progress) as t:
            from .ai.image_segment import agreement, extract_from_image
            rgb, ortho_meta = orthophoto
            img = extract_from_image(rgb, ortho_meta, ndsm=ground.ndsm)
            agree = agreement(img.footprints,
                              [b.footprint for b in extraction.buildings])
            metrics["image_extraction"] = {**img.summary(),
                                           "cross_sensor": agree}
            t.detail = {"image_footprints": len(img.footprints),
                        "agreed_with_lidar": agree["agreed"],
                        "mean_iou": agree["mean_iou"]}

    # --- stage 4: reconcile with the existing 2D cadastral layer ----------
    with _Timer(stages, "surface_parcels", progress) as t:
        parcel_objs = _add_surface_parcels(cad, enu, parcels_geojson, ground)
        t.detail = {"parcels": len(parcel_objs)}

    # --- stage 5: storey segmentation + floor-plan fusion -----------------
    plans_by_outline = _index_plans(floor_plans or [])
    building_records = []
    with _Timer(stages, "storey_segmentation", progress) as t:
        for cand in extraction.buildings:
            facade = select_facade_points(pts, cand.footprint,
                                          cand.ground_z + 0.4, cand.eave_z - 0.2)
            estimated = segment_storeys(facade, cand.ground_z, cand.eave_z)
            plan = _match_plan(cand, plans_by_outline)
            authoritative = (storeys_from_floor_plan(plan, cand.ground_z)
                             if plan else None)
            building_records.append({
                "candidate": cand, "estimated": estimated,
                "plan": plan, "authoritative": authoritative,
            })
        t.detail = {"buildings": len(building_records),
                    "with_approved_plan": sum(1 for r in building_records
                                              if r["plan"])}

    # --- stage 6: volumetric parcel delineation + ULPIN minting -----------
    with _Timer(stages, "volumetric_delineation", progress) as t:
        counts = _build_volumetric_objects(cad, enu, building_records,
                                           parcel_objs, ground)
        t.detail = counts

    # --- stage 7: subsurface infrastructure -------------------------------
    with _Timer(stages, "infrastructure", progress) as t:
        n_infra = _add_corridors(cad, enu, corridors or [])
        t.detail = {"corridors": n_infra}

    # --- stage 8: validation ----------------------------------------------
    with _Timer(stages, "validation", progress) as t:
        report = validate(cad)
        t.detail = {"findings": len(report.findings),
                    "errors": len(report.by_severity(Severity.ERROR)),
                    "warnings": len(report.by_severity(Severity.WARNING))}

    # --- accuracy against ground truth ------------------------------------
    if truth_scene is not None:
        floors_by_index = {
            r["candidate"].index: r["estimated"].n_floors
            for r in building_records
        }
        metrics["building_extraction"] = score_extraction(
            truth_scene, extraction.buildings,
            floors_by_pred_index=floors_by_index).as_dict()

    metrics["storey_comparison"] = [
        {"building": r["candidate"].index,
         "plan_id": (r["plan"] or {}).get("building_id"),
         **compare_storeys(r["estimated"], r["authoritative"])}
        for r in building_records if r["authoritative"]
    ]

    return PipelineResult(cadastre=cad, report=report, enu=enu, stages=stages,
                          metrics=metrics, terrain=ground, extraction=extraction)


# --- stage helpers -------------------------------------------------------------
def _add_surface_parcels(cad: Cadastre, enu: LocalENU,
                         parcels_geojson: Optional[dict], ground) -> list:
    """
    Import the existing 2D cadastral layer and give each parcel a z-extent.

    A legacy parcel is a flat polygon. To take part in 3D queries it needs
    depth, so it is extruded from a nominal subsurface limit to the terrain
    surface above it. The extents are marked `DERIVED` while the plan geometry
    keeps its original `EXISTING_RECORD` provenance - the horizontal boundary is
    still the authoritative one from the land record; only the vertical extent
    is an assumption we are making.
    """
    out = []
    if not parcels_geojson:
        return out

    for feat in parcels_geojson.get("features", []):
        props = feat.get("properties", {})
        coords = feat["geometry"]["coordinates"]
        ring = enu.forward_ring(coords[0])
        poly = clean_polygon(Polygon(ring))
        if poly is None:
            continue
        # terrain surface across the parcel sets the top of the surface stratum
        xs, ys = np.array(poly.exterior.coords).T
        z_surface = float(np.nanmedian(ground.dtm.sample(xs, ys)))
        obj = CadastralObject(
            object_id=props.get("parcel_id", f"P-{len(out)+1:03d}"),
            kind=ObjectKind.PARCEL,
            solid=Prism(poly, z_surface - 6.0, z_surface + 0.001),
            name=props.get("survey_no", ""),
            owner=props.get("owner", ""),
            use=props.get("land_use", ""),
            level=0, stratum=Stratum.SURFACE,
            provenance=Provenance.EXISTING_RECORD, confidence=1.0,
            attributes={
                "survey_no": props.get("survey_no", ""),
                "recorded_area_m2": props.get("area_m2"),
                "surface_level_m": round(z_surface, 3),
                "vertical_extent_note":
                    "subsurface limit is a nominal 6 m; the plan boundary is "
                    "from the existing land record",
            },
        )
        cad.add(obj, enu=enu)
        out.append(obj)
    return out


def _index_plans(plans: list[dict]) -> list[dict]:
    for p in plans:
        p["_poly"] = Polygon(p["outline"])
    return plans


def _match_plan(cand: BuildingCandidate, plans: list[dict]) -> Optional[dict]:
    """
    Match an extracted structure to an approved plan by footprint overlap.

    Matching on geometry rather than on an identifier is deliberate: the whole
    difficulty in practice is that the scan has no idea which record it is
    looking at, and a plan may be registered against a survey number whose
    boundary has since changed.
    """
    best, best_iou = None, 0.0
    for p in plans:
        poly = p["_poly"]
        inter = cand.footprint.intersection(poly).area
        if inter <= 0:
            continue
        iou = inter / cand.footprint.union(poly).area
        if iou > best_iou:
            best, best_iou = p, iou
    return best if best_iou > 0.35 else None


def _owning_parcel(footprint: Polygon, parcels: list) -> Optional[str]:
    best, best_area = None, 0.0
    for p in parcels:
        a = footprint.intersection(p.footprint).area
        if a > best_area:
            best, best_area = p, a
    return best.object_id if best else None


def _build_volumetric_objects(cad: Cadastre, enu: LocalENU, records: list,
                              parcels: list, ground) -> dict:
    """Create the building / storey / unit / air-rights hierarchy."""
    counts = {"buildings": 0, "storeys": 0, "units": 0, "air_rights": 0,
              "units_from_plan": 0, "units_inferred": 0}

    for rec in records:
        cand: BuildingCandidate = rec["candidate"]
        plan = rec["plan"]
        storeys = (rec["authoritative"] or rec["estimated"])
        est = rec["estimated"]

        bid = plan["building_id"] if plan else f"B-{cand.index + 1:02d}"
        bname = plan["building_name"] if plan else f"Structure {cand.index + 1}"
        parent_parcel = _owning_parcel(cand.footprint, parcels)

        # --- the building envelope ---------------------------------------
        z_bottom = min([s.z_bottom for s in storeys.storeys] + [cand.ground_z])
        building = CadastralObject(
            object_id=bid, kind=ObjectKind.BUILDING,
            solid=Prism(cand.footprint, z_bottom, cand.parapet_z),
            parent_id=parent_parcel, name=bname,
            owner=(plan or {}).get("owner", ""),
            level=0, stratum=Stratum.SURFACE,
            provenance=Provenance.EXTRACTED, confidence=cand.confidence,
            attributes={
                "ground_level_m": round(cand.ground_z, 3),
                "eave_level_m": round(cand.eave_z, 3),
                "roof_level_m": round(cand.roof_z, 3),
                "parapet_level_m": round(cand.parapet_z, 3),
                "habitable_height_m": round(cand.height, 3),
                "structural_height_m": round(cand.structural_height, 3),
                "storeys_above_ground": sum(1 for s in storeys.storeys if s.index >= 0),
                "storeys_below_ground": sum(1 for s in storeys.storeys if s.index < 0),
                "roof_form": cand.metrics.get("roof_form"),
                "roof_planes": cand.roof_planes,
                "extraction_metrics": cand.metrics,
                "storey_source": storeys.method,
                "storey_estimate_from_scan": est.n_floors,
                "floor_height_m": round(storeys.floor_height, 3),
                "facade_periodicity": round(est.periodicity_strength, 3),
            },
        )
        cad.add(building, enu=enu)
        counts["buildings"] += 1

        # --- storeys -------------------------------------------------------
        plan_floors = {int(f["index"]): f for f in (plan or {}).get("floors", [])}
        for st in storeys.storeys:
            sid = f"{bid}/L{st.index:+03d}"
            storey_obj = CadastralObject(
                object_id=sid, kind=ObjectKind.STOREY,
                solid=Prism(cand.footprint, st.z_bottom, st.z_top),
                parent_id=bid, name=_storey_name(st.index),
                level=st.index,
                stratum=Stratum.UNDERGROUND if st.index < 0 else Stratum.BUILDING,
                provenance=(Provenance.APPROVED_PLAN if st.source == "floor_plan"
                            else Provenance.EXTRACTED if st.source == "estimated"
                            else Provenance.ASSUMED),
                confidence=st.confidence,
                use=plan_floors.get(st.index, {}).get("use", ""),
                attributes={"floor_height_m": round(st.height, 3),
                            "level_source": st.source},
            )
            cad.add(storey_obj, enu=enu)
            counts["storeys"] += 1

            # --- units -----------------------------------------------------
            floor = plan_floors.get(st.index)
            if floor and floor.get("units"):
                for u in floor["units"]:
                    poly = clean_polygon(Polygon(u["ring"]))
                    if poly is None:
                        continue
                    uid = f"{sid}/{u['name']}"
                    unit = CadastralObject(
                        object_id=uid, kind=ObjectKind.UNIT,
                        solid=Prism(poly, st.z_bottom, st.z_top),
                        parent_id=sid, name=u["name"],
                        owner=u.get("owner", ""), use=u.get("kind", ""),
                        level=st.index,
                        stratum=(Stratum.UNDERGROUND if st.index < 0
                                 else Stratum.BUILDING),
                        provenance=Provenance.APPROVED_PLAN, confidence=1.0,
                        attributes={"carpet_area_m2": round(poly.area, 3)},
                    )
                    cad.add(unit, enu=enu,
                            preferred_unit=u.get("preferred_unit_no"))
                    counts["units"] += 1
                    counts["units_from_plan"] += 1
            else:
                # No approved plan for this storey. We do NOT invent unit
                # boundaries - subdividing a floor into flats we have never
                # seen would be fabricating legal boundaries. The storey is
                # registered as a single unallocated volume instead, and the
                # provenance says so.
                uid = f"{sid}/UNALLOCATED"
                unit = CadastralObject(
                    object_id=uid, kind=ObjectKind.UNIT,
                    solid=Prism(cand.footprint, st.z_bottom, st.z_top),
                    parent_id=sid, name="Unallocated floor volume",
                    use="unsurveyed", level=st.index,
                    stratum=(Stratum.UNDERGROUND if st.index < 0
                             else Stratum.BUILDING),
                    provenance=Provenance.EXTRACTED,
                    confidence=min(st.confidence, 0.5),
                    attributes={
                        "note": "no approved floor plan on record for this "
                                "storey; the volume is registered whole and "
                                "awaits subdivision from a sanctioned plan",
                    },
                )
                cad.add(unit, enu=enu)
                counts["units"] += 1
                counts["units_inferred"] += 1

        # --- air rights ----------------------------------------------------
        air = CadastralObject(
            object_id=f"{bid}/AIR", kind=ObjectKind.AIR_RIGHTS,
            solid=Prism(cand.footprint, cand.parapet_z,
                        cand.parapet_z + AIR_RIGHTS_HEIGHT_M),
            parent_id=parent_parcel, name=f"Air rights above {bname}",
            owner=(plan or {}).get("owner", ""), level=0,
            stratum=Stratum.AIR_RIGHTS, provenance=Provenance.DERIVED,
            confidence=0.9,
            attributes={
                "basis": "development-rights column above the built envelope",
                "column_height_m": AIR_RIGHTS_HEIGHT_M,
                "note": "extent follows the permissible height limit and is a "
                        "policy parameter, not a measurement",
            },
        )
        cad.add(air, enu=enu)
        counts["air_rights"] += 1

    return counts


def _storey_name(index: int) -> str:
    if index < 0:
        return f"Basement {abs(index)}"
    if index == 0:
        return "Ground floor"
    return f"Floor {index}"


def _add_corridors(cad: Cadastre, enu: LocalENU, corridors: list[dict]) -> int:
    """Register subsurface utility and transport corridors as volumetric objects."""
    n = 0
    for c in corridors:
        solid = sweep_corridor(
            c["alignment"], width=c["width_m"], height=c["height_m"],
            invert_levels=c["invert_levels_m"], kind=c["kind"])
        obj = CadastralObject(
            object_id=c["id"], kind=ObjectKind.INFRASTRUCTURE,
            solid=solid, name=c["name"], owner=c.get("owner", ""),
            use=c["kind"], level=0, stratum=Stratum.INFRASTRUCTURE,
            provenance=Provenance.EXISTING_RECORD, confidence=1.0,
            attributes={
                "kind": c["kind"],
                "length_m": c.get("length_m"),
                "width_m": c["width_m"],
                "height_m": c["height_m"],
                "invert_levels_m": c["invert_levels_m"],
                "min_cover_m": round(-max(c["invert_levels_m"]) - c["height_m"], 3),
            },
        )
        cad.add(obj, enu=enu)
        n += 1
    return n

"""
Generate the demonstration dataset and run the full pipeline over it.

    python scripts/build_demo.py [--clean] [--out data]

Writes into `data/`:

    raw/pointcloud.laz        simulated oblique drone-LiDAR survey
    raw/parcels.geojson       the existing 2D cadastral layer (WGS-84)
    raw/floor_plans.json      approved plans for each building
    raw/control_points.json   GNSS/CORS ground control
    raw/corridors.json        utility-authority corridor records
    raw/dem.tif, dsm.tif      elevation products (if rasterio is available)
    raw/orthophoto.png        simulated drone orthophoto
    derived/cadastre.json     the finished 3D register
    derived/validation.json   the validation report
    derived/metrics.json      measured accuracy of every AI stage
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from bhoomi3d.core.crs import GeodeticOrigin, LocalENU  # noqa: E402
from bhoomi3d.data.scene import demo_scene  # noqa: E402
from bhoomi3d.data.simulate import (simulate_control_network,  # noqa: E402
                                    simulate_corridor_records,
                                    simulate_floor_plans, simulate_lidar,
                                    simulate_parcels_geojson)
from bhoomi3d.io import (read_orthophoto, write_geotiff, write_las,  # noqa: E402
                         write_orthophoto)
from bhoomi3d.pipeline import run_pipeline  # noqa: E402


def human(n: float) -> str:
    return f"{n:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data"), help="output directory")
    ap.add_argument("--clean", action="store_true",
                    help="build a site with no injected defects")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    out = Path(args.out)
    raw, derived = out / "raw", out / "derived"
    raw.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()

    # --- 1. the site and its simulated survey ------------------------------
    print("=" * 72)
    print("  BHOOMI3D - building the demonstration dataset")
    print("=" * 72)
    scene = demo_scene(with_conflicts=not args.clean)
    print(f"\nSite: {scene.name}")
    print(f"  {scene.stats()}")
    if not args.clean:
        print("  (site carries three deliberate defects for the validator to find)")

    enu = LocalENU(GeodeticOrigin(**{k: v for k, v in scene.origin.items()}))

    print("\nSimulating the survey ...")
    t0 = time.perf_counter()
    survey = simulate_lidar(scene, seed=args.seed)
    print(f"  LiDAR: {human(survey.n)} points in {time.perf_counter()-t0:.1f}s")
    print(f"    {survey.summary()['by_class']}")
    print(f"    block datum error to be removed by GNSS control: "
          f"{survey.datum_offset}")

    control = simulate_control_network(scene, survey)
    plans = simulate_floor_plans(scene)
    parcels = simulate_parcels_geojson(scene, enu)
    corridors = simulate_corridor_records(scene)

    # --- 2. write the raw deliverables -------------------------------------
    print("\nWriting raw survey deliverables ...")
    write_las(raw / "pointcloud.laz", survey.xyz, survey.classification,
              survey.intensity, survey.return_number)
    (raw / "parcels.geojson").write_text(json.dumps(parcels, indent=1), encoding="utf-8")
    (raw / "floor_plans.json").write_text(json.dumps(plans, indent=1), encoding="utf-8")
    (raw / "control_points.json").write_text(json.dumps(control, indent=1), encoding="utf-8")
    (raw / "corridors.json").write_text(json.dumps(corridors, indent=1), encoding="utf-8")
    write_orthophoto(raw / "orthophoto.png", scene, resolution=0.20)
    print(f"  -> {raw}")

    # --- 3. run the pipeline -----------------------------------------------
    print("\nRunning the pipeline ...")

    def progress(stage: str, phase: str, detail: dict):
        if phase == "done":
            secs = detail.pop("seconds", 0)
            bits = " ".join(f"{k}={v}" for k, v in detail.items() if k != "note")
            print(f"  [{secs:6.2f}s] {stage:<24s} {bits}")

    ortho_path = raw / "orthophoto.png"
    ortho = read_orthophoto(ortho_path) if ortho_path.exists() else None

    result = run_pipeline(
        points=survey.xyz,
        origin=scene.origin,
        jurisdiction=scene.jurisdiction,
        site_name=scene.name,
        control_points=control,
        floor_plans=plans,
        parcels_geojson=parcels,
        corridors=corridors,
        return_number=survey.return_number,
        orthophoto=ortho,
        truth_scene=scene,
        truth_classification=survey.classification,
        progress=progress,
    )

    # elevation products come out of the ground filter
    write_geotiff(raw / "dem.tif", result.terrain.dtm, enu)
    write_geotiff(raw / "dsm.tif", result.terrain.dsm, enu)

    # --- 4. persist ---------------------------------------------------------
    result.cadastre.save(derived / "cadastre.json")
    (derived / "validation.json").write_text(
        json.dumps(result.report.as_dict(), indent=1), encoding="utf-8")
    (derived / "metrics.json").write_text(
        json.dumps(result.metrics, indent=1), encoding="utf-8")
    (derived / "pipeline.json").write_text(
        json.dumps(result.as_dict(), indent=1), encoding="utf-8")

    # --- 5. report ----------------------------------------------------------
    stats = result.cadastre.stats()
    print("\n" + "-" * 72)
    print("  REGISTER")
    print("-" * 72)
    print(f"  objects            {stats['objects']}")
    print(f"  by kind            {stats['by_kind']}")
    print(f"  by provenance      {stats['by_provenance']}")
    print(f"  ULPINs issued      {stats['ulpins_issued']}")
    print(f"  vertical extent    {stats['vertical_extent_m']} m")
    print(f"  total unit volume  {human(stats['total_unit_volume_m3'])} m3")

    m = result.metrics
    print("\n" + "-" * 72)
    print("  MEASURED ACCURACY (against ground truth)")
    print("-" * 72)
    if "gnss_adjustment" in m:
        g = m["gnss_adjustment"]
        print(f"  GNSS/CORS adjustment  rms={g['rms_m']} m  "
              f"scale={g['scale_ppm']} ppm  rejected={g['rejected'] or 'none'}")
    if "ground_filter" in m:
        g = m["ground_filter"]
        print(f"  Ground filter         kappa={g['kappa']}  "
              f"type-I={g['type_I_error_pct']}%  type-II={g['type_II_error_pct']}%")
    if "image_extraction" in m:
        i = m["image_extraction"]
        cs = i["cross_sensor"]
        print(f"  Imagery ({i['mode']})       {cs['agreed']}/{cs['lidar_footprints']} "
              f"footprints agree with LiDAR at IoU {cs['mean_iou']}")
    if "building_extraction" in m:
        b = m["building_extraction"]
        print(f"  Building extraction   P={b['precision']} R={b['recall']} "
              f"F1={b['f1']}  meanIoU={b['mean_iou']}")
        print(f"                        area err={b['mean_abs_area_error_pct']}%  "
              f"height RMSE={b['height_rmse_m']} m  "
              f"floor-count acc={b['floor_count_accuracy']}")
    for sc in m.get("storey_comparison", []):
        if sc.get("extra_storeys_detected"):
            print(f"  ** {sc['plan_id']}: scan finds {sc['storeys_estimated']} "
                  f"storeys, plan sanctions {sc['storeys_on_plan']} "
                  f"-> {sc['extra_storeys_detected']} unauthorised")

    rep = result.report.as_dict()
    print("\n" + "-" * 72)
    print(f"  VALIDATION  -  {'PASS' if rep['valid'] else 'FAIL'}   {rep['counts']}")
    print("-" * 72)
    for f in rep["findings"][:12]:
        print(f"  [{f['severity'].upper():<7s}] {f['rule']:<20s} {f['title']}")
        print(f"            {f['detail'][:110]}")

    print(f"\nTotal {time.perf_counter()-t_all:.1f}s.  Wrote {derived}")
    print("Start the API with:  python scripts/serve.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

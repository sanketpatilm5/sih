"""
HTTP API for the 3D cadastre.

Read paths are plain REST over the register in memory. The pipeline run is
exposed as a Server-Sent Events stream instead, because it takes tens of
seconds and a demonstration that shows each stage completing in real time is far
more convincing than a spinner followed by a result.

Everything is served in the project-local ENU frame *and* WGS-84: the viewer
wants metres, and anything that leaves the system for another GIS wants
lat/lon. Endpoints that take a position accept either.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from . import __version__
from .core.cadastre import Cadastre
from .core.crs import GeodeticOrigin, LocalENU
from .core.grid import Grid
from .core.topology import validate
from .core.ulpin import ULPIN3D, check_stability
from .io import (capabilities, cadastre_to_citygml_lite, cadastre_to_geojson,
                 read_geotiff)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
FRONTEND = ROOT / "frontend"


class SiteState:
    """
    Everything the API serves, loaded once at start-up.

    The register is small enough to hold in memory in its entirety - a few
    hundred objects - so every query is a scan over a list rather than a
    database round trip. `bhoomi3d.core.cadastre.Cadastre` is the seam where a
    PostGIS-backed store would replace this for production volumes.
    """

    def __init__(self):
        self.cadastre: Optional[Cadastre] = None
        self.enu: Optional[LocalENU] = None
        self.validation: dict = {}
        self.metrics: dict = {}
        self.pipeline: dict = {}
        self.dtm: Optional[Grid] = None
        self.dsm: Optional[Grid] = None
        self.loaded_from: Optional[str] = None
        self.error: Optional[str] = None

    def load(self, data_dir: Path = DATA) -> bool:
        derived = data_dir / "derived"
        cad_path = derived / "cadastre.json"
        if not cad_path.exists():
            self.error = (
                f"no register found at {cad_path}. Build the demonstration "
                f"dataset first:  python scripts/build_demo.py")
            return False

        self.cadastre = Cadastre.load(cad_path)
        o = self.cadastre.origin
        self.enu = LocalENU(GeodeticOrigin(
            lat=o["lat"], lon=o["lon"], height=o.get("height", 0.0),
            label=o.get("label", "origin")))

        for name, attr in (("validation", "validation"), ("metrics", "metrics"),
                           ("pipeline", "pipeline")):
            p = derived / f"{name}.json"
            if p.exists():
                setattr(self, attr, json.loads(p.read_text(encoding="utf-8")))

        for name, attr in (("dem", "dtm"), ("dsm", "dsm")):
            p = data_dir / "raw" / f"{name}.tif"
            if p.exists():
                try:
                    setattr(self, attr, read_geotiff(p, name.upper()))
                except Exception:      # a missing raster must not stop the API
                    pass

        self.loaded_from = str(data_dir)
        self.error = None
        return True

    def require(self) -> Cadastre:
        if self.cadastre is None:
            raise HTTPException(503, self.error or "no register loaded")
        return self.cadastre


STATE = SiteState()

app = FastAPI(
    title="Bhoomi3D",
    version=__version__,
    description="3D volumetric cadastre and ULPIN platform",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    STATE.load()


# --- helpers -------------------------------------------------------------------
def _resolve_xy(x: Optional[float], y: Optional[float],
                lat: Optional[float], lon: Optional[float]) -> tuple[float, float]:
    """Accept a position as either local metres or WGS-84 degrees."""
    if x is not None and y is not None:
        return x, y
    if lat is not None and lon is not None:
        return STATE.enu.forward_xy(lon, lat)
    raise HTTPException(400, "supply either x & y (local metres) or lat & lon")


def _object_payload(o, cad: Cadastre, *, with_geometry: bool = True) -> dict:
    d = o.as_dict(with_geometry=with_geometry)
    if o.ulpin:
        d["ulpin_detail"] = o.ulpin.as_dict()
    cx, cy, cz = o.centroid
    lon, lat = STATE.enu.inverse_xy(cx, cy)
    d["centroid"] = {"x": round(cx, 3), "y": round(cy, 3), "z": round(cz, 3),
                     "lat": round(lat, 8), "lon": round(lon, 8)}
    d["children"] = [c.object_id for c in cad.children(o.object_id)]
    d["ancestors"] = [a.object_id for a in cad.ancestors(o.object_id)]
    return d


# --- meta ----------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok" if STATE.cadastre else "no-data",
        "version": __version__,
        "objects": len(STATE.cadastre) if STATE.cadastre else 0,
        "data_dir": STATE.loaded_from,
        "io_capabilities": capabilities(),
        "error": STATE.error,
    }


@app.post("/api/reload")
def reload_data():
    ok = STATE.load()
    return {"reloaded": ok, "error": STATE.error,
            "objects": len(STATE.cadastre) if STATE.cadastre else 0}


@app.get("/api/site")
def site():
    """Everything the viewer needs to set up its scene."""
    cad = STATE.require()
    stats = cad.stats()
    boxes = np.array([o.bbox for o in cad.objects], dtype=float)
    extent = {
        "x_min": float(boxes[:, 0].min()), "y_min": float(boxes[:, 1].min()),
        "z_min": float(boxes[:, 2].min()), "x_max": float(boxes[:, 3].max()),
        "y_max": float(boxes[:, 4].max()), "z_max": float(boxes[:, 5].max()),
    }
    return {
        "name": cad.name,
        "stats": stats,
        "extent": extent,
        "crs": STATE.enu.as_dict(),
        "jurisdiction": cad.jurisdiction.as_dict(),
        "validation_summary": STATE.validation.get("counts", {}),
        "has_terrain": STATE.dtm is not None,
        "pipeline_stages": STATE.pipeline.get("stages", []),
    }


@app.get("/api/ground-image")
def ground_image_meta(refresh: bool = False):
    """
    An OpenStreetMap image of the site, for draping over the 3D terrain.

    Returns the extent the image covers in local metres, which is what the
    viewer needs to line it up with the buildings standing on it.
    """
    from .groundimage import build_ground_image

    cad = STATE.require()
    boxes = np.array([o.bbox for o in cad.objects], dtype=float)
    extent = {"x_min": float(boxes[:, 0].min()), "y_min": float(boxes[:, 1].min()),
              "x_max": float(boxes[:, 3].max()), "y_max": float(boxes[:, 4].max())}

    out = DATA / "derived"
    if refresh:
        for stale in ("ground.png", "ground.json"):
            (out / stale).unlink(missing_ok=True)

    try:
        meta = build_ground_image(STATE.enu, extent, out)
    except Exception as exc:
        raise HTTPException(
            502, f"could not fetch OpenStreetMap tiles: {type(exc).__name__}: {exc}")

    if meta is None:
        raise HTTPException(
            503, "no ground map available - the OpenStreetMap tile server is "
                 "unreachable or Pillow is not installed")
    return meta


@app.get("/api/ground-image.png", include_in_schema=False)
def ground_image_png():
    path = DATA / "derived" / "ground.png"
    if not path.exists():
        raise HTTPException(404, "no ground image built yet; call /api/ground-image")
    return FileResponse(path, media_type="image/png")


@app.get("/api/terrain")
def terrain(product: str = Query("dem", pattern="^(dem|dsm)$"),
            max_cells: int = 90_000):
    grid = STATE.dtm if product == "dem" else STATE.dsm
    if grid is None:
        raise HTTPException(404, f"{product.upper()} not available; it is written "
                                 f"by scripts/build_demo.py")
    return grid.as_dict(max_cells=max_cells)


# --- the register --------------------------------------------------------------
@app.get("/api/objects")
def list_objects(kind: Optional[str] = None, level: Optional[int] = None,
                 parent: Optional[str] = None, geometry: bool = True,
                 limit: int = 5000):
    cad = STATE.require()
    objs = cad.objects
    if kind:
        wanted = {k.strip() for k in kind.split(",")}
        objs = [o for o in objs if o.kind.value in wanted]
    if level is not None:
        objs = [o for o in objs if o.level == level]
    if parent:
        objs = [o for o in objs if o.parent_id == parent]
    objs = objs[:limit]
    return {"count": len(objs),
            "objects": [_object_payload(o, cad, with_geometry=geometry)
                        for o in objs]}


@app.get("/api/objects/{object_id:path}")
def get_object(object_id: str):
    cad = STATE.require()
    o = cad.get(object_id)
    if o is None:
        raise HTTPException(404, f"no object {object_id!r} in the register")
    payload = _object_payload(o, cad)
    payload["findings"] = [
        f for f in STATE.validation.get("findings", [])
        if object_id in f.get("objects", [])
    ]
    return payload


@app.get("/api/tree")
def tree():
    """The full containment hierarchy, for the viewer's layer panel."""
    cad = STATE.require()

    def node(o):
        return {
            "id": o.object_id, "kind": o.kind.value, "name": o.name or o.object_id,
            "level": o.level, "owner": o.owner,
            "ulpin": str(o.ulpin) if o.ulpin else None,
            "z_min": round(o.z_min, 2), "z_max": round(o.z_max, 2),
            "area_m2": round(o.area, 1), "volume_m3": round(o.volume, 1),
            "children": [node(c) for c in
                         sorted(cad.children(o.object_id),
                                key=lambda c: (c.kind.value, -c.level, c.object_id))],
        }

    roots = [o for o in cad.objects if not o.parent_id]
    return {"roots": [node(r) for r in
                      sorted(roots, key=lambda r: (r.kind.value, r.object_id))]}


# --- ULPIN ---------------------------------------------------------------------
@app.get("/api/ulpin/{ulpin}")
def ulpin_lookup(ulpin: str):
    """
    Resolve a 3D-ULPIN.

    The check character is verified before any lookup happens, so a mistyped
    number is rejected offline with a specific diagnosis instead of returning a
    bare 404 that leaves the user guessing whether the record exists.
    """
    cad = STATE.require()
    try:
        parsed = ULPIN3D.parse(ulpin)
    except ValueError as exc:
        raise HTTPException(400, f"invalid ULPIN: {exc}")

    obj = cad.by_ulpin(ulpin)
    payload = {"ulpin": parsed.as_dict(), "found": obj is not None}
    if obj is not None:
        payload["object"] = _object_payload(obj, cad)
        cx, cy, _ = obj.centroid
        lon, lat = STATE.enu.inverse_xy(cx, cy)
        payload["stability"] = check_stability(parsed, lat, lon)
    return payload


@app.post("/api/ulpin/validate")
def ulpin_validate(payload: dict):
    """Validate a batch of identifiers without touching the register."""
    out = []
    for text in payload.get("ulpins", []):
        try:
            p = ULPIN3D.parse(text)
            out.append({"input": text, "valid": True, "parsed": p.as_dict()})
        except ValueError as exc:
            out.append({"input": text, "valid": False, "error": str(exc)})
    return {"results": out}


# --- spatial queries -----------------------------------------------------------
@app.get("/api/column")
def column(x: Optional[float] = None, y: Optional[float] = None,
           lat: Optional[float] = None, lon: Optional[float] = None):
    """
    Everything in the vertical column through one plan position.

    This is the query the whole platform exists to answer, and the one a 2D
    cadastre cannot: stand on a square metre of ground and get back the flat
    eleven floors up, the parking bay below, the parcel itself, the air rights
    above and the metro tunnel underneath - each with its own identifier and
    owner.
    """
    cad = STATE.require()
    xx, yy = _resolve_xy(x, y, lat, lon)
    hits = cad.column_at(xx, yy)
    lon_, lat_ = STATE.enu.inverse_xy(xx, yy)
    return {
        "query": {"x": round(xx, 3), "y": round(yy, 3),
                  "lat": round(lat_, 8), "lon": round(lon_, 8)},
        "count": len(hits),
        "column": [
            {**_object_payload(o, cad, with_geometry=False),
             "span_m": [round(o.z_min, 2), round(o.z_max, 2)]}
            for o in hits
        ],
    }


@app.get("/api/at")
def at_point(z: float, x: Optional[float] = None, y: Optional[float] = None,
             lat: Optional[float] = None, lon: Optional[float] = None):
    """Which objects actually contain a given 3D point."""
    cad = STATE.require()
    xx, yy = _resolve_xy(x, y, lat, lon)
    hits = cad.at_point(xx, yy, z)
    return {"query": {"x": round(xx, 3), "y": round(yy, 3), "z": z},
            "count": len(hits),
            "objects": [_object_payload(o, cad, with_geometry=False)
                        for o in hits]}


@app.get("/api/search")
def search(q: str, limit: int = 40):
    """Free-text search over identifiers, names, owners and ULPINs."""
    cad = STATE.require()
    needle = q.strip().lower()
    if not needle:
        return {"count": 0, "results": []}

    # a well-formed ULPIN short-circuits straight to its object
    try:
        exact = cad.by_ulpin(q)
        if exact is not None:
            return {"count": 1, "results": [_object_payload(exact, cad,
                                                            with_geometry=False)]}
    except Exception:
        pass

    scored = []
    for o in cad.objects:
        hay = " ".join([o.object_id, o.name, o.owner, o.use,
                        str(o.ulpin or ""), o.kind.value]).lower()
        if needle not in hay:
            continue
        # exact id match first, then prefix, then anything
        rank = (0 if o.object_id.lower() == needle else
                1 if o.object_id.lower().startswith(needle) else
                2 if needle in (o.name or "").lower() else 3)
        scored.append((rank, o))
    scored.sort(key=lambda t: (t[0], t[1].object_id))
    return {"count": len(scored),
            "results": [_object_payload(o, cad, with_geometry=False)
                        for _, o in scored[:limit]]}


# --- validation and metrics ----------------------------------------------------
@app.get("/api/validation")
def get_validation(severity: Optional[str] = None, rule: Optional[str] = None,
                   revalidate: bool = False):
    cad = STATE.require()
    report = validate(cad).as_dict() if revalidate else STATE.validation
    if not report:
        report = validate(cad).as_dict()
    findings = report.get("findings", [])
    if severity:
        findings = [f for f in findings if f["severity"] == severity]
    if rule:
        findings = [f for f in findings if f["rule"] == rule]
    return {**report, "findings": findings}


@app.get("/api/metrics")
def get_metrics():
    """Measured accuracy of every AI stage, against ground truth."""
    return STATE.metrics or {"note": "no metrics recorded for this dataset"}


@app.get("/api/pipeline")
def get_pipeline():
    return STATE.pipeline or {"note": "no pipeline log recorded"}


# --- pipeline re-run, streamed -------------------------------------------------
@app.get("/api/pipeline/run")
async def run_pipeline_stream(with_conflicts: bool = True, seed: int = 2024):
    """
    Re-run the whole pipeline, streaming each stage as it completes.

    Server-Sent Events rather than a websocket: the traffic is one-directional
    and SSE reconnects on its own, which is the right trade for a progress feed.
    The work runs on a worker thread so the event loop stays responsive.
    """
    events: "queue.Queue[Optional[dict]]" = queue.Queue()

    def worker():
        try:
            from .data.scene import demo_scene
            from .data.simulate import (simulate_control_network,
                                        simulate_corridor_records,
                                        simulate_floor_plans, simulate_lidar,
                                        simulate_parcels_geojson)
            from .pipeline import run_pipeline

            events.put({"stage": "simulate", "phase": "start", "detail": {}})
            scene = demo_scene(with_conflicts=with_conflicts)
            enu = LocalENU(GeodeticOrigin(**scene.origin))
            survey = simulate_lidar(scene, seed=seed)
            events.put({"stage": "simulate", "phase": "done",
                        "detail": {"points": survey.n}})

            def progress(stage, phase, detail):
                events.put({"stage": stage, "phase": phase, "detail": detail})

            result = run_pipeline(
                points=survey.xyz, origin=scene.origin,
                jurisdiction=scene.jurisdiction, site_name=scene.name,
                control_points=simulate_control_network(scene, survey),
                floor_plans=simulate_floor_plans(scene),
                parcels_geojson=simulate_parcels_geojson(scene, enu),
                corridors=simulate_corridor_records(scene),
                return_number=survey.return_number,
                truth_scene=scene, truth_classification=survey.classification,
                progress=progress)

            derived = DATA / "derived"
            derived.mkdir(parents=True, exist_ok=True)
            result.cadastre.save(derived / "cadastre.json")
            (derived / "validation.json").write_text(
                json.dumps(result.report.as_dict(), indent=1), encoding="utf-8")
            (derived / "metrics.json").write_text(
                json.dumps(result.metrics, indent=1), encoding="utf-8")
            (derived / "pipeline.json").write_text(
                json.dumps(result.as_dict(), indent=1), encoding="utf-8")
            STATE.load()

            events.put({"stage": "complete", "phase": "done",
                        "detail": {"stats": result.cadastre.stats(),
                                   "validation": result.report.as_dict()["counts"],
                                   "metrics": result.metrics}})
        except Exception as exc:                       # surface it to the client
            events.put({"stage": "error", "phase": "failed",
                        "detail": {"error": f"{type(exc).__name__}: {exc}"}})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def stream():
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, events.get)
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --- 2D plan upload ------------------------------------------------------------
@app.get("/api/sample-plan")
def get_sample_plan(kind: str = Query("simple", pattern="^(simple|society)$")):
    """A ready-to-edit 2D plan, to download, change and upload back."""
    from .plan2d import sample_plan
    return JSONResponse(
        sample_plan(kind),
        headers={"Content-Disposition": f'attachment; filename="plan-{kind}.json"'})


@app.post("/api/preview-plan")
def preview_plan(payload: dict, level: int = 0):
    """
    Draw one floor of an uploaded plan, without building anything.

    A plan file is a list of coordinates, and a transposed pair is invisible in
    JSON and obvious in a drawing. This is the check before committing to a
    build - it costs milliseconds and catches the errors that would otherwise
    show up as a mangled 3D model.
    """
    from .plan2d import PlanError, expand_floors, parse_plan, render_svg
    try:
        plan = parse_plan(payload)
        levels = sorted({f["level"] for b in plan.buildings
                         for f in expand_floors(b)})
        if level not in levels and levels:
            level = levels[0]
        return {"ok": True, "summary": plan.summary(), "levels": levels,
                "level": level, "svg": render_svg(plan, level)}
    except PlanError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/build-from-plan")
def build_from_plan(payload: dict, persist: bool = True):
    """
    Turn an uploaded 2D plan into the live 3D register.

    This is the whole point of the 2D path: a municipal office that holds
    approved building plans but no survey data can still produce a validated
    volumetric register, with identifiers, on the machine in front of them.
    """
    from .plan2d import PlanError, build_register, parse_plan
    try:
        plan = parse_plan(payload)
        result = build_register(plan)
    except PlanError as exc:
        raise HTTPException(422, str(exc))

    if persist:
        derived = DATA / "derived"
        derived.mkdir(parents=True, exist_ok=True)
        result.cadastre.save(derived / "cadastre.json")
        (derived / "validation.json").write_text(
            json.dumps(result.report.as_dict(), indent=1), encoding="utf-8")
        (derived / "metrics.json").write_text(
            json.dumps(result.metrics, indent=1), encoding="utf-8")
        (derived / "pipeline.json").write_text(
            json.dumps(result.as_dict(), indent=1), encoding="utf-8")
        # the terrain rasters belong to the previous survey-based register and
        # would float over an unrelated site, so retire them with it
        for stale in ("dem.tif", "dsm.tif"):
            (DATA / "raw" / stale).unlink(missing_ok=True)
        STATE.load()

    return {
        "ok": True,
        "plan": plan.summary(),
        "stats": result.cadastre.stats(),
        "validation": result.report.as_dict()["counts"],
        "stages": [s.as_dict() for s in result.stages],
        "persisted": persist,
    }


# --- export --------------------------------------------------------------------
@app.get("/api/export/geojson")
def export_geojson(kinds: Optional[str] = None):
    cad = STATE.require()
    want = {k.strip() for k in kinds.split(",")} if kinds else None
    return JSONResponse(
        cadastre_to_geojson(cad, STATE.enu, kinds=want),
        headers={"Content-Disposition": 'attachment; filename="cadastre.geojson"'})


@app.get("/api/export/citygml")
def export_citygml():
    cad = STATE.require()
    return PlainTextResponse(
        cadastre_to_citygml_lite(cad, STATE.enu), media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="cadastre.gml"'})


@app.get("/api/export/cadastre")
def export_cadastre():
    cad = STATE.require()
    return JSONResponse(
        cad.to_document(),
        headers={"Content-Disposition": 'attachment; filename="cadastre.json"'})


# --- static frontend -----------------------------------------------------------
class RevalidatingStaticFiles(StaticFiles):
    """
    Serve the viewer with `Cache-Control: no-cache`.

    Without an explicit directive browsers fall back to heuristic caching and
    happily reuse a stale script for hours - which during development shows up
    as an edited feature that simply does nothing, with no error to chase,
    because the browser never asked the server whether the file had changed.

    `no-cache` means "revalidate before using", not "do not store": the ETag is
    still sent, so an unchanged file costs a 304 and no body. The viewer is a
    few tens of kilobytes served from localhost, so correctness is worth far
    more here than the saved round trip.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


if FRONTEND.exists():
    app.mount("/app", RevalidatingStaticFiles(directory=str(FRONTEND), html=True),
              name="app")


@app.get("/", include_in_schema=False)
def index():
    # Redirect rather than serving index.html here: the page loads its CSS and
    # modules by relative path, which only resolve correctly underneath the
    # static mount.
    if (FRONTEND / "index.html").exists():
        return RedirectResponse("/app/")
    return HTMLResponse(
        "<h1>Bhoomi3D</h1><p>API is running. The viewer was not found at "
        f"<code>{FRONTEND}</code>.</p><p>See <a href='/docs'>/docs</a>.</p>")

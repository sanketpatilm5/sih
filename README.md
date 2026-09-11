# Bhoomi3D

**A 3D volumetric cadastre and ULPIN platform.**

Today a land parcel is identified by its position on the ground. That is a
polygon, and a polygon cannot say who owns the flat on the eleventh floor, who
owns the parking bay under it, or where the metro tunnel runs beneath both. All
three occupy *the same coordinates*. On a 2D map they are simply "the same
place".

Bhoomi3D extends the Unique Land Parcel Identification Number into the vertical.
It ingests drone LiDAR, orthophotography, elevation products, GNSS/CORS control,
the existing 2D cadastral layer and approved building plans; automatically
reconstructs the built form; slices the vertical column into legally meaningful
volumes; issues each one a checksummed **3D-ULPIN**; validates the whole
register for overlaps, gaps and clearance breaches; and serves it through a REST
API and a 3D viewer.

---

## Quick start

**One command.** Creates the virtual environment, installs dependencies, builds
the demonstration dataset on first run, then starts the server.

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

macOS / Linux:

```bash
bash scripts/run.sh
```

Then open <http://127.0.0.1:8000/>. The first run takes a few minutes (package
downloads plus a one-minute dataset build); later runs start in seconds.

**Or step by step:**

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r backend/requirements.txt
```

Build the demonstration dataset and run the full pipeline over it (~30 s):

```bash
.venv/Scripts/python scripts/build_demo.py
```

Start the API and viewer:

```bash
.venv/Scripts/python scripts/serve.py
```

Then open <http://127.0.0.1:8000/> for the viewer, or `/docs` for the API.

### Two ways in

The register can be built from either of two sources, and they answer different
questions:

| | **2D plan upload** | **Drone / LiDAR survey** |
|---|---|---|
| You supply | polygons + floor heights (JSON) | LAS/LAZ cloud, orthophoto, GNSS control |
| Who has it | every building-permission office | survey contractors, SVAMITVA |
| Runtime | under a second | ~17 s per city block |
| Tells you | the **sanctioned** state | the **as-built** state |
| Finds unauthorised floors | no — nothing contradicts the plan | yes |
| Entry point | `plan2d.build_register()` | `pipeline.run_pipeline()` |

Same identifiers, same validation, same viewer — only the source of the
polygons differs. For the 2D path, generate the samples and upload one through
the viewer's **2D → 3D** tab:

```bash
.venv/Scripts/python scripts/make_samples.py
```

Or without the browser:

```bash
curl -X POST http://127.0.0.1:8000/api/build-from-plan -H "Content-Type: application/json" --data-binary @samples/plan-society.json
```

A walkthrough of that path, with the plan file and its drawing side by side, is
in [`docs/how-it-works.html`](docs/how-it-works.html).

Run the tests (~10 s):

```bash
.venv/Scripts/python -m pytest -q
```

On macOS or Linux replace `.venv/Scripts/python` with `.venv/bin/python`.

---

## What it actually does

The demonstration site is a mixed-use block in Shivajinagar, Pune: a 12-storey
housing tower with two basements, an 8-storey society block, a 5-storey
commercial plaza, a low-rise annexe with a pitched roof, mature trees, a water
main, a storm drain, and a metro tunnel running under the whole site at −17 m.

Nothing about the buildings is given to the pipeline. It is handed a point
cloud and has to work them out.

```
  drone LiDAR ──┐
  orthophoto ───┤
  GNSS/CORS  ───┼──▶  8 stages  ──▶  143-object 3D register  ──▶  viewer + API
  2D parcels ───┤                    143 3D-ULPINs
  floor plans ──┘                    validation report
```

### Measured accuracy, not asserted

Because the demonstration site is synthetic, its ground truth is known exactly,
so every stage scores itself. These numbers come from the last run and are
regenerated on every `build_demo.py`:

| Stage | Metric | Result |
|---|---|---|
| GNSS/CORS adjustment | residual RMS | **0.021 m** |
| | planted blunder | **rejected** (GCP-05) |
| Ground filtering (SMRF) | Cohen's κ | **0.991** |
| | Type I / Type II error | 0.01 % / 1.42 % |
| Building extraction | precision / recall / F1 | **1.00 / 1.00 / 1.00** |
| | mean IoU | **0.984** |
| | plan-area error | **0.88 %** |
| | height RMSE | **0.055 m** |
| Storey segmentation | floor-count accuracy | **100 %** (4/4) |
| Imagery ↔ LiDAR | footprints agreeing | **4/4** at IoU 0.86 |

Every one of these is asserted in the test suite, not just printed.

### It finds things the paper record cannot

The demonstration site carries three deliberate defects, and the validator finds
all three and nothing else:

1. **Unauthorised construction.** Shivneri Heights is built to 12 storeys but
   sanctioned for 11. The file agrees with itself perfectly; only an independent
   measurement of the built form reveals the extra floor. The pipeline recovers
   the storey count from the facade's vertical periodicity and reports ~2,700 m³
   of unregistered built volume — which it deliberately does *not* register as
   property, because an unauthorised floor has no legal existence.

2. **Overlapping ownership.** An enclosed balcony on the 4th floor of Ganga
   Residency pushes one flat 1.2 m across the party wall into its neighbour.
   Two owners, 29.9 m³ of the same space. In 2D this is invisible — both flats
   sit inside the same building outline.

3. **Clearance breach.** A re-routed storm drain passes through the commercial
   plaza's basement raft. The system reports the shortfall as a number, because
   in 3D "how far apart are these two solids" is an ordinary question with an
   exact answer.

---

## The 3D-ULPIN

```
27 - 025 - 004 - tek92et7bx - B04 - 00DY - 5
│     │     │         │        │ │    │     │
│     │     │         │        │ │    │     └── check character, ISO 7064 MOD 37,36
│     │     │         │        │ │    └──────── unit sequence within the level
│     │     │         │        │ └───────────── level index
│     │     │         │        └─────────────── stratum: G/B/U/A/I
│     │     │         └──────────────────────── geohash-10 of the centroid (1.1 m × 0.6 m)
│     │     └────────────────────────────────── tehsil / ULB code
│     └──────────────────────────────────────── district code (LGD)
└────────────────────────────────────────────── state code (LGD)
```

The design goals, and how each is met:

- **Unique across the whole column.** Horizontal identity comes from the
  geohash, vertical identity from the stratum and level, and the unit sequence
  disambiguates objects sharing a cell on one level. A flat on floor 10 and the
  metro tunnel beneath it have the same geohash and different identifiers.
- **Self-describing.** A reader can tell the state, district, tehsil, whether
  the object is above or below ground and on which level, and recover its
  approximate position, with no database lookup.
- **Error-detecting.** The check character catches **every** single-character
  substitution and **every** adjacent transposition — the two ways a clerk
  mis-keys a number off a paper record. Both properties are verified
  exhaustively in `tests/test_ulpin.py`, not sampled.
- **Stable.** A ULPIN is minted once and never changes. Re-survey drift is
  *detected and reported* against the geohash cell recorded at mint time, but
  never mutates the identifier.

Full specification: [`docs/ULPIN.md`](docs/ULPIN.md).

---

## Architecture

```
frontend/          zero-dependency WebGL2 viewer (no CDN, works offline)
backend/bhoomi3d/
  core/            geodesy, volumetric geometry, the register, validation
    crs.py         WGS-84 ellipsoid maths, local ENU, UTM, Helmert adjustment
    geometry3d.py  prisms, composite solids, exact volumes and clearances
    grid.py        georeferenced rasters (DEM/DSM/nDSM)
    ulpin.py       the identifier: geohash, check character, minting, stability
    cadastre.py    the object model and the register
    topology.py    ten validation rules
  ai/              the extraction pipeline
    ground_filter.py   SMRF bare-earth filtering
    pointfeatures.py   eigenvalue features (planarity vs scatter)
    cluster.py         DBSCAN instance segmentation, RANSAC plane fitting
    building_extract.py footprints, roof form, eave and parapet levels
    floor_segment.py   storey recovery from facade periodicity
    image_segment.py   orthophoto extraction and LiDAR fusion
    evaluate.py        scoring against ground truth
  data/            the demonstration site and its sensor simulator
  pipeline.py      the eight stages, wired together
  io.py            LAS/LAZ, GeoTIFF, GeoJSON, CityGML-flavoured export
  api.py           REST API + SSE pipeline stream
```

Design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Two decisions worth calling out

**The legal document always wins.** Where an approved floor plan exists, its
unit boundaries and slab levels are used verbatim and the scan-derived estimate
is kept only for comparison. The scan's job is to find what the paper record
does not know. Where no plan exists, the pipeline registers the storey as a
single unallocated volume rather than inventing flat boundaries — fabricating a
legal boundary is worse than admitting ignorance, so provenance is carried on
every object and the validator lists everything still awaiting ground
verification.

**Prisms, not meshes.** Almost every legal solid in a built environment is a
footprint swept between two elevations. Restricting the primitive to a prism
makes intersection volumes and clearances exact and closed-form instead of
requiring a mesh boolean library. Genuinely non-prismatic objects — a sloping
metro tunnel — are chains of prisms, which keeps every measure exact.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/site` | extent, CRS, statistics — everything the viewer needs |
| `GET /api/objects` | the register, filterable by kind, level, parent |
| `GET /api/objects/{id}` | one object with identity, lineage and findings |
| `GET /api/tree` | the full containment hierarchy |
| `GET /api/ulpin/{ulpin}` | resolve an identifier; check character verified first |
| `POST /api/ulpin/validate` | batch-validate identifiers, no lookup |
| `GET /api/column?x=&y=` | **everything in the vertical column at a position** |
| `GET /api/at?x=&y=&z=` | what contains a given 3D point |
| `GET /api/search?q=` | free text over names, owners, ids and ULPINs |
| `GET /api/validation` | the validation report, filterable |
| `GET /api/metrics` | measured accuracy of every AI stage |
| `GET /api/pipeline/run` | re-run the pipeline, streamed as Server-Sent Events |
| `GET /api/export/geojson` | 2D projection for any GIS |
| `GET /api/export/citygml` | CityGML-vocabulary export |

Positions are accepted in local ENU metres or WGS-84 degrees, interchangeably.

`GET /api/column` is the one to try first — it is the query a 2D cadastre cannot
answer at all. On the demonstration site it returns 29 objects for a single
point: the air rights, ten storeys and their flats, the building envelope, the
surface parcel, two basement levels, and the storm drain underneath.

---

## Demonstration walkthrough

1. **Open the viewer.** The register loads: 143 objects, 143 ULPINs, spanning
   −17 m to +68 m.
2. **Click a building**, then a floor, then a flat. The inspector shows its
   3D-ULPIN decoded field by field, its plan area, its volume, its z-range, its
   owner, and whether its boundary was surveyed, approved or inferred.
3. **Turn on Column probe** and click anywhere. The vertical stack appears as a
   scale strip — click any band to jump to that object.
4. **Open the Validation tab.** Five errors, each with the measured quantity
   that makes it an error. "Show in 3D" flies the camera to the offending
   geometry and outlines it in red.
5. **Open the Accuracy tab** to see every stage scored against ground truth.
6. **Press "Re-run AI pipeline"** to watch all eight stages execute live over
   Server-Sent Events.

---

## Working with real data

The simulator exists so the pipeline can be *scored*; it is not load-bearing.
To run against a real survey, call `run_pipeline()` with your own inputs:

```python
from bhoomi3d.io import read_pointcloud, read_geojson
from bhoomi3d.pipeline import run_pipeline

cloud = read_pointcloud("survey.laz")
result = run_pipeline(
    points=cloud["xyz"],
    return_number=cloud.get("return_number"),
    origin={"lat": 18.5204, "lon": 73.8567, "height": 560.0},
    jurisdiction={"state": "27", "district": "025", "tehsil": "004"},
    control_points=json.load(open("gcp.json")),
    parcels_geojson=read_geojson("parcels.geojson"),
    floor_plans=json.load(open("plans.json")),
    corridors=json.load(open("utilities.json")),
)
result.cadastre.save("cadastre.json")
```

Every stage's parameters have physical meaning and are documented at their call
site — `max_building_radius_m` must exceed the widest building on site,
`cluster_eps` must sit between the roof point spacing and the narrowest setback,
and so on.

---

## Honest limitations

- **The demonstration data is synthetic.** The accuracy figures are real
  measurements against known truth, but on simulated sensors. They establish
  that the algorithms are correct, not that they are calibrated for any
  particular real sensor. Real LiDAR has registration error, moving objects and
  systematic voids that the simulator does not reproduce.
- **The image segmenter is not a trained model.** It is a transparent
  vegetation-and-shadow rejector, and it is honest about the fact that RGB alone
  cannot distinguish a roof from a paved forecourt — the module demonstrates
  this failure explicitly and resolves it by fusing with height.
  `OnnxBackend` documents the interface a trained network would implement.
- **Basement extents come from records, not the scan.** An aerial survey cannot
  see below ground. Basement storeys are laid out from the municipal record and
  flagged with `ASSUMED` provenance rather than being silently invented.
- **The air-rights column height is a policy parameter**, not a measurement. It
  is carried as an auditable attribute.
- **Storage is a JSON document, not PostGIS.** Correct at demonstration scale
  (hundreds of objects, every query a list scan). `core/cadastre.Cadastre` is
  the seam a spatial database would replace for a city-scale deployment.
- **The CityGML export uses the standard's vocabulary but is not a
  conformance-checked document.** It says so in its own header.

---

## Licence

Built for Smart India Hackathon. No API keys. The only network access at
runtime is to OpenStreetMap, for the street map under the 3D scene and in the
Locate animation (© OpenStreetMap contributors). Without a connection the
ground falls back to a plain surface, and the Locate tab's **Offline** basemap
uses borders bundled with the app.

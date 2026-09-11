# Architecture

## The shape of the problem

Everything here follows from one observation: **the objects a cadastre needs to
record are volumes, and almost all of them are prisms.**

A flat, a floor slab, a basement bay, a shop, a parking space, a stretch of
pipeline, a segment of metro tunnel — each is a horizontal footprint swept
between two elevations. That is not a simplification for convenience; it is
what the legal descriptions actually say. A deed describes a boundary on a plan
and a floor level, not a triangle mesh.

Committing to the prism as the primitive is the single most consequential
decision in the codebase, and it pays off everywhere:

- **Intersection is exact and closed-form.** The overlap of two prisms is the
  2D intersection of their footprints swept over the overlap of their z-ranges.
  No mesh boolean library, no tolerance tuning, no degenerate-case handling.
- **Clearance is exact.** Horizontal and vertical separation are independent,
  so the minimum distance between two prisms is their Pythagorean combination.
  "How far is this foundation from the metro tunnel" has an exact answer.
- **The wire format is tiny.** The viewer receives coordinate rings plus a
  z-range and extrudes on the GPU — roughly an order of magnitude less data
  than a triangulated mesh.

Genuinely non-prismatic objects (a sloping tunnel, a stepped podium) are
`CompositeSolid` — an ordered chain of prisms. Every measure stays exact,
because the segments are disjoint by construction.

## Coordinate strategy

All geometry lives in a **project-local ENU frame** (East / North / Up, metres)
anchored on a geodetic origin.

This is not a shortcut. Areas, volumes, buffers and clearances are all metric
and directly meaningful; there is one place to apply the GNSS/CORS datum
adjustment; and the mapping back to WGS-84 is exact and reversible.

The conversion does the full geodetic → ECEF → ENU round trip rather than an
equirectangular approximation. That matters precisely because this is a *3D*
system: at 5 km from the origin the tangent plane stands 1.96 m above the
ellipsoid, and in a platform whose entire purpose is the Z axis, silently
absorbing two metres of curvature into every height would be indefensible.
`test_enu_accounts_for_curvature_not_a_flat_plane` asserts the offset is present
and correct.

No PROJ/pyproj dependency — the ellipsoid maths (Bowring's method, the Krüger
series for UTM) is implemented directly, so the platform installs on a locked-
down machine without a GDAL toolchain.

## The pipeline

Eight stages, each a pure function over the previous stage's output.

```
1  gnss_adjustment        Helmert fit against CORS control; Baarda data snooping
2  ground_filter          SMRF → DTM, DSM, nDSM, per-point ground classification
3  building_extraction    semantic scoring → DBSCAN instances → footprints
3b image_extraction       independent orthophoto extraction, cross-checked
4  surface_parcels        import the existing 2D cadastral layer, give it depth
5  storey_segmentation    facade periodicity → floors; approved plan overrides
6  volumetric_delineation build the hierarchy, mint 3D-ULPINs
7  infrastructure         sweep subsurface corridors into solids
8  validation             ten topology rules
```

### Why each algorithm

**SMRF for ground filtering** (Pingel et al., ISPRS 2013). A plain slope filter
shaves the tops off legitimately steep terrain. Progressive TIN densification is
accurate but slow. SMRF is raster-based, runs in under a second on a city block,
and has exactly two parameters with physical meaning — the largest object to
remove and the steepest plausible terrain — which a surveyor can reason about.

The progressive opening uses a **geometric** window schedule rather than a linear
one, because object detection depends on the *ratio* of window to object size;
linear stepping spends most of its passes on large windows that flag nothing new.
Square structuring elements rather than discs, because scipy evaluates
rectangular min/max filters separably — O(r) instead of O(r²) per cell, which
took this stage from 27 s to 1.7 s with identical output.

**Eigenvalue features for semantic classification** (Weinmann et al., ISPRS
2015). The decisive question is not "is this point high" — a tree is high too —
but "is this point part of a planar surface". For sorted local covariance
eigenvalues λ₁ ≥ λ₂ ≥ λ₃, planarity `(λ₂−λ₃)/λ₁` and sphericity `λ₃/λ₁` separate
roofs from canopy cleanly. Vegetation is the dominant false positive in every
automated cadastral pipeline and this is what removes it.

The scoring rule is a transparent linear combination rather than a learned
classifier. It needs no labelled training data, a surveyor can audit every term,
and it exposes the same interface a trained model would.

**DBSCAN for instance segmentation.** Semantic classification says which points
are "building"; a cadastre needs *individual* objects. Buildings are dense blobs
separated by low-density gaps, the count is unknown in advance, and leftover
scatter should be labelled noise rather than forced into a cluster — which is
exactly DBSCAN's contract.

Two details matter. Clustering runs on (x, y, z) with **z down-weighted to
0.35**: at 1.0 a tall building fragments into separate storeys, at 0.0 two
adjacent buildings merge through the ground plane. And it runs on a
**voxel-downsampled** cloud, because DBSCAN's neighbour graph grows with the
square of local density — on a 26 pts/m² roof the full cloud produces tens of
millions of pairs to answer a question that is metre-scale anyway.

**Alpha shapes for footprint tracing.** A convex hull bridges courtyards and
swallows the notch out of an L-shaped block, inflating every area it touches.
The alpha shape keeps only Delaunay triangles below a circumradius threshold and
traces the real outline.

**Rectilinear regularisation.** Real buildings are rectilinear; a footprint
traced from a point cloud is not. The outline is rotated onto its dominant wall
direction (recovered as a length-weighted circular mean of edge angles modulo
90°), simplified, snapped, and rotated back. Edges that are genuinely oblique
are left alone, and a regularisation that would change the area by more than a
quarter is rejected in favour of the honest traced outline.

**Autocorrelation for storey segmentation.** This is the stage that looks
impossible and is not. Building facades are strongly periodic in the vertical —
balcony slabs, window bands, string courses, floor-level service runs all repeat
at the storey pitch. So the facade's height histogram carries a periodic signal
whose fundamental period *is* the floor-to-floor height. Recover the period by
autocorrelation (restricted to physically plausible storey heights, with
parabolic sub-bin interpolation because the error compounds over a dozen
floors), recover the phase by matched-filtering a comb, then snap each slab to
its nearest local peak so an unusually tall ground floor does not force the
whole stack out of step.

On the demonstration site this recovers 12, 8, 5 and 2 storeys correctly,
including the 3.85 m commercial floors that a fixed 3 m assumption gets wrong.

### One subtle failure worth recording

Floor segmentation initially returned no periodic signal at all and silently
fell back to dividing height by 3 m. The cause: it was being run on the
*semantically filtered* points. The filter that removes vegetation keys on
planarity — and balconies are small, isolated, geometrically scattered
protrusions that score much like foliage. Filtering first destroyed precisely the
signal the stage depends on.

The fix is `select_facade_points()`, which draws from the **raw** cloud and takes
a ring around the wall line: raw because the filter is counterproductive here,
and a ring because a roof's tens of thousands of coplanar returns would otherwise
swamp the histogram. Periodicity strength went from 0.00 to 0.25–0.61 and
floor-count accuracy from 3/4 to 4/4.

The general lesson is in the docstring: a stage that is correct in isolation can
be wrong in composition, and "it produced a plausible number" is not evidence
that it worked.

### Sensor fusion, and being honest about it

The imagery path deliberately demonstrates its own limitation. RGB alone cannot
distinguish a roof from a paved forecourt — both are grey, flat, weakly
saturated and smooth. Run on the demonstration orthophoto, image-only extraction
returns **one 36,766 m² blob** covering the entire non-vegetated site: precision
0.00.

Rather than tune that away, the backend is framed as what colour can actually do
reliably without training data — **reject vegetation and shadow**, which are
exactly the two confusers height cannot resolve. Fused with the nDSM as a product
of evidence (a pixel must be both building-like *and* raised), the same code
finds **4/4 buildings at IoU 0.86**.

The two extractions then cross-check each other. Two sensors agreeing is
evidence; one sensor asserting something twice is not.

## The register

One object type, not a class per legal category.

```
Parcel  (surface, stratum G)
  ├── Building
  │     └── Storey ── Unit    (flat, shop, parking bay)
  └── AirRights

Infrastructure   (crosses parcels, so parented to none)
```

A surface parcel, a flat, a basement bay, a tunnel segment and a block of air
rights differ in their attributes and their place in the hierarchy, but they are
all *the same kind of thing*: a bounded volume with an identifier, an owner and
a parent. Modelling them uniformly is what lets one overlap check, one
containment rule and one viewer handle the entire column from tunnel invert to
air rights.

### Provenance is first-class

Every object records how its boundary came to exist:

| Provenance | Authoritative | Meaning |
|---|---|---|
| `SURVEYED` | yes | GNSS or total-station observation |
| `APPROVED_PLAN` | yes | a sanctioned building plan |
| `EXISTING_RECORD` | yes | the legacy 2D cadastral layer |
| `EXTRACTED` | no | inferred by the AI pipeline |
| `DERIVED` | no | computed from other objects |
| `ASSUMED` | no | a documented default, not a measurement |

This is not metadata for its own sake. A boundary derived from an approved plan
and one inferred from a point cloud are not equally authoritative, and a record
with legal consequences must never present the second as the first. The
`PROVENANCE_REVIEW` rule lists everything still awaiting ground verification,
prioritised by confidence.

### The fusion rule: the legal document wins

Where an approved floor plan exists, its unit boundaries and slab levels are used
verbatim; the scan-derived estimate is retained only for comparison. The scan's
job is to find what the paper record does not know.

Where no plan exists, the pipeline registers the storey as **a single
unallocated volume** rather than subdividing it into flats it has never seen.
Inventing a legal boundary is worse than admitting ignorance.

## Validation

Ten rules, each returning a machine-readable finding with the geometry of the
problem attached so the viewer can fly to it and draw it.

| Rule | Severity | Catches |
|---|---|---|
| `GEOM_*` | error | empty, invalid, zero-height or zero-volume solids |
| `ULPIN_MISSING` / `ULPIN_DUPLICATE` | error | identity defects |
| `VERT_OVERLAP` | error | two owners, one set of cubic metres |
| `PLAN_CONTAINMENT` / `VERT_CONTAINMENT` | error | a child outside its parent |
| `HIER_ORPHAN` | error | a dangling parent reference |
| `STOREY_GAP` / `STOREY_OVERLAP` | warn/error | storeys that do not meet |
| `UNASSIGNED_AREA` | info/warn | floor area no unit claims |
| `INFRA_CLEARANCE` | warn/error | a structure too close to a corridor |
| `UNAUTHORISED_STOREY` | error/warn | as-built exceeds the sanctioned plan |
| `AREA_MISMATCH` | warn | recorded area disagrees with geometry |
| `PROVENANCE_REVIEW` | info | inferred boundaries pending verification |

Two design points:

**The most important thing this validator does is *not* fire.** Two flats
sharing a footprint on different floors is the normal case in every apartment
block. A validator that flags it is worse than useless, because its users will
learn to ignore it. `test_stacked_flats_are_not_reported_as_overlapping` guards
that directly, and a `VERT_OVERLAP` requires the z-ranges to overlap too.

**Severity tracks evidence, not just consequence.** `UNAUTHORISED_STOREY` is an
error when the facade gave a clear periodic signal and only a warning when it did
not — because an enforcement action needs evidence, and the finding should say
which it has.

Pairwise checks use a sort-and-sweep interval index over x, which keeps
validation near-linear for the spatially separated objects a real site consists
of.

## Frontend

A self-contained WebGL2 renderer — no Three.js, no CDN, no network at runtime.
For this geometry (extruded prisms, flat shading, orbit camera, picking) a
custom renderer is a few hundred lines, and a demonstration that fails because
the venue's wifi is down is worth avoiding.

- **Ear-clipping triangulation with hole support**, so courtyards render as
  courtyards. Cadastral footprints have tens of vertices, so O(n²) is irrelevant.
- **Colour-index picking.** Object ids are rendered to an off-screen buffer and
  the pixel under the cursor is read back. Exact — it picks whatever the user can
  actually see, including a flat glimpsed through a translucent facade — which
  ray-casting against prisms would not be.
- **Two-pass picking**, solid geometry before translucent. Without it every
  click lands on the building envelope wrapping the thing the user aimed at:
  barely visible, but perfectly opaque to a pick buffer.
- **Vertex-shader explode and a cutaway plane**, so isolating a floor costs no
  re-upload.

The one non-obvious layout bug worth recording: a percentage-sized `<canvas>` in
normal flow inside a grid row feeds its own height back into the row, and grows
without bound — it reached 206,492 px. The canvas is absolutely positioned inside
an `overflow: hidden` parent so it sizes *from* the viewport without contributing
to it.

## Where the seams are

Deliberate extension points:

| Seam | Swap in |
|---|---|
| `ai.image_segment.SegmentationBackend` | a trained U-Net / Mask R-CNN via ONNX |
| `ai.pointfeatures.building_score` | a learned point classifier |
| `core.cadastre.Cadastre` | a PostGIS-backed store for city scale |
| `data.simulate` | real LAS/LAZ, GeoTIFF and GeoJSON via `io.py` |
| `pipeline.AIR_RIGHTS_HEIGHT_M` | the actual FSI / height-limit rule |
| `topology.DEFAULT_TOLERANCES` | jurisdiction-specific clearance rules |

Each is an interface, not a TODO. `OnnxBackend` in particular is deliberately
unimplemented rather than faked: it documents the exact contract a trained model
must satisfy, and raises a message saying so, because a stub that silently
returned plausible numbers would be worse than one that refuses.

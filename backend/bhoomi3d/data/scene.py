"""
The ground-truth description of a demonstration site.

Everything the platform ingests during the demo is *derived* from this scene -
the LiDAR cloud, the orthophoto, the DEM/DSM, the floor plans, the GNSS control
network. Keeping one authoritative description has a purpose beyond
convenience: because we know the true footprint, height and floor count of every
building, the extraction pipeline can be **scored** rather than admired. See
`bhoomi3d.ai.evaluate`.

The site is a block in Pune modelled on a typical mixed-use redevelopment
plot: a mid-rise housing society, a commercial tower, a low-rise annexe, mature
trees, a water main, a storm drain and a metro tunnel running underneath the
whole thing at depth. That last one is the case the current 2D cadastre cannot
express at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Site origin: Deccan / Shivajinagar, Pune.
SITE_ORIGIN = {"lat": 18.520430, "lon": 73.856744, "height": 560.0,
               "label": "Pune - Shivajinagar block (demo site)"}

JURISDICTION = {
    "state": "27", "district": "025", "tehsil": "004",
    "state_name": "Maharashtra", "district_name": "Pune",
    "tehsil_name": "Pune City (Shivajinagar)",
}


@dataclass
class UnitSpec:
    """One legally distinct unit on a floor."""

    name: str
    rect: tuple[float, float, float, float]   # x0, y0, x1, y1 in building-local m
    kind: str = "residential"
    owner: str = ""
    preferred_unit_no: Optional[int] = None


@dataclass
class FloorSpec:
    """A storey: its z-range in building-local metres, and the units on it."""

    index: int                 # 0 = ground floor, negative = basement
    z_bottom: float
    z_top: float
    units: list[UnitSpec] = field(default_factory=list)
    use: str = "residential"

    @property
    def height(self) -> float:
        return self.z_top - self.z_bottom


@dataclass
class BuildingSpec:
    """A structure: outline, position, storeys and roof form."""

    id: str
    name: str
    footprint: list[tuple[float, float]]      # building-local coords, metres
    origin: tuple[float, float]               # placement in the site frame
    rotation_deg: float
    floors: list[FloorSpec]
    roof: str = "flat"                        # "flat" | "gable"
    parapet: float = 1.1
    parcel_id: str = ""
    # Storeys present on the *sanctioned* plan. When this is fewer than the
    # storeys actually built, the site has unauthorised vertical construction -
    # which the scan can see and the paper record cannot.
    registered_floors: Optional[int] = None

    # -- geometry ------------------------------------------------------------
    def world_footprint(self) -> list[tuple[float, float]]:
        th = math.radians(self.rotation_deg)
        c, s = math.cos(th), math.sin(th)
        ox, oy = self.origin
        return [(ox + x * c - y * s, oy + x * s + y * c) for x, y in self.footprint]

    def world_rect(self, rect) -> list[tuple[float, float]]:
        x0, y0, x1, y1 = rect
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        th = math.radians(self.rotation_deg)
        c, s = math.cos(th), math.sin(th)
        ox, oy = self.origin
        return [(ox + x * c - y * s, oy + x * s + y * c) for x, y in corners]

    @property
    def z_bottom(self) -> float:
        return min(f.z_bottom for f in self.floors)

    @property
    def z_top(self) -> float:
        return max(f.z_top for f in self.floors)

    @property
    def above_ground_floors(self) -> list[FloorSpec]:
        return [f for f in self.floors if f.index >= 0]

    @property
    def height_above_ground(self) -> float:
        ag = self.above_ground_floors
        return max(f.z_top for f in ag) if ag else 0.0

    @property
    def mean_floor_height(self) -> float:
        ag = self.above_ground_floors
        return float(np.mean([f.height for f in ag])) if ag else 0.0


@dataclass
class CorridorSpec:
    """A subsurface linear utility or transport corridor."""

    id: str
    name: str
    kind: str                                  # metro | water | storm | power
    alignment: list[tuple[float, float]]
    width: float
    height: float
    invert_levels: list[float]                 # z of the corridor floor
    owner: str = ""


@dataclass
class TreeSpec:
    x: float
    y: float
    height: float
    radius: float


@dataclass
class ParcelSpec:
    """A surface cadastral parcel - what today's 2D record actually contains."""

    id: str
    ring: list[tuple[float, float]]
    owner: str
    land_use: str = "residential"
    survey_no: str = ""


@dataclass
class Scene:
    """A complete demonstration site."""

    name: str
    extent: tuple[float, float, float, float]     # x0, y0, x1, y1 metres
    buildings: list[BuildingSpec]
    corridors: list[CorridorSpec]
    trees: list[TreeSpec]
    parcels: list[ParcelSpec]
    terrain: "TerrainModel"
    origin: dict = field(default_factory=lambda: dict(SITE_ORIGIN))
    jurisdiction: dict = field(default_factory=lambda: dict(JURISDICTION))

    def building(self, bid: str) -> BuildingSpec:
        for b in self.buildings:
            if b.id == bid:
                return b
        raise KeyError(bid)

    def stats(self) -> dict:
        units = sum(len(f.units) for b in self.buildings for f in b.floors)
        return {
            "buildings": len(self.buildings),
            "storeys": sum(len(b.floors) for b in self.buildings),
            "units": units,
            "corridors": len(self.corridors),
            "parcels": len(self.parcels),
            "trees": len(self.trees),
        }


class TerrainModel:
    """
    Smooth synthetic terrain: a regional tilt plus a couple of low undulations.

    Deliberately not flat. A flat site would let a sloppy ground filter look
    perfect, and would hide the fact that floor elevations have to be referenced
    to the terrain *under each building* rather than to a single site datum.
    """

    def __init__(self, base: float = 0.0, slope: tuple[float, float] = (0.018, -0.011),
                 seed: int = 11):
        self.base = base
        self.slope = slope
        rng = np.random.default_rng(seed)
        self._waves = [
            (rng.uniform(0.6, 1.4), rng.uniform(70, 130), rng.uniform(0, 2 * math.pi),
             rng.uniform(60, 140), rng.uniform(0, 2 * math.pi))
            for _ in range(3)
        ]

    def height(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = self.base + self.slope[0] * x + self.slope[1] * y
        for amp, lx, px, ly, py in self._waves:
            z = z + amp * np.sin(2 * math.pi * x / lx + px) * \
                np.cos(2 * math.pi * y / ly + py)
        return z


# --- the demonstration site ----------------------------------------------------
def _residential_floors(n_floors: int, floor_h: float, w: float, d: float,
                        n_basements: int = 1, *, prefix: str,
                        owners: Sequence[str]) -> list[FloorSpec]:
    """Lay out a floor stack with a simple quadrant unit plan."""
    floors: list[FloorSpec] = []

    for b in range(n_basements, 0, -1):
        z_top = -(b - 1) * 3.0
        floors.append(FloorSpec(
            index=-b, z_bottom=z_top - 3.0, z_top=z_top, use="parking",
            units=[UnitSpec(f"{prefix}-B{b}-PARK", (0.4, 0.4, w - 0.4, d - 0.4),
                            kind="parking", owner="Society common",
                            preferred_unit_no=900 + b)],
        ))

    m = 0.35   # wall thickness allowance so units do not touch exactly
    for i in range(n_floors):
        z0 = i * floor_h
        use = "commercial" if i == 0 else "residential"
        units = []
        halves = [(m, m, w / 2 - m, d / 2 - m), (w / 2 + m, m, w - m, d / 2 - m),
                  (m, d / 2 + m, w / 2 - m, d - m),
                  (w / 2 + m, d / 2 + m, w - m, d - m)]
        for q, rect in enumerate(halves, start=1):
            no = (i + 1) * 100 + q if i > 0 else q
            units.append(UnitSpec(
                name=f"{prefix}-{no:03d}",
                rect=rect,
                kind="shop" if i == 0 else "residential",
                owner=owners[(i * 4 + q) % len(owners)],
                preferred_unit_no=no,
            ))
        floors.append(FloorSpec(index=i, z_bottom=z0, z_top=z0 + floor_h,
                                units=units, use=use))
    return floors


OWNERS = [
    "Rohit Deshmukh", "Aarti Kulkarni", "S. R. Patil (HUF)", "Meera Joshi",
    "Nikhil Bhosale", "Farida Shaikh", "Anand Iyer", "Priya Rane",
    "Vikram Chavan", "Sunita More", "Rahul Gaikwad", "Kavita Sathe",
]


def demo_scene(*, with_conflicts: bool = True) -> Scene:
    """
    Build the reference demonstration site.

    With `with_conflicts` (the default) the site carries three defects of the
    kind this platform exists to find. They are injected deliberately and
    documented here rather than hidden, because a validation report on a
    perfectly clean site demonstrates nothing:

    1. **Unauthorised storey.** Shivneri Heights is built to 12 floors but
       sanctioned for 11. Only a scan can catch this; the paper record agrees
       with itself.
    2. **Overlapping ownership.** A balcony enclosure on the 4th floor of Ganga
       Residency pushes one flat into its neighbour's volume - two owners, one
       set of cubic metres.
    3. **Clearance breach.** The storm drain was re-routed under the commercial
       plaza's basement raft and now passes closer than the drainage
       authority's rule allows.
    """
    terrain = TerrainModel(base=0.0)

    # --- Building A: L-shaped housing tower, 12 storeys + 2 basements ------
    a_w, a_d = 30.0, 22.0
    a_floors = _residential_floors(12, 3.10, a_w, a_d, n_basements=2,
                                   prefix="SHV", owners=OWNERS)
    building_a = BuildingSpec(
        id="B-A", name="Shivneri Heights",
        footprint=[(0, 0), (a_w, 0), (a_w, a_d), (18, a_d), (18, 34), (0, 34)],
        origin=(42.0, 40.0), rotation_deg=14.0, floors=a_floors,
        roof="flat", parcel_id="P-01",
        registered_floors=11 if with_conflicts else None,
    )

    # --- Building B: rectangular society block, 8 storeys -----------------
    b_w, b_d = 26.0, 18.0
    b_floors = _residential_floors(8, 3.00, b_w, b_d, n_basements=1,
                                   prefix="GNG", owners=OWNERS[3:])
    if with_conflicts:
        # Flat GNG-402 was extended over its neighbour's balcony line. The rect
        # deliberately reaches back across the party wall into GNG-401.
        f4 = next(f for f in b_floors if f.index == 4)
        victim = f4.units[0]
        intruder = f4.units[1]
        # a 1.2 m strip taken across the party wall - the size of an enclosed
        # balcony, which is what these disputes almost always are
        intruder.rect = (victim.rect[2] - 1.2, intruder.rect[1],
                         intruder.rect[2], intruder.rect[3])
        intruder.kind = "residential (disputed)"

    building_b = BuildingSpec(
        id="B-B", name="Ganga Residency",
        footprint=[(0, 0), (b_w, 0), (b_w, b_d), (0, b_d)],
        origin=(120.0, 46.0), rotation_deg=14.0, floors=b_floors,
        roof="flat", parcel_id="P-02",
    )

    # --- Building C: commercial tower, 5 tall storeys ----------------------
    c_w, c_d = 34.0, 20.0
    c_floors = []
    c_floors.append(FloorSpec(index=-1, z_bottom=-3.6, z_top=0.0, use="parking",
                              units=[UnitSpec("DCP-B1-PARK",
                                              (0.4, 0.4, c_w - 0.4, c_d - 0.4),
                                              kind="parking",
                                              owner="Deccan Estates Pvt Ltd",
                                              preferred_unit_no=901)]))
    for i in range(5):
        z0 = i * 3.85
        units = [
            UnitSpec(f"DCP-{i+1}0{k}", rect, kind="commercial",
                     owner="Deccan Estates Pvt Ltd", preferred_unit_no=(i + 1) * 100 + k)
            for k, rect in enumerate(
                [(0.4, 0.4, c_w / 3 - 0.3, c_d - 0.4),
                 (c_w / 3 + 0.3, 0.4, 2 * c_w / 3 - 0.3, c_d - 0.4),
                 (2 * c_w / 3 + 0.3, 0.4, c_w - 0.4, c_d - 0.4)], start=1)
        ]
        c_floors.append(FloorSpec(index=i, z_bottom=z0, z_top=z0 + 3.85,
                                  units=units, use="commercial"))
    building_c = BuildingSpec(
        id="B-C", name="Deccan Commercial Plaza",
        footprint=[(0, 0), (c_w, 0), (c_w, c_d), (0, c_d)],
        origin=(58.0, 108.0), rotation_deg=-6.0, floors=c_floors,
        roof="flat", parcel_id="P-03",
    )

    # --- Building D: low-rise annexe, gable roof ---------------------------
    d_w, d_d = 14.0, 11.0
    d_floors = [
        FloorSpec(index=i, z_bottom=i * 3.2, z_top=(i + 1) * 3.2,
                  use="residential",
                  units=[UnitSpec(f"ANX-{i+1}01", (0.3, 0.3, d_w - 0.3, d_d - 0.3),
                                  owner=OWNERS[i % len(OWNERS)],
                                  preferred_unit_no=(i + 1) * 100 + 1)])
        for i in range(2)
    ]
    building_d = BuildingSpec(
        id="B-D", name="Sahyadri Annexe",
        footprint=[(0, 0), (d_w, 0), (d_w, d_d), (0, d_d)],
        origin=(140.0, 112.0), rotation_deg=22.0, floors=d_floors,
        roof="gable", parcel_id="P-04",
    )

    buildings = [building_a, building_b, building_c, building_d]

    # --- subsurface corridors ---------------------------------------------
    corridors = [
        CorridorSpec(
            id="INF-METRO-1", name="Metro Line 3 - up tunnel", kind="metro",
            alignment=[(-10, 78), (60, 84), (130, 92), (215, 96)],
            width=6.4, height=6.4,
            invert_levels=[-17.0, -16.2, -15.4, -14.8],
            owner="Maharashtra Metro Rail Corporation",
        ),
        CorridorSpec(
            id="INF-WATER-1", name="600 mm ductile-iron water main", kind="water",
            alignment=[(20, 20), (95, 26), (170, 30)],
            width=1.4, height=1.4, invert_levels=[-2.6, -2.4, -2.3],
            owner="Pune Municipal Corporation - Water Supply",
        ),
        CorridorSpec(
            id="INF-STORM-1", name="Storm-water box drain", kind="storm",
            alignment=([(104, 8), (99, 70), (74, 120)] if with_conflicts
                       else [(104, 8), (108, 70), (112, 150)]),
            width=2.2, height=1.8,
            invert_levels=([-4.6, -5.0, -5.4] if with_conflicts
                           else [-3.4, -3.0, -2.6]),
            owner="Pune Municipal Corporation - Drainage",
        ),
    ]

    # --- trees -------------------------------------------------------------
    rng = np.random.default_rng(23)
    keep_out = [(b.origin, 34.0) for b in buildings]
    trees: list[TreeSpec] = []
    while len(trees) < 26:
        x = rng.uniform(6, 208)
        y = rng.uniform(6, 168)
        if any(math.hypot(x - ox, y - oy) < r for (ox, oy), r in keep_out):
            continue
        trees.append(TreeSpec(x, y, rng.uniform(5.5, 11.5), rng.uniform(2.4, 4.6)))

    # --- surface parcels ---------------------------------------------------
    parcels = [
        ParcelSpec("P-01", [(30, 28), (92, 28), (92, 88), (30, 88)],
                   "Shivneri Co-op Housing Society", "residential", "SN-114/2A"),
        ParcelSpec("P-02", [(110, 34), (176, 34), (176, 86), (110, 86)],
                   "Ganga Co-op Housing Society", "residential", "SN-114/3"),
        ParcelSpec("P-03", [(44, 98), (110, 98), (110, 142), (44, 142)],
                   "Deccan Estates Pvt Ltd", "commercial", "SN-115/1"),
        ParcelSpec("P-04", [(128, 100), (180, 100), (180, 140), (128, 140)],
                   "Sahyadri Trust", "residential", "SN-115/4"),
        ParcelSpec("P-05", [(30, 148), (180, 148), (180, 172), (30, 172)],
                   "Pune Municipal Corporation", "road", "SN-000/R1"),
    ]

    return Scene(
        name="Pune - Shivajinagar demonstration block",
        extent=(0.0, 0.0, 215.0, 178.0),
        buildings=buildings, corridors=corridors, trees=trees,
        parcels=parcels, terrain=terrain,
    )

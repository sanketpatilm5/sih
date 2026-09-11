"""
The 3D cadastral object model.

This is where the platform stops being a computer-vision pipeline and starts
being a land record. Everything above produces *geometry*; this module attaches
identity, ownership, hierarchy and provenance to that geometry, which is what
makes it a cadastre.

The model is deliberately one object type rather than a class per legal
category. A surface parcel, a flat, a basement bay, a stretch of metro tunnel
and a block of air rights differ in their attributes and in where they sit in
the hierarchy, but they are all *the same kind of thing*: a bounded volume of
space with an identifier, an owner and a parent. Modelling them uniformly is
what lets one overlap check, one containment rule and one viewer handle the
whole column from the tunnel invert to the air rights - which is the entire
point of the exercise.

Hierarchy
---------
::

    Parcel  (surface, stratum G)
      |
      +-- Building
      |     +-- Storey  (one per floor, including basements)
      |           +-- Unit    (flat, shop, parking bay - the legal object)
      |
      +-- AirRights   (the unbuilt column above the structure)

    Infrastructure  (tunnels, mains - crosses parcels, so parented to none)

Provenance is carried on every object because these records have legal
consequences: a boundary derived from an approved plan and one inferred from a
point cloud are not equally authoritative, and the record has to say which is
which.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Optional, Union

from shapely.geometry import Polygon

from .geometry3d import CompositeSolid, Prism
from .ulpin import ULPIN3D, JurisdictionCode, Stratum, ULPINMinter, geometry_digest

Solid = Union[Prism, CompositeSolid]


class ObjectKind(str, Enum):
    PARCEL = "parcel"
    BUILDING = "building"
    STOREY = "storey"
    UNIT = "unit"
    INFRASTRUCTURE = "infrastructure"
    AIR_RIGHTS = "air_rights"

    @property
    def default_stratum(self) -> Stratum:
        return {
            "parcel": Stratum.SURFACE,
            "building": Stratum.SURFACE,
            "storey": Stratum.BUILDING,
            "unit": Stratum.BUILDING,
            "infrastructure": Stratum.INFRASTRUCTURE,
            "air_rights": Stratum.AIR_RIGHTS,
        }[self.value]


class Provenance(str, Enum):
    """How a boundary came to be - and therefore how much weight it carries."""

    SURVEYED = "surveyed"            # GNSS/total-station observation
    APPROVED_PLAN = "approved_plan"  # a sanctioned building plan
    EXISTING_RECORD = "existing_record"   # the legacy 2D cadastral layer
    EXTRACTED = "extracted"          # inferred by the AI pipeline
    DERIVED = "derived"              # computed from other objects
    ASSUMED = "assumed"              # a documented default, not a measurement

    @property
    def is_authoritative(self) -> bool:
        return self in (Provenance.SURVEYED, Provenance.APPROVED_PLAN,
                        Provenance.EXISTING_RECORD)


@dataclass
class CadastralObject:
    """A single identified volume of space in the register."""

    object_id: str
    kind: ObjectKind
    solid: Solid
    parent_id: Optional[str] = None
    ulpin: Optional[ULPIN3D] = None
    name: str = ""
    owner: str = ""
    use: str = ""
    level: int = 0                       # storey index; 0 = ground, -1 = basement
    stratum: Optional[Stratum] = None
    provenance: Provenance = Provenance.DERIVED
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)
    stability_digest: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.stratum is None:
            self.stratum = self.kind.default_stratum
        if not self.stability_digest:
            self.stability_digest = self.compute_digest()

    # -- geometry passthroughs ----------------------------------------------
    @property
    def footprint(self):
        return self.solid.footprint

    @property
    def z_min(self) -> float:
        return self.solid.z_min

    @property
    def z_max(self) -> float:
        return self.solid.z_max

    @property
    def volume(self) -> float:
        return self.solid.volume

    @property
    def area(self) -> float:
        fp = self.solid.footprint
        return fp.area

    @property
    def bbox(self):
        return tuple(self.solid.bbox)

    @property
    def centroid(self):
        return self.solid.centroid

    def compute_digest(self) -> str:
        fp = self.solid.footprint
        coords = (fp.exterior.coords if isinstance(fp, Polygon)
                  else fp.convex_hull.exterior.coords)
        return geometry_digest(coords, self.z_min, self.z_max)

    def intersection_volume(self, other: "CadastralObject") -> float:
        return self.solid.intersection_volume(other.solid)

    def clearance(self, other: "CadastralObject") -> float:
        return self.solid.clearance(other.solid)

    # -- serialisation -------------------------------------------------------
    def as_dict(self, *, with_geometry: bool = True) -> dict:
        d = {
            "object_id": self.object_id,
            "kind": self.kind.value,
            "parent_id": self.parent_id,
            "ulpin": str(self.ulpin) if self.ulpin else None,
            "name": self.name,
            "owner": self.owner,
            "use": self.use,
            "level": self.level,
            "stratum": self.stratum.value if self.stratum else None,
            "stratum_label": self.stratum.label if self.stratum else None,
            "provenance": self.provenance.value,
            "authoritative": self.provenance.is_authoritative,
            "confidence": round(self.confidence, 3),
            "area_m2": round(self.area, 3),
            "volume_m3": round(self.volume, 3),
            "z_min": round(self.z_min, 3),
            "z_max": round(self.z_max, 3),
            "height_m": round(self.z_max - self.z_min, 3),
            "bbox": [round(v, 3) for v in self.bbox],
            "stability_digest": self.stability_digest,
            "created_at": self.created_at,
            "attributes": self.attributes,
        }
        if with_geometry:
            d["geometry"] = self.solid.as_dict()
        return d


class Cadastre:
    """
    The register: every identified object on a site, plus the ULPIN minter.

    Kept in memory and persisted as a single document. At demonstration scale
    that is the right call - the whole register is a few hundred objects, and a
    process boundary between the geometry and the identifiers would buy nothing.
    The repository interface in `bhoomi3d.store` is where a PostGIS-backed
    implementation slots in for production volumes.
    """

    def __init__(self, jurisdiction: JurisdictionCode, origin: dict,
                 name: str = "untitled site"):
        self.jurisdiction = jurisdiction
        self.origin = origin
        self.name = name
        self.minter = ULPINMinter(jurisdiction)
        self._objects: dict[str, CadastralObject] = {}
        self._children: dict[str, list[str]] = {}
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -- population ----------------------------------------------------------
    def add(self, obj: CadastralObject, *, mint: bool = True,
            enu=None, preferred_unit: Optional[int] = None) -> CadastralObject:
        """
        Register an object, minting its 3D-ULPIN.

        `enu` is the local frame, needed to convert the object's centroid to
        the lat/lon that the identifier's geohash component encodes.
        """
        if obj.object_id in self._objects:
            raise ValueError(f"object {obj.object_id!r} is already registered")

        if mint and obj.ulpin is None:
            if enu is None:
                raise ValueError("minting a ULPIN needs the local ENU frame")
            cx, cy, _ = obj.centroid
            lon, lat = enu.inverse_xy(cx, cy)
            obj.ulpin = self.minter.mint(
                lat=lat, lon=lon,
                stratum=obj.stratum or obj.kind.default_stratum,
                level=abs(obj.level), owner=obj.object_id,
                preferred_unit=preferred_unit,
            )

        self._objects[obj.object_id] = obj
        if obj.parent_id:
            self._children.setdefault(obj.parent_id, []).append(obj.object_id)
        return obj

    # -- lookup --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._objects

    def get(self, object_id: str) -> Optional[CadastralObject]:
        return self._objects.get(object_id)

    @property
    def objects(self) -> list[CadastralObject]:
        return list(self._objects.values())

    def of_kind(self, *kinds: ObjectKind) -> list[CadastralObject]:
        want = set(kinds)
        return [o for o in self._objects.values() if o.kind in want]

    def children(self, object_id: str) -> list[CadastralObject]:
        return [self._objects[c] for c in self._children.get(object_id, [])
                if c in self._objects]

    def descendants(self, object_id: str) -> list[CadastralObject]:
        out: list[CadastralObject] = []
        stack = list(self._children.get(object_id, []))
        while stack:
            cid = stack.pop()
            obj = self._objects.get(cid)
            if obj is None:
                continue
            out.append(obj)
            stack.extend(self._children.get(cid, []))
        return out

    def ancestors(self, object_id: str) -> list[CadastralObject]:
        out: list[CadastralObject] = []
        cur = self._objects.get(object_id)
        while cur and cur.parent_id:
            cur = self._objects.get(cur.parent_id)
            if cur:
                out.append(cur)
        return out

    def by_ulpin(self, text: str) -> Optional[CadastralObject]:
        """Look an object up by its 3D-ULPIN, in either notation."""
        try:
            target = ULPIN3D.parse(text).compact
        except ValueError:
            return None
        for o in self._objects.values():
            if o.ulpin and o.ulpin.compact == target:
                return o
        return None

    # -- spatial queries -----------------------------------------------------
    def at_point(self, x: float, y: float, z: float,
                 kinds: Optional[Iterable[ObjectKind]] = None
                 ) -> list[CadastralObject]:
        """
        Every object whose volume contains a point.

        This is the query a 2D cadastre cannot answer and the reason the whole
        system exists: stand at one coordinate and ask what is above you, below
        you and around you, and get back a flat, a parking bay and a metro
        tunnel rather than a single parcel id.
        """
        want = set(kinds) if kinds else None
        hits = []
        for o in self._objects.values():
            if want and o.kind not in want:
                continue
            if isinstance(o.solid, Prism):
                if o.solid.contains_point(x, y, z):
                    hits.append(o)
            else:
                if any(p.contains_point(x, y, z) for p in o.solid.parts):
                    hits.append(o)
        return sorted(hits, key=lambda o: -o.z_min)

    def column_at(self, x: float, y: float) -> list[CadastralObject]:
        """Every object in the vertical column through a plan position."""
        from shapely.geometry import Point
        p = Point(x, y)
        hits = [o for o in self._objects.values() if o.footprint.intersects(p)]
        return sorted(hits, key=lambda o: -o.z_min)

    # -- statistics ----------------------------------------------------------
    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        by_prov: dict[str, int] = {}
        for o in self._objects.values():
            by_kind[o.kind.value] = by_kind.get(o.kind.value, 0) + 1
            by_prov[o.provenance.value] = by_prov.get(o.provenance.value, 0) + 1
        units = self.of_kind(ObjectKind.UNIT)
        zs = [o.z_min for o in self._objects.values()] + \
             [o.z_max for o in self._objects.values()]
        return {
            "site": self.name,
            "objects": len(self._objects),
            "by_kind": by_kind,
            "by_provenance": by_prov,
            "ulpins_issued": self.minter.issued_count,
            "total_unit_volume_m3": round(sum(u.volume for u in units), 2),
            "total_unit_area_m2": round(sum(u.area for u in units), 2),
            "vertical_extent_m": [round(min(zs), 2), round(max(zs), 2)] if zs else None,
            "jurisdiction": self.jurisdiction.as_dict(),
            "origin": self.origin,
            "created_at": self.created_at,
        }

    # -- persistence ---------------------------------------------------------
    def to_document(self, *, with_geometry: bool = True) -> dict:
        return {
            "format": "bhoomi3d.cadastre/1",
            "name": self.name,
            "jurisdiction": self.jurisdiction.as_dict(),
            "origin": self.origin,
            "created_at": self.created_at,
            "stats": self.stats(),
            "objects": [o.as_dict(with_geometry=with_geometry)
                        for o in self._objects.values()],
        }

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_document(), fh, indent=1)

    @classmethod
    def load(cls, path) -> "Cadastre":
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        j = doc["jurisdiction"]
        cad = cls(JurisdictionCode(**j), doc["origin"], doc.get("name", ""))
        cad.created_at = doc.get("created_at", cad.created_at)
        for od in doc["objects"]:
            g = od["geometry"]
            solid = (CompositeSolid.from_dict(g) if g["type"] == "CompositeSolid"
                     else Prism.from_dict(g))
            obj = CadastralObject(
                object_id=od["object_id"],
                kind=ObjectKind(od["kind"]),
                solid=solid,
                parent_id=od.get("parent_id"),
                ulpin=ULPIN3D.parse(od["ulpin"]) if od.get("ulpin") else None,
                name=od.get("name", ""), owner=od.get("owner", ""),
                use=od.get("use", ""), level=od.get("level", 0),
                stratum=Stratum(od["stratum"]) if od.get("stratum") else None,
                provenance=Provenance(od.get("provenance", "derived")),
                confidence=od.get("confidence", 1.0),
                attributes=od.get("attributes", {}),
                stability_digest=od.get("stability_digest", ""),
                created_at=od.get("created_at", ""),
            )
            cad._objects[obj.object_id] = obj
            if obj.parent_id:
                cad._children.setdefault(obj.parent_id, []).append(obj.object_id)
            if obj.ulpin:
                cad.minter.register(obj.ulpin, obj.object_id)
        return cad

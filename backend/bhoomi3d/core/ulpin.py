"""
3D-ULPIN - a volumetric extension of the Unique Land Parcel Identification Number.

Design goals, in priority order
-------------------------------
1. **Unique** across the whole vertical column, not just the ground plane.
2. **Stable** - once minted for a legal object the identifier never changes,
   even if the object is re-surveyed and its centroid shifts slightly.
3. **Self-describing** - a human reading the number can tell which state,
   district and tehsil it belongs to, whether it is above or below ground and
   on which level, without a database lookup.
4. **Error-detecting** - a single mistyped character, or any transposition of
   two adjacent characters, is caught offline by the check character.
5. **Collision-resistant** - a deterministic minting rule plus a registry-level
   uniqueness constraint, so two authorities working offline cannot mint the
   same identifier for different objects.

Anatomy
-------
::

    27 - 018 - 004 - tvkgh8n9q2 - B05 - 0207 - K
    |     |     |         |        | |    |     |
    |     |     |         |        | |    |     +-- check character, ISO 7064 MOD 37,36
    |     |     |         |        | |    +-------- unit sequence within the level (base36)
    |     |     |         |        | +------------- level index (base36, absolute)
    |     |     |         |        +--------------- stratum: G/B/U/A/I
    |     |     |         +------------------------ geohash-10 of the horizontal centroid
    |     |     +---------------------------------- tehsil / ULB code
    |     +---------------------------------------- district code (LGD)
    +---------------------------------------------- state code (LGD)

The geohash carries the *horizontal* identity (a 10-character geohash cell is
about 1.2 m x 0.6 m), the stratum and level carry the *vertical* identity, and
the unit sequence disambiguates objects that share a cell on the same level.

Stability
---------
The geohash records where the object was when the number was minted. Because
re-survey can move a centroid across a cell boundary, the ULPIN is minted once
and then treated as immutable: `stability_digest` stores a digest of the
canonical geometry so drift can be *detected and reported* without ever
mutating the identifier. See :func:`check_stability`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

# --- alphabets ----------------------------------------------------------------
GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"  # standard, no a/i/l/o
BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

GEOHASH_PRECISION = 10  # ~1.19 m x 0.60 m cell


class Stratum(str, Enum):
    """Which vertical layer of the column a legal object occupies."""

    SURFACE = "G"          # the ground parcel itself (the classic 2D cadastre)
    BUILDING = "B"         # a unit inside a structure above ground
    UNDERGROUND = "U"      # basements, cellars, underground parking
    AIR_RIGHTS = "A"       # transferable development rights above a structure
    INFRASTRUCTURE = "I"   # utility corridors, metro tunnels, pipelines

    @property
    def label(self) -> str:
        return {
            "G": "Surface parcel",
            "B": "Above-ground unit",
            "U": "Sub-surface unit",
            "A": "Air rights",
            "I": "Infrastructure corridor",
        }[self.value]

    @property
    def is_subsurface(self) -> bool:
        return self.value in ("U", "I")


# --- geohash ------------------------------------------------------------------
def geohash_encode(lat: float, lon: float, precision: int = GEOHASH_PRECISION) -> str:
    """Encode a WGS-84 position as a geohash string."""
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude out of range: {lat}")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude out of range: {lon}")
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    out: list[str] = []
    bit = 0
    ch = 0
    even = True  # longitude is refined first
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon > mid:
                ch = (ch << 1) | 1
                lon_lo = mid
            else:
                ch <<= 1
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat > mid:
                ch = (ch << 1) | 1
                lat_lo = mid
            else:
                ch <<= 1
                lat_hi = mid
        even = not even
        bit += 1
        if bit == 5:
            out.append(GEOHASH_ALPHABET[ch])
            bit, ch = 0, 0
    return "".join(out)


def geohash_bounds(gh: str) -> tuple[float, float, float, float]:
    """Return (lat_min, lon_min, lat_max, lon_max) of a geohash cell."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for c in gh.lower():
        idx = GEOHASH_ALPHABET.find(c)
        if idx < 0:
            raise ValueError(f"invalid geohash character: {c!r}")
        for mask in (16, 8, 4, 2, 1):
            bit = 1 if idx & mask else 0
            if even:
                mid = (lon_lo + lon_hi) / 2
                if bit:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if bit:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return lat_lo, lon_lo, lat_hi, lon_hi


def geohash_decode(gh: str) -> tuple[float, float]:
    """Return the centre (lat, lon) of a geohash cell."""
    lat_lo, lon_lo, lat_hi, lon_hi = geohash_bounds(gh)
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


# --- ISO 7064 MOD 37,36 check character ---------------------------------------
def iso7064_mod37_36(payload: str) -> str:
    """
    Compute the ISO 7064 MOD 37,36 check character over a base-36 payload.

    This "pure hybrid" system is what ISBN-style identifiers use. It detects
    every single-character substitution and every transposition of two adjacent
    characters - the two dominant failure modes when a clerk keys a number off
    a paper record.
    """
    m = 36
    p = m
    for c in payload.upper():
        v = BASE36.find(c)
        if v < 0:
            raise ValueError(f"character {c!r} is not valid base36")
        s = (p + v) % m
        if s == 0:
            s = m
        p = (2 * s) % (m + 1)
    return BASE36[(m + 1 - p) % m]


def _b36(value: int, width: int) -> str:
    """Left-pad a non-negative integer as fixed-width base36."""
    if value < 0:
        raise ValueError("base36 encoding requires a non-negative value")
    out = ""
    v = value
    while v:
        v, r = divmod(v, 36)
        out = BASE36[r] + out
    out = out.rjust(width, "0")
    if len(out) > width:
        raise ValueError(f"value {value} does not fit in {width} base36 characters")
    return out


def _unb36(text: str) -> int:
    return int(text, 36)


# --- the identifier -----------------------------------------------------------
_ULPIN_RE = re.compile(
    r"^(?P<state>[0-9A-Z]{2})-(?P<district>[0-9A-Z]{3})-(?P<tehsil>[0-9A-Z]{3})-"
    r"(?P<geohash>[0-9b-hjkmnp-z]{10})-(?P<stratum>[GBUAI])(?P<level>[0-9A-Z]{2})-"
    r"(?P<unit>[0-9A-Z]{4})-(?P<check>[0-9A-Z])$"
)

MAX_LEVEL = 36 ** 2 - 1   # 1295
MAX_UNIT = 36 ** 4 - 1    # 1,679,615

# state(2) + district(3) + tehsil(3) + geohash(10) + stratum(1) + level(2)
# + unit(4) = 25 payload characters, plus one check character.
ULPIN_PAYLOAD_LEN = 25
ULPIN_COMPACT_LEN = ULPIN_PAYLOAD_LEN + 1


@dataclass(frozen=True)
class ULPIN3D:
    """A parsed 3D-ULPIN. Immutable by construction."""

    state: str
    district: str
    tehsil: str
    geohash: str
    stratum: Stratum
    level: int
    unit: int
    check: str = field(default="", compare=False)

    def __post_init__(self):
        object.__setattr__(self, "state", self.state.upper())
        object.__setattr__(self, "district", self.district.upper())
        object.__setattr__(self, "tehsil", self.tehsil.upper())
        object.__setattr__(self, "geohash", self.geohash.lower())
        if not 0 <= self.level <= MAX_LEVEL:
            raise ValueError(f"level {self.level} outside 0..{MAX_LEVEL}")
        if not 0 <= self.unit <= MAX_UNIT:
            raise ValueError(f"unit {self.unit} outside 0..{MAX_UNIT}")
        if len(self.geohash) != GEOHASH_PRECISION:
            raise ValueError(
                f"geohash must be {GEOHASH_PRECISION} characters, got {len(self.geohash)}")
        want = iso7064_mod37_36(self.payload)
        if not self.check:
            object.__setattr__(self, "check", want)
        elif self.check.upper() != want:
            raise ValueError(
                f"check character mismatch: got {self.check!r}, expected {want!r}")

    # -- rendering -----------------------------------------------------------
    @property
    def payload(self) -> str:
        """The check-protected body, with no separators."""
        return (f"{self.state}{self.district}{self.tehsil}"
                f"{self.geohash.upper()}{self.stratum.value}"
                f"{_b36(self.level, 2)}{_b36(self.unit, 4)}")

    def __str__(self) -> str:
        return (f"{self.state}-{self.district}-{self.tehsil}-{self.geohash}-"
                f"{self.stratum.value}{_b36(self.level, 2)}-{_b36(self.unit, 4)}-"
                f"{self.check}")

    @property
    def compact(self) -> str:
        """Separator-free form, for barcodes / QR codes / database keys."""
        return self.payload + self.check

    # -- derived facts -------------------------------------------------------
    @property
    def centroid(self) -> tuple[float, float]:
        """Approximate (lat, lon) recovered from the geohash, no lookup needed."""
        return geohash_decode(self.geohash)

    @property
    def cell_size_m(self) -> tuple[float, float]:
        """Ground dimensions of this ULPIN geohash cell, in metres."""
        import math
        lat_lo, lon_lo, lat_hi, lon_hi = geohash_bounds(self.geohash)
        mid_lat = math.radians((lat_lo + lat_hi) / 2)
        dy = (lat_hi - lat_lo) * 111_132.0
        dx = (lon_hi - lon_lo) * 111_320.0 * math.cos(mid_lat)
        return dx, dy

    @property
    def signed_level(self) -> int:
        """Level as a signed floor index - negative below ground."""
        return -self.level if self.stratum.is_subsurface else self.level

    def describe(self) -> str:
        lvl = ("ground level" if self.level == 0 else
               f"level {self.level} {'below' if self.stratum.is_subsurface else 'above'} ground")
        return f"{self.stratum.label}, {lvl}, unit #{self.unit}"

    def as_dict(self) -> dict:
        lat, lon = self.centroid
        return {
            "ulpin": str(self),
            "compact": self.compact,
            "state": self.state,
            "district": self.district,
            "tehsil": self.tehsil,
            "geohash": self.geohash,
            "stratum": self.stratum.value,
            "stratum_label": self.stratum.label,
            "level": self.level,
            "signed_level": self.signed_level,
            "unit": self.unit,
            "check": self.check,
            "centroid": {"lat": round(lat, 8), "lon": round(lon, 8)},
            "description": self.describe(),
        }

    # -- parsing -------------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "ULPIN3D":
        """
        Parse a formatted or compact ULPIN, validating the check character.

        Accepts both `27-018-004-tvkgh8n9q2-B05-0207-K` and the separator-free
        26+1 character form, in any letter case.
        """
        raw = "".join(text.split()).upper()
        compact = raw.replace("-", "")
        if len(compact) != ULPIN_COMPACT_LEN:
            raise ValueError(
                f"a ULPIN has {ULPIN_COMPACT_LEN} characters excluding separators, "
                f"got {len(compact)} in {text!r}")
        m = _ULPIN_RE.match(cls._punctuate(compact))
        if not m:
            raise ValueError(f"malformed ULPIN: {text!r}")
        return cls(
            state=m.group("state"),
            district=m.group("district"),
            tehsil=m.group("tehsil"),
            geohash=m.group("geohash").lower(),
            stratum=Stratum(m.group("stratum")),
            level=_unb36(m.group("level")),
            unit=_unb36(m.group("unit")),
            check=m.group("check"),
        )

    @staticmethod
    def _punctuate(compact: str) -> str:
        """Insert the canonical separators into a compact ULPIN, ready for the regex."""
        c = compact
        return (f"{c[0:2]}-{c[2:5]}-{c[5:8]}-{c[8:18].lower()}-"
                f"{c[18:21]}-{c[21:25]}-{c[25]}")

    @classmethod
    def is_valid(cls, text: str) -> bool:
        try:
            cls.parse(text)
            return True
        except (ValueError, KeyError):
            return False


# --- minting ------------------------------------------------------------------
@dataclass
class JurisdictionCode:
    """LGD-style administrative codes the identifier is namespaced under."""

    state: str
    district: str
    tehsil: str
    state_name: str = ""
    district_name: str = ""
    tehsil_name: str = ""

    def __post_init__(self):
        self.state = str(self.state).upper().rjust(2, "0")[:2]
        self.district = str(self.district).upper().rjust(3, "0")[:3]
        self.tehsil = str(self.tehsil).upper().rjust(3, "0")[:3]

    def as_dict(self) -> dict:
        return {
            "state": self.state, "district": self.district, "tehsil": self.tehsil,
            "state_name": self.state_name, "district_name": self.district_name,
            "tehsil_name": self.tehsil_name,
        }


class ULPINMinter:
    """
    Mints 3D-ULPINs and guarantees uniqueness inside one jurisdiction.

    Minting is deterministic given (centroid, stratum, level) plus the set of
    numbers already issued: the unit sequence is simply the lowest free slot in
    that geohash cell on that level. Two authorities minting independently will
    therefore collide only if they are working the same cell on the same level
    at the same time - which is exactly the case the registry-level uniqueness
    constraint in :meth:`register` is there to catch on merge.
    """

    def __init__(self, jurisdiction: JurisdictionCode,
                 issued: Optional[Iterable[str]] = None):
        self.jurisdiction = jurisdiction
        self._issued: dict[str, str] = {}   # compact ULPIN -> owning object id
        self._occupancy: dict[tuple[str, str, int], set[int]] = {}
        for text in issued or ():
            self.register(ULPIN3D.parse(text), owner="<pre-existing>")

    # -- registry ------------------------------------------------------------
    def register(self, ulpin: ULPIN3D, owner: str) -> None:
        key = ulpin.compact
        if key in self._issued and self._issued[key] != owner:
            raise ValueError(
                f"ULPIN collision: {ulpin} already issued to {self._issued[key]!r}, "
                f"cannot re-issue to {owner!r}")
        self._issued[key] = owner
        self._occupancy.setdefault(
            (ulpin.geohash, ulpin.stratum.value, ulpin.level), set()).add(ulpin.unit)

    @property
    def issued_count(self) -> int:
        return len(self._issued)

    def owner_of(self, text: str) -> Optional[str]:
        try:
            return self._issued.get(ULPIN3D.parse(text).compact)
        except ValueError:
            return None

    # -- minting -------------------------------------------------------------
    def mint(self, *, lat: float, lon: float, stratum: Stratum, level: int,
             owner: str, preferred_unit: Optional[int] = None) -> ULPIN3D:
        """
        Issue a new 3D-ULPIN for an object whose centroid is at (lat, lon).

        `preferred_unit` lets a caller ask for a semantically meaningful number
        (flat 502 -> unit 502); if that slot is taken in this cell the next free
        slot is used instead, so the identifier stays unique either way.
        """
        gh = geohash_encode(lat, lon, GEOHASH_PRECISION)
        slot_key = (gh, stratum.value, int(level))
        taken = self._occupancy.setdefault(slot_key, set())

        unit = preferred_unit if preferred_unit is not None else 0
        if unit < 0 or unit > MAX_UNIT:
            unit = 0
        while unit in taken:
            unit += 1
            if unit > MAX_UNIT:
                raise ValueError(f"geohash cell {gh} exhausted on level {level}")

        ulpin = ULPIN3D(
            state=self.jurisdiction.state,
            district=self.jurisdiction.district,
            tehsil=self.jurisdiction.tehsil,
            geohash=gh, stratum=stratum, level=int(level), unit=unit,
        )
        self.register(ulpin, owner)
        return ulpin


# --- stability ----------------------------------------------------------------
def geometry_digest(footprint_coords: Iterable[Iterable[float]],
                    z_min: float, z_max: float) -> str:
    """
    A short, canonical digest of a volumetric parcel's geometry.

    Coordinates are quantised to 1 mm before hashing so that floating-point
    noise from a re-run of the pipeline does not change the digest, while a
    genuine re-survey does.

    Adding 0.0 before formatting is not redundant: a coordinate a hair below
    zero quantises to the string "-0.000" where zero gives "0.000", so without
    this a sub-micron shift across the origin would change the digest and
    report a spurious re-survey. IEEE-754 gives -0.0 + 0.0 == +0.0.
    """
    def q(v: float) -> str:
        return f"{round(float(v), 3) + 0.0:.3f}"

    h = hashlib.blake2s(digest_size=8)
    for x, y in footprint_coords:
        h.update(f"{q(x)},{q(y)};".encode())
    h.update(f"|{q(z_min)}|{q(z_max)}".encode())
    return h.hexdigest()


def check_stability(ulpin: ULPIN3D, lat: float, lon: float) -> dict:
    """
    Report whether an object has drifted out of the geohash cell recorded in its
    ULPIN. Drift never invalidates the identifier - it raises an advisory so a
    surveyor can confirm the re-survey was intentional.
    """
    lat_lo, lon_lo, lat_hi, lon_hi = geohash_bounds(ulpin.geohash)
    inside = (lat_lo <= lat <= lat_hi) and (lon_lo <= lon <= lon_hi)
    cur_lat, cur_lon = ulpin.centroid
    import math
    dx = (lon - cur_lon) * 111_320.0 * math.cos(math.radians(lat))
    dy = (lat - cur_lat) * 111_132.0
    return {
        "stable": inside,
        "drift_m": round(math.hypot(dx, dy), 3),
        "cell_size_m": [round(v, 3) for v in ulpin.cell_size_m],
        "advisory": None if inside else (
            "centroid has moved outside the geohash cell recorded at mint time; "
            "the ULPIN remains valid and unchanged, but the re-survey should be "
            "confirmed and the mutation recorded"),
    }

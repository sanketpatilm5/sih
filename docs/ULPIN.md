# The 3D-ULPIN specification

## The problem it solves

A ULPIN identifies a land parcel — a polygon on the ground. That is sufficient
only while every legally distinct thing at a location sits at the same
elevation, which stopped being true the moment anyone built a first floor.

In a modern block, one set of ground coordinates carries:

```
              ↑  air rights (transferable development rights)
    ──────────┴──────────
         flat 1102
         flat 1002
              ⋮
         shop 001
    ═══════════════════  ground
         parking B1
         parking B2
    ─────────────────────
         water main            −2.4 m
         storm drain           −3.0 m
         metro tunnel         −15.4 m
              ↓
```

Fourteen or more legally distinct objects, several with different owners, all
sharing one parcel boundary. A 2D identifier can name the column. It cannot name
anything in it.

## Anatomy

```
27 - 025 - 004 - tek92et7bx - B04 - 00DY - 5
│     │     │         │        │ │    │     │
│     │     │         │        │ │    │     └── check character
│     │     │         │        │ │    └──────── unit sequence (base36, 4 chars)
│     │     │         │        │ └───────────── level index (base36, 2 chars)
│     │     │         │        └─────────────── stratum (1 char)
│     │     │         └──────────────────────── geohash-10 of the centroid
│     │     └────────────────────────────────── tehsil / ULB code (3 chars)
│     └──────────────────────────────────────── district code, LGD (3 chars)
└────────────────────────────────────────────── state code, LGD (2 chars)
```

25 payload characters plus one check character. The separator-free form is
26 characters and is what goes in a database key or a QR code; both forms parse.

### Fields

| Field | Width | Alphabet | Meaning |
|---|---|---|---|
| state | 2 | base36 | LGD state code (`27` = Maharashtra) |
| district | 3 | base36 | LGD district code |
| tehsil | 3 | base36 | tehsil / urban local body |
| geohash | 10 | geohash base32 | horizontal position of the centroid |
| stratum | 1 | `G B U A I` | which layer of the column |
| level | 2 | base36 | absolute level index, 0–1295 |
| unit | 4 | base36 | sequence within (cell, stratum, level), 0–1,679,615 |
| check | 1 | base36 | ISO 7064 MOD 37,36 |

### Strata

| Code | Meaning | Sign of level |
|---|---|---|
| `G` | surface parcel — the classic 2D cadastre | — |
| `B` | above-ground unit inside a structure | positive |
| `U` | sub-surface unit: basement, cellar, parking | negative |
| `A` | air rights above a structure | positive |
| `I` | infrastructure corridor: tunnel, main, duct | negative |

The stratum carries the sign, so the level field stays a plain unsigned number
and the identifier remains readable. `B04` is the fourth floor; `U02` is the
second basement.

## Design decisions

### Why a geohash rather than a sequence number

A sequential identifier carries no information and requires a central allocator
that must be online and consistent. A geohash makes the identifier
**self-locating**: `tek92et7bx` decodes to a 1.13 m × 0.60 m cell without any
lookup, so a field officer holding a paper record can confirm they are standing
in the right place. It also makes independent minting possible — two authorities
working offline in different tehsils cannot collide, because their state,
district and tehsil fields already differ.

Ten characters is the deliberate choice. Nine gives a 4.8 m cell, which is
larger than a small flat. Eleven gives 15 cm, which is finer than the survey
that produced the centroid and would waste a character on false precision.

### Why the check character is ISO 7064 MOD 37,36

These numbers get read off paper, spoken over a phone and typed into forms. The
two dominant failure modes are mis-keying one character and swapping two
adjacent ones. ISO 7064 MOD 37,36 is a "pure hybrid" system that provably
catches **100 %** of both, over the full base-36 alphabet.

`tests/test_ulpin.py` verifies this exhaustively rather than by sampling: it
generates every possible single-character substitution (910 of them for a
26-character identifier) and every adjacent transposition, and asserts that not
one is accepted.

A simple mod-10 or mod-11 digit would not do this: mod-11 fails on some
transpositions, and neither handles a base-36 alphabet.

### Why identifiers never change

This is the tension at the centre of the scheme. The geohash makes the
identifier self-locating, but it is derived from the centroid — and a re-survey
moves centroids. If the identifier were recomputed, a boundary correction of a
few centimetres could silently change the identity of a property, which would be
catastrophic for a land record. Every deed, mutation and tax record referencing
the old number would be orphaned.

So the resolution is:

- A 3D-ULPIN is **minted once** and is thereafter immutable.
- The geohash records where the object was **at mint time**.
- `check_stability()` compares the current centroid against the recorded cell
  and reports drift as an **advisory**, never as a change.

```python
>>> check_stability(ulpin, new_lat, new_lon)
{'stable': False,
 'drift_m': 118.5,
 'cell_size_m': [1.132, 0.596],
 'advisory': 'centroid has moved outside the geohash cell recorded at mint '
             'time; the ULPIN remains valid and unchanged, but the re-survey '
             'should be confirmed and the mutation recorded'}
```

Separately, `geometry_digest()` stores a quantised digest of the object's
geometry, so a re-run of the pipeline that produces identical geometry produces
an identical digest, while a genuine boundary change does not. Coordinates are
quantised to 1 mm before hashing — far below survey precision, far above
floating-point noise.

### Collision resistance

Uniqueness is enforced at three levels:

1. **Namespacing.** State, district and tehsil partition the space
   administratively, so authorities cannot collide with each other.
2. **Deterministic minting.** Within a jurisdiction, the unit sequence is the
   lowest free slot in that geohash cell on that stratum and level. Minting is
   a pure function of the position and the set already issued.
3. **Registry constraint.** `ULPINMinter.register()` refuses to issue an
   identifier already held by a different object, which is what catches a
   collision when two offline registers are merged.

`preferred_unit` lets a caller ask for a semantically meaningful sequence — flat
502 gets unit 502 — and silently falls back to the next free slot if it is
taken, so the request never compromises uniqueness.

### Capacity

Per geohash cell, per stratum, per level: 1,679,616 units. Levels run 0–1295 in
each direction. In practice the binding constraint is the cell size: a 1.1 m ×
0.6 m cell will rarely contain more than one unit centroid on a given floor, so
the sequence is almost always 0 or a semantic flat number.

## Worked examples

Taken verbatim from the demonstration register:

| Object | ULPIN | Reads as |
|---|---|---|
| Flat 502, 4th floor | `27-025-004-tek92et7bx-B04-00DY-5` | above-ground unit, level 4, unit #502 |
| Basement parking bay | `27-025-004-tek92etr7w-U01-00P1-C` | sub-surface unit, level 1 below ground, unit #901 |
| Surface parcel | `27-025-004-tek92etk2k-G00-0000-X` | surface parcel, ground level |
| Metro tunnel | `27-025-004-tek92ettcq-I00-0000-Y` | infrastructure corridor |
| Air rights above tower | `27-025-004-tek92ethqh-A00-0000-I` | air rights |

These five geohashes differ because each encodes its own object's centroid, and
these objects have genuinely different plan centres — a single flat is a
quadrant of a floor, the air-rights block spans the whole L-shaped roof, and the
tunnel runs across the site. The scheme does not depend on that: objects whose
centroids *do* fall in the same cell — a flat and the tunnel directly beneath it
— are separated by the stratum and level fields alone, which is the case
`test_tunnel_under_a_flat_gets_a_distinct_identifier` exercises directly:

```python
flat   = minter.mint(lat=lat, lon=lon, stratum=Stratum.BUILDING,       level=10, ...)
tunnel = minter.mint(lat=lat, lon=lon, stratum=Stratum.INFRASTRUCTURE, level=4,  ...)

flat.geohash == tunnel.geohash      # same ground position
flat.compact != tunnel.compact      # different identity
```

That is the whole point: identical coordinates, different property.

## API

```
GET  /api/ulpin/{ulpin}      resolve, with the check character verified first
POST /api/ulpin/validate     batch-validate without touching the register
```

A mistyped identifier returns `400` with a specific diagnosis (`check character
mismatch: got 'Z', expected 'W'`) rather than a bare `404`, because those two
answers require completely different actions from the user.

## Reference implementation

`backend/bhoomi3d/core/ulpin.py` — around 400 lines with no dependencies beyond
the standard library. `tests/test_ulpin.py` covers format, exhaustive error
detection, minting, collision refusal and stability.

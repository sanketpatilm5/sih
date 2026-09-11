import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(scope="session")
def enu():
    from bhoomi3d.core.crs import GeodeticOrigin, LocalENU
    return LocalENU(GeodeticOrigin(lat=18.520430, lon=73.856744, height=560.0))


@pytest.fixture(scope="session")
def jurisdiction():
    from bhoomi3d.core.ulpin import JurisdictionCode
    return JurisdictionCode("27", "025", "004", "Maharashtra", "Pune", "Pune City")


@pytest.fixture(scope="session")
def small_scene():
    """
    A deliberately small site.

    The full demonstration scene takes ~25 s to simulate and process, which is
    fine for a demo but far too slow to sit inside a unit-test run. This trims
    it to two buildings and thins the survey so the end-to-end test still
    exercises every stage in a few seconds.
    """
    from bhoomi3d.data.scene import demo_scene
    scene = demo_scene(with_conflicts=True)
    scene.buildings = [b for b in scene.buildings if b.id in ("B-B", "B-D")]
    scene.trees = scene.trees[:6]
    scene.extent = (100.0, 30.0, 200.0, 140.0)
    return scene

"""Abstract 2-D map: homes, schools, workplaces, gathering places.

Coordinates are arbitrary units on a ``WIDTH x HEIGHT`` canvas. Geography has no mechanistic
effect beyond identifying *where* an activity happens (for the live map); distances are not
modelled. Home positions are seeded so the map is reproducible.
"""

from __future__ import annotations

import numpy as np

from hpe.contracts.schemas import Place, WorldMap

WIDTH = 100.0
HEIGHT = 60.0

# Job family of each workplace (one workplace per training family).
WORKPLACE_FAMILIES: dict[str, str] = {
    "work_sorting_house": "sort",
    "work_market": "compare",
    "work_watch_post": "signal",
}


def civic_places(n_agents: int) -> list[Place]:
    """Fixed civic layout in the centre band of the map."""
    places = [
        Place(place_id="school_north", kind="school", name="North School", x=30.0, y=12.0),
        Place(place_id="work_sorting_house", kind="workplace", name="Sorting House", x=70.0, y=10.0),
        Place(place_id="work_market", kind="workplace", name="Market", x=50.0, y=30.0),
        Place(place_id="work_watch_post", kind="workplace", name="Watch Post", x=85.0, y=30.0),
        Place(place_id="plaza", kind="gathering", name="Plaza", x=50.0, y=46.0),
        Place(place_id="tutoring_centre", kind="tutoring_center", name="Tutoring Centre", x=22.0, y=30.0),
        Place(place_id="elder_centre", kind="elder_center", name="Elder Centre", x=78.0, y=50.0),
    ]
    if n_agents > 60:
        places.append(Place(place_id="school_south", kind="school", name="South School", x=30.0, y=50.0))
    return places


def home_position(rng: np.random.Generator) -> tuple[float, float]:
    """Homes are scattered around the edges of the map, away from the civic centre."""
    side = int(rng.integers(0, 4))
    if side == 0:
        return float(rng.uniform(4, 96)), float(rng.uniform(2, 7))
    if side == 1:
        return float(rng.uniform(4, 96)), float(rng.uniform(53, 58))
    if side == 2:
        return float(rng.uniform(2, 12)), float(rng.uniform(4, 56))
    return float(rng.uniform(90, 98)), float(rng.uniform(4, 56))


def make_home(household_id: str, rng: np.random.Generator) -> Place:
    x, y = home_position(rng)
    return Place(place_id=f"home_{household_id}", kind="home", name=f"Home {household_id}", x=x, y=y)


def build_map(places: dict[str, Place]) -> WorldMap:
    return WorldMap(width=WIDTH, height=HEIGHT, places=[places[k] for k in sorted(places)])


def _mix(a: int, b: int) -> int:
    """Stateless 32-bit integer hash (splitmix-style) for frame jitter; no RNG state consumed."""
    z = (a * 0x9E3779B1 + b * 0x85EBCA77 + 0x165667B1) & 0xFFFFFFFF
    z ^= z >> 15
    z = (z * 0x2C1B3C6D) & 0xFFFFFFFF
    z ^= z >> 12
    z = (z * 0x297A2D39) & 0xFFFFFFFF
    z ^= z >> 15
    return z


def jitter(agent_index: int, tick: int, radius: float) -> tuple[float, float]:
    h1 = _mix(agent_index, tick)
    h2 = _mix(agent_index + 7919, tick + 104729)
    return ((h1 / 0xFFFFFFFF) - 0.5) * 2 * radius, ((h2 / 0xFFFFFFFF) - 0.5) * 2 * radius

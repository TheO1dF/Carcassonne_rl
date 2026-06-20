"""
tiles.py
========

Tile configuration and board geometry for the Carcassonne base game.

This module is intentionally pure data + tiny helpers (no game logic, no Gym
dependency) so that it can be imported by the scoring engine, the board logic
and the Gym environment without any circular imports.

KEY CONCEPTS
------------

* Each tile has 4 *edges* (North, East, South, West). An edge carries a
  ``Terrain`` value (CITY / ROAD / FIELD). Two tiles may only be placed next to
  each other if their touching edges carry the *same* terrain (edge matching).

* A tile is internally made of *feature segments*. A segment is one connected
  piece of a feature that lives on the tile:
      - a CITY segment touches a set of edges (e.g. a city spanning N+E),
      - a ROAD segment touches a set of edges (a straight road touches N+S),
      - a FIELD segment occupies a set of "field positions" (see below),
      - a MONASTERY is a single central segment.
  Segments are what meeples are placed on, and what the Union-Find scoring
  engine connects across tiles.

* FIELD POSITIONS. Fields are trickier than cities/roads because a single tile
  edge can have field on *both* sides of a road. We therefore model 8 field
  "half-edge" positions around the tile, numbered CLOCKWISE starting at the
  top-left:

            0   1
          7       2
          6       3
            5   4

      - positions 0,1 lie on the NORTH edge (0 = west half, 1 = east half)
      - positions 2,3 lie on the EAST  edge (2 = north half, 3 = south half)
      - positions 4,5 lie on the SOUTH edge (4 = east  half, 5 = west  half)
      - positions 6,7 lie on the WEST  edge (6 = south half, 7 = north half)

  This is the standard "farm position" model used by most Carcassonne engines
  and is required for correct end-game farm scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, Tuple


# --------------------------------------------------------------------------- #
# Edge / terrain / feature enumerations
# --------------------------------------------------------------------------- #

# Edge direction indices. The order N, E, S, W (0, 1, 2, 3) is used everywhere
# (edge tuples, rotations, etc.). Keeping them as module constants makes the
# data tables far easier to read than magic numbers.
N, E, S, W = 0, 1, 2, 3

# Opposite edge of a given direction (used for edge matching between neighbours).
OPPOSITE = {N: S, E: W, S: N, W: E}


class Terrain(IntEnum):
    """Terrain that can sit on a tile *edge* (also used in the observation)."""
    FIELD = 1   # 0 is reserved for "empty cell" in the observation tensor
    ROAD = 2
    CITY = 3


class FeatureType(IntEnum):
    """Type of a *feature segment* (what a meeple can be placed on)."""
    CITY = 1
    ROAD = 2
    FIELD = 3
    MONASTERY = 4


# --------------------------------------------------------------------------- #
# Field-position geometry
# --------------------------------------------------------------------------- #

# The two field positions that live on each edge.
EDGE_POSITIONS: Dict[int, Tuple[int, int]] = {
    N: (0, 1),
    E: (2, 3),
    S: (4, 5),
    W: (6, 7),
}

# Which edge each field position belongs to (reverse of EDGE_POSITIONS).
POSITION_EDGE: Dict[int, int] = {0: N, 1: N, 2: E, 3: E, 4: S, 5: S, 6: W, 7: W}

# When two tiles touch, their field half-edges line up. FIELD_NEIGHBOR[d][p]
# gives, for *my* position ``p`` on *my* edge ``d``, the position on the
# neighbouring tile (which touches me on its opposite edge) that connects to it.
#
# Example: my NORTH edge (positions 0,1) meets the neighbour's SOUTH edge
# (positions 4,5). Spatially my NW corner (0) lines up with their SW corner (5),
# and my NE corner (1) lines up with their SE corner (4).
FIELD_NEIGHBOR: Dict[int, Dict[int, int]] = {
    N: {0: 5, 1: 4},
    E: {2: 7, 3: 6},
    S: {4: 1, 5: 0},
    W: {6: 3, 7: 2},
}

# (dx, dy) offset to the neighbouring cell in each direction. The board uses
# screen-style coordinates: x grows to the East, y grows to the South.
DELTA = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}


# --------------------------------------------------------------------------- #
# Tile type definition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TileType:
    """Immutable description of one of the 24 distinct base-game tiles.

    Attributes
    ----------
    name : str
        The canonical rulebook letter (A .. X).
    edges : tuple(Terrain, Terrain, Terrain, Terrain)
        Terrain on the (N, E, S, W) edges in the tile's base orientation.
    cities : tuple of (frozenset[int], bool)
        One entry per city segment: the set of edges it touches and whether it
        carries a coat-of-arms shield (worth bonus points).
    roads : tuple of frozenset[int]
        One entry per road segment: the set of edges it touches. A road that
        ends at a junction / monastery / city simply has fewer edges.
    fields : tuple of (frozenset[int], tuple[int])
        One entry per field segment: the field positions it covers and the
        indices (into ``cities``) of the city segments it is adjacent to
        (needed for farm scoring).
    monastery : bool
        Whether the tile has a central monastery (cloister).
    """
    name: str
    edges: Tuple[Terrain, Terrain, Terrain, Terrain]
    cities: Tuple[Tuple[FrozenSet[int], bool], ...] = ()
    roads: Tuple[FrozenSet[int], ...] = ()
    fields: Tuple[Tuple[FrozenSet[int], Tuple[int, ...]], ...] = ()
    monastery: bool = False

    def __deepcopy__(self, memo):
        # TileType is immutable static data; sharing it makes Board.clone()
        # (used heavily by the greedy bot's look-ahead) much cheaper.
        return self


# --------------------------------------------------------------------------- #
# Helpers to build the tile table compactly
# --------------------------------------------------------------------------- #

_EDGE_IDX = {"N": N, "E": E, "S": S, "W": W}
_TERR = {"F": Terrain.FIELD, "R": Terrain.ROAD, "C": Terrain.CITY}


def _es(s: str) -> FrozenSet[int]:
    """Build an edge set from a string such as ``"NE"`` -> {N, E}."""
    return frozenset(_EDGE_IDX[c] for c in s)


def _mk(name, edges, cities=(), roads=(), fields=(), monastery=False) -> TileType:
    """Compact constructor. ``edges`` is a 4-char string for (N, E, S, W)."""
    e = tuple(_TERR[c] for c in edges)
    c = tuple((_es(es), shield) for (es, shield) in cities)
    r = tuple(_es(es) for es in roads)
    f = tuple((frozenset(pos), tuple(adj)) for (pos, adj) in fields)
    return TileType(name, e, c, r, f, monastery)


# --------------------------------------------------------------------------- #
# The 24 distinct base-game tiles (rulebook letters A .. X)
# --------------------------------------------------------------------------- #
#
# For every tile we describe edges (N,E,S,W), then the city / road / field
# segments. Field positions follow the clockwise numbering documented at the
# top of this file. ``adj`` indices refer to the city segment list of the tile.

_TILES = [
    # A: monastery + a road leading off the south edge.
    _mk("A", "FFRF", roads=["S"],
        fields=[({0, 1, 2, 3, 4, 5, 6, 7}, ())], monastery=True),

    # B: monastery, fields all around.
    _mk("B", "FFFF", fields=[({0, 1, 2, 3, 4, 5, 6, 7}, ())], monastery=True),

    # C: a full city covering all four edges (with shield).
    _mk("C", "CCCC", cities=[("NESW", True)]),

    # D: city on the East edge, a road running straight North-South.
    _mk("D", "RCRF", cities=[("E", False)], roads=["NS"],
        fields=[({0, 5, 6, 7}, ()), ({1, 4}, (0,))]),

    # E: city on a single edge (North), field elsewhere.
    _mk("E", "CFFF", cities=[("N", False)],
        fields=[({2, 3, 4, 5, 6, 7}, (0,))]),

    # F: city spanning two opposite edges (E-W), with shield.
    _mk("F", "FCFC", cities=[("EW", True)],
        fields=[({0, 1}, (0,)), ({4, 5}, (0,))]),

    # G: city spanning two opposite edges (N-S), no shield.
    _mk("G", "CFCF", cities=[("NS", False)],
        fields=[({2, 3}, (0,)), ({6, 7}, (0,))]),

    # H: two *separate* cities on opposite edges (E and W), field between.
    _mk("H", "FCFC", cities=[("E", False), ("W", False)],
        fields=[({0, 1, 4, 5}, (0, 1))]),

    # I: two *separate* cities on adjacent edges (E and S).
    _mk("I", "FCCF", cities=[("E", False), ("S", False)],
        fields=[({0, 1, 6, 7}, (0, 1))]),

    # J: city on North, a road curving from East to South.
    _mk("J", "CRRF", cities=[("N", False)], roads=["ES"],
        fields=[({2, 5, 6, 7}, (0,)), ({3, 4}, ())]),

    # K: city on North, a road curving from South to West (mirror of J).
    _mk("K", "CFRR", cities=[("N", False)], roads=["SW"],
        fields=[({2, 3, 4, 7}, (0,)), ({5, 6}, ())]),

    # L: city on North, a road T-junction (E, S, W meeting in the centre).
    _mk("L", "CRRR", cities=[("N", False)], roads=["E", "S", "W"],
        fields=[({2, 7}, (0,)), ({3, 4}, ()), ({5, 6}, ())]),

    # M: two adjacent edges form one connected city (N-E corner), with shield.
    _mk("M", "CCFF", cities=[("NE", True)], fields=[({4, 5, 6, 7}, (0,))]),

    # N: connected corner city (N-E), no shield.
    _mk("N", "CCFF", cities=[("NE", False)], fields=[({4, 5, 6, 7}, (0,))]),

    # O: connected corner city (N-E) + a road curving S-W, with shield.
    _mk("O", "CCRR", cities=[("NE", True)], roads=["SW"],
        fields=[({4, 7}, (0,)), ({5, 6}, ())]),

    # P: connected corner city (N-E) + road S-W, no shield.
    _mk("P", "CCRR", cities=[("NE", False)], roads=["SW"],
        fields=[({4, 7}, (0,)), ({5, 6}, ())]),

    # Q: city on three edges (N, E, W) connected, with shield.
    _mk("Q", "CCFC", cities=[("NEW", True)], fields=[({4, 5}, (0,))]),

    # R: city on three edges (N, E, W) connected, no shield.
    _mk("R", "CCFC", cities=[("NEW", False)], fields=[({4, 5}, (0,))]),

    # S: city on three edges (N, E, W) + a road off the South, with shield.
    _mk("S", "CCRC", cities=[("NEW", True)], roads=["S"],
        fields=[({4}, (0,)), ({5}, (0,))]),

    # T: city on three edges (N, E, W) + road South, no shield.
    _mk("T", "CCRC", cities=[("NEW", False)], roads=["S"],
        fields=[({4}, (0,)), ({5}, (0,))]),

    # U: a straight road (North-South).
    _mk("U", "RFRF", roads=["NS"],
        fields=[({1, 2, 3, 4}, ()), ({0, 5, 6, 7}, ())]),

    # V: a curved road (South-West).
    _mk("V", "FFRR", roads=["SW"],
        fields=[({5, 6}, ()), ({0, 1, 2, 3, 4, 7}, ())]),

    # W: a road T-junction (E, S, W).
    _mk("W", "FRRR", roads=["E", "S", "W"],
        fields=[({0, 1, 2, 7}, ()), ({3, 4}, ()), ({5, 6}, ())]),

    # X: a 4-way crossroads.
    _mk("X", "RRRR", roads=["N", "E", "S", "W"],
        fields=[({0, 7}, ()), ({1, 2}, ()), ({3, 4}, ()), ({5, 6}, ())]),
]

# name -> TileType
TILE_TYPES: Dict[str, TileType] = {t.name: t for t in _TILES}

# Stable id for each tile type (used for the one-hot in the observation).
TILE_ID: Dict[str, int] = {t.name: i for i, t in enumerate(_TILES)}
NUM_TILE_TYPES = len(_TILES)

# How many copies of each tile are in the base-game bag (totals to 72).
TILE_DISTRIBUTION: Dict[str, int] = {
    "A": 2, "B": 4, "C": 1, "D": 4, "E": 5, "F": 2, "G": 1, "H": 3,
    "I": 2, "J": 3, "K": 3, "L": 3, "M": 2, "N": 3, "O": 2, "P": 3,
    "Q": 1, "R": 3, "S": 2, "T": 1, "U": 8, "V": 9, "W": 4, "X": 1,
}
assert sum(TILE_DISTRIBUTION.values()) == 72

# The game starts with one D tile face-up in the centre of the table.
START_TILE = "D"

# Maximum number of placeable feature segments on any single tile. This sizes
# the meeple part of the action space. (The crossroads X has 4 roads + 4 fields
# = 8, which is the maximum.)
MAX_FEATURE_SLOTS = max(
    len(t.cities) + len(t.roads) + len(t.fields) + (1 if t.monastery else 0)
    for t in _TILES
)


# --------------------------------------------------------------------------- #
# A tile *instance* placed on the board, with a concrete rotation
# --------------------------------------------------------------------------- #

@dataclass
class PlacedTile:
    """A concrete tile placed on the board at a particular rotation.

    Rotation is given as the number of 90-degree *clockwise* steps (0..3).
    All of the tile's geometry (edges, segments, field positions) is rotated
    on construction so the rest of the engine can work in absolute board
    orientation without ever thinking about rotation again.
    """

    tile_type: TileType
    rotation: int  # 0, 1, 2 or 3 (number of 90 deg clockwise turns)

    # --- rotated geometry (filled in __post_init__) -------------------------
    edges: Tuple[Terrain, Terrain, Terrain, Terrain] = ()
    cities: list = field(default_factory=list)   # list of (frozenset edges, shield)
    roads: list = field(default_factory=list)     # list of frozenset edges
    fields: list = field(default_factory=list)    # list of (frozenset pos, adj idx)
    monastery: bool = False

    # --- engine bookkeeping, filled when the tile is placed on a Board -------
    city_nodes: list = field(default_factory=list)   # parallel to self.cities
    road_nodes: list = field(default_factory=list)   # parallel to self.roads
    field_nodes: list = field(default_factory=list)  # parallel to self.fields
    monastery_node: int | None = None
    features: list = field(default_factory=list)     # canonical meeple order
    edge_city: dict = field(default_factory=dict)    # edge -> city node id
    edge_road: dict = field(default_factory=dict)    # edge -> road node id
    pos_field: dict = field(default_factory=dict)    # position -> field node id

    def __post_init__(self):
        r = self.rotation % 4
        b = self.tile_type.edges
        # An edge sitting at direction i after rotating CW by r came from the
        # base direction (i - r): new_edges[i] = base[(i - r) % 4].
        self.edges = tuple(b[(i - r) % 4] for i in range(4))
        # City / road segments: every edge e shifts to (e + r) % 4.
        self.cities = [
            (frozenset((e + r) % 4 for e in es), shield)
            for (es, shield) in self.tile_type.cities
        ]
        self.roads = [
            frozenset((e + r) % 4 for e in es) for es in self.tile_type.roads
        ]
        # Field positions shift by 2 per 90-degree turn: p -> (p + 2r) % 8.
        self.fields = [
            (frozenset((p + 2 * r) % 8 for p in pos), adj)
            for (pos, adj) in self.tile_type.fields
        ]
        self.monastery = self.tile_type.monastery

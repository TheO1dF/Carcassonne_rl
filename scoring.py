"""
scoring.py
==========

The scoring engine for Carcassonne, built around a **Disjoint-Set (Union-Find)**
data structure.

WHY UNION-FIND?
---------------
Cities, roads and fields are *connected components* that grow as tiles are
placed. When you place a tile whose city edge touches an existing city, those
two cities become **one** feature. Union-Find is the classic, near-O(1) data
structure for "merge two sets / ask which set an element is in", which is
exactly what we need to:

  * know when a city / road is *complete* (no open edges left),
  * detect that a meeple-placement target is already *occupied* (its component
    already contains a meeple),
  * tally final farm scores (which complete cities does a field touch?).

Each *feature segment* of each placed tile becomes one node. Nodes that belong
to the same connected feature are unioned together. Aggregate information
(number of tiles, open edges, shields, meeples, ...) is stored on the
**root** of each set and merged on union.

This module is deliberately geometry-free: the board (logic.py) decides *which*
nodes to union; this engine just maintains the sets and the per-component data.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from tiles import FeatureType


def meeple_counts(meeples: List[tuple]) -> Counter:
    """Count meeples per player on a feature.

    ``meeples`` is a list of ``(player, x, y, feature_type)`` tuples.
    """
    return Counter(m[0] for m in meeples)


def majority_winners(meeples: List[tuple]) -> List[int]:
    """Return the player(s) with the most meeples on a feature.

    In Carcassonne, when a feature scores, *all* players tied for the most
    meeples score the full value (ties are friendly). Returns an empty list if
    nobody has a meeple on the feature.
    """
    counts = meeple_counts(meeples)
    if not counts:
        return []
    top = max(counts.values())
    return [player for player, c in counts.items() if c == top]


class FeatureGraph:
    """Union-Find over feature segments + per-component aggregate data.

    Every node has an integer id. ``data[root]`` holds the aggregate info for
    the whole component (only valid on the root; non-root entries are deleted
    on merge). The fields stored per component are:

        type        : FeatureType of the component
        tiles       : set of (x, y) board cells the feature spans
        open        : number of still-open border edges (CITY/ROAD only).
                      A component is *complete* when this reaches 0.
        shields     : number of coat-of-arms shields (CITY only)
        meeples     : list of (player, x, y, feature_type) meeples on it
        adj_cities  : set of city node ids a FIELD is adjacent to (farm scoring)
        pos         : (x, y) of the tile, for MONASTERY components
        scored      : True once the feature has been scored (avoid double count)
    """

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}
        self.data: Dict[int, dict] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ #
    # Node creation
    # ------------------------------------------------------------------ #
    def new_node(self, ftype: FeatureType, open_edges: int = 0,
                 shields: int = 0, pos=None) -> int:
        """Create a fresh single-element component and return its id."""
        i = self._next_id
        self._next_id += 1
        self.parent[i] = i
        self.rank[i] = 0
        self.data[i] = {
            "type": ftype,
            "tiles": set(),
            "open": open_edges,
            "shields": shields,
            "meeples": [],
            "adj_cities": set(),
            "pos": pos,
            "scored": False,
        }
        return i

    # ------------------------------------------------------------------ #
    # Union-Find core (path compression + union by rank)
    # ------------------------------------------------------------------ #
    def find(self, x: int) -> int:
        """Return the representative (root) of x's set, compressing the path."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Second pass: point every node on the path directly at the root.
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def _merge_data(self, ra: int, rb: int) -> None:
        """Fold component rb's aggregate data into ra, then drop rb."""
        a, b = self.data[ra], self.data[rb]
        a["tiles"] |= b["tiles"]
        a["open"] += b["open"]
        a["shields"] += b["shields"]
        a["meeples"] += b["meeples"]
        a["adj_cities"] |= b["adj_cities"]
        a["scored"] = a["scored"] or b["scored"]
        if a["pos"] is None:
            a["pos"] = b["pos"]
        del self.data[rb]

    def union(self, x: int, y: int) -> int:
        """Merge the sets containing x and y; return the new root."""
        ra, rb = self.find(x), self.find(y)
        if ra == rb:
            return ra
        # Union by rank keeps the tree shallow.
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self._merge_data(ra, rb)
        return ra

    def connect(self, x: int, y: int) -> int:
        """Connect two CITY/ROAD segments that share one border edge.

        Besides merging the components, this *closes* one open edge on each
        side, so the component's ``open`` count drops by 2. This is what
        eventually drives a city/road to completion (``open == 0``).

        Note we still decrement by 2 even when x and y are already in the same
        component (a tile can close a loop onto itself).
        """
        ra, rb = self.find(x), self.find(y)
        if ra == rb:
            self.data[ra]["open"] -= 2
            return ra
        root = self.union(x, y)
        self.data[root]["open"] -= 2
        return root


# --------------------------------------------------------------------------- #
# Per-component scoring values
# --------------------------------------------------------------------------- #

def feature_value(data: dict, final: bool) -> int:
    """Point value of a completed (or end-game) CITY / ROAD component.

    * CITY  : 2 points per tile + 2 per shield when *completed during play*;
              only 1 point per tile + 1 per shield if still unfinished at the
              end of the game (``final=True``).
    * ROAD  : 1 point per tile, whether completed during play or at game end.

    Monasteries and fields need board geometry and are scored in logic.py.
    """
    t = data["type"]
    if t == FeatureType.CITY:
        per = 1 if final else 2
        return per * len(data["tiles"]) + per * data["shields"]
    if t == FeatureType.ROAD:
        return len(data["tiles"])
    return 0

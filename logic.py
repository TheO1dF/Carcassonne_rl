"""
logic.py
========

The ``Board`` ties tile geometry (tiles.py) to the Union-Find scoring engine
(scoring.py). It owns:

  * the grid of placed tiles,
  * edge-matching legality (`can_place`),
  * the actual placement, which wires up feature segments to their neighbours
    (`place`),
  * enumeration of where the current tile may go and which meeple slots are
    legal there (`candidate_cells`, `evaluate_placement`),
  * the scoring triggers, both during play (`resolve_completions`) and at the
    end of the game (`end_game_score`).

The Gym environment (env.py) is a thin wrapper around this class.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List, Tuple

from tiles import (
    N, E, S, W, OPPOSITE, DELTA, EDGE_POSITIONS, POSITION_EDGE, FIELD_NEIGHBOR,
    Terrain, FeatureType, PlacedTile, TileType,
)
from scoring import FeatureGraph, majority_winners, feature_value


class Board:
    """A square Carcassonne board of side ``size`` (in tile cells)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.grid: Dict[Tuple[int, int], PlacedTile] = {}
        self.graph = FeatureGraph()

    def clone(self) -> "Board":
        """A deep, independent copy (used by the greedy bot for 1-ply look-ahead).

        Tile *types* are shared (see ``TileType.__deepcopy__``); everything
        mutable (grid, union-find, per-component data) is duplicated.
        """
        return copy.deepcopy(self)

    # ===================================================================== #
    # Geometry helpers
    # ===================================================================== #
    def _neighbor(self, x: int, y: int, d: int) -> Tuple[int, int]:
        dx, dy = DELTA[d]
        return x + dx, y + dy

    def _neighbors8(self, x: int, y: int) -> List[Tuple[int, int]]:
        """The 8 surrounding cells (used for monastery completion)."""
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                out.append((x + dx, y + dy))
        return out

    def _monastery_count(self, x: int, y: int) -> int:
        """How many of the 8 neighbours of (x, y) are occupied."""
        return sum(1 for c in self._neighbors8(x, y) if c in self.grid)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    # ===================================================================== #
    # Placement legality (edge matching)
    # ===================================================================== #
    def can_place(self, tile_type: TileType, x: int, y: int, rotation: int) -> bool:
        """True iff ``tile_type`` at this rotation legally fits at (x, y).

        Rules: the cell must be empty and in bounds, the tile must touch at
        least one existing tile, and every touching edge must carry the same
        terrain as the neighbour's facing edge.
        """
        if not self.in_bounds(x, y) or (x, y) in self.grid:
            return False
        pt = PlacedTile(tile_type, rotation)
        touches = False
        for d in (N, E, S, W):
            nb = self.grid.get(self._neighbor(x, y, d))
            if nb is None:
                continue
            touches = True
            if pt.edges[d] != nb.edges[OPPOSITE[d]]:
                return False
        return touches

    def candidate_cells(self) -> List[Tuple[int, int]]:
        """All empty, in-bounds cells adjacent to at least one placed tile.

        The very first move (empty board) is the centre cell.
        """
        if not self.grid:
            c = self.size // 2
            return [(c, c)]
        cells = set()
        for (x, y) in self.grid:
            for d in (N, E, S, W):
                nx, ny = self._neighbor(x, y, d)
                if (nx, ny) not in self.grid and self.in_bounds(nx, ny):
                    cells.add((nx, ny))
        return list(cells)

    def has_any_placement(self, tile_type: TileType) -> bool:
        """Can ``tile_type`` be legally placed anywhere at any rotation?"""
        for (x, y) in self.candidate_cells():
            for rot in range(4):
                if self.can_place(tile_type, x, y, rot):
                    return True
        return False

    # ===================================================================== #
    # Meeple legality for a *hypothetical* placement
    # ===================================================================== #
    def evaluate_placement(self, tile_type, x, y, rotation) -> List[Tuple[str, bool]]:
        """For a candidate placement, report each placeable feature & whether
        it is already occupied.

        Returns a list of ``(kind, occupied)`` in the **canonical meeple order**
        (cities, then roads, then fields, then monastery) -- the same order used
        when actually placing a meeple, so the i-th entry corresponds to meeple
        feature index ``i``.

        A feature is "occupied" if any existing component it would merge with
        already holds a meeple. Because every existing feature is already a
        component in the graph, we can determine this by simply inspecting the
        relevant neighbour components -- no need to clone the whole board.
        """
        g = self.graph
        pt = PlacedTile(tile_type, rotation)
        result: List[Tuple[str, bool]] = []

        # --- cities -------------------------------------------------------
        for (edges, _shield) in pt.cities:
            occupied = False
            for e in edges:
                nb = self.grid.get(self._neighbor(x, y, e))
                if nb is None:
                    continue
                ncid = nb.edge_city.get(OPPOSITE[e])
                if ncid is not None and g.data[g.find(ncid)]["meeples"]:
                    occupied = True
                    break
            result.append(("city", occupied))

        # --- roads --------------------------------------------------------
        for edges in pt.roads:
            occupied = False
            for e in edges:
                nb = self.grid.get(self._neighbor(x, y, e))
                if nb is None:
                    continue
                nrid = nb.edge_road.get(OPPOSITE[e])
                if nrid is not None and g.data[g.find(nrid)]["meeples"]:
                    occupied = True
                    break
            result.append(("road", occupied))

        # --- fields -------------------------------------------------------
        for (positions, _adj) in pt.fields:
            occupied = False
            for p in positions:
                d = POSITION_EDGE[p]
                nb = self.grid.get(self._neighbor(x, y, d))
                if nb is None:
                    continue
                nfid = nb.pos_field.get(FIELD_NEIGHBOR[d][p])
                if nfid is not None and g.data[g.find(nfid)]["meeples"]:
                    occupied = True
                    break
            result.append(("field", occupied))

        # --- monastery (never connects, so never occupied) ----------------
        if pt.monastery:
            result.append(("monastery", False))

        return result

    # ===================================================================== #
    # Actually placing a tile (and optionally a meeple)
    # ===================================================================== #
    def place(self, tile_type, x, y, rotation,
              meeple_feature: int | None = None, player: int | None = None) -> PlacedTile:
        """Place a tile, wire its features to neighbours, optionally drop a meeple.

        ``meeple_feature`` is an index into the tile's canonical feature list
        (cities, roads, fields, monastery), or ``None`` for no meeple.
        """
        g = self.graph
        pt = PlacedTile(tile_type, rotation)
        self.grid[(x, y)] = pt

        # ---- 1. create one graph node per feature segment ----------------
        edge_city: Dict[int, int] = {}
        edge_road: Dict[int, int] = {}
        pos_field: Dict[int, int] = {}

        for (edges, shield) in pt.cities:
            nid = g.new_node(FeatureType.CITY, open_edges=len(edges),
                             shields=1 if shield else 0)
            g.data[nid]["tiles"].add((x, y))
            pt.city_nodes.append(nid)
            for e in edges:
                edge_city[e] = nid

        for edges in pt.roads:
            nid = g.new_node(FeatureType.ROAD, open_edges=len(edges))
            g.data[nid]["tiles"].add((x, y))
            pt.road_nodes.append(nid)
            for e in edges:
                edge_road[e] = nid

        for (positions, adj) in pt.fields:
            nid = g.new_node(FeatureType.FIELD)
            pt.field_nodes.append(nid)
            # Record which city segments (of THIS tile) the field touches.
            for ci in adj:
                g.data[nid]["adj_cities"].add(pt.city_nodes[ci])
            for p in positions:
                pos_field[p] = nid

        if pt.monastery:
            nid = g.new_node(FeatureType.MONASTERY, pos=(x, y))
            g.data[nid]["tiles"].add((x, y))
            pt.monastery_node = nid

        pt.edge_city, pt.edge_road, pt.pos_field = edge_city, edge_road, pos_field
        # Canonical meeple-target ordering: cities, roads, fields, monastery.
        pt.features = pt.city_nodes + pt.road_nodes + pt.field_nodes
        if pt.monastery_node is not None:
            pt.features.append(pt.monastery_node)

        # ---- 2. connect to each existing neighbour -----------------------
        for d in (N, E, S, W):
            nb = self.grid.get(self._neighbor(x, y, d))
            if nb is None:
                continue
            od = OPPOSITE[d]
            terr = pt.edges[d]
            # City / road segments sharing this edge get connected (closing
            # one open edge on each side).
            if terr == Terrain.CITY:
                g.connect(edge_city[d], nb.edge_city[od])
            elif terr == Terrain.ROAD:
                g.connect(edge_road[d], nb.edge_road[od])
            # Fields: both half-edges along this border join with the
            # neighbour's matching half-edges (this also happens along road
            # edges, where field sits on either side of the road).
            for p in EDGE_POSITIONS[d]:
                mf = pos_field.get(p)
                if mf is None:
                    continue
                nf = nb.pos_field.get(FIELD_NEIGHBOR[d][p])
                if nf is not None:
                    g.union(mf, nf)

        # ---- 3. optionally place a meeple --------------------------------
        if meeple_feature is not None and player is not None:
            node = pt.features[meeple_feature]
            root = g.find(node)
            ftype = g.data[root]["type"]
            g.data[root]["meeples"].append((player, x, y, int(ftype)))

        return pt

    # ===================================================================== #
    # Scoring trigger after a placement (cities / roads / monasteries)
    # ===================================================================== #
    def resolve_completions(self, x: int, y: int):
        """Score every feature that *completed* because of the tile at (x, y).

        Returns ``(points, returned)`` where:
          * points[player]   = points scored this turn,
          * returned[player] = meeples returned to that player's supply.
        """
        g = self.graph
        points: Dict[int, int] = defaultdict(int)
        returned: Dict[int, int] = defaultdict(int)
        pt = self.grid[(x, y)]

        # --- cities & roads touched by this tile --------------------------
        seen = set()
        for nid in pt.city_nodes + pt.road_nodes:
            r = g.find(nid)
            if r in seen:
                continue
            seen.add(r)
            d = g.data[r]
            if d["open"] == 0 and not d["scored"]:
                self._distribute(d, feature_value(d, final=False), points, returned)
                d["scored"] = True

        # --- monasteries: this tile may complete its own or a neighbour's -
        for (cx, cy) in [(x, y)] + self._neighbors8(x, y):
            t = self.grid.get((cx, cy))
            if t is None or t.monastery_node is None:
                continue
            d = g.data[g.find(t.monastery_node)]
            if d["scored"]:
                continue
            if self._monastery_count(cx, cy) == 8:
                self._distribute(d, 9, points, returned)  # 1 tile + 8 neighbours
                d["scored"] = True

        return points, returned

    def _distribute(self, d, value, points, returned):
        """Award ``value`` to the meeple-majority owner(s) of feature ``d`` and
        return all of its meeples to their owners."""
        for player in majority_winners(d["meeples"]):
            points[player] += value
        for (player, _x, _y, _ft) in d["meeples"]:
            returned[player] += 1
        d["meeples"] = []

    # ===================================================================== #
    # End-of-game scoring (incomplete features + farms)
    # ===================================================================== #
    def end_game_score(self) -> Dict[int, int]:
        """Score all still-meepled features once the bag is empty.

        * incomplete cities / roads / monasteries at reduced value,
        * fields: 3 points for each *completed* city the field is adjacent to.
        """
        g = self.graph
        points: Dict[int, int] = defaultdict(int)

        for r, d in list(g.data.items()):
            t = d["type"]

            if t in (FeatureType.CITY, FeatureType.ROAD):
                if d["scored"] or not d["meeples"]:
                    continue
                value = feature_value(d, final=True)
                for player in majority_winners(d["meeples"]):
                    points[player] += value

            elif t == FeatureType.MONASTERY:
                if d["scored"] or not d["meeples"]:
                    continue
                value = 1 + self._monastery_count(*d["pos"])
                for player in majority_winners(d["meeples"]):
                    points[player] += value

            elif t == FeatureType.FIELD:
                if not d["meeples"]:
                    continue
                # Count distinct *completed* cities this field supplies.
                completed_city_roots = set()
                for cid in d["adj_cities"]:
                    cr = g.find(cid)
                    cd = g.data.get(cr)
                    if cd and cd["type"] == FeatureType.CITY and cd["open"] == 0:
                        completed_city_roots.add(cr)
                value = 3 * len(completed_city_roots)
                if value > 0:
                    for player in majority_winners(d["meeples"]):
                        points[player] += value

        return points

    def end_game_scoring_meeples(self) -> Dict[int, int]:
        """Per player, count meeples that are stranded at game end on a feature
        that STILL SCORES for them (so they earn the majority/tie there).

        This mirrors :meth:`end_game_score` but counts meeples instead of points.
        It lets the reward shaper tell *productive* end-game commitments (farms
        feeding completed cities, incomplete-but-scoring cities/roads/monasteries
        -- which the champion endorses) apart from genuinely wasted meeples
        stranded on dead, zero-point features. Only the former should have their
        meeple opportunity-cost refunded; the latter stay penalised.
        """
        g = self.graph
        productive: Dict[int, int] = defaultdict(int)

        for r, d in list(g.data.items()):
            if not d["meeples"]:
                continue
            t = d["type"]

            if t in (FeatureType.CITY, FeatureType.ROAD):
                value = 0 if d["scored"] else feature_value(d, final=True)
            elif t == FeatureType.MONASTERY:
                value = 0 if d["scored"] else 1 + self._monastery_count(*d["pos"])
            elif t == FeatureType.FIELD:
                completed_city_roots = set()
                for cid in d["adj_cities"]:
                    cr = g.find(cid)
                    cd = g.data.get(cr)
                    if cd and cd["type"] == FeatureType.CITY and cd["open"] == 0:
                        completed_city_roots.add(cr)
                value = 3 * len(completed_city_roots)
            else:
                value = 0

            if value > 0:
                # Only the meeples that actually take the points (majority/tie)
                # count as productive; an out-voted contesting meeple scored 0.
                winners = set(majority_winners(d["meeples"]))
                for (player, _x, _y, _ft) in d["meeples"]:
                    if player in winners:
                        productive[player] += 1

        return dict(productive)

    # ===================================================================== #
    # Convenience: list active meeples as (player, x, y, feature_type)
    # ===================================================================== #
    def active_meeples(self) -> List[tuple]:
        out = []
        for d in self.graph.data.values():
            out.extend(d["meeples"])
        return out

    def active_meeples_detailed(self) -> List[tuple]:
        """Like :meth:`active_meeples` but each entry is
        ``(player, x, y, ftype, open_edges)``.

        ``open_edges`` is how many tile-edges the meeple's feature still needs to
        *complete* (0 for fields/monasteries, which don't close by edge-matching).
        It is the recoverability signal -- low = about to score and return the
        meeple, high = a wide-open trap -- that the policy needs to see.
        """
        out = []
        for d in self.graph.data.values():
            oe = int(d.get("open", 0))
            for (player, x, y, ftype) in d["meeples"]:
                out.append((player, x, y, ftype, oe))
        return out

    def player_open_edges(self, player: int) -> int:
        """Total open (unfilled) edges across ``player``'s meepled cities/roads.

        Lower is better -- it is the distance to completing (and recovering the
        meeples from) everything that player currently occupies. Fields and
        monasteries contribute 0 (their ``open`` is not edge-based). Used by the
        reward shaper to reward *progress toward closing* one's own features.
        """
        total = 0
        for d in self.graph.data.values():
            oe = int(d.get("open", 0))
            if oe <= 0 or not d["meeples"]:
                continue
            if any(m[0] == player for m in d["meeples"]):
                total += oe
        return total

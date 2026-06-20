"""
env.py
======

``CarcassonneEnv`` -- a Gymnasium environment for the Carcassonne base game,
designed for training Deep RL agents via **self-play** with **action masking**.

DESIGN OVERVIEW
---------------

* One *environment step* = one full player turn:
      (1) place the drawn tile at (x, y, rotation), and
      (2) optionally place a meeple on one feature of that tile.
  After the move, completed features are scored, meeples are returned, the
  turn passes to the opponent, and a fresh tile is drawn for them.

* ACTION SPACE -- a single flat ``Discrete`` index that encodes
      (board cell, rotation, meeple choice)
  via  ``action = (cell * 4 + rotation) * MEEPLE_OPTIONS + meeple_choice``.
  ``meeple_choice == 0`` means "no meeple"; ``1..MAX_FEATURE_SLOTS`` selects the
  i-th placeable feature of the just-placed tile.

* ACTION MASKING -- Carcassonne has an enormous nominal action space but very
  few legal moves at any instant (a tile only fits in a handful of spots).
  ``get_valid_actions()`` / ``action_masks()`` return a boolean mask so a
  masked-policy agent (e.g. MaskablePPO) never wastes probability on illegal
  moves. The mask is also exposed inside the observation.

* OBSERVATION -- a ``Dict`` of tensors: the board planes, the current tile,
  remaining meeples, scores, whose turn it is, and the action mask.

* REWARD -- the points the acting player scores on their turn (feature
  completions), plus their share of end-game scoring on the final step. This
  is a dense, per-player reward suitable for self-play; reshape as you like.
"""

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tiles import (
    TILE_TYPES, TILE_DISTRIBUTION, TILE_ID, NUM_TILE_TYPES, START_TILE,
    MAX_FEATURE_SLOTS, Terrain, OPPOSITE, N, E, S, W,
)
from logic import Board


# Board planes in the observation tensor (channel index -> meaning):
#   0: occupied (1 if a tile sits here)
#   1: North edge terrain   (0 empty / 1 field / 2 road / 3 city)
#   2: East  edge terrain
#   3: South edge terrain
#   4: West  edge terrain
#   5: monastery present
#   6: meeple owner here     (0 none / 1 player-0 / 2 player-1)
#   7: meeple feature type   (FeatureType value, or 0)
#   8: gap "fillability" -- for an empty FRONTIER cell, how many of the 24 tile
#      types could still legally fill it, +1 (so a DEAD hole reads 1 and empty
#      space reads 0). Lets the policy see flexible gaps vs near-dead trap holes
#      (expert board control). 0 for occupied / non-frontier cells.
NUM_CHANNELS = 10
RECOVER_CHANNEL = 8       # open edges (distance-to-completion) of a meepled feature
FILL_CHANNEL = 9
MAX_OPEN_EDGES = 8        # clamp for the recoverability plane


# --------------------------------------------------------------------------- #
# Precomputed gap-fillability table (built once at import).
# --------------------------------------------------------------------------- #
# For every edge-constraint pattern (each of N/E/S/W is a *required* terrain or
# None = free), count how many tile TYPES have some rotation that fits. A
# frontier gap's pattern comes from its placed neighbours, so this is an O(1)
# dict lookup per cell at observation time.
_TILE_ROT_EDGE_SETS = []
for _tt in TILE_TYPES.values():
    _rots = set()
    for _r in range(4):
        _rots.add(tuple(_tt.edges[(_i - _r) % 4] for _i in range(4)))
    _TILE_ROT_EDGE_SETS.append(_rots)


def _count_fitting_types(pattern) -> int:
    n = 0
    for rots in _TILE_ROT_EDGE_SETS:
        for e in rots:
            if all(p is None or e[d] == p for d, p in enumerate(pattern)):
                n += 1
                break
    return n


_TERRAIN_OPTS = (None, Terrain.FIELD, Terrain.ROAD, Terrain.CITY)
FILLABILITY_TABLE = {pat: _count_fitting_types(pat)
                     for pat in product(_TERRAIN_OPTS, repeat=4)}
MAX_FILLABILITY = max(FILLABILITY_TABLE.values())

# The observation/action WINDOW recenters on the played area every step, so the
# board never drifts into a hard edge. The old fixed grid (start at the centre,
# tiles drift outward) masked out legal moves once play reached the boundary,
# which biased the agent toward the middle. Here the underlying board is a large,
# dict-backed logical grid; the window simply crops a fixed-size, recentred view
# around the action -- bounded cost, translation-invariant, no centre bias.
WINDOW_DEFAULT = 32
LOGICAL_SIZE = 160        # logical board side; dict-backed, so cost ~= #tiles only

# Orientation of the face-up 'D' start tile. BGA opens the game with this tile
# in its base artwork orientation, which is rotation 3 in our convention
# (== IMAGE_OFFSETS["D"] in bga_translator). Using it makes a fresh game's start
# tile match BGA's -- and the replay viewer, which forces start_rotation=3.
START_ROTATION = 3


class CarcassonneEnv(gym.Env):
    """Gymnasium environment for 2-player Carcassonne self-play."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, board_size: int = WINDOW_DEFAULT, num_meeples: int = 7,
                 render_mode: Optional[str] = None) -> None:
        super().__init__()
        # ``board_size`` is the side of the square OBSERVATION/ACTION WINDOW that
        # recenters on the played area (see _window_origin). The logical board is
        # much larger (and dict-backed) so play never hits a hard edge.
        self.window = board_size
        self.size = max(LOGICAL_SIZE, board_size * 2)   # logical board side
        self.num_meeples = num_meeples
        self.render_mode = render_mode

        # Meeple sub-action: 0 = no meeple, 1..MAX_FEATURE_SLOTS = feature i-1.
        self.MEEPLE_OPTIONS = MAX_FEATURE_SLOTS + 1
        n_actions = self.window * self.window * 4 * self.MEEPLE_OPTIONS
        self.action_space = spaces.Discrete(n_actions)

        self.observation_space = spaces.Dict({
            # high covers the largest channel value (the fillability plane).
            "board": spaces.Box(0, MAX_FILLABILITY + 1,
                                (self.window, self.window, NUM_CHANNELS), np.int8),
            "current_tile": spaces.Box(0, 1, (NUM_TILE_TYPES,), np.int8),
            "current_tile_edges": spaces.Box(0, 3, (4,), np.int8),
            "meeples_remaining": spaces.Box(0, num_meeples, (2,), np.int8),
            "scores": spaces.Box(0, 10_000, (2,), np.int32),
            "current_player": spaces.Box(0, 1, (1,), np.int8),
            # Tiles left in the bag (game phase signal). 72 base tiles.
            "tiles_remaining": spaces.Box(0, 72, (1,), np.int16),
            # Count of each tile type still to be drawn (most abundant has 9).
            "bag_counts": spaces.Box(0, 10, (NUM_TILE_TYPES,), np.int8),
            "action_mask": spaces.Box(0, 1, (n_actions,), np.int8),
        })

        # State is created in reset().
        self.board: Board | None = None
        self.bag: list[str] = []
        self.scores = [0, 0]
        self.meeples_remaining = [num_meeples, num_meeples]
        self.current_player = 0
        self.current_tile_name: Optional[str] = None
        self.current_tile_type = None
        self.turn_count = 0
        self.done = True

    # ===================================================================== #
    # Action <-> (cell, rotation, meeple) encoding
    # ===================================================================== #
    def _window_origin(self):
        """Top-left ABSOLUTE (x, y) of the current observation/action window.

        The window (``self.window`` wide) is centred on the bounding box of the
        placed tiles, so the view follows the action instead of drifting toward a
        fixed edge. It's a pure function of the board state, so _build_obs /
        get_valid_actions / _decode all agree within one turn.
        """
        grid = self.board.grid if self.board is not None else None
        half = self.window // 2
        if not grid:
            c = self.size // 2
            return c - half, c - half
        xs = [x for x, _ in grid]
        ys = [y for _, y in grid]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        return cx - half, cy - half

    def _encode(self, x, y, rotation, meeple_choice) -> int:
        # ``x, y`` are ABSOLUTE board coords; the action stores WINDOW coords.
        ox, oy = self._window_origin()
        cell = (y - oy) * self.window + (x - ox)
        return (cell * 4 + rotation) * self.MEEPLE_OPTIONS + meeple_choice

    def _decode(self, action: int):
        meeple_choice = action % self.MEEPLE_OPTIONS
        rest = action // self.MEEPLE_OPTIONS
        rotation = rest % 4
        cell = rest // 4
        wx = cell % self.window
        wy = cell // self.window
        ox, oy = self._window_origin()                 # window -> ABSOLUTE coords
        return ox + wx, oy + wy, rotation, meeple_choice

    # ===================================================================== #
    # Gymnasium API: reset
    # ===================================================================== #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.board = Board(self.size)
        self.scores = [0, 0]
        self.meeples_remaining = [self.num_meeples, self.num_meeples]
        self.current_player = 0
        self.turn_count = 0
        self.done = False

        # Build and shuffle the bag (one D is used as the face-up start tile).
        bag: list[str] = []
        for name, count in TILE_DISTRIBUTION.items():
            bag.extend([name] * count)
        bag.remove(START_TILE)
        order = self.np_random.permutation(len(bag))
        self.bag = [bag[i] for i in order]

        # Place the start tile in the centre (no scoring, no meeple), in BGA's
        # opening orientation so a fresh game matches the replay viewer.
        c = self.size // 2
        self.board.place(TILE_TYPES[START_TILE], c, c, START_ROTATION)

        # Draw the first placeable tile for player 0.
        self._draw_placeable()

        return self._build_obs(), self._info()

    # ===================================================================== #
    # Gymnasium API: step
    # ===================================================================== #
    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")

        action = int(action)
        player = self.current_player
        mask = self.get_valid_actions()

        # Guard against illegal actions (a well-behaved masked agent never hits
        # this). We penalise and end the episode rather than corrupt state.
        if action < 0 or action >= self.action_space.n or not mask[action]:
            self.done = True
            return self._build_obs(), -1.0, True, False, self._info(invalid=True)

        x, y, rotation, meeple_choice = self._decode(action)
        meeple_feature = None if meeple_choice == 0 else meeple_choice - 1

        # 1. Place tile (+ optional meeple).
        self.board.place(self.current_tile_type, x, y, rotation,
                         meeple_feature, player)
        if meeple_feature is not None:
            self.meeples_remaining[player] -= 1

        # 2. Score completed features; return meeples.
        points, returned = self.board.resolve_completions(x, y)
        for p, c in returned.items():
            self.meeples_remaining[p] += c
        for p, pts in points.items():
            self.scores[p] += pts
        reward = float(points.get(player, 0))

        # 3. Hand over to the opponent and draw their tile.
        self.turn_count += 1
        self.current_player = 1 - player

        terminated = False
        if not self._draw_placeable():
            # Bag exhausted -> game over -> end-game scoring.
            end = self.board.end_game_score()
            for p, pts in end.items():
                self.scores[p] += pts
            reward += float(end.get(player, 0))
            terminated = True
            self.done = True

        return self._build_obs(), reward, terminated, False, self._info()

    # ===================================================================== #
    # Tile drawing
    # ===================================================================== #
    def _draw_placeable(self) -> bool:
        """Draw tiles until we find one that can be placed somewhere.

        Per the rules, an unplaceable tile is discarded and another is drawn.
        Returns False if the bag empties (the game then ends).
        """
        while self.bag:
            name = self.bag.pop()
            tt = TILE_TYPES[name]
            if self.board.has_any_placement(tt):
                self.current_tile_name = name
                self.current_tile_type = tt
                return True
            # else: discard and keep drawing
        self.current_tile_name = None
        self.current_tile_type = None
        return False

    # ===================================================================== #
    # Action masking
    # ===================================================================== #
    def get_valid_actions(self) -> np.ndarray:
        """Boolean mask over the action space marking every legal move.

        For each legal (cell, rotation):
          * "no meeple" is always legal,
          * placing a meeple on feature ``i`` is legal iff that feature is not
            already occupied AND the player still has a meeple in supply.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.done or self.current_tile_type is None:
            return mask

        W = self.window
        ox, oy = self._window_origin()
        tile = self.current_tile_type
        has_meeple = self.meeples_remaining[self.current_player] > 0

        for (x, y) in self.board.candidate_cells():
            wx, wy = x - ox, y - oy
            if not (0 <= wx < W and 0 <= wy < W):
                continue                       # cell lies outside the view window
            for rot in range(4):
                if not self.board.can_place(tile, x, y, rot):
                    continue
                base = ((wy * W + wx) * 4 + rot) * self.MEEPLE_OPTIONS
                mask[base] = True  # no-meeple placement
                if has_meeple:
                    for i, (_kind, occupied) in enumerate(
                            self.board.evaluate_placement(tile, x, y, rot)):
                        if not occupied:
                            mask[base + 1 + i] = True
        return mask

    def action_masks(self) -> np.ndarray:
        """Alias matching the sb3-contrib MaskablePPO convention."""
        return self.get_valid_actions()

    def valid_action_list(self) -> list[int]:
        """The legal actions as a list of integer indices (convenience)."""
        return np.flatnonzero(self.get_valid_actions()).tolist()

    # ===================================================================== #
    # Observation / info construction
    # ===================================================================== #
    def _build_obs(self) -> dict:
        win = self.window                       # NB: don't shadow W (= West edge)
        ox, oy = self._window_origin()
        board = np.zeros((win, win, NUM_CHANNELS), dtype=np.int8)

        for (x, y), pt in self.board.grid.items():
            wx, wy = x - ox, y - oy
            if not (0 <= wx < win and 0 <= wy < win):
                continue                       # tile outside the recentred view
            board[wy, wx, 0] = 1
            board[wy, wx, 1] = int(pt.edges[N])
            board[wy, wx, 2] = int(pt.edges[E])
            board[wy, wx, 3] = int(pt.edges[S])
            board[wy, wx, 4] = int(pt.edges[W])
            if pt.monastery:
                board[wy, wx, 5] = 1

        # Meeples live on feature components; stamp them onto their tile cell.
        # Channel 6 = owner (1/2), 7 = feature type, 8 = open edges the feature
        # still needs to complete (low = about to score and return the meeple).
        for (player, mx, my, ftype, open_edges) in self.board.active_meeples_detailed():
            wx, wy = mx - ox, my - oy
            if 0 <= wx < win and 0 <= wy < win:
                board[wy, wx, 6] = player + 1
                board[wy, wx, 7] = ftype
                board[wy, wx, RECOVER_CHANNEL] = min(open_edges, MAX_OPEN_EDGES)

        # Fillability plane: gap fillability for empty frontier cells (+1; dead hole = 1).
        for (cx, cy) in self.board.candidate_cells():
            wx, wy = cx - ox, cy - oy
            if 0 <= wx < win and 0 <= wy < win:
                board[wy, wx, FILL_CHANNEL] = self._cell_fillability(cx, cy) + 1

        current_tile = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        current_edges = np.zeros(4, dtype=np.int8)
        if self.current_tile_type is not None:
            current_tile[TILE_ID[self.current_tile_name]] = 1
            current_edges[:] = [int(t) for t in self.current_tile_type.edges]

        # Count of each tile type still in the bag (the current tile has already
        # been popped, so it is excluded).
        bag_counts = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        for name in self.bag:
            bag_counts[TILE_ID[name]] += 1

        return {
            "board": board,
            "current_tile": current_tile,
            "current_tile_edges": current_edges,
            "meeples_remaining": np.array(self.meeples_remaining, dtype=np.int8),
            "scores": np.array(self.scores, dtype=np.int32),
            "current_player": np.array([self.current_player], dtype=np.int8),
            "tiles_remaining": np.array([len(self.bag)], dtype=np.int16),
            "bag_counts": bag_counts,
            "action_mask": self.get_valid_actions().astype(np.int8),
        }

    def _cell_fillability(self, x: int, y: int) -> int:
        """How many of the 24 tile TYPES could still legally fill empty (x, y).

        The constraint pattern (required terrain per side, or None where there is
        no neighbour) is read from placed neighbours and looked up in the
        precomputed ``FILLABILITY_TABLE``. 0 means a dead hole no tile can fill.
        """
        pat = []
        for d in (N, E, S, W):
            nb = self.board.grid.get(self.board._neighbor(x, y, d))
            pat.append(nb.edges[OPPOSITE[d]] if nb is not None else None)
        return FILLABILITY_TABLE[tuple(pat)]

    def meeple_open_edges(self, x: int, y: int, feature_index: int) -> int:
        """Open (unfilled) edges of the city/road feature a meeple sits on.

        For a city or road placed at ``(x, y)`` and selected by ``feature_index``
        (the meeple_choice - 1), return how many tile-edges still need to be
        filled to *complete* that feature -- i.e. how far it is from scoring and
        returning the meeple. Higher = harder to complete / less recoverable.
        Returns 0 for fields, monasteries, or out-of-range indices (those don't
        complete by edge-matching).
        """
        pt = self.board.grid.get((x, y))
        if pt is None or feature_index is None:
            return 0
        nc, nr = len(pt.cities), len(pt.roads)
        if not (0 <= feature_index < nc + nr):    # only cities/roads have open edges
            return 0
        node = pt.features[feature_index]
        data = self.board.graph.data[self.board.graph.find(node)]
        return int(data.get("open", 0))

    def productive_stranded_meeples(self, player: int) -> int:
        """How many of ``player``'s still-deployed meeples SCORE at end-game.

        Call this once the game has terminated. Farms feeding completed cities
        and incomplete-but-scoring cities/roads/monasteries are *productive*
        commitments -- they earned points even though the meeple never came back
        to supply. The reward shaper refunds the meeple opportunity-cost for
        these so legitimate end-game plays aren't punished as "stranded"; only
        meeples on dead, zero-point features remain penalised.
        """
        return self.board.end_game_scoring_meeples().get(player, 0)

    def agent_open_edges(self, player: int) -> int:
        """Total open edges across ``player``'s meepled cities/roads (lower =
        closer to completing & recovering them). The reward shaper rewards
        reducing this -- i.e. *progress toward closing your own features*."""
        return self.board.player_open_edges(player)

    def _info(self, invalid: bool = False) -> dict:
        return {
            "scores": list(self.scores),
            "current_player": self.current_player,
            "meeples_remaining": list(self.meeples_remaining),
            "current_tile": self.current_tile_name,
            "turn": self.turn_count,
            "tiles_remaining": len(self.bag),
            "invalid_action": invalid,
        }

    # ===================================================================== #
    # Rendering (ASCII)
    # ===================================================================== #
    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        print(self._render_text())

    def _render_text(self) -> str:
        grid = self.board.grid
        if not grid:
            return "(empty board)"

        xs = [x for x, _ in grid]
        ys = [y for _, y in grid]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)

        # Where each meeple sits, and who owns it.
        meeple_at = {(mx, my): player
                     for (player, mx, my, _ft) in self.board.active_meeples()}

        sym = {Terrain.CITY: "C", Terrain.ROAD: "R", Terrain.FIELD: "."}
        lines: list[str] = []
        for y in range(miny, maxy + 1):
            top, mid, bot = [], [], []
            for x in range(minx, maxx + 1):
                pt = grid.get((x, y))
                if pt is None:
                    top.append("   "); mid.append("   "); bot.append("   ")
                    continue
                center = "M" if pt.monastery else "+"
                if (x, y) in meeple_at:
                    center = str(meeple_at[(x, y)])  # show meeple owner id
                top.append(f" {sym[pt.edges[N]]} ")
                mid.append(f"{sym[pt.edges[W]]}{center}{sym[pt.edges[E]]}")
                bot.append(f" {sym[pt.edges[S]]} ")
            lines.append("|".join(top))
            lines.append("|".join(mid))
            lines.append("|".join(bot))
            lines.append("-" * (4 * (maxx - minx + 1)))

        lines.append(
            f"Scores: P0={self.scores[0]}  P1={self.scores[1]}   "
            f"Meeples: P0={self.meeples_remaining[0]} P1={self.meeples_remaining[1]}   "
            f"Tiles left: {len(self.bag)}   Turn: {self.turn_count}"
        )
        return "\n".join(lines)

    # ===================================================================== #
    # Convenience accessors
    # ===================================================================== #
    @property
    def winner(self) -> Optional[int]:
        """0 / 1 for the leader, or None while tied or mid-game."""
        if self.scores[0] == self.scores[1]:
            return None
        return 0 if self.scores[0] > self.scores[1] else 1

"""
replay_env.py
=============

``CarcassonneReplayEnv`` -- a ``CarcassonneEnv`` whose tile draw order can be
*forced* to an exact sequence, so recorded games (e.g. from Board Game Arena)
can be reproduced step-by-step. This is the backbone of the Behavioural-Cloning
data pipeline: we replay a real game through the env to recover the exact
``action`` integer for every move.

Two behavioural differences from the base env when a ``tile_sequence`` is given:

  * the bag is exactly that sequence (no random distribution), and
  * ``_draw_placeable`` does **not** skip "unplaceable" tiles. In a faithful
    replay every recorded tile *was* placed by a real player, so skipping any
    tile would desynchronise the sequence from the recorded actions.
"""

from __future__ import annotations

from typing import List, Optional

import gymnasium as gym

from env import CarcassonneEnv
from tiles import TILE_TYPES, START_TILE
from logic import Board


class CarcassonneReplayEnv(CarcassonneEnv):
    """A CarcassonneEnv with a deterministic, forced tile draw order."""

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
              tile_sequence: Optional[List[str]] = None,
              start_tile: str = START_TILE, start_rotation: int = 0):
        """Reset the env.

        Parameters
        ----------
        tile_sequence : list[str] or None
            If given, the env draws exactly these tile letters in order (the
            default random ``TILE_DISTRIBUTION`` bag is ignored). May also be
            supplied via ``options["tile_sequence"]`` for Gym-API callers.
        start_tile : str
            The face-up centre tile placed before play (BGA coordinates are
            relative to it). Defaults to the standard ``D``.
        start_rotation : int
            Rotation (0..3) of the start tile. Expose this so a replay can match
            the recorded game's start-tile orientation if it differs from ours.
        """
        # Allow passing the sequence through the standard Gym ``options`` dict.
        if tile_sequence is None and options:
            tile_sequence = options.get("tile_sequence")

        # No forced sequence -> behave exactly like the base environment.
        if tile_sequence is None:
            self._forced = False
            return super().reset(seed=seed, options=options)

        # --- forced-sequence reset (mirrors CarcassonneEnv.reset) ------------
        # Seed the RNG via the Gymnasium base (skips the random-bag construction).
        gym.Env.reset(self, seed=seed)

        self.board = Board(self.size)
        self.scores = [0, 0]
        self.meeples_remaining = [self.num_meeples, self.num_meeples]
        self.current_player = 0
        self.turn_count = 0
        self.done = False

        # Pre-place the centre start tile (the origin of BGA coordinates).
        c = self.size // 2
        self.board.place(TILE_TYPES[start_tile], c, c, start_rotation)

        # Force the draw order: pop() yields the sequence front-to-back.
        self.tile_sequence = list(tile_sequence)
        self.bag = list(reversed(self.tile_sequence))
        self._forced = True
        self._draw_placeable()

        return self._build_obs(), self._info()

    def _draw_placeable(self) -> bool:
        """Draw the next tile.

        In forced mode we pop the next tile unconditionally (no placeability
        filtering), so the sequence stays perfectly aligned with the recorded
        actions. Falls back to the base behaviour outside forced mode.
        """
        if not getattr(self, "_forced", False):
            return super()._draw_placeable()

        if self.bag:
            name = self.bag.pop()
            self.current_tile_name = name
            self.current_tile_type = TILE_TYPES[name]
            return True
        # Sequence exhausted -> game ends (handled by the base step()).
        self.current_tile_name = None
        self.current_tile_type = None
        return False

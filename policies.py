"""
policies.py
===========

Pluggable "players" for evaluation and human play. Every policy implements the
same tiny interface::

    action = policy.act(env, obs)

where ``env`` is a raw :class:`CarcassonneEnv` (so the policy may inspect the
board directly) and ``obs`` is the most recent observation dict. ``act`` must
return a legal flat action index.

Provided policies:

* ``RandomPolicy`` -- uniform over the legal action mask.
* ``GreedyPolicy`` -- a 1-ply heuristic bot that prefers moves which immediately
  score points (completing a feature on which it has / places a meeple) and that
  place meeples on cities and roads rather than farms.
* ``ModelPolicy`` -- wraps a trained ``MaskablePPO`` model; presents it a
  perspective-canonicalised observation so it always plays "as Player 0".
"""

from __future__ import annotations

import numpy as np

from tiles import FeatureType
from scoring import feature_value
from rl_model import _canonicalize  # reuse the training-time perspective flip


# --------------------------------------------------------------------------- #
# Random baseline
# --------------------------------------------------------------------------- #
class RandomPolicy:
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs) -> int:
        valid = np.flatnonzero(env.get_valid_actions())
        return int(self.rng.choice(valid))


# --------------------------------------------------------------------------- #
# Greedy heuristic bot
# --------------------------------------------------------------------------- #
class GreedyPolicy:
    """1-ply greedy bot.

    For each candidate tile placement (x, y, rotation) it clones the board,
    plays the tile, and reads off:

      * the net points the placement *immediately* scores (completing features
        on which it already has a meeple -- minus any the opponent scores),
      * for each feature of the placed tile, whether it is now complete and
        free, so that dropping a meeple there would score it right away.

    The action value is then::

        value = 10 * net_immediate_points
              + 10 * value_of_a_feature_this_meeple_would_close
              +      small bonus for meeples on cities/roads/monasteries

    so the bot strongly prefers instant scoring, then meeples on real features.
    """

    name = "Greedy"

    # Preference for *where* to drop a meeple when nothing scores immediately.
    _MEEPLE_PREF = {
        int(FeatureType.CITY): 0.3,
        int(FeatureType.ROAD): 0.2,
        int(FeatureType.MONASTERY): 0.2,
        int(FeatureType.FIELD): 0.0,   # farms only pay off at game end -> avoid
    }

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs) -> int:
        valid = np.flatnonzero(env.get_valid_actions())
        cur = env.current_player
        other = 1 - cur
        tile = env.current_tile_type

        # Group legal actions by their placement (cell + rotation); meeple
        # choices within a group share one board clone.
        groups: dict = {}
        for a in valid:
            x, y, rot, meeple = env._decode(int(a))
            groups.setdefault((x, y, rot), []).append((int(a), meeple))

        best_val = -1e18
        best_actions: list[int] = []

        for (x, y, rot), actions in groups.items():
            board2 = env.board.clone()
            board2.place(tile, x, y, rot)                  # try it (no meeple)
            points, _ = board2.resolve_completions(x, y)
            net_pre = points.get(cur, 0) - points.get(other, 0)
            feat_info = self._feature_info(board2, board2.grid[(x, y)])

            for (a, meeple) in actions:
                value = 10.0 * net_pre
                if meeple != 0:
                    ftype, complete, occupied, fval = feat_info[meeple - 1]
                    if complete and not occupied:
                        value += 10.0 * fval               # close it now -> score
                    value += self._MEEPLE_PREF.get(ftype, 0.0)
                if value > best_val + 1e-9:
                    best_val, best_actions = value, [a]
                elif value > best_val - 1e-9:
                    best_actions.append(a)

        return int(self.rng.choice(best_actions))

    def _feature_info(self, board, placed):
        """Per placed-tile feature: (ftype, complete?, occupied?, score-if-closed)."""
        g = board.graph
        info = []
        for nid in placed.features:
            d = g.data[g.find(nid)]
            t = int(d["type"])
            if t in (int(FeatureType.CITY), int(FeatureType.ROAD)):
                complete = d["open"] == 0
                value = feature_value(d, final=False) if complete else 0
            elif t == int(FeatureType.MONASTERY):
                complete = board._monastery_count(*d["pos"]) == 8
                value = 9 if complete else 0
            else:  # FIELD never completes during play
                complete, value = False, 0
            info.append((t, complete, bool(d["meeples"]), value))
        return info


# --------------------------------------------------------------------------- #
# Trained-model wrapper
# --------------------------------------------------------------------------- #
class ModelPolicy:
    """Wrap a MaskablePPO model so it can play either seat.

    The model was trained as Player 0, so when it occupies seat 1 we hand it a
    canonicalised observation (it always sees itself as Player 0).
    """

    def __init__(self, model, name: str = "Model", deterministic: bool = True):
        self.model = model
        self.name = name
        self.deterministic = deterministic

    def act(self, env, obs) -> int:
        mask = env.get_valid_actions()
        view = obs if env.current_player == 0 else _canonicalize(obs)
        action, _ = self.model.predict(
            view, action_masks=mask, deterministic=self.deterministic)
        return int(action)

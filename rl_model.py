"""
rl_model.py
===========

Reusable RL building blocks for training on ``CarcassonneEnv``:

  1. ``CarcassonneFeaturesExtractor`` -- a custom SB3 ``BaseFeaturesExtractor``
     that runs a CNN over the 2D board plane and fuses it with the 1D scalar
     features (current tile, meeples, scores, ...).

  2. ``SelfPlayWrapper`` -- a ``gym.Wrapper`` that turns the 2-player,
     turn-based ``CarcassonneEnv`` into a single-agent env from Player 0's point
     of view. The opponent (Player 1) is driven automatically by an
     ``opponent_policy`` (or random valid moves), and the reward is reshaped to
     a zero-sum signal with a terminal win/loss bonus.

  3. ``mask_fn`` -- the action-mask callback for ``sb3_contrib``'s
     ``ActionMasker``.

These are kept separate from ``train.py`` so they can be imported by evaluation
scripts or notebooks without pulling in the whole training loop.

Required packages:
    torch, gymnasium, stable-baselines3, sb3-contrib, numpy
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Keys in the observation Dict that are 1D scalar feature vectors (everything
# except the 2D ``board`` plane and the ``action_mask``, which the network must
# NOT see as an input feature -- it is only used for masking the policy logits).
SCALAR_KEYS = (
    "current_tile",
    "current_tile_edges",
    "meeples_remaining",
    "scores",
    "current_player",
    "tiles_remaining",    # game phase -> early/mid/late, which expert play hinges on
    "bag_counts",         # card counting: how many of each tile type remain
)

# Rough normalisation constants so every input lands in roughly [0, 1].
BOARD_SCALE = 4.0     # fallback per-channel divisor if the layout is unexpected
EDGE_SCALE = 3.0      # terrain ids 1..3
SCORE_SCALE = 100.0   # scores are tens of points
TILES_SCALE = 72.0    # tiles remaining (0..71)
BAG_SCALE = 10.0      # per-type bag counts (most abundant tile has 9 copies)

# Per-channel divisors for the board planes (see env.py channel layout):
#   0 tile-present, 1-4 edge terrains, 5 monastery, 6 meeple owner,
#   7 meeple feature-type, 8 recoverability (open edges, 0..8),
#   9 fillability (0..~25). Channels 8/9 have large ranges and would dwarf the
#   small 0..4 planes without their own scale.
BOARD_CHANNEL_MAX = (1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 4.0, 8.0, 25.0)


# --------------------------------------------------------------------------- #
# 1. Custom feature extractor (CNN over board + MLP over scalars)
# --------------------------------------------------------------------------- #
class CarcassonneFeaturesExtractor(BaseFeaturesExtractor):
    """Fuse a CNN view of the board with the scalar game state.

    Parameters
    ----------
    observation_space : gym.spaces.Dict
        The env observation space (must contain a ``board`` Box of shape
        (H, W, C) plus the scalar keys in ``SCALAR_KEYS``).
    features_dim : int
        Size of the final feature vector handed to the policy/value heads.
    cnn_out_dim : int
        Size of the linear projection of the flattened CNN output.
    """

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256, cnn_out_dim: int = 256):
        # We must call super().__init__ with the *final* feature dim.
        super().__init__(observation_space, features_dim)

        board_space = observation_space["board"]
        h, w, c = board_space.shape          # (H, W, Channels)
        self._board_hw = (h, w)
        self._board_channels = c
        self._num_meeples = float(observation_space["meeples_remaining"].high.max())

        # Per-channel board normalisation (broadcast over B, H, W). Falls back to a
        # uniform scale if the channel count doesn't match the known layout.
        if c == len(BOARD_CHANNEL_MAX):
            scale = torch.tensor(BOARD_CHANNEL_MAX, dtype=torch.float32)
        else:
            scale = torch.full((c,), BOARD_SCALE)
        self.register_buffer("_board_scale", scale.view(1, c, 1, 1))

        # --- CNN tower (input is permuted to (B, C, H, W)) ----------------
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer the flattened CNN size with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_flat = self.cnn(dummy).shape[1]

        self.cnn_linear = nn.Sequential(nn.Linear(cnn_flat, cnn_out_dim), nn.ReLU())

        # --- size of the concatenated scalar vector -----------------------
        self._scalar_dim = int(sum(
            int(np.prod(observation_space[k].shape)) for k in SCALAR_KEYS
        ))

        # --- fusion MLP ---------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + self._scalar_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # SB3 has already converted every sub-observation to a float32 tensor.
        board = observations["board"]                 # (B, H, W, C)
        board = board.permute(0, 3, 1, 2) / self._board_scale  # -> (B, C, H, W)
        cnn_feat = self.cnn_linear(self.cnn(board))

        scalars = [
            observations["current_tile"].float(),
            observations["current_tile_edges"].float() / EDGE_SCALE,
            observations["meeples_remaining"].float() / self._num_meeples,
            observations["scores"].float() / SCORE_SCALE,
            observations["current_player"].float(),
            observations["tiles_remaining"].float() / TILES_SCALE,
            observations["bag_counts"].float() / BAG_SCALE,
        ]
        scalar_feat = torch.cat([s.reshape(s.shape[0], -1) for s in scalars], dim=1)

        return self.fusion(torch.cat([cnn_feat, scalar_feat], dim=1))


# --------------------------------------------------------------------------- #
# 2. Self-play wrapper
# --------------------------------------------------------------------------- #
def _canonicalize(obs: dict) -> dict:
    """Return ``obs`` rewritten from the *current player's* point of view.

    The learning agent is always "Player 0". When a saved opponent model (also
    trained as Player 0) is asked to move as Player 1, we must present it with
    an observation in which *it* is Player 0, otherwise it sees the board with
    the wrong ownership/score perspective. This swaps the two players'
    perspective-dependent fields (meeple owners, scores, meeple supply) and sets
    ``current_player`` to 0. Geometry (edges, monastery, the action indices) is
    perspective-independent and left untouched.
    """
    o = {k: np.array(v, copy=True) for k, v in obs.items()}

    o["scores"] = o["scores"][::-1].copy()
    o["meeples_remaining"] = o["meeples_remaining"][::-1].copy()
    o["current_player"][...] = 0

    owner = o["board"][..., 6]
    swapped = owner.copy()
    swapped[owner == 1] = 2
    swapped[owner == 2] = 1
    o["board"][..., 6] = swapped
    return o


class SelfPlayWrapper(gym.Wrapper):
    """Expose the 2-player env as a single-agent env for Player 0.

    Each step plays the agent's move and then auto-plays the opponent until it is
    the agent's turn again (or the game ends).

    The reward is the zero-sum score margin over the cycle, ``(my points) -
    (opponent points)``, plus an optional potential-based shaping term (Ng,
    Harada & Russell, 1999) on the agent's available meeples::

        F = gamma * Phi(s') - Phi(s),   Phi(s) = meeple_potential * available_meeples,
                                        Phi(terminal) = 0

    Potential-based shaping leaves the optimal policy unchanged while rewarding
    meeple recovery and pricing commitment, which speeds up learning the
    place->complete->recover loop. Set ``meeple_potential=0`` for pure zero-sum.
    A terminal ``+/- win_bonus`` rewards the win itself.
    """

    def __init__(self, env, opponent_policy=None, win_bonus: float = 10.0,
                 meeple_potential: float = 0.5, gamma: float = 0.997,
                 opponent_deterministic: bool = True):
        super().__init__(env)
        self.opponent_policy = opponent_policy
        self.win_bonus = float(win_bonus)
        # Potential-based shaping weight on available meeples (0 -> pure zero-sum).
        self.meeple_potential = float(meeple_potential)
        self.gamma = float(gamma)            # must match the model's discount
        self.opponent_deterministic = opponent_deterministic
        self.agent_id = 0  # the learning agent always plays Player 0

    # -- helpers ---------------------------------------------------------
    def set_opponent(self, opponent_policy):
        """Swap in a new opponent (e.g. the previous generation's model)."""
        self.opponent_policy = opponent_policy

    def _scores(self):
        return list(self.env.unwrapped.scores)

    def _opponent_action(self, obs: dict) -> int:
        """Choose the opponent's (Player 1's) action."""
        mask = self.env.unwrapped.get_valid_actions()
        if self.opponent_policy is None:
            # Random *valid* move.
            valid = np.flatnonzero(mask)
            return int(self.env.unwrapped.np_random.choice(valid))
        # Learned opponent: show it the board from its own perspective.
        action, _ = self.opponent_policy.predict(
            _canonicalize(obs),
            action_masks=mask,
            deterministic=self.opponent_deterministic,
        )
        return int(action)

    def _terminal_bonus(self) -> float:
        s0, s1 = self._scores()
        if s0 > s1:
            return self.win_bonus
        if s1 > s0:
            return -self.win_bonus
        return 0.0

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Per-episode meeple bookkeeping (surfaced in the terminal info dict for
        # the TensorBoard probe: recovery throughput is THE metric for the
        # place->complete->retrieve loop -- experts ~4/game, BC ~1.25/game).
        self._ep_recovered = 0
        self._ep_placed = 0
        # Player 0 always moves first, so no opponent pre-move is needed.
        return obs, info

    def step(self, action):
        u = self.env.unwrapped
        before0, before1 = self._scores()
        meeples_before = u.meeples_remaining[self.agent_id]   # available meeples
        _ax, _ay, _rot, mc = u._decode(int(action))
        placed_meeple = mc > 0                                # meeple sub-action > 0

        # Agent plays, then the opponent plays until it is the agent's turn again.
        obs, _r, terminated, truncated, info = self.env.step(action)
        while not (terminated or truncated) and u.current_player != self.agent_id:
            opp_action = self._opponent_action(obs)
            obs, _r, terminated, truncated, info = self.env.step(opp_action)

        # Zero-sum score margin over the cycle.
        after0, after1 = self._scores()
        reward = (after0 - before0) - (after1 - before1)
        meeples_now = u.meeples_remaining[self.agent_id]

        # Potential-based meeple shaping: F = gamma*Phi(s') - Phi(s),
        # Phi = meeple_potential * available_meeples, Phi(terminal) = 0.
        if self.meeple_potential:
            phi_before = self.meeple_potential * meeples_before
            phi_after = (0.0 if (terminated or truncated)
                         else self.meeple_potential * meeples_now)
            reward += self.gamma * phi_after - phi_before

        if terminated:
            reward += self._terminal_bonus()

        # Per-episode meeple stats for TensorBoard (diagnostic only). At most one
        # meeple is placed per move, so meeples recovered this cycle is the net
        # change plus what was placed.
        self._ep_placed += int(placed_meeple)
        self._ep_recovered += (meeples_now - meeples_before) + int(placed_meeple)
        if terminated or truncated:
            info = dict(info)
            info["ep_meeples_recovered"] = int(self._ep_recovered)
            info["ep_meeples_placed"] = int(self._ep_placed)
            info["ep_meeples_stranded"] = int(u.num_meeples - meeples_now)
            info["ep_meeples_productive"] = int(u.productive_stranded_meeples(self.agent_id))

        return obs, float(reward), terminated, truncated, info


# --------------------------------------------------------------------------- #
# 3. Action-mask callback for ActionMasker
# --------------------------------------------------------------------------- #
def mask_fn(env) -> np.ndarray:
    """Return the boolean action mask for the *current* state.

    ``ActionMasker`` calls this before each agent decision. ``env.unwrapped``
    reaches the underlying ``CarcassonneEnv`` regardless of how many wrappers
    sit on top.
    """
    return env.unwrapped.get_valid_actions()

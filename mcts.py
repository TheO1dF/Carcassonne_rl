"""Monte-Carlo Tree Search for 2-player Carcassonne.

Carcassonne is stochastic: the next tile is drawn at random. The bag multiset is
known (the ``bag_counts`` observation) but the draw order is hidden, so the search
determinizes -- it re-samples the next tile from the remaining multiset each time
it needs one (``rng.shuffle(bag)`` on the clone before stepping). A turn expands
as: decision (place tile + meeple) -> chance (draw) -> decision, and an action can
have several children keyed by the drawn tile.

Values are in [-1, 1] from the perspective of the player to move; players
alternate every turn, so backup is negamax. The evaluator is pluggable:
``RolloutEvaluator`` needs no trained model, ``NeuralEvaluator`` wraps a policy/
value network. ``selfplay_game`` emits ``(obs, policy_target, value)`` tuples for
the training loop in ``alphazero.py``.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import CarcassonneEnv

MARGIN_SCALE = 50.0     # a final score margin of +-50 maps to value +-1
C_PUCT = 1.5            # exploration constant in the PUCT formula
VIRTUAL_LOSS = 1.0      # discourages concurrent in-batch sims from re-treading a path


# --------------------------------------------------------------------------- #
# Game-state wrapper (cloneable, with determinized draws)
# --------------------------------------------------------------------------- #
class GameState:
    """A thin, cloneable wrapper over ``CarcassonneEnv`` for search.

    The only subtlety is :meth:`apply`, which **re-shuffles the bag** on the
    clone before stepping, so the drawn next tile is a fresh sample from the
    remaining multiset rather than the env's secret pre-shuffled order.
    """

    def __init__(self, env: CarcassonneEnv, rng: np.random.Generator):
        self.env = env
        self.rng = rng

    @property
    def to_play(self) -> int:
        return self.env.current_player

    def is_terminal(self) -> bool:
        return self.env.done

    def legal_actions(self) -> List[int]:
        return self.env.valid_action_list()

    def observation(self) -> dict:
        return self.env._build_obs()

    def outcome_value(self, perspective: int) -> float:
        """Clipped score margin from ``perspective``'s point of view, in [-1, 1]."""
        s0, s1 = self.env.scores
        margin = (s0 - s1) if perspective == 0 else (s1 - s0)
        return float(np.clip(margin / MARGIN_SCALE, -1.0, 1.0))

    def clone_env(self) -> CarcassonneEnv:
        return copy.deepcopy(self.env)

    def apply(self, action: int) -> Tuple["GameState", Tuple]:
        """Apply ``action`` and draw the next tile from a re-sampled bag.

        Returns ``(child_state, chance_key)`` where ``chance_key`` identifies the
        random outcome (the drawn tile) so the caller can de-duplicate chance
        children.
        """
        nxt = copy.deepcopy(self.env)
        self.rng.shuffle(nxt.bag)        # determinize the hidden draw order
        nxt.step(int(action))
        child = GameState(nxt, self.rng)
        # The only randomness in the transition is which tile was drawn next.
        chance_key = (nxt.current_tile_name, len(nxt.bag))
        return child, chance_key


# --------------------------------------------------------------------------- #
# Evaluators: (prior policy over legal actions, value in [-1, 1])
# --------------------------------------------------------------------------- #
class RolloutEvaluator:
    """Network-free evaluator: uniform prior + random-playout value. Needs no
    trained model, so MCTS is usable as a baseline on its own."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def evaluate(self, state: GameState) -> Tuple[Dict[int, float], float]:
        legal = state.legal_actions()
        prior = 1.0 / max(len(legal), 1)
        priors = {a: prior for a in legal}
        return priors, self._rollout(state)

    def evaluate_batch(self, states):
        # Rollouts are inherently sequential -- no batching gain, but keep the
        # interface uniform so batched MCTS works with either evaluator.
        return [self.evaluate(s) for s in states]

    def _rollout(self, state: GameState) -> float:
        """Play uniformly-random (determinized) moves to the end; return outcome
        from ``state.to_play``'s perspective."""
        env = state.clone_env()
        perspective = state.to_play
        while not env.done:
            mask = env.get_valid_actions()
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                break
            a = int(self.rng.choice(valid))
            self.rng.shuffle(env.bag)     # re-sample the next draw each ply
            env.step(a)
        s0, s1 = env.scores
        margin = (s0 - s1) if perspective == 0 else (s1 - s0)
        return float(np.clip(margin / MARGIN_SCALE, -1.0, 1.0))


class NeuralEvaluator:
    """Wraps a trained ``MaskablePPO`` policy (BC or self-play) as the AlphaZero
    evaluator: masked-softmax of the policy head for priors, the value head for
    the leaf value. Observations for Player 1 are canonicalized so the net always
    "sees itself" as Player 0 (matching how it was trained)."""

    def __init__(self, model, device: Optional[str] = None):
        import torch
        self.torch = torch
        self.policy = model.policy
        self.policy.set_training_mode(False)
        self.device = device or model.device

    def evaluate(self, state: GameState) -> Tuple[Dict[int, float], float]:
        from rl_model import _canonicalize
        torch = self.torch
        obs = state.observation()
        if state.to_play == 1:
            obs = _canonicalize(obs)
        with torch.no_grad():
            obs_t, _ = self.policy.obs_to_tensor(obs)
            features = self.policy.extract_features(obs_t)
            latent_pi, _ = self.policy.mlp_extractor(features)
            logits = self.policy.action_net(latent_pi).cpu().numpy()[0]
            value = float(self.policy.predict_values(obs_t).cpu().numpy()[0, 0])

        legal = state.legal_actions()
        z = logits[legal]
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        priors = {int(a): float(pi) for a, pi in zip(legal, p)}
        return priors, float(np.clip(value, -1.0, 1.0))

    def evaluate_batch(self, states):
        """Evaluate many leaf states in a single network forward pass, which
        amortises GPU inference latency across the batch."""
        from rl_model import _canonicalize
        torch = self.torch
        obs_list = []
        for s in states:
            o = s.observation()
            if s.to_play == 1:
                o = _canonicalize(o)
            obs_list.append(o)
        keys = obs_list[0].keys()
        batch = {k: torch.as_tensor(np.stack([o[k] for o in obs_list])).to(self.device)
                 for k in keys}
        with torch.no_grad():
            features = self.policy.extract_features(batch)
            latent_pi, _ = self.policy.mlp_extractor(features)
            logits = self.policy.action_net(latent_pi).cpu().numpy()       # (B, A)
            values = self.policy.predict_values(batch).cpu().numpy()[:, 0]  # (B,)
        out = []
        for i, s in enumerate(states):
            legal = s.legal_actions()
            z = logits[i][legal]
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            out.append(({int(a): float(pi) for a, pi in zip(legal, p)},
                        float(np.clip(values[i], -1.0, 1.0))))
        return out


# --------------------------------------------------------------------------- #
# Search tree
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("state", "to_play", "expanded", "P", "N", "W", "children")

    def __init__(self, state: GameState, to_play: int):
        self.state = state
        self.to_play = to_play
        self.expanded = False
        self.P: Dict[int, float] = {}            # prior per action
        self.N: Dict[int, int] = {}              # visit count per action
        self.W: Dict[int, float] = {}            # total value per action
        self.children: Dict[Tuple, "Node"] = {}  # (action, chance_key) -> Node

    def q(self, a: int) -> float:
        n = self.N.get(a, 0)
        return self.W[a] / n if n > 0 else 0.0

    def expand(self, priors: Dict[int, float]) -> None:
        self.P = priors
        for a in priors:
            self.N[a] = 0
            self.W[a] = 0.0
        self.expanded = True


class MCTS:
    def __init__(self, evaluator, c_puct: float = C_PUCT,
                 rng: Optional[np.random.Generator] = None,
                 dirichlet_alpha: Optional[float] = None,
                 dirichlet_frac: float = 0.25):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.rng = rng or np.random.default_rng()
        self.dirichlet_alpha = dirichlet_alpha     # set (e.g. 0.3) for self-play
        self.dirichlet_frac = dirichlet_frac

    def search(self, root_state: GameState, n_sims: int, batch_size: int = 1) -> Node:
        root = Node(root_state, root_state.to_play)
        priors, _ = self.evaluator.evaluate(root_state)
        root.expand(priors)
        if self.dirichlet_alpha:
            self._add_root_noise(root)

        if batch_size <= 1:                       # exact sequential MCTS
            for _ in range(n_sims):
                self._simulate(root)
            return root

        # Leaf-parallel MCTS: collect a batch of leaves (virtual loss makes
        # concurrent sims diverge), evaluate them in one network pass, then expand
        # and back them all up. Larger batches are faster but act on staler stats.
        done = 0
        while done < n_sims:
            b = min(batch_size, n_sims - done)
            collected = [self._select_to_leaf(root) for _ in range(b)]
            to_eval, seen = [], set()
            for _path, leaf, is_term in collected:
                if not is_term and not leaf.expanded and id(leaf) not in seen:
                    seen.add(id(leaf))
                    to_eval.append(leaf)
            valmap = {}
            if to_eval:
                results = self.evaluator.evaluate_batch([lf.state for lf in to_eval])
                for lf, (pri, val) in zip(to_eval, results):
                    lf.expand(pri)
                    valmap[id(lf)] = val
            for path, leaf, is_term in collected:
                if is_term:
                    v = leaf.state.outcome_value(leaf.to_play)
                else:
                    v = valmap.get(id(leaf))
                    if v is None:                 # already expanded (defensive)
                        _, v = self.evaluator.evaluate(leaf.state)
                self._backup(path, v)
            done += b
        return root

    def _select_to_leaf(self, root: Node):
        """Descend with virtual loss to an unexpanded or terminal leaf, returning
        ``(path, leaf, is_terminal)``. Virtual loss temporarily worsens each
        traversed edge so the next sim in the batch takes a different path."""
        path = []
        node = root
        while True:
            if node.state.is_terminal():
                return path, node, True
            if not node.expanded:
                return path, node, False
            a = self._select(node)
            node.N[a] += VIRTUAL_LOSS
            node.W[a] -= VIRTUAL_LOSS
            child_state, chance_key = node.state.apply(a)
            edge = (a, chance_key)
            child = node.children.get(edge)
            if child is None:
                child = Node(child_state, child_state.to_play)
                node.children[edge] = child
            path.append((node, a))
            node = child

    def _backup(self, path, leaf_value: float) -> None:
        """Negamax backup that also removes the virtual loss applied on descent.
        Per edge: undo (+VL was added to N, -VL to W) and apply the real visit
        (+1 to N, +v to W) -> net ``N += 1-VL``, ``W += v+VL``."""
        v = leaf_value
        for node, a in reversed(path):
            v = -v
            node.N[a] += 1 - VIRTUAL_LOSS
            node.W[a] += VIRTUAL_LOSS + v

    def _add_root_noise(self, root: Node) -> None:
        """AlphaZero root exploration: mix Dirichlet noise into the priors."""
        actions = list(root.P)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        f = self.dirichlet_frac
        for a, n in zip(actions, noise):
            root.P[a] = (1 - f) * root.P[a] + f * n

    def _simulate(self, node: Node) -> float:
        # Terminal: exact outcome from this node's mover perspective.
        if node.state.is_terminal():
            return node.state.outcome_value(node.to_play)

        # Leaf: expand with the evaluator and return its value estimate.
        if not node.expanded:
            priors, value = self.evaluator.evaluate(node.state)
            node.expand(priors)
            return value

        a = self._select(node)
        child_state, chance_key = node.state.apply(a)
        edge = (a, chance_key)
        child = node.children.get(edge)
        if child is None:
            child = Node(child_state, child_state.to_play)
            node.children[edge] = child

        # Negamax: the child's value is from the opponent's perspective.
        v = -self._simulate(child)
        node.N[a] += 1
        node.W[a] += v
        return v

    def _select(self, node: Node) -> int:
        total_n = 0
        for n in node.N.values():
            total_n += n
        sqrt_total = math.sqrt(total_n + 1)
        best_a, best_score = None, -float("inf")
        for a, p in node.P.items():
            u = self.c_puct * p * sqrt_total / (1 + node.N[a])
            score = node.q(a) + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a


# --------------------------------------------------------------------------- #
# Policy / action extraction from a searched root
# --------------------------------------------------------------------------- #
def visit_policy(root: Node) -> Dict[int, float]:
    """Normalized visit counts -- the AlphaZero policy target / improved policy."""
    total = sum(root.N.values())
    if total == 0:
        n = len(root.P)
        return {a: 1.0 / n for a in root.P}
    return {a: root.N[a] / total for a in root.N}


def best_action(root: Node, temperature: float = 0.0,
                rng: Optional[np.random.Generator] = None) -> int:
    """Pick the move. ``temperature==0`` -> most-visited (play strength);
    ``>0`` -> sample by visit^(1/T) (exploration during self-play)."""
    actions = list(root.N)
    counts = np.array([root.N[a] for a in actions], dtype=np.float64)
    if temperature <= 1e-8:
        return int(actions[int(counts.argmax())])
    probs = counts ** (1.0 / temperature)
    probs /= probs.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(actions, p=probs))


# --------------------------------------------------------------------------- #
# Convenience: choose a move for a live env, and generate self-play data
# --------------------------------------------------------------------------- #
def mcts_move(env: CarcassonneEnv, evaluator, n_sims: int,
              rng: Optional[np.random.Generator] = None,
              temperature: float = 0.0, batch_size: int = 1,
              **mcts_kwargs) -> Tuple[int, Node]:
    """Run MCTS from the current state of ``env`` and return (action, root)."""
    rng = rng or np.random.default_rng()
    state = GameState(copy.deepcopy(env), rng)
    root = MCTS(evaluator, rng=rng, **mcts_kwargs).search(state, n_sims, batch_size=batch_size)
    return best_action(root, temperature=temperature, rng=rng), root


def selfplay_game(evaluator, n_sims: int, board_size: int = 32,
                  rng: Optional[np.random.Generator] = None,
                  temperature_moves: int = 12,
                  dirichlet_alpha: float = 0.3, return_info: bool = False,
                  batch_size: int = 1):
    """Play one self-play game with MCTS on both sides and return AlphaZero
    training tuples ``(observation, visit_policy, value_target)``.

    ``value_target`` is the final game outcome (+1 win / -1 loss / 0 draw) from
    the perspective of the player who was to move in that state -- exactly the
    label the value head should regress, and ``visit_policy`` is the target the
    policy head should match. This is the data source for the training loop.
    """
    from rl_model import _canonicalize
    rng = rng or np.random.default_rng()
    env = CarcassonneEnv(board_size=board_size)
    env.reset()
    records: List[Tuple[dict, Dict[int, float], int]] = []
    ply = 0
    while not env.done:
        state = GameState(copy.deepcopy(env), rng)
        root = MCTS(evaluator, rng=rng,
                    dirichlet_alpha=dirichlet_alpha).search(state, n_sims,
                                                             batch_size=batch_size)
        pi = visit_policy(root)
        # Store the observation from "Player 0 to move" perspective (canonicalize
        # Player 1 states) so all training targets are in one consistent frame --
        # exactly how the net is used at inference. Action indices are
        # perspective-independent, so the visit policy keys stay valid.
        obs = env._build_obs()
        if env.current_player == 1:
            obs = _canonicalize(obs)
        records.append((obs, pi, env.current_player))
        temp = 1.0 if ply < temperature_moves else 0.0
        a = best_action(root, temperature=temp, rng=rng)
        env.step(a)
        ply += 1

    s0, s1 = env.scores
    z0 = 0.0 if s0 == s1 else (1.0 if s0 > s1 else -1.0)   # player-0 outcome
    # Value target is the outcome from the (now canonicalized) mover's view.
    examples = [(obs, pi, (z0 if tp == 0 else -z0)) for obs, pi, tp in records]
    if return_info:
        info = {"plies": ply, "scores": (s0, s1), "margin": s0 - s1}
        return examples, info
    return examples

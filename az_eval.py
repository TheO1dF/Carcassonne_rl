"""Head-to-head win-rate evaluation for Carcassonne models.

Plays two models against each other (or one vs a random baseline) with MCTS on
both sides at equal simulations and alternating seats, then reports W/D/L, win%,
and average score margin from model A's perspective.

    # vs random baseline:
    python az_eval.py models/model_az_iter3.zip

    # vs another model:
    python az_eval.py models/model_az_iter3.zip models/model_bc_best.zip

    # fast net-only comparison, no search:
    python az_eval.py models/model_az_iter3.zip models/model_az_iter1.zip --n-sims 0

Eval is stochastic (MCTS determinizes the hidden draw order); with --n-games 20
the 95% CI on win-rate is roughly +-22%, so raise --n-games for a confident read.
Use the same --n-sims on both sides for a fair comparison.
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np

from env import CarcassonneEnv
from mcts import GameState, NeuralEvaluator, mcts_move
from train import BOARD_SIZE, resolve_device


def load_model(path, device):
    from sb3_contrib import MaskablePPO
    from train import build_single_env
    return MaskablePPO.load(path, env=build_single_env(seed=0), device=device)


def _pick(env, evaluator, n_sims, rng, batch_size):
    """Choose a move for the current player.

    * ``evaluator is None`` -> uniform-random legal move (the Random baseline).
    * ``n_sims <= 0``       -> the network's raw policy argmax (no search; fast).
    * otherwise             -> deterministic MCTS (most-visited action).
    """
    if evaluator is None:
        return int(rng.choice(env.valid_action_list()))
    if n_sims <= 0:
        priors, _ = evaluator.evaluate(GameState(copy.deepcopy(env), rng))
        return int(max(priors, key=priors.get))
    action, _ = mcts_move(env, evaluator, n_sims, rng=rng,
                          temperature=0.0, batch_size=batch_size)
    return int(action)


def play_match(eval_a, eval_b, n_games, n_sims, rng, batch_size,
               label_a="A", label_b="B", seed=0):
    """Play ``n_games`` (alternating seats) and report results for ``eval_a``."""
    wins = draws = losses = 0
    margins = []
    for g in range(n_games):
        env = CarcassonneEnv(board_size=BOARD_SIZE)
        env.reset(seed=seed + g)
        a_seat = g % 2                       # which seat model A plays this game
        while not env.done:
            ev = eval_a if env.current_player == a_seat else eval_b
            env.step(_pick(env, ev, n_sims, rng, batch_size))
        margin = env.scores[a_seat] - env.scores[1 - a_seat]
        margins.append(margin)
        wins += margin > 0
        draws += margin == 0
        losses += margin < 0
        print(f"  game {g + 1}/{n_games}: {label_a} as seat {a_seat}  "
              f"margin {margin:+d}   (W/D/L {wins}/{draws}/{losses})", flush=True)

    wr = wins / n_games
    print(f"\n{label_a}  vs  {label_b}   over {n_games} games "
          f"({'raw-policy' if n_sims <= 0 else f'{n_sims} sims'}):")
    print(f"  W/D/L = {wins}/{draws}/{losses}   win% = {wr:.0%}   "
          f"avg margin = {np.mean(margins):+.1f}")
    return wr


def main():
    p = argparse.ArgumentParser(description="Head-to-head win-rate evaluation.")
    p.add_argument("model_a", help="model under test")
    p.add_argument("model_b", nargs="?", default=None,
                   help="opponent model; omit to play vs a Random baseline")
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--n-sims", type=int, default=60,
                   help="MCTS sims/move for BOTH sides (0 = raw policy, no search)")
    p.add_argument("--mcts-batch", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    eval_a = NeuralEvaluator(load_model(args.model_a, device), device)
    label_a = os.path.basename(args.model_a)
    if args.model_b:
        eval_b = NeuralEvaluator(load_model(args.model_b, device), device)
        label_b = os.path.basename(args.model_b)
    else:
        eval_b = None
        label_b = "Random"

    play_match(eval_a, eval_b, args.n_games, args.n_sims, rng, args.mcts_batch,
               label_a=label_a, label_b=label_b, seed=args.seed)


if __name__ == "__main__":
    main()

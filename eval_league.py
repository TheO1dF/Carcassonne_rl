"""
eval_league.py
==============

Robust evaluation of trained Carcassonne models against baseline opponents.

Loads one or more saved ``MaskablePPO`` ``.zip`` models and plays each of them
(100 games by default) against:

  * ``RandomPolicy`` -- uniform random legal moves,
  * ``GreedyPolicy`` -- the 1-ply heuristic bot from ``policies.py``.

To remove first-move advantage, seats are alternated: the model plays Player 0
in half the games and Player 1 in the other half. Reports win / draw / loss
counts, win-rate, and average scores.

Usage
-----
    python eval_league.py                         # auto-discovers models/model_gen*.zip
    python eval_league.py models/model_gen3.zip   # specific model(s)
    python eval_league.py --games 100 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from sb3_contrib import MaskablePPO

from env import CarcassonneEnv
from policies import RandomPolicy, GreedyPolicy, ModelPolicy


def play_game(env, agents, seed: int):
    """Play one full game. ``agents`` is [seat0_policy, seat1_policy]."""
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        agent = agents[env.current_player]
        action = agent.act(env, obs)
        obs, _r, terminated, truncated, _info = env.step(action)
    return int(env.scores[0]), int(env.scores[1])


def play_match(model_policy, opponent, board_size, n_games, seed):
    """Model vs opponent over ``n_games``, alternating seats. Returns a summary."""
    env = CarcassonneEnv(board_size=board_size)
    wins = draws = losses = 0
    model_score_sum = opp_score_sum = 0

    for g in range(n_games):
        model_seat = g % 2  # alternate who starts / which seat the model takes
        agents = [model_policy, opponent] if model_seat == 0 else [opponent, model_policy]
        s0, s1 = play_game(env, agents, seed=seed + g)
        model_score = s0 if model_seat == 0 else s1
        opp_score = s1 if model_seat == 0 else s0

        model_score_sum += model_score
        opp_score_sum += opp_score
        if model_score > opp_score:
            wins += 1
        elif opp_score > model_score:
            losses += 1
        else:
            draws += 1

    return {
        "games": n_games,
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": wins / n_games,
        "avg_model_score": model_score_sum / n_games,
        "avg_opp_score": opp_score_sum / n_games,
    }


def discover_models(paths):
    if paths:
        return paths
    found = sorted(glob.glob(os.path.join("models", "model_gen*.zip")))
    if not found:
        raise SystemExit("No models given and none found in ./models/. "
                         "Train some first (python train.py) or pass paths.")
    return found


def main():
    parser = argparse.ArgumentParser(description="Evaluate Carcassonne models")
    parser.add_argument("models", nargs="*", help="paths to .zip models")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample model actions instead of greedy argmax")
    args = parser.parse_args()

    model_paths = discover_models(args.models)
    deterministic = not args.stochastic

    print(f"Evaluating {len(model_paths)} model(s), {args.games} games each, "
          f"alternating seats ({'deterministic' if deterministic else 'stochastic'} policy).\n")

    header = f"{'model':<20}{'opponent':<10}{'win%':>6}{'W/D/L':>12}{'avg score':>16}"
    print(header)
    print("-" * len(header))

    for path in model_paths:
        model = MaskablePPO.load(path, device="cpu")
        board_size = model.observation_space["board"].shape[0]
        model_policy = ModelPolicy(model, name=os.path.basename(path),
                                   deterministic=deterministic)

        opponents = [
            RandomPolicy(seed=args.seed + 1),
            GreedyPolicy(seed=args.seed + 2),
        ]
        label = os.path.basename(path).replace(".zip", "")
        for opp in opponents:
            r = play_match(model_policy, opp, board_size, args.games, args.seed)
            wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
            scores = f"{r['avg_model_score']:.1f} vs {r['avg_opp_score']:.1f}"
            print(f"{label:<20}{opp.name:<10}{r['win_rate'] * 100:>5.0f}%"
                  f"{wdl:>12}{scores:>16}")
        print()


if __name__ == "__main__":
    main()

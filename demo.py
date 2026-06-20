"""
demo.py
=======

A self-contained sanity check + example of how to drive ``CarcassonneEnv``
with a random, action-masked agent. It also asserts a couple of invariants
that catch most scoring / meeple bugs:

  * meeples are conserved: remaining + on-board == num_meeples for each player,
  * scores never go negative.

Run:  python demo.py            (one verbose game)
      python demo.py 200        (200 silent games, report wins + timing)
"""

import sys
import time

import numpy as np

from env import CarcassonneEnv


def random_masked_action(env, rng) -> int:
    """Pick a uniformly-random *legal* action using the env's mask."""
    valid = np.flatnonzero(env.action_masks())
    return int(rng.choice(valid))


def check_invariants(env):
    on_board = [0, 0]
    for (player, _x, _y, _ft) in env.board.active_meeples():
        on_board[player] += 1
    for p in (0, 1):
        assert env.meeples_remaining[p] + on_board[p] == env.num_meeples, (
            f"meeple leak for player {p}: "
            f"{env.meeples_remaining[p]} + {on_board[p]} != {env.num_meeples}"
        )
        assert env.scores[p] >= 0, "negative score!"


def play_one(seed: int, verbose: bool = False) -> tuple[int, int, int]:
    env = CarcassonneEnv(board_size=30)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)

    terminated = False
    steps = 0
    while not terminated:
        action = random_masked_action(env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        check_invariants(env)
        steps += 1
        if verbose and steps % 10 == 0:
            print(f"  turn {steps:3d}: scores={env.scores} "
                  f"tiles_left={info['tiles_remaining']}")

    if verbose:
        print("\nFinal board:\n")
        env.render()
    return env.scores[0], env.scores[1], steps


def main():
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        t0 = time.time()
        wins = [0, 0, 0]  # p0, p1, draws
        for g in range(n):
            s0, s1, _ = play_one(seed=g, verbose=False)
            wins[0 if s0 > s1 else 1 if s1 > s0 else 2] += 1
        dt = time.time() - t0
        print(f"Played {n} games in {dt:.1f}s ({dt / n * 1000:.1f} ms/game)")
        print(f"P0 wins: {wins[0]}   P1 wins: {wins[1]}   draws: {wins[2]}")
    else:
        print("Playing one verbose random game...\n")
        s0, s1, steps = play_one(seed=42, verbose=True)
        print(f"\nGame finished in {steps} turns. Final score: P0={s0}, P1={s1}")


if __name__ == "__main__":
    main()

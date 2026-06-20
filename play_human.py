"""
play_human.py
=============

Play Carcassonne against a trained MaskablePPO model from the command line.

Each turn the script:
  * renders the ASCII board,
  * on your turn, lists the legal tile placements (cell + rotation) and then the
    legal meeple options for the placement you pick,
  * on the AI's turn, asks the model for a move and shows what it played.

Usage
-----
    python play_human.py models/model_gen3.zip
    python play_human.py models/model_gen3.zip --seat 1   # you play second
"""

from __future__ import annotations

import argparse
import os
import random

from sb3_contrib import MaskablePPO

from env import CarcassonneEnv
from tiles import Terrain
from policies import ModelPolicy

ROT_LABEL = {0: "0", 1: "90", 2: "180", 3: "270"}
TERR_CHAR = {Terrain.CITY: "C", Terrain.ROAD: "R", Terrain.FIELD: "."}


# --------------------------------------------------------------------------- #
# Pretty-printing helpers
# --------------------------------------------------------------------------- #
def describe_tile_edges(tile_type, rotation: int) -> str:
    """e.g. 'N:C E:R S:. W:.' for the tile at a given rotation."""
    from tiles import PlacedTile
    pt = PlacedTile(tile_type, rotation)
    n, e, s, w = (TERR_CHAR[t] for t in pt.edges)
    return f"N:{n} E:{e} S:{s} W:{w}"


def list_placements(env):
    """Distinct legal (x, y, rotation) placements for the current tile."""
    seen = {}
    for a in env.valid_action_list():
        x, y, rot, _meeple = env._decode(a)
        seen.setdefault((x, y, rot), True)
    return sorted(seen.keys())


def list_meeple_options(env, x, y, rot):
    """Legal meeple choices for a chosen placement.

    Returns a list of (meeple_choice, label). Index 0 is always 'no meeple'.
    """
    options = [(0, "no meeple")]
    if env.meeples_remaining[env.current_player] > 0:
        feats = env.board.evaluate_placement(env.current_tile_type, x, y, rot)
        for i, (kind, occupied) in enumerate(feats):
            if not occupied:
                options.append((i + 1, f"meeple on {kind} (feature #{i})"))
    return options


def prompt_int(prompt: str, lo: int, hi: int) -> int:
    """Read an int in [lo, hi] from stdin, re-prompting on bad input."""
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  please enter a number between {lo} and {hi}.")


# --------------------------------------------------------------------------- #
# Turn handlers
# --------------------------------------------------------------------------- #
def human_turn(env):
    """Interactively build and return the human's action index."""
    tile = env.current_tile_name
    print(f"\nYour tile: {tile}   ({describe_tile_edges(env.current_tile_type, 0)} "
          f"at rotation 0)")
    print(f"Meeples left -- you: {env.meeples_remaining[env.current_player]}")

    placements = list_placements(env)
    print(f"\nLegal placements ({len(placements)}):")
    for idx, (x, y, rot) in enumerate(placements):
        print(f"  [{idx:2d}] cell ({x:2d},{y:2d}) rot {ROT_LABEL[rot]:>3}deg   "
              f"{describe_tile_edges(env.current_tile_type, rot)}")
    choice = prompt_int("Choose a placement #: ", 0, len(placements) - 1)
    x, y, rot = placements[choice]

    options = list_meeple_options(env, x, y, rot)
    print("\nMeeple options:")
    for idx, (_mc, label) in enumerate(options):
        print(f"  [{idx}] {label}")
    mopt = prompt_int("Choose a meeple option #: ", 0, len(options) - 1)
    meeple_choice = options[mopt][0]

    return env._encode(x, y, rot, meeple_choice)


def ai_turn(env, ai, obs):
    """Let the AI move; print a short description of what it did."""
    action = ai.act(env, obs)
    x, y, rot, meeple = env._decode(action)
    meeple_desc = "no meeple" if meeple == 0 else f"meeple on feature #{meeple - 1}"
    print(f"\nAI plays tile {env.current_tile_name} at ({x},{y}) "
          f"rot {ROT_LABEL[rot]}deg, {meeple_desc}.")
    return action


# --------------------------------------------------------------------------- #
# Main game loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Play Carcassonne vs a trained AI")
    parser.add_argument("model", help="path to a saved MaskablePPO .zip model")
    parser.add_argument("--seat", type=int, default=0, choices=(0, 1),
                        help="which player you are (0 = move first)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    board_size = model.observation_space["board"].shape[0]
    ai = ModelPolicy(model, name="AI", deterministic=True)

    human_seat = args.seat
    ai_seat = 1 - human_seat
    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)

    env = CarcassonneEnv(board_size=board_size, render_mode="human")
    obs, _ = env.reset(seed=seed)

    print("=" * 60)
    print(f"Carcassonne vs {os.path.basename(args.model)}")
    print(f"You are Player {human_seat}; the AI is Player {ai_seat}.")
    print("Board cells are (x, y); x grows East, y grows South.")
    print("=" * 60)

    terminated = truncated = False
    while not (terminated or truncated):
        print("\n" + env._render_text())
        if env.current_player == human_seat:
            action = human_turn(env)
        else:
            action = ai_turn(env, ai, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["scores"] != [0, 0]:
            print(f"Scores -> P0: {info['scores'][0]}   P1: {info['scores'][1]}")

    print("\n" + "=" * 60)
    print("GAME OVER")
    print(env._render_text())
    s = env.scores
    if s[human_seat] > s[ai_seat]:
        print(f"\nYou WIN {s[human_seat]} - {s[ai_seat]}!")
    elif s[ai_seat] > s[human_seat]:
        print(f"\nThe AI wins {s[ai_seat]} - {s[human_seat]}. Better luck next time!")
    else:
        print(f"\nIt's a draw, {s[0]} - {s[1]}.")


if __name__ == "__main__":
    main()

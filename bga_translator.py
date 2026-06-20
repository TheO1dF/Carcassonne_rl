"""
bga_translator.py
=================

Convert a raw Board Game Arena (BGA) Carcassonne replay into a standardized
**Kifu** (game record): a list of exact environment ``action`` integers ready for
replay or Behavioural Cloning.

Deterministic rotation mapping (no search / calibration)
--------------------------------------------------------
Two hand-verified tables make the translation pure math:

* ``BGA_TO_LOCAL`` -- BGA tile *type* id (1..24) -> our rulebook letter.
* ``IMAGE_OFFSETS`` -- per local letter, the number of 90deg CW steps our base
  tile differs from BGA's default artwork orientation.

BGA's ``ori`` is **1-indexed** (1=N, 2=E, 3=S, 4=W); ours is **0-indexed** (0..3
clockwise steps). So we convert *before* any mod-4 arithmetic:

    local_x   = bga_x + (board centre)             # env.size // 2, the logical
    local_y   = bga_y + (board centre)             #   board centre (BGA start=0,0)
    base_rot  = (bga_ori - 1) % 4                  # BGA 1..4 -> our 0..3
    local_rot = (base_rot + IMAGE_OFFSETS[local_letter]) % 4

(``board_size`` here is the observation/action WINDOW that recenters on play; the
absolute coordinates above use the larger logical board centre, ``env.size//2``.)

The start tile is always BGA type 15 ('D'), pre-placed at
``start_rotation = IMAGE_OFFSETS['D']`` so the board origin matches BGA's.

Pipeline
--------
parse JSON -> ``BGAMove`` model (strict ``playTile`` + ``playPartisan`` pairing by
``args['id']``) -> compute action -> validate against ``get_valid_actions()`` ->
``env.step`` -> save ``kifu_<timestamp>.json`` (``tile_sequence`` + ``actions``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tiles import TILE_TYPES
from replay_env import CarcassonneReplayEnv


# --------------------------------------------------------------------------- #
# Hardcoded, hand-verified mappings
# --------------------------------------------------------------------------- #
# BGA tile *type* id (1..24) -> our local rulebook letter.
BGA_TO_LOCAL = {
    "1": "T", "2": "S", "3": "N", "4": "M", "5": "P", "6": "O", "7": "G", "8": "F",
    "9": "I", "10": "H", "11": "E", "12": "K", "13": "J", "14": "L", "15": "D", "16": "U",
    "17": "V", "18": "W", "19": "X", "20": "B", "21": "A", "22": "C", "23": "R", "24": "Q",
}
LOCAL_TO_BGA = {v: k for k, v in BGA_TO_LOCAL.items()}

# Number of 90deg CW rotations our LOCAL base tile differs from BGA's default
# artwork. Applied as local_rot = (base_rot + IMAGE_OFFSETS[letter]) % 4, where
# base_rot = (bga_ori - 1) % 4. Hand-calibrated against the official sprite sheet.
IMAGE_OFFSETS = {
    "A": 0, "B": 0, "C": 0, "D": 3, "E": 0, "F": 0, "G": 3, "H": 3,
    "I": 2, "J": 0, "K": 0, "L": 0, "M": 3, "N": 3, "O": 3, "P": 3,
    "Q": 0, "R": 0, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0,
}

# BGA meeple target -> our local feature kind (BGA "abbey" == our monastery).
BGA_TARGET_TO_KIND = {
    "city": "city",
    "road": "road",
    "field": "field",
    "farm": "field",
    "abbey": "monastery",
    "cloister": "monastery",
    "monastery": "monastery",
}

DEFAULT_BOARD_SIZE = 32            # observation/action WINDOW; must match train.BOARD_SIZE
                                  # for Behavioural Cloning (recenters on play, span <= ~24)
START_LETTER = "D"                 # BGA always opens with type 15 ('D')


def bga_ori_to_rot(ori: int) -> int:
    """Convert a 1-indexed BGA orientation (1..4) to our 0-indexed step (0..3)."""
    return (int(ori) - 1) % 4


def local_rotation(bga_ori: int, local_letter: str) -> int:
    """Deterministic local rotation (0..3): base_rot + the tile's image offset."""
    return (bga_ori_to_rot(bga_ori) + IMAGE_OFFSETS[local_letter]) % 4


# Start tile 'D' is pre-placed at this rotation so our origin matches BGA's.
START_ROTATION = IMAGE_OFFSETS[START_LETTER]


class TranslationError(Exception):
    """Raised when a recorded game cannot be fully reproduced as legal actions."""


# --------------------------------------------------------------------------- #
# Game-record model
# --------------------------------------------------------------------------- #
@dataclass
class BGAMove:
    """One player turn: a ``playTile`` plus its optional ``playPartisan`` meeple."""
    tile_type: str                       # BGA type id as string, e.g. "15"
    x: int                               # BGA x (relative to the start tile)
    y: int                               # BGA y
    ori: int                             # BGA orientation 1..4
    player_name: Optional[str] = None
    tile_id: Optional[object] = None     # BGA tile instance id (links the meeple)
    target: Optional[str] = None         # meeple target kind: city/road/field/abbey
    pos: Optional[object] = None         # BGA visual meeple position index
    raw: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Robust BGA JSON parsing  ->  list[BGAMove]
# --------------------------------------------------------------------------- #
def _get(d: dict, *keys, default="__required__"):
    """Return the first present, non-None key (BGA field names vary)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    if default == "__required__":
        raise KeyError(f"none of {keys} found in {list(d.keys())}")
    return default


def _norm_target(value) -> Optional[str]:
    return None if value is None else str(value).strip().lower()


def parse_bga_log(raw) -> Tuple[List[str], List[BGAMove]]:
    """Parse a BGA replay into ``(players, moves)``.

    Accepts the real BGA shape (arrays of packets whose ``data`` lists hold
    ``playTile`` / ``playPartisan`` notifications) and a simplified
    ``{"players", "moves"}`` shape used for testing.
    """
    if isinstance(raw, dict) and isinstance(raw.get("moves"), list):
        moves = [_move_from_simplified(m) for m in raw["moves"]]
    else:
        moves = _moves_from_notifications(list(_iter_notifications(raw)))

    if not moves:
        raise TranslationError(
            "No moves parsed -- adapt _iter_notifications()/_moves_from_"
            "notifications() to your BGA export's field names.")

    return _derive_players(raw, moves), moves


def _iter_notifications(raw):
    """Yield ``{"type":..., "args":...}`` notifications from a BGA log.

    Drills through the usual containers: a top-level list of packets, or
    ``raw["data"]["data"]`` / ``["logs"]`` / ``raw["data"]``. Each packet's
    ``data`` list holds the notifications.
    """
    logs = raw
    if isinstance(raw, dict):
        data = raw.get("data", raw)
        if isinstance(data, dict):
            logs = data.get("data") or data.get("logs") or data.get("packets") or []
        elif isinstance(data, list):
            logs = data
        else:
            logs = []
    if not isinstance(logs, list):
        return
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("data"), list):           # a packet of notifications
            for notif in entry["data"]:
                if isinstance(notif, dict) and "type" in notif:
                    yield notif
        elif "type" in entry:                              # a bare notification
            yield entry


def _moves_from_notifications(notifs: List[dict]) -> List[BGAMove]:
    """Strictly pair playTile + playPartisan by the tile ``args['id']``."""
    moves: List[BGAMove] = []
    by_id: dict = {}
    current: Optional[BGAMove] = None

    for notif in notifs:
        ntype = str(notif.get("type", "")).lower()
        args = notif.get("args", notif) or {}

        if ntype in ("playtile", "tileplayed"):
            tile_id = _get(args, "id", "tile_id", "tileId", default=None)
            move = BGAMove(
                tile_type=str(_get(args, "type", "tile_type", "tileType")),
                x=int(_get(args, "x")),
                y=int(_get(args, "y")),
                ori=int(_get(args, "ori", "orientation", "rotation", default=1)),
                player_name=_get(args, "player_name", "player", "player_id", default=None),
                tile_id=tile_id,
                raw=dict(args),
            )
            moves.append(move)
            current = move
            if tile_id is not None:
                by_id[str(tile_id)] = move

        elif ntype in ("playpartisan", "placemeeple", "meeplemoved",
                       "partisanplayed", "meepleplaced"):
            # The meeple references the same tile via args["id"].
            tile_id = _get(args, "id", "tile_id", "tileId", default=None)
            move = by_id.get(str(tile_id), current) if tile_id is not None else current
            if move is not None:
                move.target = _norm_target(_get(args, "target", "location",
                                                "feature", "placement", default=None))
                move.pos = _get(args, "pos", "position", "spot", "index", default=None)

    return moves


def _move_from_simplified(d: dict) -> BGAMove:
    return BGAMove(
        tile_type=str(_get(d, "type", "tile_type", "tileType")),
        x=int(_get(d, "x")),
        y=int(_get(d, "y")),
        ori=int(_get(d, "ori", "orientation", "rotation", default=1)),
        player_name=_get(d, "player_name", "player", default=None),
        tile_id=_get(d, "id", "tile_id", default=None),
        target=_norm_target(_get(d, "target", "meeple", default=None)),
        pos=_get(d, "pos", "position", default=None),
        raw=d,
    )


def _derive_players(raw, moves: List[BGAMove]) -> List[str]:
    """Players in seat order = order of first appearance in the move log."""
    order: List[str] = []
    for m in moves:
        if m.player_name is not None and str(m.player_name) not in order:
            order.append(str(m.player_name))
    if len(order) >= 2:
        return order[:2]
    if isinstance(raw, dict) and isinstance(raw.get("players"), list) and raw["players"]:
        return [str(_player_label(p)) for p in raw["players"]][:2]
    while len(order) < 2:
        order.append(f"player_{len(order)}")
    return order


def _player_label(p):
    if isinstance(p, dict):
        return p.get("name") or p.get("player_name") or p.get("id") or "player"
    return p


# --------------------------------------------------------------------------- #
# Meeple resolution
# --------------------------------------------------------------------------- #
# Edge meeple-spot location in BGA's 3x3 tile grid, as a row-major *reading
# index* (row*3 + col), computed in BGA's BASE orientation:
#
#       0 . 1 . 2          N=(1,0)->1   W=(0,1)->3   E=(2,1)->5   S=(1,2)->7
#       3 . 4 . 5          so edges are ordered  N, W, E, S.
#       6 . 7 . 8
#
# BGA numbers a tile's meeple spots 1.. grouped by kind (monastery, roads,
# cities, fields). Within a group: roads/cities by ascending edge reading index;
# fields by ascending half-edge position (clockwise from the top-left corner).
_EDGE_READ = {0: 1, 1: 5, 2: 7, 3: 3}                       # N, E, S, W (tiles.py)

# ROADS order their stubs the same way EXCEPT a North stub sorts LAST, not first
# -- validated against the crossroads ('X') tile, the only multi-road tile that
# has a North road (W/L only have E/S/W). So X roads order W, E, S, N.
_ROAD_READ = {0: 9, 1: 5, 2: 7, 3: 3}                       # N last, then W, E, S


def feature_order_by_pos(tile_type, offset: int) -> List[int]:
    """OUR canonical feature indices in BGA ``pos`` order (``pos - 1`` -> index).

    A BGA meeple is a *child object* of the tile: it sits on a fixed spot that
    rotates with the tile, so ``pos`` selects a feature **independently of the
    on-board rotation**. BGA numbers spots by kind group (monastery, roads,
    cities, fields) in BGA's *base* orientation -- our base tile rotated by
    ``offset = IMAGE_OFFSETS[letter]``. Reverse-engineered from real BGA logs and
    validated against the recorded per-move and end-game scores.

    Our own canonical feature order (``evaluate_placement`` / ``place``) is
    cities, roads, fields, monastery; this function returns indices into it.
    """
    nc, nr, nf = len(tile_type.cities), len(tile_type.roads), len(tile_type.fields)

    def road_key(edges):                   # roads: North stub sorts last
        return min(_ROAD_READ[(e + offset) % 4] for e in edges)

    def edge_key(edges):                   # cities: smallest spot reading index
        return min(_EDGE_READ[(e + offset) % 4] for e in edges)

    def field_key(positions):
        # Usually the smallest BGA-base half-edge position. BUT a field that wraps
        # the NW boundary (contains BOTH 0 and 7) and is bigger than the 2-position
        # {0,7} corner is a SIDE field whose position 0 is incidental to the wrap;
        # order it by its real arc start (treat 0 as 8). This makes the straight
        # road tile 'U' split as East-field (pos n) then West-field (pos n+1),
        # which `min` alone gets backwards because the West field touches pos 0.
        bga = {(p + 2 * offset) % 8 for p in positions}
        if 0 in bga and 7 in bga and len(bga) > 2:
            return min(8 if p == 0 else p for p in bga)
        return min(bga)

    order: List[int] = []
    if tile_type.monastery:
        order.append(nc + nr + nf)         # monastery is last in OUR order
    order += [nc + i for i in sorted(range(nr), key=lambda i: road_key(tile_type.roads[i]))]
    order += [i for i in sorted(range(nc), key=lambda i: edge_key(tile_type.cities[i][0]))]
    order += [nc + nr + i
              for i in sorted(range(nf), key=lambda i: field_key(tile_type.fields[i][0]))]
    return order


def resolve_meeple(env, tile_type, lx, ly, rot, target, bga_type, pos, offset
                   ) -> Tuple[int, Optional[str]]:
    """Map a BGA meeple (``target`` kind + ``pos`` spot) to our ``meeple_choice``.

    BGA's ``pos`` indexes a fixed per-tile meeple spot (see
    ``feature_order_by_pos``), which lets us pick the *exact* feature even when a
    tile has several of the same kind (W's three roads, H's two cities, X's four
    fields). Returns ``(meeple_choice, warning_or_None)``; ``0`` means no meeple.
    """
    if target is None:
        return 0, None

    kind = BGA_TARGET_TO_KIND.get(str(target).lower())
    if kind is None:
        return 0, f"unknown target '{target}' (tile {bga_type}) -> no meeple"

    features = env.board.evaluate_placement(tile_type, lx, ly, rot)

    # --- primary: resolve via the BGA spot index -------------------------------
    if pos is not None:
        try:
            p = int(pos)
        except (TypeError, ValueError):
            p = None
        order = feature_order_by_pos(tile_type, offset)
        if p is not None and 1 <= p <= len(order):
            fidx = order[p - 1]
            fkind, occupied = features[fidx]
            if fkind == kind:
                if occupied:
                    return 0, (f"tile {bga_type} pos={p} ('{kind}') already "
                               f"occupied -> no meeple")
                return fidx + 1, None
            # pos/target disagree -> fall through to kind matching with a warning.

    # --- fallback: first unoccupied feature of the target kind -----------------
    matches = [i for i, (fkind, occupied) in enumerate(features)
               if fkind == kind and not occupied]
    if not matches:
        return 0, f"no free '{kind}' feature (tile {bga_type} pos={pos}) -> no meeple"
    if len(matches) > 1:
        warn = (f"ambiguous meeple: tile {bga_type} target '{kind}' pos={pos} "
                f"matches features {matches}; picking first ({matches[0]})")
        return matches[0] + 1, warn
    return matches[0] + 1, None


# --------------------------------------------------------------------------- #
# Deterministic translation  ->  Kifu game record
# --------------------------------------------------------------------------- #
def translate(players, moves: List[BGAMove], board_size: int = DEFAULT_BOARD_SIZE,
              start_rotation: Optional[int] = None, on_error: str = "abort",
              verbose: bool = True, **_ignored) -> dict:
    """Translate the recorded game with the deterministic per-tile rotation map.

    ``start_rotation`` defaults to ``IMAGE_OFFSETS['D']`` (the BGA start tile is
    always type 15 = 'D'). On an illegal move, ``on_error="abort"`` raises
    ``TranslationError`` (used by the replay viewer); ``"partial"`` keeps the
    good prefix. ``**_ignored`` keeps signature-compat with older callers.
    """
    if start_rotation is None:
        start_rotation = START_ROTATION

    tile_sequence = [BGA_TO_LOCAL[m.tile_type] for m in moves]
    env = CarcassonneReplayEnv(board_size=board_size)
    env.reset(tile_sequence=tile_sequence, start_rotation=start_rotation)
    half = env.size // 2

    actions: List[int] = []
    warnings: List[str] = []
    complete = True
    fail = None

    for idx, m in enumerate(moves):
        local = BGA_TO_LOCAL[m.tile_type]
        tile = TILE_TYPES[local]
        env.current_tile_name = local           # force the recorded tile
        env.current_tile_type = tile

        # --- coordinate ---
        lx, ly = m.x + half, m.y + half
        if not (0 <= lx < env.size and 0 <= ly < env.size):
            fail = (idx, f"out of bounds local=({lx},{ly}); increase --board-size")
            complete = False
            break

        # --- rotation: 1-indexed BGA ori -> 0-indexed base_rot -> + image offset.
        #     Never feed raw m.ori into the mod-4 math. ---
        base_rot = bga_ori_to_rot(m.ori)
        assert 0 <= base_rot <= 3
        rot = (base_rot + IMAGE_OFFSETS[local]) % 4
        assert 0 <= rot <= 3

        # --- meeple ---
        meeple_choice, warn = resolve_meeple(env, tile, lx, ly, rot, m.target,
                                             m.tile_type, m.pos, IMAGE_OFFSETS[local])
        if warn:
            warnings.append(f"move {idx}: {warn}")
            if verbose:
                print(f"  [warn] move {idx}: {warn}")

        # --- encode, validate, step ---
        action = int(env._encode(lx, ly, rot, meeple_choice))
        mask = env.get_valid_actions()
        if action >= len(mask) or not mask[action]:
            fail = (idx, f"illegal action (tile '{local}', rot {rot}, "
                         f"meeple {meeple_choice})")
            complete = False
            break

        env.step(action)
        actions.append(action)
        if verbose:
            print(_step_line(idx, m, local, lx, ly, base_rot, rot, meeple_choice, env))

    if verbose and fail is not None:
        _report_invalid(env, fail[0], moves[fail[0]], fail[1])
    if verbose:
        print(f"\nTranslated {len(actions)}/{len(moves)} moves "
              f"(complete={complete}, start_rotation={start_rotation}, "
              f"{len(warnings)} warning(s)).")

    # Final scores + winner seat (None on a tie) -- used by Behavioural-Cloning
    # "winner-only" filtering.
    final_scores = list(env.scores)
    winner_seat = (0 if env.scores[0] > env.scores[1]
                   else (1 if env.scores[1] > env.scores[0] else None))

    kifu = {
        "players": list(players),
        "tile_sequence": tile_sequence[:len(actions)],
        "actions": actions,
        "board_size": board_size,
        "start_rotation": start_rotation,
        "complete": complete,
        "warnings": warnings,
        "final_scores": final_scores,
        "winner_seat": winner_seat,
    }
    if not complete and on_error == "abort":
        raise TranslationError(
            f"Could not fully reproduce the game "
            f"({len(actions)}/{len(moves)} moves). See debug above.")
    return kifu


# --------------------------------------------------------------------------- #
# Pretty console output
# --------------------------------------------------------------------------- #
def _step_line(idx, m, local, lx, ly, base_rot, rot, meeple_choice, env) -> str:
    name = str(m.player_name)[:12] if m.player_name is not None else f"P{idx % 2}"
    meeple = "none" if meeple_choice == 0 else f"{m.target or '?'}(#{meeple_choice - 1})"
    return (f"[{idx:3d}] {name:<12} tile {local:<2} "
            f"BGA({m.x:>3},{m.y:>3}) ori{m.ori}->base{base_rot} "
            f"-> local({lx:>2},{ly:>2}) rot{rot}  "
            f"meeple:{meeple:<12} scores={env.scores}")


def _report_invalid(env, idx, move, reason):
    local = BGA_TO_LOCAL.get(move.tile_type, "?")
    off = IMAGE_OFFSETS.get(local, "?")
    print("\n" + "!" * 70)
    print(f"INVALID MOVE #{idx}: {reason}")
    print(f"  BGA   : type={move.tile_type} ('{local}') x={move.x} y={move.y} "
          f"ori={move.ori} target={move.target} pos={move.pos}  IMAGE_OFFSET={off}")
    half = env.size // 2
    seen = {}
    for a in env.valid_action_list():
        gx, gy, gr, _gm = env._decode(a)
        seen.setdefault((gx, gy, gr), True)
    print(f"  tile '{local}' currently has {len(seen)} legal (x,y,rot) spots; e.g.:")
    for (gx, gy, gr) in list(seen)[:8]:
        print(f"     local ({gx},{gy}) rot {gr}  ==  BGA (x={gx - half}, y={gy - half})")
    print("!" * 70 + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Translate a BGA Carcassonne replay into a Kifu action file.")
    parser.add_argument("input", help="path to the raw BGA JSON replay")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE,
                        help="env board size (must match your BC env)")
    parser.add_argument("--start-rotation", type=int, default=None, choices=(0, 1, 2, 3),
                        help="override the start-tile rotation (default IMAGE_OFFSETS['D'])")
    parser.add_argument("--on-error", choices=("abort", "partial"), default="abort",
                        help="on an illegal move: abort, or keep the good prefix")
    parser.add_argument("--out", default=None, help="output kifu path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    players, moves = parse_bga_log(raw)
    print(f"Parsed {len(moves)} moves; players = {players}")

    try:
        kifu = translate(players, moves, board_size=args.board_size,
                         start_rotation=args.start_rotation, on_error=args.on_error)
    except TranslationError as exc:
        print(f"\nTranslation aborted: {exc}")
        sys.exit(1)

    out = args.out or f"kifu_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(kifu, f, indent=2)

    print(f"\nWrote {out}: {len(kifu['actions'])} actions, "
          f"start_rotation={kifu['start_rotation']}, "
          f"complete={kifu['complete']}, warnings={len(kifu['warnings'])}")


if __name__ == "__main__":
    main()

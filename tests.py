"""
tests.py
========

Targeted unit tests for the scoring engine. Random self-play rarely *completes*
features, so these construct exact board scenarios and check that completion,
meeple return, and end-game / farm scoring all produce the right numbers.

Run:  python tests.py     (prints PASS/FAIL per test)
"""

from logic import Board
from tiles import TILE_TYPES

A = TILE_TYPES["A"]   # road -> monastery (road terminates at the cloister)
B = TILE_TYPES["B"]   # plain monastery
E = TILE_TYPES["E"]   # city on a single (North) edge
F = TILE_TYPES["F"]   # city spanning E-W, with a shield


def test_road_completion():
    """Two A-tiles facing each other -> a 2-tile road that closes (= 2 pts)."""
    b = Board(11)
    # A1: road on its South edge, meeple on the road (feature index 0).
    b.place(A, 5, 5, 0, meeple_feature=0, player=0)
    # A2 directly below, rotated 180 so its road sits on the North edge.
    b.place(A, 5, 6, 2)
    points, returned = b.resolve_completions(5, 6)
    assert points[0] == 2, f"road score should be 2, got {points[0]}"
    assert returned[0] == 1, "meeple should be returned on completion"
    assert points[1] == 0


def test_city_completion():
    """Two E-tiles face to face -> a closed 2-tile city (2 pts/tile = 4)."""
    b = Board(11)
    # E1: city on North, meeple on the city (feature index 0).
    b.place(E, 5, 5, 0, meeple_feature=0, player=1)
    # E2 above it, rotated 180 so its city faces South and connects.
    b.place(E, 5, 4, 2)
    points, returned = b.resolve_completions(5, 4)
    assert points[1] == 4, f"city score should be 4, got {points[1]}"
    assert returned[1] == 1


def test_city_with_shield():
    """F (E-W city, shield) closed by an E-tile on each side.

    3 tiles * 2 + 1 shield * 2 = 8 points.
    """
    b = Board(11)
    b.place(F, 5, 5, 0, meeple_feature=0, player=0)  # meeple on the city
    b.place(E, 6, 5, 3)        # east neighbour, city facing West
    p1, _ = b.resolve_completions(6, 5)
    assert p1.get(0, 0) == 0, "city is not complete yet after one side"
    b.place(E, 4, 5, 1)        # west neighbour, city facing East -> closes it
    p2, ret = b.resolve_completions(4, 5)
    assert p2[0] == 8, f"shielded city should score 8, got {p2[0]}"
    assert ret[0] == 1


def test_monastery_completion():
    """A monastery fully surrounded by 8 tiles scores 9."""
    b = Board(11)
    # Centre B with a meeple on the monastery (feature index 1: [field, monastery]).
    b.place(B, 5, 5, 0, meeple_feature=1, player=0)
    offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    last = None
    for (dx, dy) in offsets:
        b.place(B, 5 + dx, 5 + dy, 0)
        last = (5 + dx, 5 + dy)
    points, returned = b.resolve_completions(*last)
    assert points[0] == 9, f"monastery should score 9, got {points[0]}"
    assert returned[0] == 1


def test_farm_scoring():
    """A farmer on a field adjacent to one completed city scores 3 at game end."""
    b = Board(11)
    # E1 with a meeple on its FIELD (feature index 1: [city, field]).
    b.place(E, 5, 5, 0, meeple_feature=1, player=0)
    # Close the city with E2 above (no meeple).
    b.place(E, 5, 4, 2)
    b.resolve_completions(5, 4)            # city closes (no meeple on it -> 0 pts)
    end = b.end_game_score()
    assert end[0] == 3, f"farm by 1 completed city should score 3, got {end[0]}"


def test_incomplete_city_endgame():
    """An unfinished city with a meeple scores 1 point per tile at game end."""
    b = Board(11)
    b.place(E, 5, 5, 0, meeple_feature=0, player=1)  # lone open city, 1 tile
    b.resolve_completions(5, 5)
    end = b.end_game_score()
    assert end[1] == 1, f"incomplete 1-tile city should score 1, got {end[1]}"


def main():
    tests = [
        test_road_completion,
        test_city_completion,
        test_city_with_shield,
        test_monastery_completion,
        test_farm_scoring,
        test_incomplete_city_endgame,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())

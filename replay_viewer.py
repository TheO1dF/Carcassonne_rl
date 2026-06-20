"""
replay_viewer.py
================

A visual replay viewer for BGA Carcassonne JSON logs, built on pygame.

It glues together three existing pieces:
  * ``bga_translator`` -- parse the raw BGA log + translate it into the exact
    ``tile_sequence`` and ``action`` integers (the same validation used for
    Behavioural Cloning, so an illegal move aborts here too).
  * ``CarcassonneReplayEnv`` -- step those actions to reconstruct any position.
  * ``play_gui`` -- reuse its sprite-based tile/meeple rendering, camera maths
    and colour constants (no code duplicated).

Controls
--------
  Right Arrow / Left-click  : next step
  Left Arrow  / Right-click : previous step
  Spacebar                  : toggle auto-play (1 step / 500 ms)
  Home / End                : jump to start / end
  Esc                       : quit

Usage
-----
    python replay_viewer.py path/to/bga_log.json [--board-size 20]
"""

from __future__ import annotations

import argparse
import json
import sys

import pygame

# --- reuse rendering, camera maths and colours from the GUI (no duplication) --
from play_gui import (
    CarcassonneGUI, feature_anchor,
    BG, VIEW_BG, HUD_BG, TEXT, TEXT_DIM, HUMAN_COL, AI_COL,
    VIEW_W, WIN_H, HUD_W, WIN_W,
)
from replay_env import CarcassonneReplayEnv
from bga_translator import (
    parse_bga_log, translate, TranslationError, BGA_TO_LOCAL, BGA_TARGET_TO_KIND,
)

AUTOPLAY_MS = 500


def describe_move(step_index: int, move, players) -> str:
    """Human-readable action log line for a single BGA move.

    The acting seat alternates (player 0 starts), matching the env.
    """
    seat = step_index % 2
    name = str(players[seat]) if seat < len(players) else f"Player {seat}"
    letter = BGA_TO_LOCAL[move.tile_type]
    kind = BGA_TARGET_TO_KIND.get((move.target or "").lower())
    meeple = f"with meeple on {kind.capitalize()}" if kind else "with no meeple"
    return f"{name} placed Tile '{letter}' at ({move.x}, {move.y}) {meeple}"


class ReplayViewer:
    """Scrubbable, auto-playable viewer over a translated BGA game."""

    # Reuse the GUI's rendering + camera methods directly (they only touch
    # attributes this class also provides: _sprites, _tile_cache, screen,
    # human_seat, env, camera, state, placements, pending, last_move).
    _load_sprites = CarcassonneGUI._load_sprites
    _tile_surface = CarcassonneGUI._tile_surface
    _draw_meeple = CarcassonneGUI._draw_meeple
    _tile_meeple = CarcassonneGUI._tile_meeple
    _compute_camera = CarcassonneGUI._compute_camera
    _w2s = CarcassonneGUI._w2s
    _s2w = CarcassonneGUI._s2w

    def __init__(self, players, tile_sequence, actions, descriptions,
                 board_size: int, start_rotation: int = 0, headless: bool = False,
                 incomplete: bool = False):
        self.players = players
        self.tile_sequence = tile_sequence
        self.actions = actions
        self.descriptions = descriptions
        self.total = len(actions)
        # True if translation stopped early (the next move, index == total, was
        # illegal in our engine). Shown in the HUD so the failure is obvious.
        self.incomplete = incomplete
        # The actions were encoded against this start-tile orientation, so every
        # reset below must reproduce it (else replayed actions become illegal).
        self.start_rotation = start_rotation

        self.env = CarcassonneReplayEnv(board_size=board_size)
        self.current_step = 0

        # Attributes the borrowed GUI methods expect.
        self._tile_cache = {}
        self.human_seat = 0          # -> player 0 blue, player 1 red
        self.camera = (0.0, 0.0, 64)
        self.state = None            # disables the GUI camera's PLACE branch
        self.placements = {}
        self.pending = None
        self.last_move = None        # (seat, (x, y)) of the most recent move

        # Auto-play.
        self.autoplay = False
        self._last_auto = 0

        pygame.init()
        flags = pygame.HIDDEN if headless else 0
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
        pygame.display.set_caption("Carcassonne -- BGA replay viewer")
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_sm = pygame.font.SysFont("consolas", 15)
        self.font_big = pygame.font.SysFont("consolas", 26, bold=True)
        self.clock = pygame.time.Clock()

        # Slice the tile sprite sheet (needed by the borrowed _tile_surface);
        # requires the display to exist for convert_alpha(), hence after set_mode.
        self._load_sprites()

        # Initialise the env at step 0 (start tile placed, no actions applied).
        self.env.reset(tile_sequence=self.tile_sequence, start_rotation=self.start_rotation)
        self.current_step = 0
        self.last_move = None

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def go_to_step(self, target_step: int):
        """Move the env to ``target_step`` actions applied.

        Gym envs can't step backwards, so going back resets and fast-forwards;
        going forward just applies the new actions.
        """
        target_step = max(0, min(target_step, self.total))

        if target_step < self.current_step:
            # Rewind: reset and replay from the beginning.
            self.env.reset(tile_sequence=self.tile_sequence, start_rotation=self.start_rotation)
            for i in range(target_step):
                self.env.step(self.actions[i])
        else:
            # Fast-forward: only the new actions.
            for i in range(self.current_step, target_step):
                self.env.step(self.actions[i])

        self.current_step = target_step
        if target_step == 0:
            self.last_move = None
        else:
            i = target_step - 1
            x, y, _rot, _m = self.env._decode(self.actions[i])
            self.last_move = (i % 2, (x, y))

    def next_step(self):
        if self.current_step < self.total:
            self.go_to_step(self.current_step + 1)

    def prev_step(self):
        if self.current_step > 0:
            self.go_to_step(self.current_step - 1)

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #
    def _handle_event(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_RIGHT:
                self.next_step()
            elif event.key == pygame.K_LEFT:
                self.prev_step()
            elif event.key == pygame.K_SPACE:
                self.autoplay = not self.autoplay
                self._last_auto = pygame.time.get_ticks()
            elif event.key == pygame.K_HOME:
                self.go_to_step(0)
            elif event.key == pygame.K_END:
                self.go_to_step(self.total)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < VIEW_W:
            if event.button == 1:        # left-click on board -> next
                self.next_step()
            elif event.button == 3:      # right-click on board -> prev
                self.prev_step()
        return True

    def _update(self):
        if not self.autoplay:
            return
        now = pygame.time.get_ticks()
        if now - self._last_auto >= AUTOPLAY_MS:
            if self.current_step < self.total:
                self.next_step()
                self._last_auto = now
            else:
                self.autoplay = False    # stop at the end

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw(self):
        self._compute_camera()
        _ox, _oy, tile = self.camera

        self.screen.fill(BG)
        pygame.draw.rect(self.screen, VIEW_BG, (0, 0, VIEW_W, WIN_H))

        # Placed tiles + their meeples (reusing the GUI's sprite renderers).
        for (x, y), pt in self.env.board.grid.items():
            sx, sy = self._w2s(x, y)
            self.screen.blit(self._tile_surface(pt.tile_type, pt.rotation, tile), (sx, sy))
            m = self._tile_meeple(x, y)
            if m is not None:
                fi, player = m
                nx, ny = feature_anchor(pt, fi)
                self._draw_meeple((int(sx + nx * tile), int(sy + ny * tile)),
                                  player, max(6, int(tile * 0.12)))

        # Highlight the most recent move.
        if self.last_move is not None:
            seat, cell = self.last_move
            sx, sy = self._w2s(*cell)
            col = HUMAN_COL if seat == 0 else AI_COL
            pygame.draw.rect(self.screen, col, (sx, sy, tile, tile), max(2, tile // 22))

        self._draw_hud()
        pygame.display.flip()

    # ----- HUD ---------------------------------------------------------
    def _text(self, s, pos, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(s, True, color), pos)

    def _wrap(self, text, font, max_w):
        lines, cur = [], ""
        for word in text.split(" "):
            trial = (cur + " " + word).strip()
            if font.size(trial)[0] <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _draw_hud(self):
        x0 = VIEW_W
        pad = x0 + 20
        pygame.draw.rect(self.screen, HUD_BG, (x0, 0, HUD_W, WIN_H))

        self._text("REPLAY VIEWER", (pad, 16), self.font_big)
        self._text(f"Step {self.current_step} / {self.total}", (pad, 52), self.font)
        if self.incomplete:
            tag = "INCOMPLETE" if self.current_step < self.total else \
                  f"INCOMPLETE: move {self.total} invalid"
            self._text(tag, (pad, 72), self.font_sm, AI_COL)

        # Players + current scores.
        s = self.env.scores
        actor = (self.current_step - 1) % 2 if self.current_step > 0 else -1
        self._text("Players", (pad, 92), self.font_sm, TEXT_DIM)
        self._draw_player_row(pad, 114, 0, s[0], actor == 0)
        self._draw_player_row(pad, 138, 1, s[1], actor == 1)

        # Action log for the current step.
        self._text("Current move", (pad, 184), self.font_sm, TEXT_DIM)
        if self.current_step == 0:
            self._text("- game start -", (pad, 206), self.font, TEXT_DIM)
        else:
            y = 206
            for line in self._wrap(self.descriptions[self.current_step - 1],
                                   self.font_sm, HUD_W - 40):
                self._text(line, (pad, y), self.font_sm)
                y += 20

        # Auto-play status.
        ap_col = (110, 220, 120) if self.autoplay else TEXT_DIM
        self._text(f"Auto-play: {'ON' if self.autoplay else 'OFF'}",
                   (pad, WIN_H - 150), self.font, ap_col)

        # Controls.
        help_lines = [
            "->/L-click : next step",
            "<-/R-click : prev step",
            "Space      : auto-play",
            "Home/End   : start/end",
            "Esc        : quit",
        ]
        y = WIN_H - 118
        for ln in help_lines:
            self._text(ln, (pad, y), self.font_sm, TEXT_DIM)
            y += 21

    def _draw_player_row(self, x, y, seat, score, is_actor):
        col = HUMAN_COL if seat == 0 else AI_COL
        pygame.draw.circle(self.screen, col, (x + 6, y + 9), 6)
        name = str(self.players[seat]) if seat < len(self.players) else f"P{seat}"
        marker = ">" if is_actor else " "
        self._text(f"{marker} {name[:14]:<14} {score:>3} pts", (x + 20, y), self.font_sm)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if not self._handle_event(event):
                    running = False
            self._update()
            self._draw()
            self.clock.tick(60)
        pygame.quit()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def load_replay(path: str, board_size: int):
    """Parse + translate a BGA log into (players, tile_sequence, actions, descriptions).

    Uses ``on_error="partial"`` so that an illegal/unsupported move does NOT abort
    the whole replay: we keep the legal prefix and can still open the viewer to
    inspect the board right up to the failure. ``TranslationError`` is only raised
    if the log can't be parsed at all (no moves), in which case ``main`` exits.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    players, moves = parse_bga_log(raw)
    # "partial" -> stop at the first illegal move but return the good prefix
    # (the offending move's debug is still printed by the translator).
    kifu = translate(players, moves, board_size=board_size, on_error="partial")
    n = len(kifu["actions"])
    descriptions = [describe_move(i, moves[i], players) for i in range(n)]

    if not kifu["complete"]:
        print(f"\n[!] Replay is INCOMPLETE: {n}/{len(moves)} moves reproduced; "
              f"move {n} could not be replayed (see the debug above). Opening the "
              f"viewer on the valid prefix -- scrub to the end (step {n}) to inspect "
              f"the board state just before the failure.")

    return (players, kifu["tile_sequence"], kifu["actions"], descriptions,
            kifu["start_rotation"], not kifu["complete"])


def main():
    parser = argparse.ArgumentParser(description="Visual replay viewer for BGA logs")
    parser.add_argument("input", help="path to the raw BGA JSON log")
    parser.add_argument("--board-size", type=int, default=32,
                        help="observation/action window; must match the size the "
                             "actions were translated with (default 32)")
    args = parser.parse_args()

    try:
        players, tile_sequence, actions, descriptions, start_rotation, incomplete = \
            load_replay(args.input, args.board_size)
    except TranslationError as exc:
        print(f"\nCannot build replay: {exc}")
        sys.exit(1)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read '{args.input}': {exc}")
        sys.exit(1)

    if not actions:
        print("No legal moves to display (even the first move is invalid).")
        sys.exit(1)

    print(f"Loaded replay: {len(actions)} move(s) playable, players = {players}, "
          f"start_rotation = {start_rotation}")
    ReplayViewer(players, tile_sequence, actions, descriptions, args.board_size,
                 start_rotation=start_rotation, incomplete=incomplete).run()


if __name__ == "__main__":
    main()

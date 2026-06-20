"""
play_gui.py
===========

A 2D pygame GUI for playing Carcassonne against a trained ``MaskablePPO`` model.

Tiles are drawn from a sprite sheet (``tiles.png`` -- a 6x4 grid of the 24 tile
types, see ``SPRITE_MAPPING``). Everything else (meeples, legal-move highlights,
grid outlines, the HUD) is still drawn with ``pygame.draw``.

The human turn is a two-step state machine, because ``env.step`` wants a single
integer encoding ``(x, y, rotation, meeple_choice)``:

  * STATE_HUMAN_PLACE  -- highlight legal cells; hover to preview; right-click to
    cycle that cell's legal rotations; left-click to confirm ``(x, y, rotation)``.
  * STATE_HUMAN_MEEPLE -- the confirmed tile is shown but NOT yet committed; pick
    a meeple feature (colored dots on the tile, or the HUD buttons) or "Skip
    Meeple". Only then is the action encoded and ``env.step`` called.

The AI turn queries the model, steps the env, and pauses briefly so you can see
what it played. Human meeples are BLUE, AI meeples are RED.

Requires: pygame (or pygame-ce), plus the training stack to load the model.

Usage
-----
    python play_gui.py models/model_gen3.zip
    python play_gui.py models/model_gen3.zip --seat 1     # you play second
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame

from env import CarcassonneEnv
from tiles import PlacedTile, Terrain, FeatureType, N, E, S, W


# --------------------------------------------------------------------------- #
# Colours & layout constants
# --------------------------------------------------------------------------- #
FIELD_GREEN = (108, 168, 86)
CITY_BROWN = (158, 102, 52)
CITY_OUTLINE = (104, 62, 28)
ROAD_GRAY = (225, 225, 222)
ROAD_OUTLINE = (120, 120, 118)
GRID_LINE = (52, 60, 52)
SHIELD_COL = (224, 226, 238)

BG = (24, 26, 32)
VIEW_BG = (38, 46, 40)
HUD_BG = (30, 33, 42)
TEXT = (235, 237, 242)
TEXT_DIM = (160, 165, 178)
HUMAN_COL = (54, 110, 230)     # blue
AI_COL = (214, 60, 56)         # red
HILITE = (248, 240, 130)       # legal-cell highlight
PENDING_BORDER = (80, 200, 255)

KIND_COLOR = {
    "city": CITY_BROWN,
    "road": (110, 110, 110),
    "field": (70, 140, 52),
    "monastery": (212, 172, 104),
}

VIEW_W, WIN_H = 840, 780
HUD_W = 300
WIN_W = VIEW_W + HUD_W
MAX_TILE = 112
MIN_TILE = 18

THINK_MS = 500     # AI "thinking" pause before it moves
SHOW_MS = 950      # pause after the AI moves so you can see it

# Normalised (0..1) tile geometry, reused by both rendering and meeple anchors.
CITY_POLY = {
    N: [(0, 0), (1, 0), (0.82, 0.45), (0.18, 0.45)],
    E: [(1, 0), (1, 1), (0.55, 0.82), (0.55, 0.18)],
    S: [(0, 1), (1, 1), (0.82, 0.55), (0.18, 0.55)],
    W: [(0, 0), (0, 1), (0.45, 0.82), (0.45, 0.18)],
}
EDGE_MID = {N: (0.5, 0.0), E: (1.0, 0.5), S: (0.5, 1.0), W: (0.0, 0.5)}
CENTER = (0.5, 0.5)
# The 8 field half-edge positions (clockwise from top-left), pulled inward.
FIELD_PT = {
    0: (0.28, 0.16), 1: (0.72, 0.16), 2: (0.84, 0.28), 3: (0.84, 0.72),
    4: (0.72, 0.84), 5: (0.28, 0.84), 6: (0.16, 0.72), 7: (0.16, 0.28),
}

# Tile sprite sheet: a 6-column x 4-row grid of the 24 tile types. Each tile's
# letter maps to its (row, col) cell; the slicing happens in CarcassonneGUI.
SPRITE_SHEET = "tiles.png"
SPRITE_COLS, SPRITE_ROWS = 6, 4
SPRITE_MAPPING = {
    'T': (0, 0), 'S': (0, 1), 'N': (0, 2), 'M': (0, 3), 'P': (0, 4), 'O': (0, 5),
    'G': (1, 0), 'F': (1, 1), 'I': (1, 2), 'H': (1, 3), 'E': (1, 4), 'K': (1, 5),
    'J': (2, 0), 'L': (2, 1), 'D': (2, 2), 'U': (2, 3), 'V': (2, 4), 'W': (2, 5),
    'X': (3, 0), 'B': (3, 1), 'A': (3, 2), 'C': (3, 3), 'R': (3, 4), 'Q': (3, 5),
}

# The sprite art is in BGA's default orientation, which differs from our env's
# rot=0 by this many 90deg clockwise steps for some tiles. We subtract it when
# rendering so env rotation r shows the matching artwork.
IMAGE_OFFSETS = {
    "A": 0, "B": 0, "C": 0, "D": 3, "E": 0, "F": 0, "G": 3, "H": 3,
    "I": 2, "J": 0, "K": 0, "L": 0, "M": 3, "N": 3, "O": 3, "P": 3,
    "Q": 0, "R": 0, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0,
}

# Human-turn state machine.
STATE_PLACE = "place"
STATE_MEEPLE = "meeple"
STATE_AI = "ai"
STATE_OVER = "over"


def _mean(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def feature_anchor(placed: PlacedTile, feature_index: int):
    """Normalised (0..1) anchor point inside the tile for a feature segment.

    ``feature_index`` follows the canonical order cities, roads, fields,
    monastery -- the same order used by ``env._decode``'s meeple choice and by
    ``Board.evaluate_placement``.
    """
    nc, nr, nf = len(placed.cities), len(placed.roads), len(placed.fields)
    if feature_index < nc:                                  # city
        edges, _shield = placed.cities[feature_index]
        return _lerp(_mean([EDGE_MID[e] for e in edges]), CENTER, 0.40)
    fi = feature_index - nc
    if fi < nr:                                             # road
        edges = placed.roads[fi]
        return _lerp(_mean([EDGE_MID[e] for e in edges]), CENTER, 0.25)
    fi -= nr
    if fi < nf:                                             # field
        positions, _adj = placed.fields[fi]
        return _mean([FIELD_PT[p] for p in positions])
    return CENTER                                           # monastery


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class CarcassonneGUI:
    def __init__(self, env: CarcassonneEnv, ai_policy, human_seat: int = 0,
                 seed=None, headless: bool = False):
        self.env = env
        self.ai = ai_policy
        self.human_seat = human_seat
        self.ai_seat = 1 - human_seat
        self.seed = seed

        pygame.init()
        flags = pygame.HIDDEN if headless else 0
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
        pygame.display.set_caption("Carcassonne -- you vs AI")
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_sm = pygame.font.SysFont("consolas", 15)
        self.font_big = pygame.font.SysFont("consolas", 26, bold=True)
        self.clock = pygame.time.Clock()

        self._tile_cache: dict = {}
        self.camera = (0.0, 0.0, MAX_TILE)   # (offset_x, offset_y, tile_px)

        self._load_sprites()
        self._reset_game()

    def _load_sprites(self):
        """Slice the tile sprite sheet into one subsurface per tile letter.

        The sheet is a ``SPRITE_COLS`` x ``SPRITE_ROWS`` grid; each sub-tile's
        width/height is the sheet size divided by the column/row count. Each
        ``SPRITE_MAPPING[letter] = (row, col)`` cell is extracted into
        ``self._sprites[letter]`` for use by ``_tile_surface``.
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SPRITE_SHEET)
        sheet = pygame.image.load(path).convert_alpha()
        tile_w = sheet.get_width() // SPRITE_COLS
        tile_h = sheet.get_height() // SPRITE_ROWS
        self._sprites: dict = {}
        for letter, (row, col) in SPRITE_MAPPING.items():
            rect = pygame.Rect(col * tile_w, row * tile_h, tile_w, tile_h)
            self._sprites[letter] = sheet.subsurface(rect).copy()

    # ------------------------------------------------------------------ #
    # Game / turn management
    # ------------------------------------------------------------------ #
    def _reset_game(self):
        self.obs, _ = self.env.reset(seed=self.seed)
        self.last_move = None          # (who, (x, y))
        self.ai_desc = ""
        self.pending = None            # (x, y, rotation) confirmed, not committed
        self.pending_placed = None
        self.pending_feats = []        # [(feature_index, kind), ...]
        self.selected_cell = None
        self.rot_idx = 0
        self.placements = {}
        if self.env.current_player == self.human_seat:
            self._begin_human_turn()
        else:
            self._begin_ai_turn()

    def _begin_human_turn(self):
        """Recompute legal placements for the human and enter the PLACE state."""
        placements: dict = {}
        for a in self.env.valid_action_list():
            x, y, rot, _m = self.env._decode(a)
            placements.setdefault((x, y), set()).add(rot)
        self.placements = {cell: sorted(rots) for cell, rots in placements.items()}
        self.selected_cell = None
        self.rot_idx = 0
        self.pending = None
        self.state = STATE_PLACE

    def _begin_ai_turn(self):
        self.state = STATE_AI
        self.ai_phase = "thinking"
        self.ai_timer = pygame.time.get_ticks()
        self._ai_terminated = False

    def _confirm_placement(self, x, y, rot):
        """Left-click confirmed a (cell, rotation). Decide meeple step or commit."""
        feats = self.env.board.evaluate_placement(self.env.current_tile_type, x, y, rot)
        mask = self.env.get_valid_actions()
        options = []
        for fi, (kind, _occ) in enumerate(feats):
            action = self.env._encode(x, y, rot, fi + 1)
            if action < len(mask) and mask[action]:
                options.append((fi, kind))

        self.pending = (x, y, rot)
        self.pending_placed = PlacedTile(self.env.current_tile_type, rot)
        self.pending_feats = options
        if options:                       # there is something to claim -> ask
            self.state = STATE_MEEPLE
        else:                             # nothing claimable -> just place it
            self._commit(0)

    def _commit(self, meeple_choice):
        """Encode the full action for the pending placement and step the env."""
        x, y, rot = self.pending
        action = self.env._encode(x, y, rot, meeple_choice)
        self.obs, _r, terminated, _t, _i = self.env.step(action)
        self.last_move = ("human", (x, y))
        self.pending = None
        if terminated:
            self.state = STATE_OVER
        else:
            self._begin_ai_turn()

    def _do_ai_move(self):
        """Query the model and commit the AI's move."""
        action = self.ai.act(self.env, self.obs)
        x, y, rot, m = self.env._decode(int(action))
        name = self.env.current_tile_name
        self.obs, _r, terminated, _t, _i = self.env.step(int(action))
        self.last_move = ("ai", (x, y))
        meeple_txt = "no meeple" if m == 0 else f"meeple #{m - 1}"
        self.ai_desc = f"AI: tile {name} @ ({x},{y}) rot{rot * 90} ({meeple_txt})"
        self._ai_terminated = terminated

    def _update(self):
        """Drive the AI's timed think/show phases."""
        if self.state != STATE_AI:
            return
        now = pygame.time.get_ticks()
        if self.ai_phase == "thinking" and now - self.ai_timer >= THINK_MS:
            self._do_ai_move()
            self.ai_phase = "showing"
            self.ai_timer = now
        elif self.ai_phase == "showing" and now - self.ai_timer >= SHOW_MS:
            if self._ai_terminated:
                self.state = STATE_OVER
            elif self.env.current_player == self.human_seat:
                self._begin_human_turn()
            else:
                self._begin_ai_turn()   # (defensive; shouldn't happen in 2p)

    # ------------------------------------------------------------------ #
    # Camera (auto-fit so every relevant cell stays visible)
    # ------------------------------------------------------------------ #
    def _compute_camera(self):
        cells = set(self.env.board.grid.keys())
        if self.state == STATE_PLACE:
            cells |= set(self.placements.keys())
        if self.pending is not None:
            cells.add((self.pending[0], self.pending[1]))
        if self.last_move is not None:
            cells.add(self.last_move[1])

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        minx, maxx, miny, maxy = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
        bw, bh = (maxx - minx + 1), (maxy - miny + 1)

        tile = min(VIEW_W // bw, WIN_H // bh, MAX_TILE)
        tile = max(tile, MIN_TILE)
        total_w, total_h = bw * tile, bh * tile
        off_x = (VIEW_W - total_w) // 2 - minx * tile
        off_y = (WIN_H - total_h) // 2 - miny * tile
        self.camera = (off_x, off_y, tile)

    def _w2s(self, x, y):
        off_x, off_y, tile = self.camera
        return int(off_x + x * tile), int(off_y + y * tile)

    def _s2w(self, px, py):
        off_x, off_y, tile = self.camera
        return int((px - off_x) // tile), int((py - off_y) // tile)

    # ------------------------------------------------------------------ #
    # Sprite-sheet tile rendering
    # ------------------------------------------------------------------ #
    def _tile_surface(self, tile_type, rotation, size):
        """Return the tile's sprite, scaled to ``size`` and rotated, cached.

        The art comes from the sprite sheet (``SPRITE_MAPPING[tile_type.name]``):
        we ``smoothscale`` to ``size`` x ``size`` for quality, then rotate.
        The art is in BGA's base orientation, which differs from our env's rot=0
        by ``IMAGE_OFFSETS[name]`` clockwise steps, so the steps to apply are
        ``(rotation - IMAGE_OFFSETS[name]) % 4``. Pygame rotates
        *counter-clockwise*, so the angle is ``-90 * steps``. (A 90deg multiple
        keeps a square square, so the surface stays ``size`` x ``size``.) A thin
        grid outline is kept so adjacent tiles read clearly.
        """
        key = (tile_type.name, rotation, size)
        if key in self._tile_cache:
            return self._tile_cache[key]

        sprite = self._sprites[tile_type.name]
        scaled = pygame.transform.smoothscale(sprite, (size, size))
        sprite_rot_steps = (rotation - IMAGE_OFFSETS[tile_type.name]) % 4
        surf = pygame.transform.rotate(scaled, -90 * sprite_rot_steps)

        pygame.draw.rect(surf, GRID_LINE, surf.get_rect(), max(1, size // 32))
        self._tile_cache[key] = surf
        return surf

    def _tile_meeple(self, x, y):
        """Return (feature_index, player) of the meeple on the tile at (x,y), or None."""
        pt = self.env.board.grid[(x, y)]
        g = self.env.board.graph
        for fi, node in enumerate(pt.features):
            root = g.find(node)
            for (mp, mx, my, _ft) in g.data[root]["meeples"]:
                if mx == x and my == y:
                    return fi, mp
        return None

    def _draw_meeple(self, center, player, radius):
        color = HUMAN_COL if player == self.human_seat else AI_COL
        pygame.draw.circle(self.screen, (250, 250, 250), center, radius + 2)
        pygame.draw.circle(self.screen, color, center, radius)

    # ------------------------------------------------------------------ #
    # Clickable geometry (shared by event handling and drawing)
    # ------------------------------------------------------------------ #
    def _meeple_dots(self):
        """List of (center, radius, meeple_choice) for the pending tile."""
        if self.pending is None:
            return []
        x, y, _rot = self.pending
        _ox, _oy, tile = self.camera
        sx, sy = self._w2s(x, y)
        radius = max(9, int(tile * 0.13))
        dots = []
        for (fi, _kind) in self.pending_feats:
            nx, ny = feature_anchor(self.pending_placed, fi)
            dots.append(((int(sx + nx * tile), int(sy + ny * tile)), radius, fi + 1))
        return dots

    def _hud_buttons(self):
        """List of (rect, payload) for HUD buttons in the current state."""
        buttons = []
        bx, bw, bh = VIEW_W + 20, HUD_W - 40, 34
        y = WIN_H - 220
        if self.state == STATE_MEEPLE:
            for (fi, kind) in self.pending_feats:
                buttons.append((pygame.Rect(bx, y, bw, bh), ("meeple", fi + 1, kind)))
                y += bh + 8
            buttons.append((pygame.Rect(bx, y, bw, bh), ("skip", 0, None)))
        elif self.state == STATE_OVER:
            buttons.append((pygame.Rect(bx, WIN_H - 90, bw, 40), ("restart", 0, None)))
        return buttons

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #
    def _handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_r and self.state == STATE_OVER:
                self._reset_game()
            return True

        if event.type == pygame.MOUSEMOTION and self.state == STATE_PLACE:
            cell = self._s2w(*event.pos)
            if cell in self.placements:
                if cell != self.selected_cell:
                    self.selected_cell = cell
                    self.rot_idx = 0
            elif event.pos[0] < VIEW_W:
                self.selected_cell = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_click(event.button, event.pos)
        return True

    def _handle_click(self, button, pos):
        # HUD buttons work in any state they're shown.
        for rect, payload in self._hud_buttons():
            if rect.collidepoint(pos):
                self._activate_button(payload)
                return

        if self.state == STATE_PLACE:
            cell = self._s2w(*pos)
            if cell not in self.placements:
                return
            rots = self.placements[cell]
            if button == 3:                       # right-click: cycle rotation
                if cell != self.selected_cell:
                    self.selected_cell, self.rot_idx = cell, 0
                else:
                    self.rot_idx = (self.rot_idx + 1) % len(rots)
            elif button == 1:                     # left-click: confirm placement
                if cell != self.selected_cell:
                    self.selected_cell, self.rot_idx = cell, 0
                self._confirm_placement(cell[0], cell[1], rots[self.rot_idx])

        elif self.state == STATE_MEEPLE:
            if button == 3:                       # right-click: cancel back to place
                self.pending = None
                self.state = STATE_PLACE
                return
            for (center, radius, choice) in self._meeple_dots():
                if (pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2 <= radius ** 2:
                    self._commit(choice)
                    return

    def _activate_button(self, payload):
        kind = payload[0]
        if kind == "meeple":
            self._commit(payload[1])
        elif kind == "skip":
            self._commit(0)
        elif kind == "restart":
            self._reset_game()

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw(self):
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, VIEW_BG, (0, 0, VIEW_W, WIN_H))
        _ox, _oy, tile = self.camera

        # Placed tiles + their meeples.
        for (x, y), pt in self.env.board.grid.items():
            sx, sy = self._w2s(x, y)
            self.screen.blit(self._tile_surface(pt.tile_type, pt.rotation, tile), (sx, sy))
            m = self._tile_meeple(x, y)
            if m is not None:
                fi, player = m
                nx, ny = feature_anchor(pt, fi)
                self._draw_meeple((int(sx + nx * tile), int(sy + ny * tile)),
                                  player, max(6, int(tile * 0.12)))

        if self.state == STATE_PLACE:
            self._draw_placement_overlay(tile)
        elif self.state == STATE_MEEPLE:
            self._draw_meeple_overlay(tile)

        # Highlight the most recent move.
        if self.last_move is not None:
            who, cell = self.last_move
            sx, sy = self._w2s(*cell)
            col = AI_COL if who == "ai" else HUMAN_COL
            pygame.draw.rect(self.screen, col, (sx, sy, tile, tile), max(2, tile // 24))

        self._draw_hud()
        pygame.display.flip()

    def _draw_placement_overlay(self, tile):
        # Legal cells.
        overlay = pygame.Surface((tile, tile), pygame.SRCALPHA)
        overlay.fill((*HILITE, 60))
        for cell in self.placements:
            sx, sy = self._w2s(*cell)
            self.screen.blit(overlay, (sx, sy))
            pygame.draw.rect(self.screen, HILITE, (sx, sy, tile, tile), 2)
        # Hover preview at the selected rotation.
        if self.selected_cell is not None:
            cell = self.selected_cell
            rot = self.placements[cell][self.rot_idx]
            sx, sy = self._w2s(*cell)
            surf = self._tile_surface(self.env.current_tile_type, rot, tile).copy()
            surf.set_alpha(200)
            self.screen.blit(surf, (sx, sy))
            pygame.draw.rect(self.screen, PENDING_BORDER, (sx, sy, tile, tile), 3)

    def _draw_meeple_overlay(self, tile):
        x, y, rot = self.pending
        sx, sy = self._w2s(x, y)
        self.screen.blit(self._tile_surface(self.env.current_tile_type, rot, tile), (sx, sy))
        pygame.draw.rect(self.screen, PENDING_BORDER, (sx, sy, tile, tile), 3)
        # Claimable feature dots.
        feats_by_choice = {fi + 1: kind for fi, kind in self.pending_feats}
        for (center, radius, choice) in self._meeple_dots():
            kind = feats_by_choice.get(choice, "field")
            pygame.draw.circle(self.screen, KIND_COLOR[kind], center, radius)
            pygame.draw.circle(self.screen, HUMAN_COL, center, radius, 3)
            pygame.draw.circle(self.screen, (255, 255, 255), center, radius, 1)

    # ----- HUD ---------------------------------------------------------
    def _text(self, s, pos, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(s, True, color), pos)

    def _draw_hud(self):
        x0 = VIEW_W
        pygame.draw.rect(self.screen, HUD_BG, (x0, 0, HUD_W, WIN_H))
        pad = x0 + 20

        self._text("CARCASSONNE", (pad, 16), self.font_big)
        you_first = "Player 0" if self.human_seat == 0 else "Player 1"
        self._text(f"You are {you_first}", (pad, 50), self.font_sm, TEXT_DIM)

        # Scores / meeples.
        s = self.env.scores
        m = self.env.meeples_remaining
        self._text("Scores", (pad, 86), self.font)
        self._draw_player_row(pad, 110, 0, s[0], m[0])
        self._draw_player_row(pad, 134, 1, s[1], m[1])

        # Whose turn.
        turn_y = 174
        if self.state in (STATE_PLACE, STATE_MEEPLE):
            self._text("YOUR TURN", (pad, turn_y), self.font, HUMAN_COL)
        elif self.state == STATE_AI:
            self._text("AI's turn...", (pad, turn_y), self.font, AI_COL)

        # Current drawn tile.
        if self.env.current_tile_type is not None and self.state != STATE_OVER:
            self._text("Drawn tile", (pad, 208), self.font_sm, TEXT_DIM)
            size = 110
            surf = self._tile_surface(self.env.current_tile_type, 0, size)
            self.screen.blit(surf, (pad, 228))
            self._text(f"'{self.env.current_tile_name}'", (pad + size + 14, 270), self.font)

        # Instructions / status.
        self._draw_status(pad)

        # Buttons.
        for rect, payload in self._hud_buttons():
            self._draw_button(rect, payload)

        if self.state == STATE_OVER:
            self._draw_gameover(pad)

    def _draw_player_row(self, x, y, player, score, meeples):
        col = HUMAN_COL if player == self.human_seat else AI_COL
        pygame.draw.circle(self.screen, col, (x + 6, y + 9), 6)
        label = "You" if player == self.human_seat else "AI "
        self._text(f"{label}  {score:>3} pts   meeples {meeples}", (x + 20, y), self.font_sm)

    def _draw_status(self, pad):
        y = 360
        lines = []
        if self.state == STATE_PLACE:
            lines = ["Hover a yellow cell.",
                     "Right-click: rotate.",
                     "Left-click: place tile."]
        elif self.state == STATE_MEEPLE:
            lines = ["Claim a feature:",
                     "click a dot or a button.",
                     "Right-click: cancel."]
        elif self.state == STATE_AI:
            lines = [self.ai_desc] if self.ai_desc else ["AI is thinking..."]
        for ln in lines:
            self._text(ln, (pad, y), self.font_sm, TEXT_DIM)
            y += 22

    def _draw_button(self, rect, payload):
        kind = payload[0]
        mouse = pygame.mouse.get_pos()
        hot = rect.collidepoint(mouse)
        base = (70, 78, 96) if not hot else (96, 106, 130)
        pygame.draw.rect(self.screen, base, rect, border_radius=6)
        if kind == "meeple":
            label = f"Claim {payload[2]}"
            pygame.draw.circle(self.screen, KIND_COLOR[payload[2]],
                               (rect.x + 16, rect.centery), 8)
            self._text(label, (rect.x + 32, rect.y + 7), self.font_sm)
        elif kind == "skip":
            self._text("Skip meeple", (rect.x + 14, rect.y + 7), self.font_sm)
        elif kind == "restart":
            self._text("New game (R)", (rect.x + 14, rect.y + 9), self.font)

    def _draw_gameover(self, pad):
        s = self.env.scores
        you, ai = s[self.human_seat], s[self.ai_seat]
        if you > ai:
            msg, col = "YOU WIN!", HUMAN_COL
        elif ai > you:
            msg, col = "AI WINS", AI_COL
        else:
            msg, col = "DRAW", TEXT
        self._text("GAME OVER", (pad, 300), self.font_big)
        self._text(msg, (pad, 334), self.font_big, col)
        self._text(f"You {you}  -  AI {ai}", (pad, 372), self.font)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        running = True
        while running:
            self._compute_camera()
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
def main():
    parser = argparse.ArgumentParser(description="Play Carcassonne vs a trained AI (GUI)")
    parser.add_argument("model", help="path to a saved MaskablePPO .zip model")
    parser.add_argument("--seat", type=int, default=0, choices=(0, 1),
                        help="which player you are (0 = move first)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Imported here so the (heavy) RL stack only loads when actually playing.
    from sb3_contrib import MaskablePPO
    from policies import ModelPolicy

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")

    model = MaskablePPO.load(args.model, device="cpu")
    board_size = model.observation_space["board"].shape[0]
    env = CarcassonneEnv(board_size=board_size)
    ai = ModelPolicy(model, name="AI", deterministic=True)

    CarcassonneGUI(env, ai, human_seat=args.seat, seed=args.seed).run()


if __name__ == "__main__":
    main()

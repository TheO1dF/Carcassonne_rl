# Carcassonne RL Environment (base game)

A local, standalone **Gymnasium** environment for the board game *Carcassonne*
(base game, 72 tiles), built for training Deep RL agents via 2-player self-play
with **action masking**.

The hard parts of Carcassonne — edge-matching tile placement and connected-component
(city / road / field) scoring — are implemented with strict validation and a
**Disjoint-Set (Union-Find)** engine.

```
tiles.py    Tile data: the 24 distinct tiles (A–X), edge/feature enums,
            field-position geometry, rotations, the 72-tile distribution.
scoring.py  Disjoint-Set (Union-Find) + per-component aggregates and the
            scoring value rules.
logic.py    Board: edge-match legality, placement/feature wiring, valid-move
            enumeration, in-game completion scoring and end-game/farm scoring.
env.py      CarcassonneEnv(gym.Env): observation/action spaces, masking, step/
            reset/render.
demo.py     Random masked-agent rollout + meeple-conservation invariant checks.
tests.py    Targeted unit tests for the scoring engine.
```

## Install & run

```bash
pip install -r requirements.txt

python tests.py        # scoring engine unit tests
python demo.py         # one verbose random game (ASCII board)
python demo.py 200     # 200 silent games, win counts + timing
```

## The model

### Tiles (`tiles.py`)
Each tile has 4 **edges** (N, E, S, W), each carrying a `Terrain`
(`CITY` / `ROAD` / `FIELD`). Two tiles fit only if touching edges share the
same terrain. Internally a tile is a set of **feature segments** (city / road /
field / monastery); segments are what meeples sit on and what the scoring graph
connects across tiles.

Fields use the standard **8 "half-edge" positions** (2 per side, numbered
clockwise from the top-left) so that a road can correctly split a field — this
is what makes end-game farm scoring correct.

Rotation (0/90/180/270, clockwise) is applied once when a tile is placed, so the
rest of the engine works in absolute board orientation.

### Scoring engine (`scoring.py`)
A `FeatureGraph` (Union-Find) gives every feature segment a node. Placing a tile
**unions** its segments with matching neighbours. Each component's root stores
`tiles`, `open` (open border edges), `shields`, `meeples`, and field→city
adjacency. A city/road is **complete** when `open == 0`. Scoring:

| Feature | During play (complete) | End of game (incomplete) |
|---|---|---|
| City | 2 × tiles + 2 × shields | 1 × tiles + 1 × shields |
| Road | 1 × tiles | 1 × tiles |
| Monastery | 9 (tile + 8 neighbours) | 1 + occupied neighbours |
| Field (farm) | — (end only) | 3 × completed adjacent cities |

Meeple placement is validated against the connected component: you cannot place
a meeple on a feature whose component already contains one.

## Gymnasium API (`env.py`)

One **step = one full turn**: place the drawn tile `(x, y, rotation)` and
optionally place a meeple on a feature of that tile. Then completed features
score, meeples return, the turn passes to the opponent, and a new tile is drawn.

### Action space — flat `Discrete` + masking
```
action = (cell * 4 + rotation) * MEEPLE_OPTIONS + meeple_choice
cell   = wy * window + wx          # (wx, wy) are WINDOW coords (see below)
meeple_choice: 0 = no meeple, 1..MAX_FEATURE_SLOTS = i-th feature of the tile
```
Cells are addressed in a **recentering window**, not absolute board coords:
`env._encode/_decode` translate between the two using the current window origin
(`_window_origin`). The board itself is a large dict-backed logical grid, so play
never hits a hard edge; the fixed-size window just crops a view that recenters on
the played area each step (translation-invariant, no centre bias).
The nominal space is huge but only a handful of moves are ever legal, so use the
mask:

```python
mask = env.action_masks()          # bool array, length == action_space.n
                                   # (also env.get_valid_actions(), and in obs)
```

`get_valid_actions()` enforces edge-matching **and** per-feature meeple legality
(occupied components and empty meeple supply are masked out).

### Observation — `Dict` of tensors
| key | shape | meaning |
|---|---|---|
| `board` | `(window, window, 8)` int8 | per-cell (recentred window): occupied, 4 edge terrains, monastery, meeple owner, meeple feature type |
| `current_tile` | `(24,)` int8 | one-hot of the drawn tile type |
| `current_tile_edges` | `(4,)` int8 | its N/E/S/W edge terrains |
| `meeples_remaining` | `(2,)` int8 | per player |
| `scores` | `(2,)` int32 | per player |
| `current_player` | `(1,)` int8 | whose turn it is |
| `action_mask` | `(n_actions,)` int8 | the legal-move mask |

### Reward
The acting player's points scored that turn (feature completions), plus their
share of end-game scoring on the final step. Dense and per-player — reshape
(e.g. zero-sum `score - opponent_score`) as your training setup prefers.

## Example: random masked rollout

```python
import numpy as np
from env import CarcassonneEnv

env = CarcassonneEnv(board_size=32)   # 32 = observation/action window side
obs, info = env.reset(seed=0)
done = False
while not done:
    legal = np.flatnonzero(env.action_masks())
    action = int(np.random.choice(legal))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.render()
print("Final scores:", env.scores)
```

## Training notes
- For masked policies, `env.action_masks()` matches the `sb3-contrib`
  **MaskablePPO** convention; the mask is also in the observation for custom
  agents.
- `board_size` (default 32) is the **observation/action window** side, not a hard
  board limit. The window recenters on the played area each step, so it only needs
  to exceed a game's *span* (~20–24 for a 2p base game), not the old grid's 2×drift.
  This removes the centre bias the old fixed grid caused (boundary cells got masked,
  so the agent never learned edge play). It MUST be identical across generations and
  match the window used to translate any BC kifus. Bump it if `span > window-2` ever
  occurs (rare); the cost scales with `window²` so 32 stays small.
  Placements outside the grid are simply never offered as legal actions.
- `num_meeples` defaults to 7 (base game).

## Training (MaskablePPO self-play)

A full RL training pipeline is included:

```
rl_model.py    CarcassonneFeaturesExtractor (CNN over board + MLP over scalars),
               SelfPlayWrapper (2-player -> single-agent + zero-sum reward),
               mask_fn (action-mask callback).
policies.py    RandomPolicy, GreedyPolicy (1-ply heuristic bot), ModelPolicy.
train.py       MaskablePPO + 8-way SubprocVecEnv self-play league.
eval_league.py Play saved models vs Random/Greedy baselines, report win rates.
play_human.py  Play against a trained model from the CLI.
```

Install the RL extras and run:

```bash
pip install stable-baselines3 sb3-contrib tensorboard
# GPU build of PyTorch (CUDA 12.4) -- the default `torch` wheel is CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Train a self-play league (8 parallel envs by default, GPU learner, 2M steps/gen)
python train.py                                   # 3 generations, device=cuda
python train.py --generations 4 --timesteps 3000000 --n-envs 12

# Resume a generation after Ctrl+C (checkpoints saved every 100k steps)
python train.py --start-gen 2 \
    --resume-model models/checkpoints/ckpt_gen2_600000_steps.zip

# Evaluate saved models against Random and Greedy baselines (100 games each)
python eval_league.py                             # auto-discovers models/model_gen*.zip
python eval_league.py models/model_gen3.zip --games 100

# Play against a trained model yourself
python play_human.py models/model_gen3.zip --seat 0   # text/CLI
python play_gui.py   models/model_gen3.zip --seat 0   # 2D pygame GUI (mouse)
```

### Graphical play (`play_gui.py`)
A `pygame` GUI that renders the 72 tiles **procedurally** (no image assets):
green fields, gray roads, brown city wedges, a house for monasteries. The human
turn is a two-step state machine — **left-click** a highlighted legal cell to
place (**right-click** cycles that cell's legal rotations), then claim a feature
by clicking a colored dot / HUD button or "Skip Meeple". The camera auto-fits so
the whole board stays visible; human meeples are blue, AI meeples red. Install
with `pip install pygame-ce` (or `pygame`).

Environment stack: `CarcassonneEnv → SelfPlayWrapper → Monitor → ActionMasker`.

- **Feature extractor** — permutes `board` `(B,H,W,C)→(B,C,H,W)`, runs a 3-layer
  CNN, and fuses the flattened result with the normalised scalar features
  (`current_tile`, `current_tile_edges`, `meeples_remaining`, `scores`,
  `current_player`) through an MLP. `action_mask` is excluded from the network
  input and used only to mask the policy logits.
- **Self-play wrapper** — each agent `step()` plays Player 0's move then
  auto-plays Player 1 via `opponent_policy` (or random valid moves if `None`).
  Reward is zero-sum, `Δ(my score) − Δ(opp score)`, with dense shaping
  (`+0.5` for placing a meeple, `−0.1` for a turn that neither scores nor places
  one) and a small `±10` terminal win/loss bonus. A saved opponent model sees a
  perspective-canonicalised observation so it always plays "as Player 0".
- **League** — Gen 1 trains vs a random opponent; each later generation trains
  against the frozen model from the previous generation (`model_gen{N}.zip`).
- **Parallelism & GPU** — training runs **8 `SubprocVecEnv` workers** by default
  (`--n-envs` to change) that feed a **GPU learner** (`device="cuda"`, large
  `batch_size`, `n_epochs=10`). The opponent is passed to each worker as a
  *path string* and `MaskablePPO.load`-ed **inside** the worker on **CPU** — so
  no un-picklable model crosses the process boundary, and we avoid spawning a
  CUDA context per worker. Each worker also caps torch to 1 thread to avoid
  oversubscribing the cores. (On Windows, each spawned worker re-imports the
  CUDA PyTorch build, so too many workers can exhaust RAM — 8 is a safe default.)
- **Resumable** — a `CheckpointCallback` saves to `models/checkpoints/` every
  100k steps, and Ctrl+C saves a recovery model. Continue with
  `--start-gen N --resume-model <ckpt>.zip`; the script trains only the steps
  remaining for that generation, then proceeds through the rest of the league.
  Note: the default `torch` wheel is **CPU-only** (`+cpu`); install a CUDA build
  to actually use the GPU (`train.py` prints `torch.cuda.is_available()` and the
  GPU name on startup, and falls back to CPU with a warning otherwise).

### Evaluation & baselines (`policies.py`, `eval_league.py`)
- **`RandomPolicy`** — uniform over the legal action mask.
- **`GreedyPolicy`** — a 1-ply heuristic bot: clones the board to look one move
  ahead, strongly prefers moves that *immediately* score (close a feature it has
  / can place a meeple on) and otherwise places meeples on cities/roads rather
  than farms. (It beats `RandomPolicy` ~10:0.)
- **`ModelPolicy`** — wraps a trained model and canonicalises the observation so
  it can play either seat.
- `eval_league.py` plays each model 100 games vs each baseline, **alternating
  seats** to cancel first-move advantage, and reports win/draw/loss + avg scores.

> Note: `BOARD_SIZE` in `train.py` must stay constant across generations (it
> fixes the observation shape and the discrete action-space size). The default
> 2M steps/gen is a starting point — Carcassonne's large masked action space
> benefits from substantially more training for strong play.

## Behavioural-Cloning data pipeline (BGA replays)

To bootstrap from human games, BGA replay logs are converted into "Kifu" records
of exact env `action` integers.

```
replay_env.py     CarcassonneReplayEnv: forces an exact tile draw order so a
                  recorded game can be reproduced step-by-step.
bga_translator.py BGA JSON log -> kifu_<timestamp>.json {players, tile_sequence,
                  actions, board_size, ...}.
```

```bash
python bga_translator.py path/to/bga_log.json --board-size 32   # window; match training
python replay_viewer.py   path/to/bga_log.json --board-size 32   # visual scrubber
```

`replay_viewer.py` is a pygame viewer that parses+translates a BGA log (reusing
`bga_translator`), reconstructs any position by stepping `CarcassonneReplayEnv`,
and renders with `play_gui`'s procedural tiles/meeples + camera (no duplicated
code). Scrub with ←/→ or mouse clicks, `Space` toggles auto-play (500 ms/step),
`Home`/`End` jump to the ends; the HUD shows `Step X / Y`, player scores, and the
current move ("Alice placed Tile 'U' at (2, -1) with meeple on Road"). Stepping
back resets and fast-replays (Gym envs can't step backward). If a move can't be
reproduced, translation keeps the valid prefix (`on_error="partial"`) and the
viewer still opens on it with an `INCOMPLETE` HUD tag, so you can scrub to the
board state right before the failure.

- `CarcassonneReplayEnv.reset(tile_sequence=[...])` ignores the random bag and
  draws those tiles in order (and, in forced mode, does **not** skip tiles, so
  the sequence stays aligned with the recorded actions).
- The translator parses BGA `playTile`+`playPartisan` notifications into a
  `BGAMove` game-record model (strictly paired by `args["id"]`; also accepts a
  simplified `{"players","moves"}` shape) and maps everything **deterministically**
  via two hand-verified tables:
  - tile id → local letter via `BGA_TO_LOCAL`;
  - `local = bga + env.size//2` (the logical board centre; BGA start = 0,0);
  - rotation `= (base_rot + IMAGE_OFFSETS[letter]) % 4`, where `base_rot =
    (ori - 1) % 4` converts BGA's 1-indexed `ori` (1–4) to our 0-indexed step
    (0–3) **before** any mod-4 math, and `IMAGE_OFFSETS` holds the per-letter
    90° CW offset between our base tile and BGA's default artwork;
  - start tile placed at `start_rotation = IMAGE_OFFSETS["D"] = 3` (BGA always
    opens with type 15 = D);
  - meeple: BGA's `pos` indexes a fixed per-tile spot that rotates *with* the
    tile (a child object), so it picks the exact feature even when a tile has
    several of one kind (W's three roads, H's two cities, X's four fields).
    `feature_order_by_pos` numbers spots by kind group (monastery, roads, cities,
    fields) in BGA's base orientation — cities by 3×3-grid reading order
    (N, W, E, S); roads the same but with a North stub **last** (so the X
    crossroads orders W, E, S, N); fields by half-edge position (clockwise from
    the top-left), except a *wrapping side field* (contains both half-edges 0 and
    7 and is bigger than the `{0,7}` corner) sorts by its real arc start — so the
    straight-road `U` splits East-field then West-field. All matching `pos`
    1, 2, 3… (`abbey`→monastery); falls back to first-free-of-kind if `pos` absent.
  Each action is then **validated against `get_valid_actions()`** before
  stepping. Illegal moves print rich debug; `--on-error` aborts or keeps the
  prefix.
- `board_size` (the window, 32) and `start_rotation` are stored in the Kifu so the
  replay viewer and BC env reproduce the exact same board; it must match training.
- Validated end-to-end on a real BGA log (`bga_replay.json`): all 71 moves
  translate with **0 warnings**, and replaying the Kifu reproduces BGA's exact
  per-move and final scores (**81 / 94**).

## Scope / simplifications
- Base game only (no expansions, no river).
- 2 players, designed for self-play.
- The start tile is a `D` tile at the board centre.

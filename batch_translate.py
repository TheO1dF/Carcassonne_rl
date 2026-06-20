"""
batch_translate.py
==================

Batch-convert the raw BGA replay JSONs in ``raw_bga_replays/`` into clean Kifu
game records in ``kifus/``, applying the **winner-only** filter for Behavioural
Cloning.

A kifu is saved ONLY if it is a clean, decisive, standard base game:
  * ``complete`` is True              -- every recorded move reproduced legally,
  * ``len(actions) == 71``            -- a full 2-player base game (72 tiles, 71
                                         placements after the face-up start tile),
  * ``winner_seat is not None``       -- not a tie (we clone the *winner*).

Unparseable / corrupted replays are skipped without stopping the batch.

Usage
-----
    python batch_translate.py
"""

from __future__ import annotations

import glob
import json
import os

from bga_translator import parse_bga_log, translate

try:                                   # nice progress bar if available
    from tqdm import tqdm
except ImportError:                    # graceful fallback to periodic prints
    tqdm = None


RAW_DIR = "raw_bga_replays"
OUT_DIR = "kifus"
BOARD_SIZE = 32                        # must match train.BOARD_SIZE (the window)
FULL_GAME_ACTIONS = 71                 # 72 base tiles - 1 face-up start tile


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        print(f"No raw replays found in '{RAW_DIR}/'. Run bga_scraper.py first.")
        return

    saved = 0
    # Reasons a file was not saved, for a useful end-of-run summary.
    skipped = {"incomplete": 0, "not_71": 0, "tie": 0, "error": 0}

    iterator = tqdm(files, desc="translating", unit="file") if tqdm else files
    for i, path in enumerate(iterator):
        name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            players, moves = parse_bga_log(raw)
            # on_error="partial" -> never raise; we inspect the result instead.
            kifu = translate(players, moves, board_size=BOARD_SIZE,
                             on_error="partial", verbose=False)

            if not kifu["complete"]:
                skipped["incomplete"] += 1
            elif len(kifu["actions"]) != FULL_GAME_ACTIONS:
                skipped["not_71"] += 1
            elif kifu["winner_seat"] is None:
                skipped["tie"] += 1
            else:
                with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
                    json.dump(kifu, f, ensure_ascii=False)
                saved += 1
        except Exception as exc:                       # noqa: BLE001 - skip & continue
            skipped["error"] += 1
            if tqdm is None:
                print(f"  [skip] {name}: {type(exc).__name__}: {exc}")

        # periodic progress when tqdm isn't installed
        if tqdm is None and (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(files)} processed, {saved} saved")

    total = len(files)
    print(f"\nSaved {saved}/{total} clean kifus into '{OUT_DIR}/'.")
    print(f"  skipped: {skipped['incomplete']} incomplete, "
          f"{skipped['not_71']} not-71-moves, {skipped['tie']} ties, "
          f"{skipped['error']} errors.")


if __name__ == "__main__":
    main()

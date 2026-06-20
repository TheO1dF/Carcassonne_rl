"""
bga_scraper.py
==============

Batch-download Carcassonne replay JSON logs for one player from Board Game
Arena (BGA), to feed the Behavioural-Cloning pipeline (``bga_translator.py``).

It drives a real browser with Playwright (sync API), reusing a saved login so
it never needs your password in the script. It follows BGA's real navigation:

  1. open the player's history PRE-FILTERED to Carcassonne (BGA's
     ``/gamestats?...&game_id=1`` page) and read the ``/table?table=<id>`` links,
     paginating "See more" until ``--limit`` is reached,
  2. for each table: ``/gamereview?table=<id>`` -> follow it to that player's
     replay view (``/archive/replay/<date>/?table=<id>&player=<id>``); the date
     segment can't be guessed, so we follow the link/redirect,
  3. pull the inline replay JSON out of the page source (the largest object after
     the ``g_bgalangages`` marker),
  4. save it to ``raw_bga_replays/<table_id>.json``.

It only keeps Carcassonne games: the history URL is already filtered to
Carcassonne (``game_id=1``), AND the downloaded log must actually contain
Carcassonne tile moves before it is saved (a bulletproof final safeguard). It is
deliberately polite and robust: it skips tables already downloaded and retries
each flaky table up to 3 times, so one bad table never stops the batch. Requests
follow a **"Burst and Deep Rest"** rhythm to respect BGA's token-bucket rate
limit -- short pauses between downloads, then a multi-minute rest every
``--batch-size`` replays (see ``--rest-mins``). (Stays strictly sequential on
purpose -- parallel downloads would risk an IP ban.)

Fast by design (HTML-only): on the **replay** page it blocks heavy assets AND
external scripts (.js), so the BGA game engine / WebGL never initialise -- the
browser treats the archive page as a dumb text document. It then waits for
``domcontentloaded``, which fires almost instantly yet guarantees the entire
multi-MB document (and the inline JSON) has finished streaming. The **history**
page loads with no blocking and the intermediate **/gamereview** page uses a
middle "review" tier (heavy assets blocked, scripts allowed) so their JS-rendered
links appear fast without dragging in images/avatars/CSS.

Setup
-----
    pip install playwright
    playwright install chromium

    # log in once interactively and save the session (do this in a desktop env):
    playwright codegen https://boardgamearena.com --save-storage bga_state.json

Usage
-----
    python bga_scraper.py --player 90943003 --limit 100
    python bga_scraper.py --player 90943003 --limit 100 --batch-size 20 --rest-mins 6
    python bga_scraper.py --player 90943003 --limit 5 --headful

Notes
-----
* ``--player`` is the **numeric** BGA user id shown in the profile URL
  (``/player?id=90943003``), not the display name.
* BGA's history DOM and redirects change over time. If table discovery comes up
  empty, tweak ``HISTORY_URL`` / the "See more" selectors in ``_click_see_more``.
  On an extraction miss the raw page HTML is dumped to ``raw_bga_replays/_debug_*``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

try:
    from playwright.sync_api import (
        sync_playwright,
        TimeoutError as PlaywrightTimeoutError,
    )
except ImportError:
    sys.exit(
        "Playwright is not installed. Run:\n"
        "    pip install playwright\n"
        "    playwright install chromium"
    )


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE = "https://boardgamearena.com"
STATE_FILE = "bga_state.json"
OUT_DIR = "raw_bga_replays"

# A marker that appears in the page source *before* the inline replay log. The
# extractor anchors here and returns the LARGEST JSON object after it, so the
# anchor only has to precede the replay -- it doesn't have to be the replay's own
# variable. ``g_bgalangages`` (the languages dict) is reliable and proven; the
# replay's own ``globalThis.g_gamelogs`` was less robust (its value isn't always a
# brace-literal the scanner can grab from right after the name).
JSON_ANCHOR = "g_bgalangages"

# BGA's game-history page, PRE-FILTERED to Carcassonne (game_id=1). Every
# /table?table=<id> link on this page is a Carcassonne match, so no game-name
# DOM checking is needed.
HISTORY_URL = BASE + "/gamestats?player={player}&opponent_id=0&game_id=1&finished=0"

# Scope table-link scraping to the actual results list. A bare a[href*="table="]
# is far too greedy -- it also matches links in the site header / notifications
# (#topbar) and the active-games sidebar (a *different*, in-progress game). These
# container-scoped selectors keep us to the match-history results only. (If a
# future BGA layout breaks this, "#page-content a[href*='table=']" is a broader
# fallback to try.)
TABLE_LINK_SELECTOR = (
    "#gamelist a[href*='table='], .palmares_game a[href*='table=']"
)

# BGA's servers can be very slow; be generous and rely on polling for the JSON.
NAV_TIMEOUT_MS = 60_000
REDIRECT_TIMEOUT_MS = 60_000
ANCHOR_TIMEOUT_MS = 30_000        # how long to poll the page source for the inline JSON

# Multi-tiered request blocking. The replay JSON is inline in the document, so on
# the archive page nothing external is needed; but the profile and intermediate
# /gamereview pages render their links with JS/CSS and must keep those.
#   "review"  -> block heavy assets but ALLOW scripts (dynamic links still render)
#   "archive" -> also block scripts: HTML-only, fastest (engine/WebGL never start;
#                the inline <script> with the JSON is part of the document, so it
#                is NEVER blocked)
_BLOCK_REVIEW = {"image", "media", "font", "stylesheet", "websocket"}
_BLOCK_ARCHIVE = _BLOCK_REVIEW | {"script"}

# Current tier: "none" (allow all), "review", or "archive". Set by scrape() and
# open_replay_page(); read by the route handler below.
_BLOCK_MODE = "none"


def _block_heavy(route) -> None:
    """Route handler: abort assets according to the current ``_BLOCK_MODE`` tier."""
    rt = route.request.resource_type
    if _BLOCK_MODE == "archive" and rt in _BLOCK_ARCHIVE:
        route.abort()
    elif _BLOCK_MODE == "review" and rt in _BLOCK_REVIEW:
        route.abort()
    else:
        route.continue_()


# --------------------------------------------------------------------------- #
# Step 3 helper -- pull the embedded replay JSON out of the page source
# --------------------------------------------------------------------------- #
def _balanced_object(s: str, start: int):
    """Return the balanced ``{...}`` substring of ``s`` beginning at ``start``.

    Tracks string literals (single or double quoted) and escapes so braces
    inside strings don't confuse the depth count. Returns ``None`` if the brace
    never closes.
    """
    depth = 0
    in_str = False
    quote = ""
    escaped = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == quote:
                in_str = False
        elif c in ("'", '"'):
            in_str = True
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def extract_replay_json(html: str):
    """Extract the replay log object embedded in an archive-replay page.

    BGA writes the replay data into an inline ``<script>`` as a big object
    literal. We anchor on ``JSON_ANCHOR`` (``g_bgalangages``, which appears just
    before it), then scan forward for balanced ``{...}`` blocks and return the
    **largest** one that ``json.loads`` accepts -- the massive replay log dwarfs
    any small objects (e.g. the languages dict) around it.
    """
    match = re.search(JSON_ANCHOR, html)
    start = match.end() if match else 0

    best = None
    best_len = 0
    n = len(html)
    i = start
    while True:
        b = html.find("{", i)
        if b == -1:
            break
        block = _balanced_object(html, b)
        if block is None:
            break                       # an unbalanced brace -> nothing usable left
        try:
            obj = json.loads(block)
            if isinstance(obj, (dict, list)) and len(block) > best_len:
                best, best_len = obj, len(block)
            i = b + len(block)          # skip the whole block we just consumed
        except json.JSONDecodeError:
            i = b + 1                    # not JSON (plain JS) -> try the next brace
        # Nothing further can beat what we already have.
        if n - i <= best_len:
            break

    return best


def is_carcassonne_replay(data) -> bool:
    """True iff the replay log contains Carcassonne tile-placement notifications.

    A definitive game check: the profile listing can mislabel a table, but only
    Carcassonne emits ``playTile`` / ``tilePlayed`` notifications. We scan the
    parsed object's nested ``type`` fields (bounded, so a huge non-Carcassonne
    log can't run away). This is what ultimately guarantees we never save a
    Seasons / Agricola / ... replay.
    """
    targets = {"playtile", "tileplayed"}
    stack = [data]
    budget = 500_000
    while stack and budget > 0:
        budget -= 1
        cur = stack.pop()
        if isinstance(cur, dict):
            t = cur.get("type")
            if isinstance(t, str) and t.lower() in targets:
                return True
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return False


# --------------------------------------------------------------------------- #
# Step 1 -- discover Carcassonne table ids for the player
# --------------------------------------------------------------------------- #
def fetch_table_ids(page, player: str, limit: int) -> list[str]:
    """Return up to ``limit`` Carcassonne ``table_id``s from the filtered history.

    The page is pre-filtered to Carcassonne (``game_id=1``), so every numeric
    ``/table?table=<id>`` link is a Carcassonne match -- no game-name checking
    needed. We simply paginate: click "See more" until enough ids are loaded (or
    the history is exhausted), then return the first ``limit``.
    """
    url = HISTORY_URL.format(player=player)
    print(f"[gamestats] {url}")
    page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)

    # The list is injected by JS; wait for at least one REAL numeric table link
    # in the RESULTS list (the static HTML only holds hidden
    # /table?table={PLACEHOLDER} templates, and we ignore header/sidebar links).
    try:
        page.wait_for_function(
            """(sel) => [...document.querySelectorAll(sel)]
                        .some(a => /[?&]table=\\d+/.test(a.getAttribute('href') || ''))""",
            arg=TABLE_LINK_SELECTOR,
            timeout=NAV_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        print("[gamestats] no Carcassonne tables found -- check --player id / login.")

    def current_ids() -> list[str]:
        """Unique numeric table ids from the RESULTS list, in document order."""
        hrefs = page.eval_on_selector_all(
            TABLE_LINK_SELECTOR,
            "els => els.map(a => a.getAttribute('href') || '')")
        ids, seen = [], set()
        for h in hrefs:
            m = re.search(r"[?&]table=(\d+)", h)
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                ids.append(m.group(1))
        return ids

    ids = current_ids()
    print(f"[gamestats] {len(ids)} table(s) loaded")

    # Paginate until we have enough, the button is gone, or no new rows appear.
    guard = 0
    while len(ids) < limit and guard < 50:
        guard += 1
        if not _click_see_more(page):
            print("[gamestats] no more results to load.")
            break
        page.wait_for_timeout(1000)            # let the new rows render
        grown = current_ids()
        if len(grown) <= len(ids):
            print("[gamestats] history exhausted.")
            break
        ids = grown
        print(f"[gamestats] {len(ids)} table(s) loaded")

    return ids[:limit]


def _click_see_more(page) -> bool:
    """Click a 'see more results' control if present. Returns True if clicked."""
    for sel in ("#see_more", "#seemore", "#see_more_tables", "a#see_more",
                "text=See more", "text=More results"):
        try:
            el = page.locator(sel).first
            if el.count() > 0 and el.is_visible():
                el.click()
                page.wait_for_timeout(800)        # let the new rows render
                return True
        except PlaywrightTimeoutError:
            return True
        except Exception:
            continue
    return False


# --------------------------------------------------------------------------- #
# Step 2 -- land on the /archive/replay/ page for a table
# --------------------------------------------------------------------------- #
def open_replay_page(page, table_id: str, player_id: str) -> str:
    """Reach that player's archive replay and return its HTML source.

    Targets ``/archive/replay/<date>/?table=<id>&player=<player_id>``. The
    ``<date>`` segment can't be guessed, so we read the replay link's ``href``
    off the review page and navigate to it **explicitly**.

    Bulletproofed for BGA's slow/flaky servers:
      * the intermediate ``/gamereview`` page uses the "review" tier (heavy assets
        blocked but scripts allowed) so it renders the link fast without dragging
        in avatars / images / CSS;
      * every ``goto`` timeout is swallowed -- the HTML is usually already there,
        and we poll the source for the inline JSON regardless;
      * the final archive page uses the "archive" tier (scripts blocked too) and
        ``wait_until="domcontentloaded"`` so the whole inline JSON has streamed.
    """
    global _BLOCK_MODE
    review_url = f"{BASE}/gamereview?table={table_id}"
    print(f"[table {table_id}] {review_url}")

    # Step A: "review" tier -- block heavy assets but keep scripts so the link
    # renders (avatars/images/CSS were what made this page excruciatingly slow).
    _BLOCK_MODE = "review"

    # Step B: a goto timeout is non-fatal -- the HTML is often fully present.
    try:
        page.goto(review_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        pass

    # gamereview sometimes redirects straight to the archive replay.
    if "/archive/replay/" in page.url:
        _BLOCK_MODE = "archive"
        return _wait_for_anchor_html(page)

    # Step C: read the replay link's href (prefer this player's view). BGA renders
    # this link via slow JS, and ``locator.count()`` is instantaneous -- it would
    # report "not found" before the link exists and refresh/abort prematurely. So
    # first explicitly WAIT for the link to be attached to the DOM.
    try:
        page.wait_for_selector('a[href*="/archive/replay/"]', state="attached",
                               timeout=15_000)
    except PlaywrightTimeoutError:
        pass

    # That wait may have ended because gamereview auto-redirected us to the
    # archive page (the "refresh" you saw) -- if so, the data is already here.
    if "/archive/replay/" in page.url:
        _BLOCK_MODE = "archive"
        return _wait_for_anchor_html(page)

    def _find_href():
        for sel in (f'a[href*="/archive/replay/"][href*="player={player_id}"]',
                    'a[href*="/archive/replay/"]'):
            loc = page.locator(sel).first
            if loc.count() > 0:
                h = loc.get_attribute("href")
                if h:
                    return h
        return None

    href = _find_href()
    if not href:
        page.wait_for_timeout(2000)
        href = _find_href()
    if not href:
        raise RuntimeError("no /archive/replay/ link found on the review page")

    if href.startswith("http"):
        replay_url = href
    elif href.startswith("/"):
        replay_url = BASE + href
    else:
        replay_url = BASE + "/" + href

    # Step D: "archive" tier -- block heavy assets AND external scripts for the
    # final page. With the engine/WebGL never loading, it is just a text document.
    _BLOCK_MODE = "archive"

    # Step E: wait for "domcontentloaded". Because external scripts are blocked it
    # fires almost instantly, yet (unlike "commit") it guarantees the ENTIRE multi-
    # MB document -- and the inline JSON -- has finished streaming, so the {...}
    # balancing never sees a truncated payload. A timeout is still non-fatal since
    # _wait_for_anchor_html polls the source anyway.
    try:
        page.goto(replay_url, wait_until="domcontentloaded", timeout=REDIRECT_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        pass

    return _wait_for_anchor_html(page)


def _wait_for_anchor_html(page, timeout_ms: int = ANCHOR_TIMEOUT_MS,
                          interval_ms: int = 200) -> str:
    """Poll the live page source until the inline replay payload is present.

    The replay JSON lives in an inline ``<script>``, so it is in the source as
    soon as the document body arrives -- no need to wait for any page events. We
    poll until ``JSON_ANCHOR`` (``g_bgalangages``) appears, then return the HTML
    (or return after the timeout, so the caller's extractor can still try / dump
    a debug page).
    """
    deadline = time.time() + timeout_ms / 1000.0
    html = page.content()
    while JSON_ANCHOR not in html and time.time() < deadline:
        page.wait_for_timeout(interval_ms)
        html = page.content()
    return html


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def scrape(player: str, limit: int, headful: bool,
           batch_size: int = 20, rest_mins: float = 6.0) -> None:
    global _BLOCK_MODE
    os.makedirs(OUT_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        context = browser.new_context(storage_state=STATE_FILE)
        context.set_default_timeout(NAV_TIMEOUT_MS)
        # Route is always registered; it blocks per the current _BLOCK_MODE tier.
        context.route("**/*", _block_heavy)
        page = context.new_page()

        # History first: "none" -> load fully so the JS-rendered match list appears.
        _BLOCK_MODE = "none"
        try:
            table_ids = fetch_table_ids(page, player, limit)
        except Exception as exc:                       # noqa: BLE001 - keep going / report
            print(f"[fatal] could not read history for '{player}': {exc}")
            browser.close()
            return

        print(f"[gamestats] {len(table_ids)} Carcassonne table(s) to fetch: {table_ids}")
        if not table_ids:
            print("[gamestats] nothing to download. Check --player id or login state.")
            browser.close()
            return

        # Replays: open_replay_page sets the per-page tier itself ("review" for
        # gamereview, then "archive" for the final inline-JSON page).
        saved = 0
        processed = 0                                  # tables we actually fetched
        for table_id in table_ids:
            out_path = os.path.join(OUT_DIR, f"{table_id}.json")

            # 1. Deduplicate: skip tables already downloaded (huge time saver on
            #    repeat runs). No navigation, so no polite delay needed.
            if os.path.exists(out_path):
                print(f"[table {table_id}] already exists, skipping.")
                continue

            # "Burst and Deep Rest": short polite pauses between requests, then a
            # multi-minute rest every `batch_size` downloads to refill BGA's
            # token bucket and avoid the rate-limit ban.
            if processed > 0:
                if processed % batch_size == 0:
                    delay = (rest_mins * 60.0) + random.uniform(0.0, 180.0)
                    print(f"\n[zZz] Batch of {batch_size} reached. Deep resting for "
                          f"{delay / 60.0:.1f} minutes to refill BGA tokens...\n")
                    time.sleep(delay)
                else:
                    delay = random.uniform(15.0, 25.0)
                    print(f"  [pause] {delay:.1f}s ...")
                    time.sleep(delay)
            processed += 1

            # 2. Up to 3 attempts: BGA's gamereview->archive step is flaky.
            done = False
            last_html = None
            for attempt in range(3):
                if attempt:
                    time.sleep(2)                      # back off before retrying
                try:
                    last_html = open_replay_page(page, table_id, player)
                    data = extract_replay_json(last_html)
                    if data is None:
                        raise RuntimeError("no replay JSON in page source")

                    # Definitive guard: never save a non-Carcassonne replay.
                    if not is_carcassonne_replay(data):
                        print(f"[table {table_id}] NOT Carcassonne (no tile "
                              f"moves) -- skipping, not saving.")
                        done = True
                        break

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    saved += 1
                    print(f"[table {table_id}] saved -> {out_path}")
                    done = True
                    break
                except Exception as exc:               # noqa: BLE001 - retry, then move on
                    if attempt < 2:
                        print(f"[table {table_id}] attempt {attempt + 1} failed, "
                              f"retrying... ({exc})")
                    else:
                        print(f"[table {table_id}] attempt {attempt + 1} failed. ({exc})")

            # 3. All attempts exhausted -> report and dump the page for debugging.
            if not done:
                print(f"[table {table_id}] FAILED after 3 attempts -- skipping.")
                if last_html is not None:
                    debug = os.path.join(OUT_DIR, f"_debug_{table_id}.html")
                    with open(debug, "w", encoding="utf-8") as f:
                        f.write(last_html)
                    print(f"[table {table_id}] dumped page to {debug} for inspection.")

        browser.close()
        print(f"\nDone: saved {saved} replay(s) into '{OUT_DIR}/'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-scrape Carcassonne replay JSONs from Board Game Arena.")
    parser.add_argument("--player", required=True,
                        help="target player's numeric BGA user id (e.g. 90943003)")
    parser.add_argument("--limit", type=int, default=10,
                        help="maximum number of replays to download (default: 10)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="replays to fetch before a deep rest (default: 20)")
    parser.add_argument("--rest-mins", type=float, default=6.0,
                        help="base minutes to deep-rest between batches (default: 6.0)")
    parser.add_argument("--headful", action="store_true",
                        help="show the browser window (useful for debugging redirects)")
    args = parser.parse_args()

    if not os.path.exists(STATE_FILE):
        sys.exit(
            f"Login state '{STATE_FILE}' not found. Create it once with:\n"
            f"    playwright codegen https://boardgamearena.com "
            f"--save-storage {STATE_FILE}\n"
            "(log in inside the window that opens, then close it.)"
        )

    scrape(args.player, args.limit, args.headful,
           batch_size=args.batch_size, rest_mins=args.rest_mins)


if __name__ == "__main__":
    main()

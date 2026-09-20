"""Play Codenames against the real spymaster, on localhost.

The published artifact runs a JavaScript port of the model over ~20 boards whose
raw feature inputs were exported ahead of time -- it has to, because a browser
cannot hold the 11,145 x 25 x 3 similarity tensor. That buys portability at the
cost of a fixed board pool. Here there is no such constraint: this serves one
page and calls `LearnedListenerSpymaster` directly, so every board is freshly
generated and the clue is the same one the arena would see.

Boards are addressed by seed, and the client sends the seed plus the revealed
words back with every request. The server therefore keeps no per-game state:
`Board.generate(seed)` is deterministic, so replaying the reveal set
reconstructs the position exactly. Refreshing the page mid-game loses nothing,
and two browsers can play the same seed without interfering.

The k=1 substitution rule (`k1_max_similarity`) is ON by default here and OFF
in `configs/spymasters.json`. That is deliberate: the rule is directionally
positive but not significant in the arena (22-18 head to head, p = 0.73;
identical 33-7 against the centroid), while against a human it is clearly
better -- it is what makes the one-word clue for Bat be BASEBALL rather than
whichever safe clue wins by the third decimal. Pass --no-k1 to turn it off.

Usage:
    python scripts/tools/play_server.py            # http://127.0.0.1:8000
    python scripts/tools/play_server.py --port 8080 --no-k1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, OpponentBoardView, Role
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

PAGE = Path(__file__).parent / "webplay" / "index.html"
ROLE_CODE = {Role.OWN: "a", Role.OPPONENT: "b", Role.NEUTRAL: "n", Role.ASSASSIN: "x"}

# LightGBM and the similarity tensor are not thread-safe to share carelessly,
# and a second request mid-scoring would interleave nothing useful anyway --
# the page issues one clue request at a time. One lock, held for the ~0.7 s.
_LOCK = threading.Lock()


class Engine:
    def __init__(self, k1: bool):
        self.sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
        self.spymaster = LearnedListenerSpymaster(k1_max_similarity=k1)
        self.k1 = k1

    def new_board(self) -> dict:
        seed = random.randrange(1, 2**31)
        return self.describe(Board.generate(seed=seed))

    @staticmethod
    def describe(board: Board) -> dict:
        return {
            "seed": board.seed,
            "words": list(board.words),
            "roles": [ROLE_CODE[c.role] for c in board.cards],
        }

    def clue(self, seed: int, revealed: list[str], turn: str) -> dict:
        board = Board.generate(seed=seed)
        on_board = {w.lower(): w for w in board.words}
        board.revealed = {on_board[w.lower()] for w in revealed if w.lower() in on_board}
        # Roles are stored from team A's perspective; team B plays the same
        # board through the view that swaps OWN and OPPONENT, exactly as the
        # two-team game loop does.
        view = board if turn == "a" else OpponentBoardView(board)
        ctx = TurnContext(board=view, turn_index=len(board.revealed))
        t0 = time.time()
        with _LOCK:
            picks = self.spymaster.top_clues(ctx, self.sims, 1)
        if not picks:
            return {"clue": None}
        word, number, score = picks[0]
        return {"clue": word, "number": int(number), "score": float(score),
                "ms": int((time.time() - t0) * 1000)}


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):      # one line per clue, not per asset
            if "/api/clue" in (args[0] if args else ""):
                sys.stderr.write("  %s\n" % (fmt % args))

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                html = PAGE.read_text(encoding="utf-8").replace(
                    "__K1__", "on" if engine.k1 else "off")
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            try:
                if self.path == "/api/board":
                    seed = req.get("seed")
                    return self._json(engine.describe(Board.generate(seed=int(seed)))
                                      if seed is not None else engine.new_board())
                if self.path == "/api/clue":
                    return self._json(engine.clue(int(req["seed"]),
                                                  list(req.get("revealed") or []),
                                                  req.get("turn", "a")))
            except Exception as exc:                       # surface it in the page
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            self._json({"error": "no such endpoint"}, 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-k1", dest="k1", action="store_false",
                    help="disable the k=1 max-similarity substitution")
    ap.add_argument("--no-open", dest="open_browser", action="store_false")
    ap.set_defaults(k1=True)
    args = ap.parse_args()

    print("loading the model...", flush=True)
    engine = Engine(k1=args.k1)
    url = f"http://{args.host}:{args.port}/"
    print(f"listening on {url}   (k=1 max-similarity rule: {'on' if args.k1 else 'off'})")
    print("Ctrl-C to stop.")
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

"""Play Codenames against the real spymaster, on localhost -- and run a blind
one-clue evaluation of spymasters against a human guesser.

The published artifact runs a JavaScript port of the model over ~20 boards whose
raw feature inputs were exported ahead of time -- it has to, because a browser
cannot hold the 11,145 x 25 x 3 similarity tensor. That buys portability at the
cost of a fixed board pool. Here there is no such constraint: this serves the
pages and calls `LearnedListenerSpymaster` directly, so every board is freshly
generated and the clue is the same one the arena would see.

Two pages:

    /       a full game. Pick the spymaster from a menu (SPYMASTERS below);
            the choice applies from the next clue.
    /eval   the blind study. One position, one clue, you guess, next. Each
            clue comes from one of `--eval-arms`, assigned here and never sent
            to the browser. Every turn is appended to `cache/human_eval.jsonl`;
            scripts/tools/analyze_human_eval.py reads it.

**Why the eval is built the way it is**, since each choice protects the data:

- *The arm and the key stay on the server.* The page gets the words, the roles
  of cards already revealed, and the clue -- nothing else. Each pick is a round
  trip that returns that one card's role. A guesser cannot learn which
  spymaster they are facing, or the key, even from devtools.
- *Arms are assigned in shuffled blocks of one each*, so the two stay balanced
  to within one position however long a session runs.
- *Stopping is a first-class outcome.* A human who no longer sees a connection
  passes; gpt-oss never does, which is exactly why the arena cannot evaluate the
  outside option. The Stop button is what records it.
- *There is no skip.* If people skipped bad clues more than good ones, the
  skips would depend on the arm -- the same arm-specific censoring that made
  the guesser-refusal discards a worry in the arena sweeps.
- *Positions span a game.* 0-8 non-assassin cards are pre-revealed at random,
  as the listener's training positions were, rather than always turn one.

Boards in the full game are addressed by seed, and the client sends the seed
plus the revealed words back with every request, so that page keeps no
server-side state. The eval page does keep state -- it has to, to hold the key.

The k=1 tiebreak (`k1_tiebreak`) is ON by default here and OFF in
`configs/spymasters.json`. Among clues already within `k1_tie_tolerance` of
optimal under the full board-aware reward, it plays the one most obviously tied
to the word it means -- BASEBALL for Bat, rather than whichever safe clue wins
by the third decimal. Pass --no-k1 to turn it off.

Usage:
    python scripts/tools/play_server.py                    # http://127.0.0.1:8000
    python scripts/tools/play_server.py --eval-arms incumbent,decoy_out25
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Two layouts: in the repo this file sits at scripts/tools/, and in a demo
# bundle (scripts/tools/make_demo_bundle.py) it sits beside `codenames/` at the
# top level. Pick whichever actually contains the package rather than assuming
# a depth, so the bundle does not depend on where it was unzipped.
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next((c for c in (_HERE.parents[1], _HERE) if (c / "codenames").is_dir()), _HERE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.learned_listener import ListenerBundle, LearnedListenerSpymaster

PAGES = {"/": "index.html", "/index.html": "index.html", "/eval": "eval.html"}
WEB = Path(__file__).parent / "webplay"
ROLE_CODE = {Role.OWN: "a", Role.OPPONENT: "b", Role.NEUTRAL: "n", Role.ASSASSIN: "x"}
DEFAULT_EVAL_LOG = DEFAULT_CACHE_DIR / "human_eval.jsonl"

# Every spymaster the pages can use. The decoy variants share ONE loaded
# booster: they differ only in `outside_n`, which is read at scoring time, so
# offering all of them costs about the memory of the two model files. An entry
# whose model file is missing is simply not offered -- the laptop demo bundle
# may not carry the decoy booster.
SPYMASTERS: dict[str, dict] = {
    "incumbent": {"label": "Incumbent", "model": "listener_gbt.txt", "outside_n": 0,
                  "about": "The deployed listener. No decoy training, no outside option."},
    "decoy": {"label": "Decoy-trained", "model": "listener_gbt_decoy.txt", "outside_n": 0,
              "about": "Retrained with 5,039 decoy positions; outside option off."},
    **{f"decoy_out{n}": {
        "label": f"Decoy-trained, outside n={n}", "model": "listener_gbt_decoy.txt",
        "outside_n": n,
        "about": f"Prices 'the guesser picks something the clue never meant' as {n} "
                 f"random vocabulary words ending the turn at zero."}
       for n in (5, 10, 25, 50)},
}

# Clue computation holds one lock: LightGBM and the tensor are shared, and a
# single user never needs two clues at once. ~0.7-1.1 s per clue.
_LOCK = threading.Lock()


class Engine:
    def __init__(self, k1: bool, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.k1 = k1
        self.sims = SimilarityTensor.load(cache_dir)
        self.clue_stats = ClueStats.load(cache_dir=cache_dir)
        self._bundles: dict[str, ListenerBundle] = {}
        self._spymasters: dict[str, LearnedListenerSpymaster] = {}

    def available(self) -> list[str]:
        return [k for k, v in SPYMASTERS.items() if (self.cache_dir / v["model"]).exists()]

    def spymaster(self, key: str) -> LearnedListenerSpymaster:
        """Built on first use and kept. Call under _LOCK."""
        if key not in SPYMASTERS or key not in self.available():
            raise ValueError(f"unknown or unavailable spymaster {key!r}; have {self.available()}")
        if key not in self._spymasters:
            spec = SPYMASTERS[key]
            if spec["model"] not in self._bundles:
                self._bundles[spec["model"]] = ListenerBundle.load(
                    self.cache_dir, self.cache_dir / spec["model"])
            self._spymasters[key] = LearnedListenerSpymaster(
                outside_n=spec["outside_n"], k1_tiebreak=self.k1,
                bundle=self._bundles[spec["model"]], clue_stats=self.clue_stats,
                cache_dir=self.cache_dir)
        return self._spymasters[key]

    @staticmethod
    def describe(board: Board) -> dict:
        return {"seed": board.seed, "words": list(board.words),
                "roles": [ROLE_CODE[c.role] for c in board.cards]}

    def new_board(self) -> dict:
        return self.describe(Board.generate(seed=random.randrange(1, 2**31)))

    def best_clue(self, board, key: str, turn_index: int) -> tuple[str, int, float] | None:
        ctx = TurnContext(board=board, turn_index=turn_index)
        with _LOCK:
            picks = self.spymaster(key).top_clues(ctx, self.sims, 1)
        if not picks:
            return None
        word, number, score = picks[0]
        return word, int(number), float(score)

    def clue(self, seed: int, revealed: list[str], turn: str, key: str) -> dict:
        board = Board.generate(seed=seed)
        on_board = {w.lower(): w for w in board.words}
        board.revealed = {on_board[w.lower()] for w in revealed if w.lower() in on_board}
        # Roles are stored from team A's perspective; team B plays the same
        # board through the view that swaps OWN and OPPONENT, exactly as the
        # two-team game loop does.
        view = board if turn == "a" else OpponentBoardView(board)
        t0 = time.time()
        got = self.best_clue(view, key, turn_index=len(board.revealed))
        if got is None:
            return {"clue": None}
        return {"clue": got[0], "number": got[1], "score": got[2], "spymaster": key,
                "ms": int((time.time() - t0) * 1000)}


class EvalTurn:
    """One guessing turn under real Codenames rules, with the key held here.

    Pure bookkeeping, no model, so the rules can be tested on their own: a
    wrong pick ends the turn; a right one allows another, up to number + 1;
    finding every remaining own word ends it; Stop ends it at once.
    """

    ENDED_BY = {"b": "opponent", "n": "neutral", "x": "assassin"}

    def __init__(self, roles: list[str], revealed: list[bool], number: int):
        self.roles, self.revealed, self.number = roles, revealed, number
        self.picks: list[int] = []
        self.ended_by: str | None = None

    @property
    def done(self) -> bool:
        return self.ended_by is not None

    def own_left(self) -> int:
        return sum(1 for i, r in enumerate(self.roles)
                   if r == "a" and not self.revealed[i] and i not in self.picks)

    def pick(self, i: int) -> str:
        if self.done:
            raise ValueError("the turn is over")
        if not 0 <= i < len(self.roles) or self.revealed[i] or i in self.picks:
            raise ValueError("that card cannot be picked")
        self.picks.append(i)
        role = self.roles[i]
        if role != "a":
            self.ended_by = self.ENDED_BY[role]
        elif self.own_left() == 0:
            self.ended_by = "cleared"
        elif len(self.picks) >= self.number + 1:
            self.ended_by = "exhausted"
        return role

    def stop(self) -> None:
        if self.done:
            raise ValueError("the turn is over")
        self.ended_by = "stopped"

    def own_found(self) -> int:
        return sum(1 for i in self.picks if self.roles[i] == "a")


class EvalStudy:
    """Serves positions, assigns arms blind, records outcomes."""

    MAX_PREREVEAL = 8

    def __init__(self, engine: Engine, arms: list[str], log_path: Path):
        self.engine, self.arms, self.log_path = engine, arms, log_path
        self.session = secrets.token_hex(4)
        self._lock = threading.Lock()
        self._block: list[str] = []
        self._pending: dict[str, dict] = {}
        self._served = 0
        self._recorded = 0

    def _next_arm(self) -> str:
        with self._lock:
            if not self._block:
                self._block = list(self.arms)
                random.shuffle(self._block)
            return self._block.pop()

    def next(self, player: str) -> dict:
        arm = self._next_arm()
        while True:                      # redraw the rare position with nothing to clue
            seed = random.randrange(1, 2**31)
            board = Board.generate(seed=seed)
            rng = random.Random(seed ^ 0x5EED)
            roles = [ROLE_CODE[c.role] for c in board.cards]
            pool = [i for i, r in enumerate(roles) if r != "x"]
            pre = set(rng.sample(pool, rng.randint(0, self.MAX_PREREVEAL)))
            if sum(1 for i, r in enumerate(roles) if r == "a" and i not in pre) < 2:
                continue
            board.revealed = {board.words[i] for i in pre}
            got = self.engine.best_clue(board, arm, turn_index=len(pre))
            if got is not None:
                break
        clue, number, score = got
        token = secrets.token_hex(8)
        revealed = [i in pre for i in range(len(roles))]
        with self._lock:
            self._served += 1
            self._pending[token] = {
                "turn": EvalTurn(roles, revealed, number), "arm": arm, "seed": seed,
                "clue": clue, "number": number, "score": score, "player": player,
                "words": list(board.words), "t_shown": time.time(), "t_first": None,
                "position": self._served,
            }
        return {"token": token, "words": list(board.words), "clue": clue, "number": number,
                "revealed": [roles[i] if revealed[i] else None for i in range(len(roles))]}

    def _finish(self, token: str, p: dict) -> dict:
        turn: EvalTurn = p["turn"]
        spec = SPYMASTERS[p["arm"]]
        first = turn.roles[turn.picks[0]] == "a" if turn.picks else None
        row = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "session": self.session, "player": p["player"], "position": p["position"],
            "seed": p["seed"], "arm": p["arm"], "model": spec["model"],
            "outside_n": spec["outside_n"], "k1_tiebreak": self.engine.k1,
            "clue": p["clue"], "number": p["number"], "score": p["score"],
            "pre_revealed": [w for w, r in zip(p["words"], turn.revealed) if r],
            "own_at_start": sum(1 for i, r in enumerate(turn.roles)
                                if r == "a" and not turn.revealed[i]),
            "picks": [{"word": p["words"][i], "role": turn.roles[i]} for i in turn.picks],
            "ended_by": turn.ended_by, "own_found": turn.own_found(),
            "first_pick_own": first, "stopped": turn.ended_by == "stopped",
            "ms_to_first_pick": (int((p["t_first"] - p["t_shown"]) * 1000)
                                 if p["t_first"] else None),
            "ms_total": int((time.time() - p["t_shown"]) * 1000),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            del self._pending[token]
            self._recorded += 1
            recorded = self._recorded
        return {"done": True, "ended_by": turn.ended_by, "own_found": turn.own_found(),
                "key": turn.roles, "recorded": recorded}

    def _get(self, token: str) -> dict:
        p = self._pending.get(token)
        if p is None:
            raise ValueError("unknown or finished position -- deal a new one")
        return p

    def pick(self, token: str, index: int) -> dict:
        p = self._get(token)
        if p["t_first"] is None:
            p["t_first"] = time.time()
        role = p["turn"].pick(index)
        out = {"role": role}
        if p["turn"].done:
            out.update(self._finish(token, p))
        return out

    def stop(self, token: str) -> dict:
        p = self._get(token)
        p["turn"].stop()
        return self._finish(token, p)

    def recorded(self) -> int:
        """Turns recorded by this server run -- the page's session counter."""
        return self._recorded


def make_handler(engine: Engine, study: EvalStudy | None, default_key: str):
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
            path = self.path.split("?")[0]
            if path in PAGES:
                html = (WEB / PAGES[path]).read_text(encoding="utf-8").replace(
                    "__K1__", "on" if engine.k1 else "off")
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/spymasters":
                return self._json({"default": default_key, "spymasters": [
                    {"id": k, **{f: SPYMASTERS[k][f] for f in ("label", "about")}}
                    for k in engine.available()]})
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
                                                  req.get("turn", "a"),
                                                  req.get("spymaster") or default_key))
                if self.path.startswith("/api/eval/"):
                    if study is None:
                        return self._json({"error": "the eval study is not enabled -- "
                                           "its arms need model files this install lacks"}, 400)
                    if self.path == "/api/eval/next":
                        out = study.next(str(req.get("player") or "")[:40])
                        out["recorded"] = study.recorded()
                        return self._json(out)
                    if self.path == "/api/eval/pick":
                        return self._json(study.pick(str(req["token"]), int(req["index"])))
                    if self.path == "/api/eval/stop":
                        return self._json(study.stop(str(req["token"])))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
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
                    help="disable the k=1 similarity tiebreak")
    ap.add_argument("--spymaster", default="incumbent", choices=sorted(SPYMASTERS),
                    help="the game page's starting choice")
    ap.add_argument("--eval-arms", default="decoy,decoy_out25",
                    help="comma-separated spymasters the blind study compares. The default "
                         "differs ONLY in the outside option -- same booster -- which is the "
                         "comparison the arena sweep made and gpt-oss could not settle.")
    ap.add_argument("--eval-log", type=Path, default=DEFAULT_EVAL_LOG)
    ap.add_argument("--no-open", dest="open_browser", action="store_false")
    ap.set_defaults(k1=True)
    args = ap.parse_args()

    print("loading the model...", flush=True)
    engine = Engine(k1=args.k1)
    have = engine.available()
    default_key = args.spymaster if args.spymaster in have else have[0]
    arms = [a.strip() for a in args.eval_arms.split(",") if a.strip()]
    missing = [a for a in arms if a not in have]
    study = None if missing or len(arms) < 2 else EvalStudy(engine, arms, args.eval_log)

    url = f"http://{args.host}:{args.port}/"
    print(f"game:  {url}   spymasters available: {', '.join(have)}")
    if study:
        print(f"study: {url}eval   arms {' vs '.join(arms)} (blind)   -> {args.eval_log}")
    else:
        print(f"study: disabled -- arms {arms} need {missing or 'at least two entries'}")
    print("Ctrl-C to stop.")
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, study, default_key))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

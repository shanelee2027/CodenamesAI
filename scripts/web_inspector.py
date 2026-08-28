"""Local web UI wrapping the inspector (SCOPE.md §M3) in a browser.

Same data as scripts/inspector.py -- same Board, same SimilarityTensor, same
guesser pool -- just served over a tiny local HTTP server (stdlib only, no
new dependency) instead of printed to a terminal. This lets you click cards
to reveal them and re-type clues interactively instead of re-running a CLI
command each time.

Usage:
    python scripts/web_inspector.py [--port 8000]
Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from codenames.board import Board, Role, is_legal_clue
from codenames.codemasters import CentroidCodemaster, LinearScorerCodemaster, RandomCodemaster
from codenames.game import ROLE_REWARD
from codenames.guessers import load_pool
from codenames.similarity import SimilarityTensor
from inspector import BASELINE_ROLE_WEIGHTS, ROLE_LABELS, baseline_score

HTML_PATH = Path(__file__).parent / "webui" / "inspector.html"

SIMS = SimilarityTensor.load()
POOL = load_pool()

# Populated in main() once --checkpoint (if given) is known -- request
# handling just reads whatever's here, so this must be set before the
# server starts serving.
CODEMASTERS: dict = {
    "random": RandomCodemaster(seed=0),
    "centroid": CentroidCodemaster(seed=0),
    "linear_scorer": LinearScorerCodemaster(),
}


def _nan_to_none(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _board_words_payload(board: Board) -> list[dict]:
    return [
        {"word": w, "role": ROLE_LABELS[board.role_of(w)], "revealed": board.is_revealed(w)}
        for w in board.words
    ]


def _make_board(seed: int, reveal: list[str]) -> Board:
    board = Board.generate(seed=seed)
    for word in reveal:
        if word:
            board.reveal(word)
    return board


def build_state_response(seed: int, reveal: list[str]) -> dict:
    board = _make_board(seed, reveal)
    return {"seed": seed, "words": _board_words_payload(board)}


def build_query_response(seed: int, clue: str, reveal: list[str], top: int, guesser_top: int) -> dict:
    board = _make_board(seed, reveal)

    response: dict = {
        "seed": seed,
        "clue": clue,
        "legal": is_legal_clue(clue, board.words),
        "in_vocab": clue.lower() in SIMS.clue_index,
        "spaces": SIMS.spaces,
        "words": _board_words_payload(board),
    }
    if not response["in_vocab"]:
        return response

    for w in response["words"]:
        values = SIMS.similarity(clue, w["word"])
        w["sims"] = {space: _nan_to_none(v) for space, v in zip(SIMS.spaces, values)}
        valid = values[~np.isnan(values)]
        w["mean_sim"] = float(valid.mean()) if len(valid) else None

    top_per_space = {}
    for space in SIMS.spaces:
        pairs = []
        for w in board.words:
            v = SIMS.similarity(clue, w, space=space)
            if not np.isnan(v):
                pairs.append((w, float(v)))
        pairs.sort(key=lambda p: -p[1])
        top_per_space[space] = [
            {"word": w, "role": ROLE_LABELS[board.role_of(w)], "value": v} for w, v in pairs[:top]
        ]
    response["top_per_space"] = top_per_space

    unrevealed = [w for w in board.words if not board.is_revealed(w)]
    guessers = []
    for name, entry in POOL.items():
        ranked = entry.guesser.rank_candidates(clue, unrevealed, SIMS)[:guesser_top]
        guessers.append(
            {
                "name": name,
                "held_out": entry.held_out,
                "picks": [{"word": w, "role": ROLE_LABELS[board.role_of(w)]} for w in ranked],
            }
        )
    response["guessers"] = guessers

    total, role_means = baseline_score(SIMS, board, clue)
    response["baseline"] = {
        "total": total,
        "role_means": {ROLE_LABELS[r]: role_means[r] for r in Role},
        "weights": {ROLE_LABELS[r]: BASELINE_ROLE_WEIGHTS[r] for r in Role},
    }
    return response


def _parse_reveal(query: dict) -> list[str]:
    raw = query.get("reveal", [""])[0]
    return raw.split(",") if raw else []


def build_give_clue_response(seed: int, reveal: list[str], codemaster_name: str) -> dict:
    if codemaster_name not in CODEMASTERS:
        return {"error": f"unknown codemaster {codemaster_name!r}, choices: {list(CODEMASTERS)}"}
    board = _make_board(seed, reveal)
    clue, number = CODEMASTERS[codemaster_name].give_clue(board, SIMS)
    return {"codemaster": codemaster_name, "clue": clue, "number": number}


def _simulate_turn(board: Board, clue: str, number: int, guesser, sims: SimilarityTensor) -> dict:
    """What actually happens if (clue, number) is played against one
    guesser, from the current board state -- same stop-on-first-miss /
    number+1-attempts logic as codenames.game.play_turn, but read-only
    (peeks at role_of, never reveals) since the same board is reused
    across every guesser in one request."""
    candidates = [w for w in board.words if not board.is_revealed(w)]
    ranked = guesser.rank_candidates(clue, candidates, sims)
    attempts = ranked[: number + 1]

    guesses = []
    reward = 0.0
    ended_reason = "no_guesses"
    for word in attempts:
        role = board.role_of(word)
        guesses.append({"word": word, "role": ROLE_LABELS[role]})
        reward += ROLE_REWARD[role]
        if role != Role.OWN:
            ended_reason = role.value
            break
    else:
        if attempts:
            ended_reason = "exhausted_guesses"

    return {"guesses": guesses, "reward": reward, "ended_reason": ended_reason}


def build_simulate_response(seed: int, reveal: list[str], clue: str, number: int) -> dict:
    board = _make_board(seed, reveal)
    results = [
        {"name": name, **_simulate_turn(board, clue, number, entry.guesser, SIMS)}
        for name, entry in POOL.items()
    ]
    return {"clue": clue, "number": number, "results": results}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:  # keep stdout quiet
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path in ("/", "/inspector.html"):
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/state":
            seed = int(query.get("seed", ["42"])[0])
            self._send_json(build_state_response(seed, _parse_reveal(query)))
            return

        if parsed.path == "/api/query":
            seed = int(query.get("seed", ["42"])[0])
            clue = query.get("clue", [""])[0].strip()
            top = int(query.get("top", ["10"])[0])
            guesser_top = int(query.get("guesser_top", ["5"])[0])
            if not clue:
                self._send_json({"error": "clue is required"}, status=400)
                return
            self._send_json(build_query_response(seed, clue, _parse_reveal(query), top, guesser_top))
            return

        if parsed.path == "/api/codemasters":
            self._send_json({"codemasters": list(CODEMASTERS)})
            return

        if parsed.path == "/api/give_clue":
            seed = int(query.get("seed", ["42"])[0])
            codemaster_name = query.get("codemaster", [""])[0]
            response = build_give_clue_response(seed, _parse_reveal(query), codemaster_name)
            self._send_json(response, status=400 if "error" in response else 200)
            return

        if parsed.path == "/api/simulate":
            seed = int(query.get("seed", ["42"])[0])
            clue = query.get("clue", [""])[0].strip()
            number = int(query.get("number", ["1"])[0])
            if not clue:
                self._send_json({"error": "clue is required"}, status=400)
                return
            self._send_json(build_simulate_response(seed, _parse_reveal(query), clue, number))
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="scorer checkpoint from scripts/train_scorer.py -- adds a 'learned' codemaster if given"
    )
    parser.add_argument("--risk-aversion", type=float, default=None, help="miss_penalty for the learned codemaster (default: -10.0, see codenames.scorer)")
    args = parser.parse_args()

    if args.checkpoint is not None:
        from codenames.codemasters import LearnedCodemaster

        learned_kwargs = {} if args.risk_aversion is None else {"miss_penalty": args.risk_aversion}
        CODEMASTERS["learned"] = LearnedCodemaster(args.checkpoint, **learned_kwargs)
        print(f"loaded learned codemaster from {args.checkpoint}")

    server = ThreadingHTTPServer(("localhost", args.port), Handler)
    print(f"Inspector web UI running at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

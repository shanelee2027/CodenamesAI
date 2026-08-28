"""Local web UI wrapping the inspector (SCOPE.md §M3) in a browser.

Same data as scripts/inspector.py -- same Board, same SimilarityTensor, same
guesser pool -- just served over a tiny local HTTP server (stdlib only, no
new dependency) instead of printed to a terminal. This lets you click cards
to reveal them and re-type clues interactively instead of re-running a CLI
command each time.

Also lets you test codemasters (random/centroid, an "oracle:numberbatch"
upper-bound explorer -- see codenames/codemasters/oracle.py -- and any
trained "learned" checkpoint found under cache/checkpoints/ or
cache/m9/checkpoints/noise_*/, auto-discovered at startup, no flag needed
for the common case -- currently the noise-sweep variants, see
docs/log.md) and simulate the resulting turn against every guesser. Two
things stay adjustable at request time with no retraining, since neither
is baked into the trained model: the 4 reward values a learned
codemaster's clue choice (and the simulated turn's displayed reward) are
scored against (own/neutral/opponent/assassin -- see
codenames/scorer.py's module docstring), and which noise-level guesser
pool a turn gets simulated against (one of NOISE_LEVELS, independent of
which noise level the codemaster itself was *trained* under). A third
knob, `max_rarity`, screens candidate clues by CLUE_RARITY_PERCENTILE --
derived once at startup from wordfreq's conversational/subtitle-weighted
word frequencies (not raw web-corpus rank, which badly overrates place
names -- see docs/log.md) -- so an obscure pick like "confectionery" can
be filtered out without retraining anything either.

Usage:
    python scripts/web_inspector.py [--port 8000]
Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import argparse
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from wordfreq import zipf_frequency
from codenames.board import Board, Role, is_legal_clue
from codenames.codemasters import CentroidCodemaster, OracleCodemaster, RandomCodemaster
from codenames.game import ROLE_REWARD
from codenames.guessers import load_pool
from codenames.guessers.registry import DEFAULT_POOL_CONFIG
from codenames.scorer import DEFAULT_MISS_PENALTY, OWN_REWARD
from codenames.similarity import SimilarityTensor
from inspector import BASELINE_ROLE_WEIGHTS, ROLE_LABELS, baseline_score

HTML_PATH = Path(__file__).parent / "webui" / "inspector.html"

SIMS = SimilarityTensor.load()

# The same discrete noise levels scripts/run_ablation_study.py's
# --noise-levels sweep trains against -- picking any other value would
# need a fresh guesser pool AND wouldn't correspond to any trained
# learned:noise_* codemaster, so the play-time noise dial is restricted
# to exactly these rather than a free-form number.
NOISE_LEVELS = [0.0, 0.03, 0.06, 0.08, 0.1, 0.15]
DEFAULT_NOISE = 0.03
_BASE_POOL_CONFIG = json.loads(DEFAULT_POOL_CONFIG.read_text())


def _pool_config_at_noise(noise_std: float) -> dict:
    config = copy.deepcopy(_BASE_POOL_CONFIG)
    for entry in config["guessers"]:
        if entry.get("type") == "noisy":
            entry["params"]["noise_std"] = noise_std
    return config


# One pool per noise level, built once at startup (load_pool accepts an
# in-memory config dict, not just a file path -- see registry.py) so a
# simulate request just picks one rather than reconstructing guessers
# per request.
POOLS_BY_NOISE = {level: load_pool(_pool_config_at_noise(level)) for level in NOISE_LEVELS}
POOL = POOLS_BY_NOISE[DEFAULT_NOISE]


def _build_clue_rarity_percentile(clue_words: list[str]) -> dict[str, float]:
    """0.0 = the most common word in the clue vocabulary, ~100.0 = the
    rarest -- lets the UI filter out obscure clues like "confectionery".

    Originally derived from GloVe's own frequency-ordered file position,
    but that was a bad proxy for "a person would recognize this word" --
    proper nouns (city names especially) get mentioned constantly in the
    news/web/Wikipedia text GloVe was trained on regardless of whether an
    average speaker actually knows them, so e.g. "Stuttgart" and
    "Helsinki" both landed in the top 10% by that measure (confirmed
    empirically, not assumed -- see docs/log.md). `wordfreq.zipf_frequency`
    blends subtitle/conversational-text frequency in alongside web text
    specifically to correct for that skew (subtitle frequency is the
    standard psycholinguistic fix for "recognizable word" vs. "frequently
    printed word"), fully offline after install (bundled data, no network
    calls at runtime).

    Percentile is computed within the clue vocabulary itself (not all of
    wordfreq's English vocabulary), since that's the pool an actual filter
    choice is made over -- clue_words already skews toward moderately-
    common words by construction (build_similarity_tensor.py's top-N +
    intersection filtering), so a percentile against the full English
    lexicon would make even a fairly obscure Codenames clue look
    deceptively "common."
    """
    scores = np.array([zipf_frequency(w, "en") for w in clue_words])
    order = np.argsort(-scores)  # descending: highest zipf (most common) first
    percentile = np.empty(len(clue_words), dtype=np.float64)
    percentile[order] = np.arange(len(clue_words)) / len(clue_words) * 100.0
    return dict(zip(clue_words, percentile))


CLUE_RARITY_PERCENTILE = _build_clue_rarity_percentile(SIMS.clue_words)


def _discover_checkpoints() -> dict[str, Path]:
    """Scan scripts/run_ablation_study.py's noise-sweep checkpoints for
    trained models, so the web UI can offer "learned" codemasters without
    needing a --checkpoint flag for the common case. Restricted to
    noise_*/ specifically (not every subdirectory under
    cache/m9/checkpoints/) so the dropdown stays limited to the permanent
    noise-level variants even if a full ablation study run (drop-space,
    pool-sensitivity, etc. -- not kept as UI options) leaves its other
    checkpoints on disk too. `cache/blend_pool/checkpoints/` is a second,
    explicitly-named exception: the single-guesser weighted-blend pool
    variant (configs/guesser_pool_blend.json), intentionally listed here
    rather than matched by a wildcard, same reasoning as the noise_*
    restriction -- only checkpoints meant to be permanent UI options
    should show up automatically."""
    found: dict[str, Path] = {}
    default = Path("cache/checkpoints/scorer_best.pt")
    if default.exists():
        found["checkpoints"] = default
    for path in sorted(Path("cache/m9/checkpoints").glob("noise_*/scorer_best.pt")):
        found[path.parent.name] = path
    blend_path = Path("cache/blend_pool/checkpoints/scorer_best.pt")
    if blend_path.exists():
        found["blend"] = blend_path
    return found


def _load_learned_codemasters() -> dict:
    from codenames.codemasters import LearnedCodemaster

    learned = {}
    for label, path in _discover_checkpoints().items():
        try:
            learned[f"learned:{label}"] = LearnedCodemaster(path)
        except Exception as e:
            # e.g. linear_baseline's checkpoint holds a LinearScorer, whose
            # state dict doesn't match Scorer's architecture -- skip rather
            # than fail the whole server over one incompatible checkpoint.
            print(f"skipping checkpoint {path} ({label}): {e}")
    return learned


# CODEMASTERS is finalized before the server starts serving (main() may add
# an explicit --checkpoint on top of whatever auto-discovery found).
CODEMASTERS: dict = {
    "random": RandomCodemaster(seed=0),
    "centroid": CentroidCodemaster(seed=0),
}
if "numberbatch" in SIMS.spaces:
    CODEMASTERS["oracle:numberbatch"] = OracleCodemaster(space="numberbatch")
CODEMASTERS.update(_load_learned_codemasters())


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


# Query-param name -> LearnedCodemaster attribute. "risk_aversion" keeps
# its established name (-> miss_penalty, the assassin value) rather than
# being renamed "assassin_reward" everywhere, since that's the field the
# UI has always called it and matches SCOPE's own "risk aversion" framing.
_REWARD_PARAMS = {
    "risk_aversion": "miss_penalty",
    "own_reward": "own_reward",
    "neutral_reward": "neutral_reward",
    "opponent_reward": "opponent_reward",
}


def _apply_reward_overrides(codemaster, overrides: dict[str, str]) -> None:
    """Set any of LearnedCodemaster's 4 reward attributes directly on the
    shared instance from raw (possibly empty) query-string values -- fine
    for a single-user local dev tool. No-op for codemasters without these
    attributes (everything except LearnedCodemaster)."""
    for param, attr in _REWARD_PARAMS.items():
        value = overrides.get(param, "")
        if value and hasattr(codemaster, attr):
            setattr(codemaster, attr, float(value))


# Over-fetch pool when a rarity filter is active: top_k_clues' own
# candidate pool (clue_search._CANDIDATE_POOL) already defaults to 200,
# so asking for a few hundred more is close to free computationally (the
# forward pass scoring the whole vocabulary already happened; this only
# affects how many of clue_search's already-sorted candidates get walked
# for legality + the rarity check).
_RARITY_FETCH_POOL = 300


def build_give_clue_response(
    seed: int,
    reveal: list[str],
    codemaster_name: str,
    reward_overrides: dict[str, str] | None = None,
    top_k: int = 1,
    max_rarity: float = 100.0,
) -> dict:
    """`max_rarity` (0-100, default 100 = no filtering) excludes clues
    above that CLUE_RARITY_PERCENTILE -- e.g. max_rarity=50 keeps only
    the more-common half of the clue vocabulary, screening out obscure
    picks like "confectionery". Only applies to codemasters exposing
    top_k_clues (i.e. not RandomCodemaster, which has no ranking to
    filter); may return fewer than top_k if the over-fetch pool doesn't
    contain that many eligible clues, same as top_k_legal_clues' own
    "fewer than k" case."""
    if codemaster_name not in CODEMASTERS:
        return {"error": f"unknown codemaster {codemaster_name!r}, choices: {list(CODEMASTERS)}"}
    codemaster = CODEMASTERS[codemaster_name]
    _apply_reward_overrides(codemaster, reward_overrides or {})
    board = _make_board(seed, reveal)

    filtering = max_rarity < 100.0 and hasattr(codemaster, "top_k_clues")
    if (top_k > 1 or filtering) and hasattr(codemaster, "top_k_clues"):
        fetch_k = max(top_k, _RARITY_FETCH_POOL) if filtering else top_k
        candidates = codemaster.top_k_clues(board, SIMS, fetch_k)
        if filtering:
            candidates = [c for c in candidates if CLUE_RARITY_PERCENTILE.get(c[0], 100.0) <= max_rarity]
        clues = [
            {"clue": c, "number": n, "score": s, "rarity_percentile": CLUE_RARITY_PERCENTILE.get(c)}
            for c, n, s in candidates[:top_k]
        ]
    else:
        clue, number = codemaster.give_clue(board, SIMS)
        clues = [{"clue": clue, "number": number, "score": None, "rarity_percentile": CLUE_RARITY_PERCENTILE.get(clue)}]

    return {"codemaster": codemaster_name, "clues": clues}


def _simulate_turn(board: Board, clue: str, number: int, guesser, sims: SimilarityTensor, reward_table: dict) -> dict:
    """What actually happens if (clue, number) is played against one
    guesser, from the current board state -- same stop-on-first-miss /
    number-attempts logic as codenames.game.play_turn, but read-only
    (peeks at role_of, never reveals) since the same board is reused
    across every guesser in one request. `reward_table` lets the
    displayed reward reflect whatever reward overrides the UI is
    currently exploring, rather than always the true ROLE_REWARD."""
    candidates = [w for w in board.words if not board.is_revealed(w)]
    ranked = guesser.rank_candidates(clue, candidates, sims)
    attempts = ranked[:number]

    guesses = []
    reward = 0.0
    ended_reason = "no_guesses"
    for word in attempts:
        role = board.role_of(word)
        guesses.append({"word": word, "role": ROLE_LABELS[role]})
        reward += reward_table[role]
        if role != Role.OWN:
            ended_reason = role.value
            break
    else:
        if attempts:
            ended_reason = "exhausted_guesses"

    return {"guesses": guesses, "reward": reward, "ended_reason": ended_reason}


def build_simulate_response(
    seed: int, reveal: list[str], clue: str, number: int, reward_overrides: dict[str, str] | None = None, noise: float = DEFAULT_NOISE
) -> dict:
    overrides = reward_overrides or {}
    reward_table = {
        Role.OWN: float(overrides.get("own_reward") or OWN_REWARD),
        Role.NEUTRAL: float(overrides.get("neutral_reward") or ROLE_REWARD[Role.NEUTRAL]),
        Role.OPPONENT: float(overrides.get("opponent_reward") or ROLE_REWARD[Role.OPPONENT]),
        Role.ASSASSIN: float(overrides.get("risk_aversion") or DEFAULT_MISS_PENALTY),
    }
    pool = POOLS_BY_NOISE.get(noise, POOL)
    board = _make_board(seed, reveal)
    results = [
        {"name": name, **_simulate_turn(board, clue, number, entry.guesser, SIMS, reward_table)}
        for name, entry in pool.items()
    ]
    return {"clue": clue, "number": number, "noise": noise, "results": results}


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
            reward_overrides = {param: query.get(param, [""])[0] for param in _REWARD_PARAMS}
            top_k = int(query.get("top_k", ["1"])[0])
            max_rarity = float(query.get("max_rarity", ["100"])[0])
            response = build_give_clue_response(seed, _parse_reveal(query), codemaster_name, reward_overrides, top_k, max_rarity)
            self._send_json(response, status=400 if "error" in response else 200)
            return

        if parsed.path == "/api/simulate":
            seed = int(query.get("seed", ["42"])[0])
            clue = query.get("clue", [""])[0].strip()
            number = int(query.get("number", ["1"])[0])
            noise = float(query.get("noise", [str(DEFAULT_NOISE)])[0])
            reward_overrides = {param: query.get(param, [""])[0] for param in _REWARD_PARAMS}
            if not clue:
                self._send_json({"error": "clue is required"}, status=400)
                return
            self._send_json(build_simulate_response(seed, _parse_reveal(query), clue, number, reward_overrides, noise))
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="extra scorer checkpoint to load as 'learned', e.g. one outside the auto-scanned "
        "cache/checkpoints/ and cache/m9/checkpoints/*/ locations. Not required for the common "
        "case -- checkpoints in those locations are picked up automatically at startup (see the "
        "'learned:<name>' entries in the codemaster dropdown). Risk aversion is set from the web UI, not a flag.",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        from codenames.codemasters import LearnedCodemaster

        CODEMASTERS["learned"] = LearnedCodemaster(args.checkpoint)
        print(f"loaded learned codemaster from {args.checkpoint}")

    server = ThreadingHTTPServer(("localhost", args.port), Handler)
    print(f"Inspector web UI running at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

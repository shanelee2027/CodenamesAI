# Working log

Record what was expected vs. what actually happened as work proceeds,
oldest first, one entry per piece of work. The log starts at the first
threshold-based baseline; the board, similarity tensor, guesser and
evaluation infrastructure it builds on are described in `README.md` and
`docs/iteration-architecture.md`. See `docs/worklog.md` for a short
checklist of what has been tried, `docs/versions/` for the models, and
`docs/design-decisions.md` for standing rationale.


## z_threshold baseline and cache/clue_stats.npz

Built the fifth baseline (`docs/versions/z_threshold.md` has the design
and sanity check). Two new modules, one script, one registry entry:
`codenames/clue_stats.py` (`ClueStats` dataclass -- per-clue mean/std over
all 400 board words, a rarity percentile, and `z_for_board`),
`scripts/data/build_clue_stats.py` (writes `cache/clue_stats.npz` +
`cache/clue_stats_meta.json`), `codenames/spymasters/z_threshold.py`
(`ZThresholdSpymaster`), and a `z_threshold` entry in
`configs/spymasters.json` with `"roles": ["baseline"]`.

Expected: percentile-to-z-threshold conversion at construction
(`statistics.NormalDist`, no scipy), a risk term over Gaussian noise in
cosine space weighted by each role's real reward magnitude so an assassin
costs 10x a neutral at equal margin, and the two required fallbacks (no
valid clue -> ignore role thresholds; no eligible clue -> force the
single best own word). All landed as designed; 21 new tests
(`tests/test_clue_stats.py`, `tests/test_z_threshold_spymaster.py`)
cover threshold filtering, the `MAX_CLUE_NUMBER` cap, revealed-word
exclusion, both fallbacks, determinism, legality, the assassin-vs-neutral
risk asymmetry, rarity filtering, and `ClueStats` rejecting a vocabulary
hash mismatch. 335 tests pass (314 + 21).

**Judgment call: `ClueStats`/`clue_stats` as an extra keyword-only
constructor argument.** The spec's seven tunable parameters don't include
a cache location or a way to inject a pre-built `ClueStats`, but tests
need small synthetic fixtures rather than the real 256MB tensor + 2.7MB
cache, and `top_clues(ctx, sims, k)`'s signature is fixed by
`spymasters/base.py` -- there's nowhere else to pass one in per-call. Added
`cache_dir`/`clue_stats` as keyword-only, defaulting to the real load path,
so registry/production use is unaffected and tests can hand in a
synthetic instance directly.

**Sanity check surprised in one way, worth flagging even though it
checked out.** Every one of 10 real holdout boards announced number=4
(the max) for its top clue. Verified this isn't a single runaway "hub"
word dominating every board -- the true max own-word count across the
whole 111,440-word vocabulary per board was only 5-7, and hundreds to
~1,300 distinct clues cleared the k>=4 bar per board. It's simply common
for *some* clue among an 11,000-candidate pool to beat 4 of a random
9-word own set at only a 90th-percentile bar. Not a bug, but it means
`own_top=0.10` with `MAX_CLUE_NUMBER=4` makes "announce 4" the modal
outcome on a fresh board under the defaults -- worth a sweep before this
baseline's numbers get quoted as typical.

**Environment note, not a design issue:** this worktree's `cache/`
started empty (correctly gitignored) while the real ~256MB similarity
tensor lives in the main checkout's `cache/`. Symlinked the four
similarity-tensor files into this worktree's `cache/` to run
`build_clue_stats.py` and the sanity check against real data without
copying 267MB. Separately hit a real footgun worth recording: running
`python scripts/data/build_clue_stats.py` directly puts the script's own
directory first on `sys.path`, and an editable install of this package
(pointing at the main checkout) then shadows this worktree's `codenames/`
package entirely -- the first attempt silently computed against and wrote
into the *main checkout's* `cache/`, not this worktree's. Caught it by
checking where the output landed, deleted the stray files there, and
reran with the worktree root explicitly inserted ahead of the script's
directory on `sys.path`. Anyone running a `scripts/` entry point directly
(not via pytest, which doesn't hit this) in a worktree with an editable
install pointed elsewhere should watch for this.

## expected_words baseline: replaces z_threshold entirely

Implemented the threshold-free baseline `docs/versions/expected_words.md`
describes: `codenames/spymasters/expected_words.py`
(`ExpectedWordsSpymaster`), registry entry `expected_words`, and deleted
`z_threshold.py`/its test/its config entry/its registry entry/its doc
rather than keeping both around. One metric,
`score(clue, k) = gain(k) - penalty(k)`, replaces the old own/neutral/
opponent/assassin thresholds and their three-stage fallback chain --
every `(clue, k)` pair now scores finite, so there's no "no valid clue"
state to fall back out of.

**Refactored the metric's core algebra into a standalone pure function,
`gain_and_penalty(a, b, costs, tau_gain, tau_pen)`, not required by the
spec.** It takes plain numpy arrays (no `ClueStats`/`SimilarityTensor`),
so the sub-linear `gain` term -- the property the spec explicitly says
not to "simplify" away -- can be unit-tested against hand-computed
z-scores directly, instead of only observable indirectly through a full
board/clue-vocabulary fixture. `_score_all_clues` just calls it. Judgment
call, not a spec requirement; flagging since it's a structural choice
someone reviewing the diff might ask about.

**Judgment call: the spec's "assassin proximity penalised ~10x a
neutral" doesn't match `ROLE_REWARD` as written.** `ROLE_REWARD` gives
neutral=-0.2, opponent=-1.0, assassin=-10.0 -- assassin costs the
*opponent* role exactly 10x at equal margin, but a *neutral* 50x (10.0 /
0.2), not 10x. Wrote both tests: one asserting the exact 10x ratio
against `opponent` (matching the spec's number precisely), and one
asserting the actual 50x ratio against `neutral` (matching the spec's
role, documenting the real number rather than asserting a false "10x").
Did not change `ROLE_REWARD` -- that's `codenames/game.py`'s own reward
formula, out of scope here, and the task said implement the design
exactly.

**Verification numbers** (40 fresh `load_holdout_wordlist()` boards):
mean announced number 2.025 (expected ~2.0), distribution concentrated
on k=2 (37/40), median assassin margin +4.32 (expected ~+4.3), worst
+2.40 (expected ~+2.4) -- all within the validated design's targets.
Smoke test (`run_two_team_arena.py --n-boards 100`) against all three
synthetic guessers played legally and finished every game; assassin-hit
rate 0.0%/14.0%/23.0% for numberbatch/glove/wikipedia2vec respectively,
reproducing `z_threshold.md`'s cross-space-assassin pattern as expected
(inherited, not re-derived -- this model still selects on a single
space).

Hit the same editable-install-shadows-the-worktree footgun the previous
log entry already flagged, from the other direction: running
`scripts/pipeline/run_two_team_arena.py` directly picked up the *main
checkout's* `codenames` package (still had `z_threshold`, not
`expected_words`) even after clearing `__pycache__`, because the script's
own directory (not the repo root) is what Python puts on `sys.path[0]`.
Fixed by running with `PYTHONPATH=.` from the repo root rather than
editing `sys.path` in the script itself. 340 tests pass (335 - 17
deleted z_threshold tests + 19 new expected_words tests + 3 pre-existing
uncommitted changes already in this worktree at session start).

## Sizing the first paid evaluation

Before spending anything on the LLM guesser, measured what 50 two-team
games (baseline spymaster on both sides) would actually cost. Simulated
the games locally with a wrapper guesser that builds the exact prompt
`LLMGuesser` would send, records its size, then delegates the ranking to
`noisy_numberbatch` -- so the call count and prompt sizes are real
without spending anything.

**Measured:** 14.3 guesser calls per game (min 10, max 22), **715 calls**
for 50 games; mean 17.0 candidate words per call (not 25 -- most turns
happen after some words are revealed); prompt 533 chars ~ 165 tokens.
`LLMGuesser.max_tokens` is 512, which bounds worst-case output at 366k
tokens for the whole run regardless of how much the model thinks.

Priced the proposal to have the model return only the `n` words it would
guess instead of the full ranking. The full JSON array averages **158
chars ~ 44 tokens**; top-n only averages 11 chars ~ 3 tokens. Saving is
~41 tokens/call, **~$2.20 of a ~$20-27 run (~10%)**, because thinking
tokens bill as output and don't shrink when the answer does -- the model
still weighs all 17 words to name 1. Not taken: the full ranking is the
only way to see *where* the assassin sat when a clue fails, which is the
data that calibrates sigma, and `_parse_ranking`'s board-order fallback
would silently become the normal path for the unranked words. The
non-cost argument for top-n (it's the task a real guesser performs) is a
separate methodology question, not a budget one.

Note these dollar figures assume $15/$75 per Mtok and, more importantly,
*estimate* the thinking/text split rather than measuring it.
`scripts/tools/probe_llm_cost.py` settles it for a few cents: it runs
real board positions through the API, splits `usage.output_tokens` into
visible text (measured with the free `count_tokens` endpoint) and
inferred thinking, and projects to 715 calls. It also doubles as the
first live exercise of the eval path, which until now has only ever run
against mocks -- its last stage makes one real `LLMGuesser` call and
checks that the response parses, the disk cache writes, and a fresh
instance reads the same ranking back.

Rejected using the Claude subscription (`claude -p` / the Agent SDK)
instead of the API. It would be free at the margin, but ships the harness
system prompt and tool definitions with every 165-token request, spawns a
process per call, and -- the reason that actually decides it -- gives up
exact model pinning. `CLAUDE.md` already names `cache/llm_store.db` as
the only thing making past evaluations reproducible; "Messages API,
model=claude-opus-5, effort=medium" is defensible in a viva, "shelled out
to my IDE assistant" is not.

## Clue legality was letting near-identical clues through

Found while reading the first live Opus responses: for the clue `centre`
the guesser answered `["Center"]` -- the board word, in British spelling.
`is_legal_clue` only rejected substring/stem overlap, and neither
"centre" nor "center" contains the other.

Not a one-off. Over 60 fresh holdout boards, **6 (10%)** of the
baseline's chosen clues were forms of an intended word: `mexican`/Mexico,
`canadian`/Canada (x2), `changing`/Change, `led`/Lead, plus
`centre`/Center from the probe. Every one is illegal at a real table, and
every one scored **z between +8.8 and +12.5** against a typical winning
clue of +4 to +6 -- they win *because* they are the same word. Left in,
they would have inflated the paid evaluation by ~10% of turns, and Opus
takes them instantly.

**Rejected: fuzzy string similarity.** `SequenceMatcher(...).ratio() >=
0.8` was the obvious catch-all and is badly wrong here. It forbade
`able`/Marble, `after`/Water, `agree`/Green, `am`/Arm, `bar`/Bear,
`bat`/Beat -- coincidental letter overlap, not shared roots -- putting
**7.0%** of the admissible pool out of reach to catch two extra cases.
Short words are where it fails worst, since a single shared letter pair
moves the ratio a long way.

**Taken: shared prefix, plus targeted normalization.** Measured over the
full 11,145-clue admissible pool x 400 board words:

| rule | catches | pool affected |
|---|---|---|
| prefix >= 4 | 4/6 | 5.16% |
| **prefix >= 5 + spelling fold** | **4/6** | **1.04%** |
| prefix >= 6 + spelling fold | 1/6 | 0.18% |
| prefix >= 5 or ratio >= 0.8 | 5/6 | 6.99% |

Five characters is the knee. Added alongside it: British/American folding
(`-re`->`-er`, `-our`->`-or`, `-ise`->`-ize`) and suffix stripping with
`-e` restoration and consonant de-doubling, which reaches `coding`/Code
and `batter`/Bat.

Two mistakes worth recording, both caught by tests rather than by
reasoning. First, stripped stems were initially compared by *substring*,
which made `amazing`/Amazon illegal -- "amaz" sits inside "amazon"
sharing no root. Stems now participate only in equality and prefix
comparisons; whole surface forms keep the substring rule, where
containment means something. Second, spelling was folded before stripping
but not after, so `centres` never reached `center` and survived the first
fix; the fold now runs on both.

**Result:** 6/60 boards -> 1/60. The survivor is `led`/Lead, an irregular
past tense no dependency-free rule sees, documented as a known gap
alongside mouse/mice. Cost is a mean 2.4% of the clue pool per board
(min 1.6%, max 3.4%). 376 tests pass.

## Sonnet vs. Opus as the evaluation guesser

Asked whether the cheaper model would be much worse. Measured rather than
assumed: 25 real positions spanning openings through late game, each
model handed the exact prompt `LLMGuesser` sends, each ranking scored the
way `play_turn` would (walk it, count own words until the first non-own
word). `scripts/tools/compare_guesser_models.py`.

| model | own/turn | assassin | opponent | neutral | clean |
|---|---|---|---|---|---|
| claude-opus-5, medium | 1.36 | 0 | 2 | 4 | 19 |
| claude-sonnet-5, medium | 1.40 | 0 | 1 | 4 | 20 |

Paired difference **-0.04 own/turn, se 0.091, t = -0.44**, 95% CI
[-0.22, +0.14]. They produced identical guess sequences on 18/25
positions and the same top word on 22/25; of the 5 positions where yield
differed, Sonnet was ahead on 3. Assassin rank was median 12 for both,
reaching the top 3 twice for Opus and once for Sonnet.

No detectable difference, at roughly a fifth the cost (~$3.20 vs ~$16 per
50 games). **The sample cannot rule out a small gap** -- +/-0.22 on a base
of 1.4 is +/-16%, so this excludes "a lot worse", not "slightly worse".

Kept Opus for the headline evaluation anyway. Cost is not the binding
constraint at $16 paid once into a cache, and "evaluated against the
strongest available guesser" is the more defensible claim when the number
is being defended orally. Sonnet is the right choice for development
runs, where the same positions get replayed often.

**Correction: a claimed validation of the Poisson-binomial change was a
bug in the reporting script.** The first version of this entry reported
that only ~32% of revealed own words were the spymaster's intended top-k,
and read that as evidence for scoring N rather than a fixed top-k set.
Wrong. `collect_positions` built `intended` as `own[:number]` in *board
order*, never sorting by similarity -- so for the clue HILARIOUS it named
Ketchup (z -0.1) as intended while Comic sat at z +9.5. Almost every
"surprise" was the script mislabelling the correct answer.

Corrected: the intended hit rate is **76% (Opus) / 81% (Sonnet)**, and
only **2 of 25** positions had any unintended own word taken:

- `EXTREME 3` (seed 5008): took Cold +4.0, Scale +3.7, Center +2.3 where
  Nut +2.6 was intended -- a 0.3 near-tie, yield unchanged at 3/3.
- `TROUSERS 2` (seed 5021): took Pants +10.3 and **Fly +0.9** over Suit
  +5.1. Fly is the zip on a pair of trousers; Numberbatch has the
  category relation to Suit but not the part-of relation to Fly. Yield
  unchanged at 2/2.

The Poisson-binomial fix stands on its own terms -- it is verified
against a 3M-sample Monte Carlo to +/-0.002 (see 783f7da), which is a
mathematical result and does not depend on this measurement. It simply
does not have the empirical support claimed here. If anything the
opposite is worth chasing: 76-81% agreement with our own z-ordering
suggests **sigma = 2.5 is too pessimistic**, and is the first real
evidence available for choosing it.

## Polysemy is where a static embedding can't follow the guesser

Chased the one interesting divergence from the model comparison. On seed
5021, clue `TROUSERS 2`, the guesser took **Fly** (+0.9) over **Suit**
(+5.1) -- the zip on a pair of trousers. All three spaces rank it the
same way, so this is not a Numberbatch quirk:

| space | Pants | Suit | Fly |
|---|---|---|---|
| numberbatch | +10.32 | +5.15 | +0.88 |
| glove | +7.79 | +3.72 | +0.60 |
| wikipedia2vec | +7.50 | +3.44 | +0.62 |

Probing `Fly` against clues aimed at each of its senses shows why (z,
numberbatch / glove / wikipedia2vec): airplane 4.79/3.60/3.51, mosquito
4.56/2.03/2.93, insect 4.46/1.97/2.09, wing 4.29/2.03/2.53, fishing
2.86/2.34/2.44 -- against zipper 1.70/0.21/0.90, pants 1.11/0.77/0.83,
trousers 0.88/0.60/0.62, button -0.22/0.40/0.39.

Five senses (insect, aviation, fly-fishing, a baseball fly ball, the
garment) share one vector, which sits near the centroid of the frequent
ones. The garment sense is the rarest and contributes almost nothing --
even `zipper`, the most direct probe available, reaches only +1.70 in the
best space. The association is not absent, it is swamped. An LLM
disambiguates from context and simply does not have this failure.

Two consequences:

**The cross-space veto will not address this class of error.** It targets
spaces *disagreeing*; here all three agree and are all wrong together,
because they share the one-vector-per-type limitation and similar
training corpora. Still worth building for the assassin problem -- just
not for this.

**It is a structured deviation, not the noise the model assumes.**
`expected_words` posits zhat = z + eps with eps ~ N(0, sigma^2)
independent across words. Polysemy produces word-specific systematic
offsets: `Fly` is reliably underrated under any garment clue, in every
space, every time. This is the weakest assumption in
docs/clue-selection-theory.html and the one most likely to be challenged;
better to have measured it than to be asked about it cold.

It cuts favourably too. When the guesser sees a connection we don't, we
collect a word we didn't plan for -- and since 783f7da the objective
scores *any* own word clearing D, so that upside is counted. Before it,
it was not.

## 2026-09-15 — centroid vs expected_words, 100 games against Claude Sonnet

First head-to-head between two spymasters rather than self-play, and the
first paid run since the training pipeline was removed. 50 boards, each
played twice with the sides swapped (team A holds 9 words and moves
first, team B holds 8 — a one-sided run measures the seating), guesser
`claude-sonnet-5` at effort medium on both sides. 100 games, 220s at 48
workers, ~1,660 new API calls (~$8). Recorded in `cache/llm_store.db`
under `centroid-vs-expected_words+llm-sonnet|A=...`.

**Expected:** `expected_words` to win clearly. It is the model the
project is built around, it has an explicit distractor penalty, and
`centroid` has no assassin-avoidance of any kind.

**Got:** a coin flip on win rate, and a large, real difference in how the
games were lost.

| | win% (95% CI) | assassin% (95% CI) | mean k | own/clue | own% |
|---|---|---|---|---|---|
| `centroid` | 49 [39.4, 58.7] | 13 [7.8, 21.0] | 1.47 | 1.14 | 81.0 |
| `expected_words` | 51 [41.3, 60.6] | 3 [1.0, 8.5] | 1.16 | 1.07 | 93.2 |

The win rate is nothing: 51–49, CIs almost entirely overlapping. The
paired view is blunter still — of 50 boards, only **17 were won by the
same model under both seatings** (9 `expected_words`, 8 `centroid`), and
the other 33 flipped with the seat. Board and seating dominate the
spymaster difference at this sample size.

The assassin rate is real: 13% vs 3%, z = 2.61, **p = 0.009**. And the
mechanism is visible in the transcripts rather than inferred. Of
`centroid`'s 13 assassin losses, **7 were on the guesser's very first
pick** — `wool` → Australia, `queen` → England, `shoe` → Boot,
`smurfs` → Comic, `austria` → Czech, `filed` → Chocolate,
`straight` → Ruler. The clue's single nearest board word *was* the
assassin. That is exactly what a centroid of own-word vectors with no
distractor term should do, and it had never been demonstrated against a
real listener.

**Why the safety doesn't convert into wins.** `expected_words` is
strictly better per guess (93.2% own vs 81.0%) and 4x safer, but slower:
mean announced number 1.16 vs 1.47, words revealed per clue 1.07 vs 1.14.
It buys safety with pace, and over a full game the two cancel almost
exactly. Worth stating plainly because "safer model, same win rate" is a
result about the *reward function*, not about the model — at
`sigma = 2.5` the penalty term is pricing the assassin high enough to
suppress large `k` nearly everywhere.

**Caveat that limits what this shows.** Sonnet at medium effort is the
listener `sigma = 2.5` was itself chosen against
(`scripts/tools/sweep_sigma.py`, see `docs/versions/expected_words.md`).
So `expected_words` is playing to a listener it was calibrated for and
`centroid` is not. The frozen Opus suite remains the untainted
comparison; this run is a diagnostic, not an evaluation result.

**Incidental:** `expected_words` played the clue `rn` (seed 5), which is
not a word. The clue vocabulary is the intersection of three embedding
spaces with a rarity filter, and that still admits tokenization debris.
A part-of-speech / real-word filter on the clue vocabulary is worth
doing before any headline number is quoted.

## 2026-09-15 — why the matchup's mean k (1.16) is far below the sweep's (1.72)

Chasing a discrepancy noticed in the matchup above: `expected_words`
announced a mean number of 1.16 there, against 1.72 reported for
sigma=2.5 in `docs/clue-selection-theory.tex`'s sweep and 1.83 in
783f7da's commit message. Both numbers are correctly computed. They
measure different board distributions, and the difference is a flaw in
how sigma was chosen.

Reproduced the 1.72 exactly by running the current model over
`scripts/tools/sweep_sigma.py::positions()`, so the tex is right about
what it measured. Fresh boards give 1.54, and the clue vocabulary is not
involved: the default 400-word list and the 150-word holdout both give
1.54 on fresh boards.

**The cause.** `positions()` builds mid-game boards by revealing
`(i * 2) % 13` words chosen **uniformly at random** from all 25. But 16
of the 25 cards are distractors, so uniform reveals clear distractors
about 1.8x faster than own words, and the board gets *easier* as it
progresses. Real play does the opposite: the guesser picks an own word
93% of the time, so own words deplete first and the distractor field
stays dense. The board gets *harder*.

Live distractors at the same number of live own words:

| own left | `positions()` | real games |
|---|---|---|
| 9 | 14.5 | 16.0 |
| 8 | 13.1 | 15.1 |
| 7 | 12.1 | 14.4 |
| 6 | 10.4 | 13.2 |
| 5 | 10.2 | 12.3 |
| 4 |  9.8 | 10.9 |

**Decomposition of the 1.72 -> 1.16 gap:**

- **0.41 (72%)** — the positions are easier at a matched own-count, per
  the table above, so the model announces bigger numbers on them.
- **0.15 (28%)** — which positions occur at all. Real games spend 30% of
  turns at <= 3 own words left, where `k <= min(own, 4)` forces the
  number to 1; `positions()` puts 3% of its boards there and its
  `remaining(OWN) >= 2` filter excludes the rest. (That filter alone is
  minor: excluding those turns from the real games moves 1.16 to 1.17.)

Announced k by own words remaining, real games, 592 turns:

| own left | 9 | 8 | 7 | 6 | 5 | 4 | <=3 |
|---|---|---|---|---|---|---|---|
| turns | 50 | 62 | 65 | 77 | 80 | 78 | 180 |
| mean k | 1.54 | 1.56 | 1.22 | 1.12 | 1.06 | 1.05 | 1.00 |

**Why this matters beyond the discrepancy.** sigma=2.5 was selected by
that sweep — it is the only free parameter of the project's only model,
and it was tuned on a distribution of boards that real games never
visit. The sweep's reward-per-turn curve therefore peaks where it does
under systematically easy positions. Whether the peak moves under
realistic positions is unmeasured; the fix is to build sweep positions by
playing real games and snapshotting them, rather than by revealing cards
at random. That re-run costs API money and has not been done.

`positions()`'s own docstring says the intent was "boards spanning the
arc of a game, not just openings," so this is an implementation flaw in
the approximation, not a deliberate choice.

## 2026-09-17 — sigma re-swept on simulated positions: no basis to move it

Acting on the previous entry: `positions()` built sweep boards by
revealing cards at random, which made them easier than real play, so the
sigma that the only model's only free parameter was set to had been
chosen on a distribution real games never visit. Re-ran the sweep with
each sigma scored on positions its *own* play reaches
(`sweep_sigma.py --simulate 25`, Claude Sonnet at medium effort, 100
positions per sigma, 881 calls, ~$4).

**The correction validated first, for free.** Mean announced number at
sigma=2.5: 1.72 under random reveals, **1.20** under simulated positions,
1.22 under snapshots of recorded games — against **1.16** measured in the
100 real games of the centroid matchup. The diagnosis was right and the
new position source reproduces real play. The sweep's own table then
reported 1.16 at sigma=2.5, matching the matchup exactly.

**Expected:** a shifted optimum, since the old boards were systematically
easy. (I guessed it would move up, toward more caution. Wrong on two
counts.)

**Got:** no optimum worth acting on.

| sigma | announced | delivered | reward | s.e. | assassin | clean |
|---|---|---|---|---|---|---|
| 0.25 | 3.67 | 1.39 | 0.232 | 0.281 | 8 | 13 |
| 0.5 | 3.37 | 1.37 | 0.488 | 0.271 | 5 | 20 |
| **1.0** | 2.78 | 1.69 | **1.106** | 0.235 | 3 | 47 |
| 1.5 | 1.95 | 1.34 | 0.852 | 0.198 | 3 | 63 |
| 2.0 | 1.39 | 1.15 | 0.978 | 0.119 | 1 | 83 |
| 2.5 | 1.16 | 1.06 | 0.984 | 0.058 | 0 | 90 |
| 3.0 | 1.10 | 1.03 | 0.984 | 0.049 | 0 | 93 |
| 4.0 | 1.01 | 0.96 | 0.926 | 0.034 | 0 | 95 |
| 6.0 | 1.00 | 0.92 | 0.798 | 0.114 | 1 | 92 |
| 10.0 | 1.00 | 0.71 | 0.400 | 0.163 | 2 | 71 |

sigma=1.0 has the highest mean, and is not significantly better than
anything: vs 1.5 p=0.41, vs 2.0 p=0.63, vs 2.5 p=0.62, vs 3.0 p=0.61, vs
4.0 p=0.45. Its apparent lead is an artifact of its own variance.

**The errors are strongly heteroscedastic, and that is the lesson.**
Reward sd is 2.35 at sigma=1.0 against 0.58 at sigma=2.5, because a
larger announced number means occasional -10 assassin turns. Taking the
argmax of a noisy curve whose arms have 4-5x different variance
systematically favours the high-variance arm -- it has the most chances
to look good by luck. The original sweep reported no standard errors at
all, which is how sigma=2.5 (1.314) came to look like a clean peak over
2.0 (1.172) and 3.0 (1.174) when those three differ by less than one
s.e. The table now prints s.e. per row.

**Separating them is not affordable.** sigma=1.0 vs sigma=2.5 would need
~3,100 positions per arm for 80% power, roughly $253 for those two arms
alone, against $4 for the whole sweep at n=100.

**Decision: keep sigma=2.5.** Not because it won -- nothing won -- but
because it is statistically indistinguishable from the nominal best while
having the tightest error bar (0.058), zero assassin hits in 100
positions, and 90/100 clean finishes. Moving it would be chasing noise.
sigma=3.0 is an equally defensible choice on identical evidence (0.984,
s.e. 0.049, 93 clean); there is no reason to prefer either.

What genuinely changed is confidence in the number rather than the number
itself: sigma=2.5 was previously justified by a peak that the corrected
positions do not reproduce (1.314 there, 0.984 here), and is now
justified by being indistinguishable from every plausible alternative
while carrying the least risk.

**Incidental, on CPU cost.** The sweep needs ~7,000 full-vocabulary clue
searches at ~530ms each -- an hour serially. torch defaults to 8 intra-op
threads and they return almost nothing on this shape of work: 604ms on
one thread against 526ms on eight, 1.15x for 8 cores. One process per
core with `OMP_NUM_THREADS=1` set *before* torch imports (setting
`torch.set_num_threads(1)` afterwards is too late -- the OpenMP pool
already exists and busy-waits) brings it to ~7 min of CPU phase. Kernel
time stayed high (39 min) even after the fix, so something else is still
contending; not chased further.

## 2026-09-17 — sigma=1.5 beats sigma=2.5 by 19 points, and the sweep ranked them backwards

`expected_words[sigma=1.5]` vs `centroid`, 50 boards x both seatings, Claude
Sonnet at medium effort -- the same seeds, opponent and listener as the
sigma=2.5 matchup already in the store, so the two are directly comparable.

| | win% (95% CI) | assassin% | mean k | own/clue | own% |
|---|---|---|---|---|---|
| sigma=2.5 | 51 [41.3, 60.6] | 3 | 1.16 | 1.07 | 93.2 |
| **sigma=1.5** | **70 [60.4, 78.1]** | 6 | 1.93 | **1.33** | 78.9 |

+19pp, z = 2.75, **p = 0.006**. Paired by board it is the same story: sigma=1.5
won more games on 21 boards, fewer on 6, tied on 23 (sign test p = 0.006).

The mechanism is pace. sigma=1.5 is less accurate per guess (78.9% own against
93.2%) and hits the assassin twice as often (6% against 3%), but delivers 1.33
own words per clue against 1.07, and over a full game that compounds into a
decisive lead. sigma=2.5's caution was buying accuracy that did not pay.

**The part that matters beyond the parameter: the per-turn reward proxy ranked
these backwards.** The simulated sweep scored sigma=1.5 at 0.852 -- the *worst*
of the 1.0-3.0 range -- against sigma=2.5's 0.984. In real games sigma=1.5 wins
19 points more. That proxy is the only basis on which sigma has ever been
chosen, and it is now known to be anti-correlated with game outcomes over at
least part of the range. The likely cause: per-turn reward charges -10 for an
assassin and +1 per word, which overweights rare catastrophes against the
compounding value of pace, and it scores isolated turns that are never played
forward, so it cannot see a game lost on the clock.

Not yet acted on. Two things first: this is one weak opponent (faster play beats
a weak opponent more reliably than a strong one), and sigma=1.0 is untested but
announces bigger still (2.19 pooled) -- it may be better again or past the peak.

## 2026-09-17 — cache-blocking the clue search: tried, measured, reverted

`gain_and_penalty` scores all ~11k candidates at once, allocating ~68MB for one
intermediate. Candidates never interact, so blocking into cache-sized slices is
an exact refactor. An isolated benchmark showed 3.55x at block=128, bit-identical
output (max|diff| = 0).

**The 3.55x was a benchmarking error.** That test ran one process against
worst-case *fresh-board* array dimensions while 16 arc workers were saturating
memory bandwidth, so it measured a starved process rather than the workload.
A/B on the real thing -- the arc tool, 16 workers, 8 games per sigma each way:

    unblocked     94.69s wall, 1264.64 user, 33.79 sys
    blocked(128)  94.41s wall, 1272.04 user, 24.21 sys

No difference, and none on an idle single process either (507ms unblocked vs
525ms blocked). Real arc turns are mostly mid-game, where n_own and n_non_own
have shrunk and the arrays are already cache-resident; the optimization solved a
problem this workload does not have. Reverted rather than kept as harmless
complexity in the model's hot path.

**Where the 20 minutes actually goes**, since that was the original question:
85% core occupancy, so the pool is not the limit. Each clue search costs 0.60s
of CPU alone but 1.71s when 16 run at once -- memory-bandwidth contention -- so
effective speedup is 4.6x on 16 cores, not 16x. ~10,000 clue searches at that
rate is the runtime. Making it faster needs a cheaper search, not more workers,
and candidate blocking is not that lever.

## 2026-09-17 — gpt-oss-120B as a cheap listener: two silent bugs before any data

Goal: a listener cheap enough to afford a sigma sweep with enough games to
settle the sigma=1.5 vs 2.5 question that the per-turn proxy got backwards.
Picked `openai/gpt-oss-120b` on DeepInfra, wired through a new
`OpenAICompatGuesser`, and sampled 5 real recorded positions beside Sonnet
before spending anything at scale.

**Expected:** a qualitative read on how the open-weight model interprets clues.
**Got:** rankings that were identical to the board's own order in all five
positions, for *both* models on some of them. Two independent bugs.

**Bug 1 — the guesser fabricated rankings.** gpt-oss-120b is a reasoning model:
chain of thought goes in `reasoning_content`, the answer in `content`. Left
unset, `reasoning_effort` defaults to medium; with `max_tokens=512` it spent all
512 on reasoning and returned `finish_reason="length"` with `content=""`.
`LLMGuesser._parse_ranking` is deliberately tolerant -- it backfills anything the
model omitted from board order -- so an empty response became a complete,
well-formed, entirely fabricated ranking, cached to `cache/llm_store.db` as if
paid for. Nothing in the output said so.

Board order is role-shuffled (`codenames/board.py`), so in play this degrades to
a *random* guesser, not a biased one. That is the dangerous part: a sweep run
this way would have reported "the open-weight model is a weak listener" and
looked entirely normal. It would not have crashed.

Fixed by making `OpenAICompatGuesser` strict where the Anthropic guesser is
tolerant: reject `finish_reason="length"`, empty content, or a response naming
fewer than `min_coverage` (0.8) of the candidates; retry once at double budget,
then raise. Deliberately *not* done by changing `_parse_ranking`, which the
Anthropic path and 5,742 cached rows depend on and which is correct for its own
case. Defaults moved to `reasoning_effort="low"`, `max_tokens=4096`. Eight
regression tests added; the 5 poisoned rows were deleted from the store.

**Bug 2 — the sampler leaked the answer.** `sample_guesser.py` built its
candidate list by iterating the recorded board, which is stored grouped by role,
so the prompt listed every own word, then every opponent word, then neutrals.
That both hands the model block structure to exploit and makes a board-order
fallback score a perfect 4/4 for team A and 0/4 for team B -- which is exactly
the pattern the first run showed. Real play passes `board.words` order. The
original order is not recoverable from the record, so the fix is a per-position
shuffle seeded by (seed, clue).

**After both fixes, on the same 5 positions:** gpt-oss-120b took 11/16 intended
words, Sonnet 5 took 11/16. Identical top-3 on `munich` and `infrared`; gpt-oss
beat Sonnet 4/4 vs 3/4 on `accidentally` and ranked the assassin lower (safer) on
`mice` (8 vs 5). n=5 is anecdotal and is not evidence of parity -- it is only
evidence that the cheap model reads clues in the same kind of way.

**Cost, measured head to head on one 25-word position rather than from token
prices:** Sonnet 5 at effort=medium $0.00477/call (250 in, 427 out);
gpt-oss-120b at effort=low $0.000102 (555 completion tokens) -- **47x cheaper**;
at effort=medium $0.000775 (4509 completion tokens), only 6x cheaper. Per token
the open-weight model is ~55x cheaper, so most of the advantage is spent on
reasoning tokens. The effort setting, not the sticker price, is what decides
whether this is worth doing -- and an earlier note in this repo claiming
"~1/150th of Sonnet" was a per-token figure that ignored that entirely.

**Standing caution.** The tolerant parser is the right default for a guesser
that either answers or errors. It is the wrong default for anything that can
return a well-formed empty response. Any future listener on a reasoning model
needs the strict path, and any listener at all should be sampled on a handful of
real positions before a paid run -- this cost $0.0005 to find and would have
cost the whole sweep to miss.

## 2026-09-17 — sigma measured from cached rankings: ~2.0, and it is not constant

`expected_words`'s `sigma` had never been measured. It was picked from the
announced-number distribution on fresh boards, and the per-turn proxy that
later re-picked it turned out to rank values backwards against real outcomes.
Meanwhile DeepInfra caps us at ~0.6 calls/s (measured: 429 "Model busy" at
32-way concurrency), which puts a 4-arm game sweep at ~11 hours.

Shane's idea, and it is the better experiment: the model says a listener
perceives `z_w + eps_w` and picks the max, and cache/llm_store.db already
holds 5,747 rankings bought by earlier runs. That is a direct observation of
the thing sigma parameterises. No new API calls at all.

**Estimator.** Thurstone Case V with *known* utilities -- the z-scores come
from the tensor, so sigma is the only unknown. Scored on the listener's top
pick, where conditioning on the winner's own noise makes the field
independent and the probability is an exact 1-D Gaussian integral.

Two cheaper extensions were implemented, measured against planted sigmas, and
discarded. The "probability this is the max of what remains, multiplied along
the ranking" factorisation is exact only under Gumbel noise (Luce's axiom),
not Gaussian; the pairwise composite conditions each pair on the winner
having already won, which is a selection effect:

    true sigma   exact top-1   sequential   pairwise(top3)
          0.50         0.493       --             0.415
          1.00         1.003       --             0.783
          2.00         2.004      ~2.9            1.547
          4.00         4.121       --             2.927

Only the exact top-1 likelihood survives. `tests/test_listener_fit.py` plants
sigmas and checks recovery before any number off real data is believed.

**Result** (space=numberbatch, turn rankings only):

    listener                          n     sigma      95% LI   obs top1  sim top1
    claude-sonnet-5+effort=medium  3403     2.056  [2.00,2.11]     0.764     0.746
    claude-sonnet-5                 770     2.009  [1.90,2.13]     0.683     0.671
    deepinfra/gpt-oss-120b+low      352     2.195  [2.04,2.37]     0.798     0.770

**The calibration check passes**: simulated top-1 rates land within ~2 points
of observed and mean z-ranks within ~0.2, so sigma is measuring noise rather
than absorbing gross misspecification. Effort barely moves Sonnet (2.06 vs
2.01, overlapping intervals).

**gpt-oss-120b's interval overlaps Sonnet's.** That is much stronger support
for the cheap listener than the n=5 qualitative sample, and it cost nothing.

**But sigma is not constant**, which the single-parameter model assumes:

    by announced k:   k=1 1.97   k=2 2.00   k=3 2.17   k=4 2.63
    by board size:  9-13 1.84  14-18 2.04  19-25 2.18

Board size is already inside the model (more candidates, more chances a
distractor wins on noise), so sigma rising with n means real errors grow
faster than Gaussian noise predicts -- heavier tails. Two confounds keep this
from being a clean decomposition: k was chosen by the spymaster, so k=4 clues
are a selected subset where it believed four words were reachable and part of
that 2.63 is its own optimism regressing to the mean; and k and board size
are themselves correlated, since big boards are early game.

**What this does and does not settle.** The shipped sigma is 2.5; the
descriptive fit is ~2.06. But real games said sigma=1.5 beats sigma=2.5
70-30 against centroid. Those are not in conflict -- they answer different
questions. The listener genuinely behaves like sigma~2.0, and a spymaster
apparently still plays better believing sigma=1.5, i.e. being optimistic,
announcing bigger, and moving faster. In a race, tempo is worth more than the
per-turn expected-value objective (with its -10 assassin charge) credits.
Descriptive sigma and playing sigma are separate quantities and must be
reported as such; this measures only the first.

**Open, and now concrete:** sigma(k) rising with k means one global sigma is a
compromise that is over-optimistic about large-k clues. A `sigma = a + b*k`
variant is a real, data-grounded candidate for the next model -- but the
selection confound above has to be handled first, since fitting sigma(k) on
clues whose k the spymaster chose would bake its own optimism into the fit.

## 2026-09-17 — sigma=2.0 played for real: the measured sigma is not the best sigma

The listener fit put Sonnet at sigma=2.06 [2.00, 2.11]. Shipped is 2.5, and
the only head-to-head said 1.5 beat 2.5. So the obvious question was what the
*measured* value actually does in games. sigma=2.0 had never been played --
only simulated in the clue-number arc.

Ran it on the same footing as the others: 50 boards (seeds 0-49), sides
swapped, vs centroid, Sonnet at effort=medium. 1,014 newly billed calls out
of 1,120 clues (~10% cache hits on shared opening positions), ~$4.84.

    sigma   mean k (sonnet)   mean k (noisy_glove)   win% vs centroid
      1.5              1.93                   1.90              70.0%
      2.0              1.42                   1.43              57.0%
      2.5              1.16                   1.21              51.0%

Paired on matched (board seed, side), sign test on discordant pairs:

    1.5 vs 2.0   25-12 discordant   p=0.047
    1.5 vs 2.5   30-11 discordant   p=0.0043
    2.0 vs 2.5   17-11 discordant   p=0.345

**sigma=1.5 beats both; 2.0 and 2.5 are indistinguishable.** Three tests, so
under Bonferroni (0.0167) only 1.5-vs-2.5 survives -- 1.5-vs-2.0 is
suggestive, not established. Still one weak opponent, and the pairing is on
board and side only: the games diverge after the first move, so this is
matched, not controlled.

**The headline: descriptive sigma and playing sigma are genuinely different
numbers, now measured separately.** The listener really does behave like
sigma~2.06, and a spymaster that believes it plays no better than one that
believes 2.5, while one that believes 1.5 -- announcing 1.93 words a clue
instead of 1.42 -- wins more. Modelling the listener accurately is not the
objective; winning the race is, and tempo is worth more than the per-turn
expected-value objective (with its -10 assassin charge) credits.

**Useful side finding: mean clue number barely depends on the listener.**
1.42 vs 1.43 at sigma=2.0, 1.93 vs 1.90 at sigma=1.5, 1.16 vs 1.21 at 2.5 --
Sonnet against free noisy_glove. Clue choice never consults the guesser; only
the trajectory does, and the trajectories are close enough that the mean
survives. Mean k can therefore be swept for free from here on, and only win
rate needs paid games.

Not changing the shipped sigma on this. 1.5 is ahead on one weak opponent,
and sigma=1.0 remains untested and announces bigger still -- it may be better
again or past the peak. That, plus a second opponent, is what a shipping
decision needs.

## 2026-09-17 — the sigma curve has a plateau, and the failure mode moves along it

Filled in sigma=1.0 and 1.25 against centroid, Sonnet effort=medium, same 50
seeds and side-swap as every other arm. 2,043 billed calls, ~$9.75.

    sigma  win%  mean k | win/words  win/gift | loss/words  loss/assassin
      1.0   69%    2.67 |        56        13 |         17             14
     1.25   70%    2.27 |        59        11 |         22              8
      1.5   70%    1.93 |        60        10 |         24              6
      2.0   57%    1.42 |        47        10 |         41              2
      2.5   51%    1.16 |        38        13 |         46              3

**Win rate plateaus across sigma 1.0-1.5** at 69-70%, then falls off a cliff:
57% at 2.0, 51% at 2.5. Paired sign tests on matched (seed, side) put the
three plateau arms at p=1.0 against each other -- genuinely indistinguishable,
not merely close.

**But the failure mode moves along the plateau.** From 1.5 to 1.0,
losses-on-words fall 24 -> 17, so the extra tempo really does win more races;
assassin deaths climb 6 -> 14 and eat the entire gain. sigma=1.0 converts race
losses into catastrophic losses at close to par. The assassin difference is
established at 1.0 (14/100 vs 3/100 at sigma=2.5, Fisher p=0.009) where it was
not at 1.5 (6/100, p=0.498).

So within the plateau **sigma=1.25-1.5 dominates sigma=1.0**: the same win
rate for roughly half the assassin rate. That is a risk-adjusted preference of
exactly the kind docs/design-decisions.md argues for reporting separately
rather than collapsing into one number.

**The control worked.** Wins handed over by centroid's own assassin are flat
across every arm (13, 11, 10, 10, 13), which they must be -- our sigma cannot
change how centroid plays. So none of the spread is opponent-blunder luck; it
is all in the on-words columns.

**Multiplicity.** Ten pairwise tests. Under Bonferroni (alpha=0.005) only
1.5-vs-2.5 (p=0.0043) survives; 1.0-vs-2.5 (0.0096) and 1.25-vs-2.5 (0.0079)
do not. The weight of evidence is that all three plateau arms beat both
high-sigma arms consistently, not any single p-value.

**Where this leaves the shipping decision.** Shipped is 2.5, which is now
clearly the worst setting tested. The descriptive fit (sigma~2.06) is also in
the bad region -- more evidence that modelling the listener accurately is not
the objective. The plateau means the choice inside 1.25-1.5 cannot be made on
win rate against this opponent and should be made on risk, or on a second
opponent that is not centroid. Still not changing the config on one weak
opponent; that needs a docs/versions/ entry and a stronger test.

## 2026-09-17 — the model re-gives clues that just failed (found by random sampling)

Shane asked for a random sample of sigma=1.5 games rather than selected ones.
The curated picks had shown one game repeating 'forbes'; the random draw of 6
had a repeat in 2 of them, which prompted counting it properly.

**Clue repetition is systematic and scales with aggression:**

    arm          games with a repeat   repeated clues / all clues
    sigma=1.0              38%                14.2%
    sigma=1.25             32%                 9.0%
    sigma=1.5              30%                 8.6%
    sigma=2.0              25%                 6.6%
    sigma=2.5              19%                 4.1%
    centroid                 -                 3.2%

**Every repeat follows a miss.** Of 43 repeats at sigma=1.5: 24 came after a
neutral, 19 after an opponent word, and *zero* after a clean
`exhausted_guesses` turn. So the model never re-uses a clue that worked -- it
re-uses only clues that have just been shown not to work.

The mechanism is plain once seen: the failed guess removes a distractor from
the board, which *raises* that clue's score, so it returns as the argmax. The
guesser's demonstrated misreading is nowhere in the model's state, because
`expected_words` scores each turn from the board alone. One game gave 'forbes'
n=3 (took a neutral), then 'forbes' n=4 -- escalating the number after a
failure -- and hit the assassin.

The repeats do badly: 15/43 got no own word at all, 24/43 ended in another
miss, 1 in the assassin. 28/43 got at least one own word, so it is not pure
waste -- but re-running a known-failed experiment is not why.

This is exactly the gap docs/design-decisions.md names as the open direction:
"a clue only has to be distinguishable from the clues already given." It also
explains part of why low sigma is risky: bigger k means more partial failures,
which means more distractor removals, which feeds the repetition.

**Why this is a good next model.** The fix needs no lookahead and no LLM:
exclude (or penalise) clues already given in this game. `TurnContext` already
carries `turn_index`, so the plumbing for game state exists. Clue choice and
mean k are free to measure (established earlier today: mean k is within 0.01-0.05
between Sonnet and a free listener), so the whole change can be developed and
screened at zero cost, with paid games only to confirm the win rate.

## 2026-09-17 — distilled listener, tier 1: it works, it barely helps, and it found two bugs

Shane's idea: train a local model on a cheap LLM's rankings so sweeps cost
nothing. Built the pipeline and trained tier 1 on the 4,664 cached Sonnet
positions we already own.

**Formulation.** Conditional logit (McFadden). The model emits one scalar
utility per candidate word; a softmax over the board turns those into a
choice; the response is *which word the teacher picked* -- categorical over a
choice set that changes size every turn, so there is no regression target and
no fixed-width output. Each position contributes `k` choice events
(Plackett-Luce to depth k), which is exactly as deep as the game ever reads a
ranking: measured, 58.5% of turns read one word, 100% read at most four.

Features are computed ONCE on the full board and reused across the PL steps,
because `rank_candidates` is called once per turn and the game reads the top k
off a single ordering -- so training and inference score identically.

Note this uses Plackett-Luce as the *model*, after rejecting the same
factorisation as a biased *approximation* when estimating sigma earlier today.
Different roles: there it had to recover a Gaussian parameter, here `f` is an
arbitrary learned function with no Gaussian claim.

**Result: +0.0108 pooled, +0.0077 on step-1, over ranking by numberbatch z.**
Real but small, with early stopping and a board-seed split.

**Bug 1, mine, in the feature design.** Within a board,
`rank_numberbatch`, `gaptop_numberbatch` and all three `p_max_sigma` features
are monotone transforms of `z_numberbatch` -- measured within-group rank
correlation exactly 1.000. Four more (`k`, `n_candidates`, `peak_z`,
`lead_margin`) are constant within a board. So **nine of 21 features cannot
reorder anything**; they can only gate interactions. The `p_max_sigma`
features I argued hardest for -- the Gaussian model's own prediction -- encode
board context, and board context cancels in a within-board softmax. Only
`word_mean_sim` (rho 0.23), `word_sd_sim` (0.25), `rival_min_space` (0.41) and
the two weaker spaces (~0.51) carry independent ordering information.

**Bug 2, an evaluation bug that made a null model look excellent.** The target
sits at index 0 of every group (features are built in the teacher's ranked
order) and `np.argmax` returns the FIRST maximum -- so a constant-scoring
model grades 100% correct by tie-breaking. It surfaced as early stopping
choosing iteration 1 for every configuration, and as *more* regularisation
scoring *better* (8 leaves 0.675 > 15 leaves 0.649 > 31 leaves 0.633), which
is the giveaway: heavier regularisation means more tied leaves means more free
credit. Fixed by scoring ties as 1/(number tied), the expectation under random
tie-breaking. `tests/test_listener_features.py` asserts a constant model
scores chance.

**Also measured.** Without early stopping the model reaches 0.9951 training
accuracy while validation *falls below* baseline -- capacity was never the
problem. And the custom objective was a Python loop over ~6k groups per
boosting round; vectorising with `reduceat` gave a bit-identical 30x speedup
(21.70 -> 0.73 ms/round), which is the speedup that was actually available
here. GPU would not help: 97k rows by 21 features is far too small, and a
Python objective forces a host round-trip every round regardless.

**What this says about the plan.** Tier 1 was largely the baseline wearing 21
hats. The features carrying genuinely new within-board information are the
ones deferred to tier 2 -- cohesion (thematic grouping, for the k>=3 regime
where embeddings collapse to 0.36 top-1) and entity similarity (the 867k
discarded Wikipedia2Vec ENTITY vectors). Those are the next test, not more
data.

**Training distribution is a known gap.** Every cached position came from a
game where a spymaster chose a clue it believed was good, so the model has
never seen a clue that is irrelevant to the board, or one that points at the
opponent's words or the assassin. A spymaster's search scores ~111k candidate
clues per turn and most are bad, so collection must deliberately sample junk
and adversarial clues, not just clues a spymaster liked.

## 2026-09-17 — 8.6k adversarial positions bought; two feature hypotheses, both null

Collected 8,648 generated positions from gpt-oss-120b (2.6 h, $0.88, 4.2%
rejected by the strict guard). Deliberately adversarial: clues drawn 40/20/10/
10/20 across own / opponent / assassin / neutral / uniform-random, on boards
revealed to a random depth, from the 250-word training list at seeds >= 1e6.
Rejection was near-uniform by kind (93.6% retained on junk clues, 96-97%
elsewhere), so the adversarial half survived intact.

**The data did what it was bought to do.** Baseline agreement (rank by
numberbatch z) fell from 0.6030 on game-derived Sonnet positions to 0.4276 on
these -- numberbatch is a much worse predictor of a listener when the clue
points at the opponent or at nothing, which is exactly the regime a
spymaster's search must get right and which no game-derived data contains.

**But the model's lift barely moved.** +0.0139 pooled on 15,976 training
choice events against +0.0108 on 5,951. 2.7x the data, +0.003. That is a
feature ceiling, not a data ceiling, and it means another 10k positions would
have been wasted money.

**Cohesion (tier 2) is null.** "Is w part of the cluster the clue points at",
mean z of w against the clue's top-5 candidates, plus its rank and its margin
over w's own clue similarity:

    without cohesion   all 0.4416 (+0.0139)   step-1 0.6392 (+0.0060)
    with cohesion      all 0.4414 (+0.0137)   step-1 0.6396 (+0.0065)

`cohesion_minus_own` came 4th of 24 by gain importance -- the trees used it
heavily -- and held-out accuracy did not move. Gain importance measures
training-loss reduction, not generalisation, and this is a clean example of
the difference.

**Where that leaves the distilled listener.** It is barely beating "rank by
numberbatch z", which is a guesser the project already has for free. And the
sigma fit says a listener *is* numberbatch plus N(0, 2.06) (2.20 for gpt-oss),
with calibration passing. If that model is right, the calibrated distillation
is `noisy_numberbatch` at the fitted sigma -- no training, no features, no
data purchase -- and the GBT's only job is to beat it. That comparison is free
and is the next thing to run, ahead of extracting entity vectors.

Also note top-1 agreement may be the wrong yardstick: a guesser only has to
make the same GAME decisions, and the real test is whether it reproduces the
sigma ladder (69/70/70/57/51) better than noisy_glove's 54/41/26.

## 2026-09-17 — the teacher is 84-99% self-consistent: the gap is missing knowledge, not noise

Shane's read, against mine. I had argued the distillation plateau might be
irreducible: vague clues have no right answer, so 0.35 could already be the
ceiling. Measured it instead -- 750 collected positions re-asked with the
response cache bypassed, 150 per clue kind, $0.08 and 12 minutes.

    kind        n    teacher self-agreement   our model   headroom
    own       149      0.987 [0.97, 1.00]        0.729      26 pts
    assassin  145      0.952 [0.92, 0.99]        0.721      23 pts
    opponent  150      0.947 [0.91, 0.98]        0.726      22 pts
    neutral   149      0.933 [0.89, 0.97]        0.740      19 pts
    random    144      0.840 [0.78, 0.90]        0.352      49 pts

**The noise hypothesis is dead.** gpt-oss reproduces its own top pick on a
*meaningless* clue 84% of the time, and on targeted clues 93-99%. Weighting by
dataset composition the ceiling is ~0.94 against our ~0.64: about 30 points of
reproducible structure the features do not capture, worst by far on junk clues.

This is consistent with the earlier free estimate -- cross-model agreement
(0.756 on vague positions) lower-bounds self-consistency, and self-consistency
indeed came in above it.

**What it does and does not establish.** Self-consistency measures determinism,
not learnability: a teacher can be perfectly reproducible while relying on
knowledge no embedding holds. So this rules out "the residual is noise" and
makes "the residual is knowledge we lack" the live hypothesis -- which is what
justifies spending on new sources rather than more data or more model.

**Also measured: both teachers have a prompt primacy bias.** Mean normalised
position of the chosen word in the prompt list (0.5 = unbiased):

    gpt-oss-120b     0.4609 [0.454, 0.468]
    claude-sonnet-5  0.4772 [0.468, 0.486]

Both exclude 0.5. Part of what our features cannot explain is not semantic at
all -- it is where a word sits in the prompt. No embedding space will supply
it. Candidate order is board order, which is role-shuffled, so this adds noise
rather than bias to game outcomes, but it is a real property of the measuring
instrument and it is a free feature if we want fidelity over cleanliness.

**Also: adversarial clues are not the hard case.** Clues aimed at the
opponent (0.715 baseline) or the assassin (0.713) are predicted as well as
clues aimed at one's own words (0.727). The hard case is the absence of any
anchor: junk clues sit at 0.320. That was not what the adversarial collection
was expected to show.

**Next, in order.** SWOW free-association norms first: "what word comes to
mind given this cue" is precisely the junk-clue regime, where there is no
strong semantic anchor but the teacher is still 84% determined. Then the
Wikipedia2Vec ENTITY vectors already on disk, then ConceptNet graph edges.
Coverage of our 11,145-word clue pool has to be checked before any of them,
since a source that covers a tenth of the pool cannot move a pooled metric.

## 2026-09-17 — human association data closes half the gap: +23.7 points

Shane's call, over my recommendation to drop the distillation. He was right.

**The source.** SWOW-EN18 (De Deyne et al. 2019): 1.39M cue->response pairs
over 12,217 cues from ~90k participants -- what word people actually say when
cued with another word. That is the listener's task measured on humans, and it
is a different *kind* of evidence from an embedding. Embeddings measure
similarity (words used in similar contexts); association measures relatedness
(words that come to mind together). `nikon` and `Olympus` are not similar,
they are associated. Three embedding spaces cannot encode that, which is why
every feature block derived from them was redundant or null.

Checked the cheaper USF norms first and they are too thin: 32.4% clue-pool
coverage, 0.13 board words per 25-word board at one hop. SWOW: 59.2% coverage,
100% of board words reachable, 0.57 at one hop and **15.6 of 25 at two hops**.
Two hops is what makes it usable -- the density lives in intermediate words
that are on no board. Paths combine by sum of products, so several weak routes
accumulate rather than all but one being discarded (one sparse matmul,
scripts/data/build_swow_tables.py, 2.87M two-hop entries, 13.7 MB).

**Result, same 8,609 positions and same board-seed split:**

    baseline (numberbatch z)     pooled 0.4276   step-1 0.6332
    before SWOW                  pooled 0.4416   step-1 0.6392   (+0.014 / +0.006)
    with SWOW                    pooled 0.6642   step-1 0.7930   (+0.237 / +0.160)

`swow2_rank` is the top feature by gain, 4x the next. Against the measured
self-consistency ceiling (~0.94), the model has gone from 0.64 to 0.79 -- about
half the available headroom, from one feature block.

**Missing is NaN, never 0.** 41% of the clue pool never appears as a SWOW cue.
Zero would assert "these words are unrelated", which is a strong and wrong
claim; NaN says "no evidence" and LightGBM learns a split direction for it.
Conflating the two is the standard way a sparse source poisons a dense feature
set.

**What this says about the earlier null results.** Tier 1 and cohesion were
not failures of method -- they were all functions of the same three embedding
spaces, so they could only recombine information already present. The lesson
is that feature engineering within one information source has a low ceiling,
and the diagnosis that matters is which *kind* of evidence is missing.

Data is CC BY-NC-ND 3.0, lives under gitignored data/, and only the derived
table is cached. Cite De Deyne, Navarro, Perfors, Brysbaert & Storms (2019),
Behavior Research Methods.

**By clue kind, with SWOW** (same positions, board-seed split; ceiling is the
measured teacher self-consistency):

    kind         n    baseline   model    lift    ceiling   left
    random    1861      0.320    0.667   +0.347    0.840    0.173
    own       3390      0.727    0.863   +0.136    0.987    0.124
    opponent  1605      0.715    0.842   +0.128    0.947    0.105
    neutral    894      0.737    0.843   +0.106    0.933    0.090
    assassin   537      0.713    0.832   +0.119    0.952    0.120

The junk-clue regime was the whole story, exactly as the earlier diagnosis
said: 0.320 -> 0.667 where the 49-point gap lived, with every other category
gaining a uniform +11 to +14. Weighted remaining headroom is ~12.7 points.

Worth noting for the defense: the association data helps most precisely where
embeddings have least to say. A clue with no strong semantic anchor still has
strong human associations, and that is what a listener follows.

## 2026-09-17 — CORRECTION: the SWOW and entity results above were a leak

**The two entries above (commits 942d6ec, 308bf00) report numbers that are
wrong.** They are left in place because this log records what was believed and
when, but nothing in them should be cited.

**The bug.** `_ranks(np.nan_to_num(x, nan=-inf))` on a column that is entirely
NaN returns `[0, 1/(n-1), 2/(n-1), ...]` -- the candidate's position in the
list -- because `argsort` on equal values returns index order. Features were
extracted with candidates in the TEACHER'S RANKED ORDER and the target at
index 0 of every group, so that column *was* the answer. It fired on the 41%
of clues SWOW does not cover and the 71% entity does not, which is most of the
data.

**What the numbers actually are**, with candidates shuffled and `_ranks` made
NaN- and tie-safe:

    baseline (numberbatch z alone)          pooled 0.4276   step-1 0.6332
    embedding-derived features only (~24)   pooled 0.4414   (+0.014)
    all 32, with SWOW and entity            pooled 0.4589   (+0.031)

So SWOW plus entity contribute about **+1.7 points**, not the +23.7 claimed.
Leak-free ablation:

    all 32                 0.4589      pruned 20 (with entity)  0.4545
    no entity (29)         0.4555      pruned 17 (no entity)    0.4506
    SWOW + entity ALONE    0.2944   <- far WORSE than numberbatch alone

That last row is the clearest refutation: under the leak "SWOW only" scored
0.607. These sources are a weak supplement, not a replacement.

**Root cause, and it is the second time.** The target sat at a fixed index, so
any positional artifact became the answer. The earlier tie-breaking bug in the
accuracy metric had the same cause -- a constant model scored 100%. Patching
`_ranks` fixes this instance; SHUFFLING the candidate order before extracting
features kills the class, and that is what is now done. `targets` tracks where
each of the teacher's picks landed.

**What survives from the entries above**, because none of it used features:
teacher self-consistency (84-99%), the sigma fit and ladder, the prompt
primacy bias, and every coverage measurement (SWOW 59.2% of clues, 15.6 of 25
board words at two hops; USF too thin at 32.4%; entity 29.2% and 18% of pairs).

**What does not survive**: that association data closes half the distillation
gap, the by-kind table (junk clues 0.320 -> 0.667), and the Sonnet replication
at +0.138. All three need re-measuring leak-free.

**Lesson worth keeping.** Both bugs were caught by distrusting an
implausibly good number -- but only after it had been reported and committed.
A feature covering 18% of pairs cannot be worth +19 points, and one block
should not take a model from +1.4 to +23.7. That reflex needs to fire before
the write-up, not after.

## 2026-09-17 — which accuracy number is representative: neither of the two I was quoting

Shane pushed back that step-1 accuracy under-represents play, since clues with
k>1 make the guesser read deeper. He is right, and checking the weights showed
our pooled figure is wrong in the other direction.

    step   baseline   model   our weight   game weight
      1      0.6332  0.6614      40.6%        65.1%
      2      0.3371  0.3777      29.7%        27.0%
      3      0.2434  0.2639      19.3%         6.6%
      4      0.2240  0.2605      10.3%         1.2%

Game weights come from recorded turns: 58.5% read one word, 31.4% two, 8.2%
three, 1.9% four, so across 1000 turns there are 1000 step-1 events, 415
step-2, 101 step-3, 19 step-4.

Our pooled metric weights steps by the COLLECTED data's clue-number mix, and
`collect_listener_data.py` samples k uniformly from 1-4. That puts 29.6% of the
weight on steps 3-4 where play reads them 7.8% of the time -- so pooled is
pessimistic, and step-1 is optimistic.

    pooled (our k mix)   baseline 0.4276   model 0.4589
    step-1 only          baseline 0.6332   model 0.6614
    GAME-weighted        baseline 0.5517   model 0.5824   <- report this one

**Accuracy collapses with depth: 0.66, 0.38, 0.26, 0.26.** Predicting the
teacher's second choice is nearly twice as hard as its first. Every feature is
a variant of "how close is this word to the clue", which explains a top pick
and not a runner-up.

The lift is stable across all three framings (+0.031, +0.028, +0.031), so the
choice of metric moves the headline by 12 points but not the verdict on
whether a feature block helps.

**And all three are still proxies.** In a real turn the guesser stops at the
first non-own word, so step 2 only happens if step 1 was right: errors truncate
rather than average. A model at 0.66 then 0.38 does not deliver their mean, it
delivers a turn that usually ends after one word. Only playing games measures
that, which is the argument for the ladder replay that has still never been run.

## 2026-09-18 — tripled the data again: pooled accuracy identical to 4 decimals

Bought 16,951 more gpt-oss rankings (4.4 h, $1.73, 4.1% rejected), taking the
store to 25,714 and the training set to 19,032 positions / 47,389 choice
events -- 2.95x the previous run, with SWOW and entity features in place.

    positions   choice events   pooled   step-1
        6,450          15,976   0.4589   0.6614
       19,032          47,389   0.4589   0.6554

**Pooled accuracy is identical to four decimal places, and step-1 is slightly
worse.** Feature-to-data ratio is now 595 positions per feature, which is
comfortable, and it changed nothing.

The hypothesis being tested was that partial-coverage sources need more
examples to exploit -- SWOW covers 59% of clues, entity 29%, so a feature
present a third of the time gets a third of the effective sample. That was
wrong; tripling the sample did not help them.

**Three data scales, three feature blocks, same plateau.** This is a feature
ceiling and no longer an open question. The remaining budget should not go on
collection.

Where that leaves the distilled listener: 0.4589 pooled, 0.6554 step-1, ~0.58
game-weighted, against a measured teacher self-consistency ceiling of ~0.94.
It recovers roughly a third of the way from chance to what the teacher can
reproduce of itself, and no amount of the evidence available to it closes the
rest.

## Per-block ablation: what is numberbatch alone worth?

The `baseline` line the training script has always printed is the *raw*
`z_numberbatch` argmax, which conflates two different questions: how much does
a learned model add over a linear rule, and how much do the non-numberbatch
sources add over numberbatch. Added `--blocks` to
`scripts/pipeline/train_listener.py` so the four blocks can be selected
independently; columns are dropped after extraction, so every run below sees
identical rows, groups, split and seed, and the only thing that changes is what
the trees may look at. gpt-oss teacher, 47,171 train / 15,766 val choice events.

| model                     | features | pooled | step-1 |
|---------------------------|---------:|-------:|-------:|
| raw numberbatch z argmax  |        1 | 0.4229 | 0.6364 |
| GBT, numberbatch only     |       15 | 0.4387 | 0.6419 |
| + glove, wiki2vec         |       24 | 0.4446 | 0.6462 |
| + SWOW, entity            |       23 | 0.4526 | 0.6604 |
| all blocks                |       32 | 0.4611 | 0.6618 |

Paired bootstrap over the 15,766 validation events (2,000 draws), since every
model is scored on identical boards and an unpaired SE overstates the spread:

    nb          -> nb+spaces    +0.0058  [+0.0019, +0.0099]
    nb          -> nb+swow+ent  +0.0139  [+0.0098, +0.0180]
    nb+spaces   -> all          +0.0165  [+0.0120, +0.0210]
    nb+swow+ent -> all          +0.0084  [+0.0044, +0.0126]

All four intervals exclude zero, so every block is carrying something. The
ordering is the useful part: the association/entity block is worth about three
times what the two extra embedding spaces are worth (+0.0139 vs +0.0058), and
it carries essentially all of the step-1 gain (+0.0240 of the +0.0254). That is
consistent with the original motivation -- glove and wiki2vec measure the same
kind of thing numberbatch does, so they mostly re-ask a question already
answered, while free-association is a different kind of evidence.

**This corrects a claim made earlier in the session.** After the leak fix I
described the SWOW/entity/cohesion group as worth ~+1.7 points and as "the kind
of feature block that has now produced nothing twice". The +1.7 figure came from
a leave-one-out against a different feature set; measured properly as a block,
SWOW+entity is the single largest contributor of the three additions
(0.4446 -> 0.4611), not the smallest. The leak destroyed the original +23.7
claim, but it did not make the block worthless, and I overcorrected.

The blocks are slightly superadditive (spaces alone +0.0058, swow+ent alone
+0.0139, both +0.0224 > 0.0197), which is what you would expect if the trees use
one source to decide when to trust another.

## Scoring the listener as a probability model instead of an argmax

Shane's observation, and it reframes the whole distillation effort: top-1
accuracy is not what the spymaster consumes. `expected_words` never looks at
the listener's favourite word -- it needs P(guess w) over the whole board for
gain and penalty, and a turn where the top word is 95% likely and one where it
is 40% likely are completely different turns that score identically under
accuracy.

The incumbent is therefore not "numberbatch argmax". It is the Gaussian
`perceived = z + N(0, sigma)` that `expected_words` already assumes, i.e.
`listener_features.p_is_max`, and the honest question is whether the GBT's
softmax beats it as a *distribution*. `scripts/tools/eval_listener_calibration.py`,
gpt-oss teacher, 15,766 validation choice events, every baseline's free
parameter fitted on held-out training boards:

| model                      | log loss | Brier  | top-1  | mean conf |
|----------------------------|---------:|-------:|-------:|----------:|
| uniform                    |   2.5413 | 0.9084 | 0.0929 |    0.0916 |
| numberbatch softmax T=1.4  |   1.9052 | 0.7268 | 0.4229 |    0.4304 |
| Gaussian p_is_max s=2.5    |   1.8961 | 0.7262 | 0.4229 |    0.3869 |
| GBT                        |   1.7286 | 0.6784 | 0.4535 |    0.4581 |

Gaussian -> GBT is +0.1675 nats, 95% CI [+0.1588, +0.1764] by paired bootstrap.
Against uniform the Gaussian captures 0.645 nats and the GBT 0.813, so the
distilled model recovers **26% more information than the incumbent** -- a much
larger relative gain than the +3.1 accuracy points, because accuracy was
measuring the wrong thing.

**The GBT needs no temperature correction: T* = 1.0.** The group-softmax
objective is a proper scoring rule, so it produces calibrated probabilities
directly. ECE 0.0087, and the reliability table is close to the diagonal in
every bin. This was nearly missed: fitting the temperature on the GBT's *own*
training scores gives T*=0.8 and makes validation worse (1.7443 vs 1.7164),
because overfit training scores make the fit conclude the model should be
sharpened. The temperature has to be fitted on boards the model never saw.

**The Gaussian is miscalibrated in a way that matters for clue choice.**
ECE 0.0550, six times the GBT's, and it is not a uniform shift:

    predicted 0.35 -> actual 0.43   (underconfident in the middle)
    predicted 0.45 -> actual 0.53
    predicted 0.85 -> actual 0.74   (overconfident at the top)
    predicted 0.95 -> actual 0.82

It compresses everything toward the middle -- it calls the top word >0.8 on
only 8.1% of turns where the GBT does so on 14.0% -- and then over-trusts the
confident end it does reach. Both halves push `expected_words` the same way:
the safe turns it cannot recognise as safe, and the turns it does call safe are
riskier than it thinks. That is a plausible contributor to the assassin deaths
observed at low sigma, and it is a bug the accuracy metric could never surface.

The practical consequence is that wiring the GBT into `expected_words` in place
of `p_is_max` is now the best-motivated next change to the spymaster -- and,
unlike the feature work, it does not need any more teacher data.

### Clarification: three different sigmas, and one coincidence

The calibration table above reports the incumbent as "Gaussian p_is_max
sigma=2.5", which invites the reading that it is the shipped config value.
It is not. That sigma was grid-searched against the gpt-oss rankings, and it
landed on 2.5 only because the grid was coarse (..., 2.0, 2.5, 3.0). On a fine
grid the pooled optimum is **2.4**, and the agreement with the config was an
artefact of grid resolution. Grid widened in the script so it cannot mislead
again.

Three quantities in this project are all called sigma:

1. **Config sigma = 2.5** -- what `expected_words` assumes and what the
   `noisy_glove` synthetic guesser plays at. A *shipping* choice, currently
   the worst of the five values the arena ladder tested.
2. **Descriptive sigma** -- which Gaussian best *describes* a real listener.
   `listener_fit.py` reports 2.06 for Sonnet and 2.20 for gpt-oss by exact
   top-1 MLE.
3. **The calibration fit** -- which Gaussian best predicts the teacher's pick
   by log loss over all choice events.

(2) and (3) estimate the same thing by different criteria on different events,
and they disagree because the answer depends on which events you score:

        gpt-oss, fine grid      pooled NLL      step-1 NLL
        sigma = 2.1               1.9061          1.3430  <- best step-1
        sigma = 2.2               1.9000          1.3452
        sigma = 2.4               1.8955  <- best  1.3579
        sigma = 2.5               1.8961          1.3674

Step-1 alone wants 2.1, matching listener_fit's 2.20 top-1 MLE as it should --
same estimand. Pooling later steps pushes the answer to 2.4: choices from a
partly-revealed board are noisier than the opening pick, so a single Gaussian
splits the difference. That is itself a small argument against the fixed-sigma
model, which has no way to express "later steps are noisier".

None of these is the *playing* sigma. The arena ladder had Sonnet winning most
at 1.25--1.5 while behaving descriptively like 2.06; a listener model and a
listener assumption you should optimise against are different objects.

**The calibration conclusion is unaffected.** The Gaussian's NLL at its true
optimum 2.4 is 1.8955, against 1.8961 at 2.5 -- 0.0006 nats, next to the GBT's
0.1675 advantage. The curve is flat across 2.2--2.6, so no choice of sigma
rescues it; the deficit is the functional form, not the parameter.

## Does the listener hold up on a mostly-revealed board?

Shane asked how the model generalises when fewer than 25 words remain. The
honest worry was data: the collector reveals a random number of words, so
small candidate sets are rarer (2--8 words is 22% of training choice events,
2--4 alone only 4%). And the Gaussian already shows an n-dependence -- step-1
alone wants sigma 2.1 while pooling pushes it to 2.4.

Answer: **board size is not the problem. Step index is.** Reported as nats
captured over uniform, because chance moves with n and raw log loss across
bins is not comparable. `scripts/tools/eval_listener_calibration.py` now
prints this breakdown.

    candidates      step      n  available   Gauss     GBT  GBT share
     2- 8      1 (first)   1080      1.876   0.935   1.057     56.4%
     2- 8             2+   2418      1.703   0.092   0.196     11.5%
     9-16      1 (first)   2482      2.509   1.200   1.342     53.5%
     9-16             2+   3646      2.509   0.188   0.373     14.9%
    17-25      1 (first)   2763      3.039   1.474   1.671     55.0%
    17-25             2+   3377      3.006   0.359   0.559     18.6%

At step 1 the GBT captures 56.4% / 53.5% / 55.0% of the available information
as the board shrinks -- flat to within noise. The Gaussian is equally flat at
~48--50%. Neither model degrades with board size at all in relative terms, and
the GBT's edge over the Gaussian survives everywhere (paired, +0.110 at 2--8
words, +0.168 at 9--16, +0.199 at 17--25, all CIs clear of zero).

What collapses is depth into the turn:

      step      n  available   Gauss     GBT
         1   6325      2.632   1.275   1.437
         2   4704      2.547   0.373   0.559
         3   3167      2.455   0.106   0.271
         4   1570      2.331   0.020   0.152

The Gaussian captures 0.020 nats at step 4 -- indistinguishable from knowing
nothing. The GBT is better at every depth but also falls off a cliff.

**This is a property of the teacher, not of the models.** Once gpt-oss has
taken the words it actually wants, the rest of its ranking is close to
arbitrary, and it matches the measured self-consistency: re-asking the same
position agreed on the top pick 84--99% of the time depending on clue kind, and
there is no reason the tail should be anywhere near that stable.

**Two consequences worth acting on.**

1. Every pooled number in this log is dominated by near-noise. Steps 3 and 4
   are 4,737 of 15,766 validation events (30%) carrying 0.271 and 0.152 nats.
   The pooled 0.4611 accuracy and 1.7286 log loss are real but they are mostly
   a measurement of how well we fit a coin flip.
2. The PL expansion weights step 4 exactly as heavily as step 1 in the
   objective, so a meaningful share of model capacity is spent fitting the
   teacher's arbitrary tail. Mean k in real games is 1.42, so steps 3+ barely
   occur in play. Down-weighting or truncating later steps is a training-side
   change needing no new data, and it is the obvious next experiment.

## Raw rival z-scores: a negative result, reverted

Shane's question: does the hand-engineering earn its keep, or would the trees
do better handed the similarities themselves? Every competition feature here
collapses the board to order statistics (`rival_min_space`, `gaptop_*`,
`lead_margin`), so the test is to add the raw numbers and see.

Two design problems had to be solved before it was worth running.

**Fixed columns per board word leak position.** Laying out `sim_word1,
sim_word2, ...` makes the column index list position, which is arbitrary --
the same failure that produced the `_ranks` leak. Fixed by sorting rivals and
emitting order statistics, which is permutation-invariant by construction.

**Sorting per space destroys cross-space structure**, which Shane spotted
before it was built: a board where one rival leads in every space and a board
where two rivals each lead in one produce identical per-space sorted vectors,
and telling those apart is the entire point of `rival_min_space`. Fixed by
ranking rivals *once*, by mean z across the three spaces, then emitting each
rival's full triple in that shared order. The mean is only a sort key; no value
is averaged into a feature. Ties fall through to numberbatch, glove, wiki2vec
rather than to `argsort`'s index order.

Depth 5 (Shane's call), so 15 features, 32 -> 47.

    32 (no raw5)   trees=177   R2 0.3249   step-1 R2 0.5517
    47 (+raw5)     trees=186   R2 0.3229   step-1 R2 0.5498

    pooled log-loss change: -0.0051  95% CI [-0.0076, -0.0028]
    step-1                  -0.0050  95% CI [-0.0083, -0.0016]

Both intervals lie entirely below zero, so this is a real if small **harm**,
not a null. Accuracy could not resolve it (0.4589 -> 0.4588 pooled), which is
another instance of the metric being the wrong one.

Best explanation: the 15 columns are largely redundant with order statistics
the model already has, and they dilute `feature_fraction=0.8` -- each tree now
samples 38 of 47 columns rather than 26 of 32, so a larger share of every
tree's candidate splits are on near-duplicate information.

Reverted. This is the fourth feature block to measure null or negative
(Tier-1 redundant, cohesion null, entity ~2% of SHAP contribution, raw5
negative). The only block that ever moved the metric materially was SWOW, and
the pattern across all five is that a genuinely different *kind* of evidence
helps and another view of distributional similarity does not.

## Reverse SWOW: the largest single feature gain so far

Hypothesis from the block pattern: four blocks measured null or negative and
the only one that moved anything was SWOW, so what helps is a different *kind*
of evidence, not another view of distributional similarity. The cheapest test
of that was sitting in data already on disk.

Association is asymmetric. "nurse" cues "doctor" far more often than "doctor"
cues "nurse". `build_swow_tables.py` only ever walked clue -> intermediate ->
board word, which asks the *spymaster's* question: does the clue bring this
word to mind? The reverse walk asks the *listener's* question: does this word
bring the clue to mind -- and the listener is the thing being modelled. Same
edges, same intermediate layer, traversed the other way. No download, no new
data, one extra sparse multiply (rev one-hop 39,131 nnz, rev two-hop 1,806,450).

Five features: `swow_rev1`, `swow_rev2`, `swow_rev2_rank`, `swow_rev2_share`,
and `swow_asym` (the gap between the directions, as shares rather than raw
strengths -- the two walks traverse different numbers of edges and their scales
are not comparable).

    32 baseline          trees=177   R2 0.3249   step-1 R2 0.5517
    37 +reverse          trees=282   R2 0.3365   step-1 R2 0.5628
    32 rev-only (-fwd)   trees=231   R2 0.3282   step-1 R2 0.5528

    pooled gain: +0.0294 nats  95% CI [+0.0250, +0.0342]
    step-1 gain: +0.0291 nats  95% CI [+0.0227, +0.0354]

**Reverse is worth more than forward.** Forward SWOW was +0.0139 nats when it
went in; reverse adds +0.0294 *on top of* forward and everything else. And the
rev-only arm (0.3282) beats the fwd-only baseline (0.3249) on equal feature
count, so this is not simply more columns. Direction is doing real work, and
the direction that matters is the listener's.

Worth stating plainly for the defence: this was free. The data has been on disk
since the SWOW download and half of it was being discarded because the build
script was written from the spymaster's point of view.

## LM pointwise mutual information: +0.019 nats, and complementary to SWOW

SWOW's weakness is coverage: 12,217 cues, so 41% of the clue pool has no row
and the feature is NaN. The hypothesis was that a language model measures the
same syntagmatic axis -- which words come to mind together -- from a corpus
rather than from people, and covers the whole vocabulary.

**Cross-encoders were tried first and rejected.** `cross-encoder/stsb-roberta-large`
scores bare word pairs in a 0.009-0.075 band and puts schmidt/table above
schmidt/scorpion. STS is trained for paraphrase similarity between sentences,
which is the wrong objective and out of distribution on single words; sentence
templating widened the range but not the discrimination. Deleted rather than
kept as a null arm, because it never got as far as a measurement.

**PMI, not the raw conditional.** The conditional alone is dominated by
tokenisation: "platypus" is " plat" + "ypus" and the second token is nearly
certain given the first, so a length-normalised score ranked it above "camera"
as an associate of "nikon", and king/stapler above king/queen. Summing instead
of averaging keeps that bias, but the unconditional term carries exactly the
same bias and it cancels:

    PMI = log P(w | "The word <clue> reminds me of the word")
        - log P(w | "The word reminds me of the word")

On a probe set every related pair then outranks every unrelated one, and it
recovers schmidt -> scorpion (2.19 vs 0.79 for schmidt -> table), the
encyclopedic link the entity vectors were added for and largely failed to give.

gpt2-large on the RTX 5080, 11,145 clues x 400 board words in 22.8 min, 100%
coverage. Four features: `pmi`, `pmi_rank`, `pmi_gaptop`, `pmi_share`.

    37 no-PMI          trees=282   R2 0.3365   step-1 R2 0.5628
    41 +PMI            trees=298   R2 0.3439   step-1 R2 0.5687
    31 PMI, no SWOW    trees=190   R2 0.3177   step-1 R2 0.5379

    pooled gain: +0.0188 nats  95% CI [+0.0147, +0.0230]
    step-1 gain: +0.0155 nats  95% CI [+0.0099, +0.0209]

**PMI does not replace SWOW, it adds to it.** Dropping both SWOW blocks and
keeping PMI scores 0.3177, well below the 0.3365 that SWOW alone reaches, so
the corpus version is the weaker of the two despite its full coverage -- human
association is measuring something a language model does not reproduce. Both
are worth keeping.

Running total for the session: R2 0.3249 -> 0.3439 on 32 -> 41 features,
step-1 0.5517 -> 0.5687, entirely from evidence sources rather than from more
views of the three embedding spaces. gpt2-large is the smallest sensible model
here and an obvious upgrade path if this line is pushed further.

## Two more embedding spaces: small but real, and my prediction was wrong

Shane asked whether newer versions of the three tensor spaces exist. Mostly
not: GloVe has had no release since 2014, Numberbatch 19.08 is current, and
Wikipedia2Vec's newest pretrained dump is the enwiki_20180420 we already use.
But the tensor carries `glove.6B` -- Wikipedia+Gigaword, 400k tokens, the
*weakest* GloVe release -- where `glove.840B` (Common Crawl, 2.2M cased) exists.
Added that plus fastText (Common Crawl, 2M words, subword).

Built as side tables (`cache/extra_sims.npz`), not as tensor slots. The tensor's
vocabulary is deliberately the intersection of its spaces, so adding to it would
shrink the legal clue set and silently invalidate every cached rollout and every
number in this log. Z-scoring happens at build time over all 400 board words, so
the feature cannot accidentally become "normalised over whatever is still
unrevealed". Coverage is 100% of the clue pool and 396/400 board words -- the
four misses are the multi-word board entries, same as everywhere else.

    41 no-extra            trees=298   R2 0.3439   step-1 R2 0.5687
    47 +extra              trees=382   R2 0.3469   step-1 R2 0.5722
    38 extra, no glove/wiki trees=198  R2 0.3437   step-1 R2 0.5674

    pooled gain: +0.0078 nats  95% CI [+0.0042, +0.0112]
    step-1 gain: +0.0093 nats  95% CI [+0.0047, +0.0140]

**I predicted this would land inside the noise and it did not.** The claim was
that five null blocks had established that another view of distributional
similarity buys nothing; the interval here excludes zero comfortably. The
correct version of the claim is weaker: another view of distributional
similarity buys *little*, about a quarter of what reverse SWOW bought and under
half of LM PMI, but not nothing.

The third arm is the interesting one. Dropping glove and wiki2vec from the
tensor features while keeping the two new spaces scores 0.3437 against the
41-feature baseline's 0.3439 -- statistically the same model on three fewer
features. So glove840 and fastText are straight substitutes for glove.6B and
wiki2vec, and it is only their *union* that adds anything.

Session running total: R2 0.3249 -> 0.3469 (32 -> 47 features), step-1
0.5517 -> 0.5722. Of the +0.022, roughly +0.012 came from the two association
directions, +0.007 from LM PMI, +0.003 from the new embedding spaces.

## Concreteness norms and WordNet taxonomy: both help, and partly overlap

Two blocks, measured separately so either could come back null.

**Concreteness (Brysbaert, Warriner & Kuperman 2014).** Motivated directly by
the SHAP attribution: `word_mean_sim` and `word_sd_sim` rank second and third
by contribution per feature, and both describe the word with no clue involved.
That says the model wants word-level priors, and concreteness is the
best-attested one -- a listener told "animal" and looking at LION and SPIRIT has
a reason to prefer the one they can picture. The same file carries SUBTLEX
frequency and percent-known, so all three priors came free. Five features:
`conc`, `conc_rank`, `conc_sd`, `pct_known`, `log_freq`. Board coverage 91.5%.

**WordNet.** Neither distributional nor associative: a hand-built is-a hierarchy
knows LION and WHALE are both mammals with no corpus evidence and no human
free-association. Sense ambiguity is handled by taking the max over sense pairs
rather than disambiguating -- which is not a compromise here but the right
model, since a listener who sees a link acts on it whether or not it was the
sense the spymaster meant, and that is exactly how Codenames goes wrong.
`lcs_depth` is emitted alongside the Wu-Palmer ratio because "both are dogs"
(depth 13) and "both are entities" (depth 1) are very different evidence and
the ratio alone does not separate them. Three features. Coverage was much better
than feared: board 98.8%, clues 94.1%, 62.2% of pairs with a nonzero score.

    47 neither     trees=382   R2 0.3469   step-1 R2 0.5722
    52 +norms      trees=170   R2 0.3496   step-1 R2 0.5716
    50 +wordnet    trees=344   R2 0.3496   step-1 R2 0.5751
    55 both        trees=158   R2 0.3507   step-1 R2 0.5734

    +norms   +0.0067 nats  95% CI [+0.0032, +0.0103]
    +wordnet +0.0068 nats  95% CI [+0.0035, +0.0099]
    +both    +0.0097 nats  95% CI [+0.0058, +0.0133]

Equal in size, and sub-additive: together they buy 0.0097 of a possible 0.0135,
so roughly 30% of what each carries the other also carries.

**The two help in different places.** WordNet improves step-1 (0.5722 ->
0.5751) while concreteness slightly does not (0.5716), yet concreteness improves
the pooled figure just as much. So the norms are earning their keep at steps 2+,
which is what you would expect: once the clue's own signal is spent on the
obvious word, what is left to separate the remaining candidates is properties of
the words themselves.

Note the tree counts collapse from 382 to 158 once both blocks are in. The
model reaches its plateau far sooner with these features available, which is
another sign they carry information the rest of the set was working hard to
approximate.

Session running total: R2 0.3249 -> 0.3507 (32 -> 55 features), step-1
0.5517 -> 0.5734.

## Pruning: 19 of 55 features go for free, and extract() halves

`scripts/tools/prune_listener_features.py`. Leave-one-out rather than SHAP,
because SHAP says how much the fitted model *uses* a feature and not whether it
could do without it -- and with five overlapping association and similarity
blocks that is the entire question. Selection on the calibration boards, final
comparison on val.

    calib: full 0.3583 (55 feat) -> pruned 0.3566 (36 feat)   delta -0.00169
    val  : full 0.3495           -> pruned 0.3500             delta +0.00047

Nineteen features are free to drop. Accuracy moves the other way by a hair
(0.4775 -> 0.4739 val top-1) while R2 does not, which is the two metrics
disagreeing inside the noise; R2 is the one the model is trained on and the one
the spymaster consumes.

**What went, and why it makes sense:**

- **All three within-board `rank_*` columns.** The `gaptop_*` columns already
  encode the ordering *and* how far behind each word is; a rank throws the
  distance away. Same story for `swow2_rank`, `swow_rev2_rank`, `cohesion_rank`,
  `conc_rank`, `ft_rank`, `pmi_rank` -- seven of the nineteen are ranks whose
  underlying value is also present.
- **Two of the three `p_max` sigmas.** Carried originally so the trees could
  interpolate between noise levels; 1.0 and 3.0 are both leave-one-out negative.
- **Three of the four PMI features**, leaving only `pmi_gaptop`. That is the
  right survivor: PMI is a log ratio, so a difference is a ratio of ratios.
- **`rival_min_space` and `gap_vs_rival_min`**, which is a straightforward
  negative result on a feature block this project argued for at length in
  listener_features.py's own docstring.

**The prune buys real inference time**, unusually for a parsimony exercise:
`extract()` drops from 1647 to 752 us per clue, because two of the three
49-point `p_is_max` quadratures are gone and those dominated the cost. A
full-pool turn falls from 18.4 s to 8.4 s.

Verified numerically rather than by eye: a snapshot of all 55 columns over nine
(board, clue, k) cases was taken before editing `extract`, and every surviving
column matches it exactly.

SHAP on the pruned model, for the record -- note `pmi_gaptop` alone now carries
13.9% of total attribution:

    block              n  sum |SHAP|   share  per feature
    nb                11      0.8507   36.0%       0.0773
    pmi                1      0.3292   13.9%       0.3292
    spaces             4      0.2564   10.9%       0.0641
    extraspaces        5      0.2381   10.1%       0.0476
    swowrev            4      0.2114    9.0%       0.0528
    norms              4      0.2027    8.6%       0.0507
    swow               3      0.1947    8.2%       0.0649
    wordnet            2      0.0562    2.4%       0.0281
    entity             2      0.0204    0.9%       0.0102

**A process note worth keeping.** The first run of the prune tool deadlocked
for 16 minutes with zero output: it fitted the full-feature baseline in the
parent before creating the worker pool, which starts LightGBM's OpenMP threads,
and the forked children then inherited a mutex held by a thread that does not
exist in them. Eight workers at exactly 00:00:00 CPU was the tell, and
instantaneous %CPU was not -- cumulative CPU time is the diagnostic. The
baseline is now a pool job like any other, and the hazard is documented in
`loo`'s docstring. The earlier hyperparameter sweep avoided this by accident.

## Early stopping was watching the wrong metric

Found while measuring the lexical blocks, and it matters more than they do.

The model is **trained** on group softmax (log loss), **reported** on McFadden
R2, and was **early-stopped** on tie-aware group accuracy. Three different
quantities. Accuracy is a step function that plateaus and jitters, so the
stopping point wandered badly: arms differing by three features stopped
anywhere between 138 and 652 trees.

The noise that introduced was large enough to invent a result. Measuring three
new blocks under the old procedure gave each one a clear individual gain and
the combination of all three a clear *loss* (-0.0067, CI entirely below zero) --
an apparently interesting interaction that was nothing but a premature stop at
138 trees.

    block            before (acc stop)        after (log-loss stop)
    36 baseline      0.3512  200 trees        0.3531  393 trees
    37 +polysemy     0.3538  377              0.3538  377
    41 +orth         0.3537  652              0.3544  393
    40 +gloss        0.3540  327              0.3542  370
    44 all           0.3486  138              0.3538  390

Stopping now uses `mcfadden_on`, which is the reported metric and -- since the
null is a constant for a fixed validation set -- exactly equivalent to stopping
on log loss, i.e. on the training objective. Tree counts settle into a 370-393
band. The baseline alone gains +0.0019 for free.

**What this invalidates.** Every block measurement in this session was made
under accuracy-based stopping. The large gains are far outside this noise and
stand: reverse SWOW +0.0294, LM PMI +0.0188. The small ones are not, and are
re-measured in the next entry: extra embedding spaces (+0.0078), concreteness
norms (+0.0067), WordNet (+0.0068), and the 55->36 prune, whose leave-one-out
values were all under 0.005 and therefore comparable to the stopping noise.

## Re-measuring every block under log-loss stopping

Leave-one-block-out against the full 44-feature set, so every number is
directly comparable and produced under the corrected stopping rule. Positive =
what dropping that block costs, in nats.

    block dropped    n      R2  step-1   cost of dropping
    nb              11  0.3437  0.5690   +0.0256 [+0.0203,+0.0310] REAL
    swowrev          4  0.3474  0.5727   +0.0162 [+0.0118,+0.0210] REAL
    swow             3  0.3483  0.5727   +0.0141 [+0.0095,+0.0184] REAL
    norms            5  0.3498  0.5750   +0.0103 [+0.0069,+0.0141] REAL
    pmi              1  0.3501  0.5752   +0.0095 [+0.0055,+0.0136] REAL
    extraspaces      5  0.3506  0.5776   +0.0082 [+0.0045,+0.0119] REAL
    spaces           4  0.3518  0.5764   +0.0052 [+0.0018,+0.0087] REAL
    wordnet          2  0.3530  0.5774   +0.0020 [-0.0013,+0.0051] null
    lexical          7  0.3538  0.5778   +0.0001 [-0.0029,+0.0031] null
    entity           2  0.3546  0.5802   -0.0021 [-0.0051,+0.0010] null

Seven of ten blocks are real. Notably `pmi` is one feature (`pmi_gaptop`) doing
+0.0095, the best value-per-feature in the set. The three that had to be
re-measured survive: extraspaces, norms and spaces all still clear zero.

**The joint check reversed the obvious conclusion, which is the point of
having one.** wordnet, lexical and entity are each individually null, and the
tempting move is to drop all eleven features. Measured over three seeds:

    44 full     R2 0.3545  (0.3538, 0.3544, 0.3553)   step-1 0.5793
    33 pruned   R2 0.3516  (0.3513, 0.3517, 0.3517)   step-1 0.5759

    cost of dropping all three: +0.0075 nats  95% CI [+0.0047, +0.0102]

Dropping them **costs** 0.0029 R2. Each is individually redundant because the
other two cover it, and removing all three removes information nothing else
carries. Leave-one-out is systematically blind to this and would have thrown
away a real +0.0075; the lesson is that a block is only droppable once the
*combination* has been measured, never on its own LOO value.

(The verdict label in the checking script was written with the sign backwards
and printed "dropping HELPS". The R2 columns are unambiguous and were what
caught it. Noting it because a mislabelled verdict is exactly the kind of thing
that survives into a writeup.)

All 44 features stay. Best val R2 this session: **0.3545** (3-seed mean), from
0.3249 at the start, with step-1 0.5517 -> 0.5793.

**What was and was not affected by the stopping bug.** `prune_listener_features.py`
and `sweep_listener_params.py` both already early-stopped on McFadden R2, so the
55->36 prune and the hyperparameter sweep were never affected -- an earlier note
in this log saying the prune needed re-measuring was wrong. Only the ad-hoc
block-ablation scripts went through `train_listener.train`, and those are the
ones re-run here.

## Down-weighting later choice events: +0.010 nats at step 1, free

The PL expansion turns one teacher ranking into k choice events and weighted
them all equally. Two independent arguments say it should not: steps 3 and 4
score R2 0.11 and 0.07 because the teacher's ranking tail is near-arbitrary
once it has taken the words it wants, and they are 30% of the training signal;
and mean k in real games is 1.42, so those steps barely occur in play.

Four schemes, weights applied inside the objective (explicitly, rather than via
LightGBM's `weight=`, so the scaling cannot depend on whether a given version
forwards dataset weights into a custom objective's output):

    scheme                trees   pooled   step-1   step-2
    uniform (current)       390   0.3538   0.5795   0.2563
    step-1 only             155   0.3275   0.5790   0.2212
    1/(1+step)              351   0.3525   0.5842   0.2530
    0.5^step                354   0.3502   0.5833   0.2513
    0.75^step               380   0.3539   0.5834   0.2561

    0.75^step  step-1 +0.0102 [+0.0061,+0.0143]   pooled +0.0003 [-0.0026,+0.0032]
    1/(1+step) step-1 +0.0123 [+0.0078,+0.0170]   pooled -0.0034 [-0.0066,+0.0000]

Adopted `0.75**step`: a real step-1 gain with pooled statistically unchanged.

**The `step-1 only` arm is the informative one, and it refutes the strong form
of the hypothesis.** If later steps were pure noise, training on step 1 alone
should be best for step 1. It is not -- it is indistinguishable at step 1
(-0.0014, CI spanning zero) while destroying pooled (-0.0668). So the noisy
tail is still useful training signal and merely must not dominate. Truncation
is the wrong move; down-weighting is the right one. Worth stating plainly for
the defence, because "later steps are noise, so drop them" is the intuitive
conclusion and it is wrong.

Val top-1 with weighting: 0.4759 pooled, 0.6727 step-1, against 0.4589/0.6599
at the start of the session.

## Seed ensembling, and a second hyperparameter sweep that changed nothing

**Ensembling: +0.0066 nats, real.** Five models differing only in seed,
combined by averaging raw scores.

    single-model mean  R2 0.3540  (spread 0.3535-0.3543)
    2-model            R2 0.3556   step-1 0.5845
    3-model            R2 0.3559   step-1 0.5847
    5-model            R2 0.3566   step-1 0.5850

    5-model vs mean single: +0.0066 nats  95% CI [+0.0065, +0.0066]

The interval is unusually tight because it is paired against the seed average
rather than against one draw. Most of the gain is the second model (1->2 is
+0.0016 R2, 2->3 is +0.0003, 3->5 is +0.0007), so 2-3 members is the sensible
operating point if this is ever wired in. Geometric (average raw scores) and
mixture (average probabilities) averaging are a wash: 0.3566 vs 0.3567 pooled,
reversed at step 1. I expected the mixture to win on log loss, being the
principled predictive distribution; it does not, meaningfully.

**Second sweep: no change adopted.** The first sweep ran at 32 features and put
all ten of its top configs at lambda_l2=10.0, the largest value tried, so the
optimum was unresolved at the grid edge. Re-run at 44 features with step
weighting and l2 extended to 300:

    leaves  min_leaf     l2  trees   calib R2
        31       250   10.0   1166     0.3587   <- sweep winner
        31       100   30.0    975     0.3577
       127       100   10.0   ~400     0.3562   <- current, rank 34 of 48

The edge worry resolves: the winner sits at l2=10 with the top ten spread over
10-100, so the optimum was inside the original range all along. The apparent
change was num_leaves 127 -> 31.

But the val gain was +0.0008, which is exactly the seed spread measured above,
and the sweep picked on one seed. Re-measured over three seeds, paired:

    current 127/100/10  trees [380, 404, 386]   R2 0.3538   step-1 0.5830
    swept    31/250/10  trees [1311,1107, 991]  R2 0.3545   step-1 0.5814

    +0.0018 nats  95% CI [-0.0001, +0.0037]  indistinguishable

Indistinguishable pooled, *worse* at step 1, and roughly 3x the trees to
predict. Kept 127/100/10. This is the third time this session that a
single-seed difference would have driven a change that does not survive
replication -- after the phantom block interaction and the mislabelled joint
prune. Any config decision from here needs multiple seeds.

## Learning rate, PMI templates, and two rejected ideas

Four things tried. Two landed, two did not, and the one that landed biggest was
the one I had never looked at.

**Learning rate: +0.0027 R2, and it had been fixed at 0.05 since the first fit.**
Three seeds per arm:

    lr=0.1     R2 0.3505   step-1 0.5791
    lr=0.05    R2 0.3538   step-1 0.5830   <- previous
    lr=0.02    R2 0.3559   step-1 0.5844
    lr=0.01    R2 0.3565   step-1 0.5849   <- adopted, ~2300 trees
    lr=0.005   R2 0.3570   step-1 0.5853      ~4800 trees
    lr=0.003   R2 0.3569   step-1 0.5850      ~7300 trees

Monotone to 0.005, then flat. 0.005 beats 0.01 by +0.0005, inside the seed
noise band, for 2.1x the trees -- inference cost the spymaster pays on every
clue, so 0.01 is the operating point. Worth recording plainly: this is a bigger
gain than four of the evidence sources that cost hours of work and gigabytes of
download, and it is the one major hyperparameter never swept.

The round cap was raised 3000 -> 8000 at the same time. At lr=0.01 the model
wants ~2300 trees; a cap that binds looks exactly like convergence.

**PMI template averaging: +0.0030 R2.** One prompt is one arbitrary way of
asking. On a 14-pair probe no single template separates related from unrelated
pairs cleanly -- each gets some hard pair backwards, but a different one -- so
averaging cancels the prompt-specific part. Three templates, chosen by probe
margin (`Things related to {}:`, `The word {} reminds me of the word`,
`Word association. {} ->`); `{} and` and `{} makes me think of` were measured
and dropped.

    single-template  R2 0.3565 (0.3565, 0.3565, 0.3565)  step-1 0.5849
    3-template avg   R2 0.3595 (0.3596, 0.3594, 0.3595)  step-1 0.5866

    +0.0075 nats  95% CI [+0.0052, +0.0099]

Tested at lr=0.01 rather than earlier, because measuring a feature change at a
stale learning rate measures the wrong model. Note how tight the seed spread
is at this learning rate -- 0.0002, against 0.0008 at lr=0.05.

**Ensembling: rejected, and it was mostly an artifact.** At lr=0.05 a 5-seed
ensemble bought +0.0026 R2. At lr=0.01 it buys +0.0006. A lower learning rate
is itself a variance-reduction mechanism and absorbs most of what ensembling
was doing, so the case for 5x the training and inference disappears.

**Diverse ensembling: rejected.** Members varying in `num_leaves` and
`feature_fraction` rather than seed were indistinguishable from seed-only
(+0.0003 nats, CI [-0.0004, +0.0011]). The hypothesis that decorrelated members
would beat seed jitter is simply not supported here.

Session total: R2 0.3249 -> 0.3595, step-1 0.5517 -> 0.5866.

## The clean holdout: what twenty-five selections on val actually cost

1,910 positions were collected from a board-seed range beyond everything used
for training or selection (`collect_listener_data.py --start 40000`), after
every modelling decision was frozen. The planner reported 0 of 1,910 already
cached, confirming the boards were untouched. $0.19, 31.5 min, 41 errors.
Trained on the old train and val together (those decisions are already made),
scored once.

    model                    test R2    step-1
    uniform                   0.0000    0.0000
    Gaussian p_is_max 2.4     0.2260    0.4423
    distilled GBT             0.3458    0.5583   (seeds .3456 .3460 .3459)

    Gaussian -> GBT: +0.3039 nats  95% CI [+0.2808, +0.3272]

**The val number was optimistic, by about what you would guess.** Pooled
0.3595 -> 0.3458 (-0.0137), step-1 0.5866 -> 0.5583 (-0.0283). That gap is
larger than most individual feature blocks bought, which is the honest cost of
selecting ~25 times on one validation set. Any future claim about this model
should quote 0.346, not 0.360.

**But the comparison that matters got better, not worse.** The Gaussian
incumbent drops further on these boards than the GBT does -- 0.2541 -> 0.2260
against 0.3595 -> 0.3458 -- so the test set is simply harder, and most of the
GBT's shortfall is difficulty rather than overfitting. Measured against the
baseline on the same events:

                     val      test
    GBT - Gaussian   0.105    0.120
    GBT / Gaussian   1.41x    1.53x

On boards nothing was ever selected on, the distilled listener captures 53%
more of the available information than the model `expected_words` currently
assumes. That is the defensible headline, and it is stronger than the
within-val version of the same claim.

Caveat on precision: 4,671 test choice events against 15,766 in val, so the
intervals are wider. The +0.30 nat gap over the Gaussian is far outside them.

## Expected reward under the learned listener

`docs/clue-selection-learned.tex` (and its PDF) derives the expected reward of
a clue when the guesser model is the distilled scoring function rather than
Gaussian noise on similarities; `codenames/pl_reward.py` implements it.

The question that motivated it: the listener gives a distribution over the
guesser's *next* pick, but a turn is a sequence -- each first pick opens a
subtree of second picks with renormalised probabilities, and so on, at
O(n**K) paths. Enumerating that is not affordable inside a clue search.

**It does not have to be enumerated.** The distilled model is a Plackett-Luce
model: `train_listener.py` computes the feature matrix once per position and
later steps merely drop rows, so scores do not depend on which words remain --
removal changes only the normalising sum. Under that, giving each word an
independent Exp(exp(score)) clock reproduces the whole selection tree exactly,
by memorylessness. The turn then becomes minima of independent exponentials:

    T = min over non-team clocks,  W = argmin,  N = #{own words before T}

and the guesser reveals min(k, N) of ours, hitting W exactly when N < k. This
is deliberately the same shape as spymasters/expected_words.py's derivation, so
the two guesser models are swappable behind one `(gain, penalty)` interface.

**It is cheaper than the Gaussian version, not dearer.** For independent
exponentials the minimum and its argument are independent, so which non-team
word ends the turn does not depend on when -- the expected miss cost is one
constant per clue instead of a hazard ratio at every quadrature point, and |B|
leaves the integral. And u = exp(-Lambda t) maps the integral to [0,1] exactly,
so there is no interval to pick and no tail to wave away, unlike the Gaussian
model's [t_0, t_M] with its GRID_PAD sigmas.

**The exactness claim is tested, not asserted.** `tests/test_pl_reward.py`
brute-forces the selection tree by recursion on small boards and demands
agreement, across six (own, bad, k) shapes and every k in one pass, plus a
200k-trial simulation of the race itself. All pass. Also asserted: shift
invariance of the scores, monotonicity and the min(k, N) cap, and no overflow
at +/-300 logits.

**Quadrature is the only approximation, and it is first order**, because
p_i(u) carries u**(lambda_i/Lambda) whose derivative is unbounded at u = 0.
Error against brute force roughly halves per doubling: 4.7e-4 (gain) and
2.8e-3 (penalty) at the default 96 cells. That sounds marginal but the search
consumes the *ordering* of clues, and on 500 competing clues over a 9-own /
16-other board the argmax, the top ten and the chosen k all match an 8192-cell
reference from 48 cells upward. 96 costs 7 ms per 500 clues.

Not yet wired into a spymaster: that needs the two-stage shortlist (extract()
is 752 us/clue, so scoring all 11,145 is 8.4 s/turn against 0.17 s for a
shortlist of 100) and belongs in its own module.

## First real games: the learned listener against Sonnet

`learned_listener` (docs/clue-selection-learned.tex + codenames/pl_reward.py,
searched two-stage in codenames/spymasters/learned_listener.py) against the
`expected_words` baseline, 20 boards played twice with the sides swapped,
Claude Sonnet 5 as the guesser for both. 40 games, 204 s.

    spymaster           win%    as A    as B  assassin%  clues  mean k  own/clue    own%
    expected_words     10.0%   10.0%   10.0%       2.5%    192    1.29      1.15   90.9%
    learned_listener   90.0%   90.0%   90.0%       0.0%    207    1.97      1.56   84.6%

36-4. Identical as A and as B, so the side advantage is not doing the work.

The mechanism is visible in the columns: the learned listener gives more
ambitious clues (mean k 1.97 against 1.29) and converts more words per clue
(1.56 against 1.15) while being slightly *less* precise per guess (84.6%
against 90.9%). It is not guessing better, it is trusting the guesser further
and being right often enough that the trade pays. It also never hit the
assassin, against 2.5% for the baseline.

**Checked for a positional leak before believing it.** The spymaster passes
candidates as own-words-first, while training used a per-position shuffle --
exactly the asymmetry that produced the `_ranks` leak earlier in this project.
Feeding shuffled candidate orders and comparing scores under the inverse
permutation gives a maximum deviation of 0.0: the features are exactly
order-invariant, so own-first ordering hands the model nothing.

**Two caveats, both real.**

The baseline here runs at sigma=2.5, which the sigma ladder measured as the
worst of five values tested; sigma=1.5 won 70-30 against centroid. So part of
this margin is beating a badly configured opponent rather than beating the
approach, and a sigma=1.5 rematch is running.

`number` is passed to the feature extractor as K_max = min(n_own, 4) before the
clue's own best k is known, so the scores used to evaluate every k were
computed under one value of that feature. `k` is constant within a board and
therefore cannot change the softmax over words -- it only gates tree
interactions -- so this cannot leak which word is ours, but it does mean the
per-k rewards share one scoring pass rather than each getting its own.

## The sigma=1.5 rematch, extended: 72% over 100 games

Extending the learned_listener vs expected_words[sigma=1.5] arm from 20 boards
to 50 settles what 40 games could not. Win rate 72%, 95% Wilson [0.63, 0.80] --
the interval now excludes 0.5, where at 40 games it was 65% and p = 0.07 on the
paired sign test.

The head-to-head summary against all three opponents, Sonnet guessing:

    opponent                games   win%   95% CI
    centroid                   40    82%   [0.68, 0.91]
    expected_words sigma=2.5   40    90%   [0.77, 0.96]
    expected_words sigma=1.5  100    72%   [0.63, 0.80]

**The mechanism is clean at equal ambition.** Against sigma=1.5 the two models
give almost identically bold clues -- mean k 1.97 against 1.93 -- so the win is
not aggression:

    model                    mean k   own%   opp%   assassin%
    expected_words sigma=1.5   1.93    79%    10%        6.0%
    learned_listener           1.97    87%     5%        1.0%

Same number of words attempted per clue, eight points more of them correct,
half the opponent words, and a sixth of the assassin losses.

`notebooks/arena_results.ipynb` renders all of this from cache/llm_store.db
with no API calls: the sigma ladder with Wilson intervals, mean k and assassin
rate across the ladder, the three head-to-heads, and guess composition by role.
It deduplicates by (label, seed) -- the sigma=1.5 arm was extended in place, so
its first 20 seeds appear twice in the table and would otherwise be
double-counted.

### Per-turn efficiency metrics

Three metrics added to the results notebook, all per turn (a turn is one clue):
own cards achieved, card advantage (own minus opponent), and reward under the
game's own scoring (+1 own, -0.2 neutral, -1 opponent, -10 assassin).

    model              own/turn  advantage/turn  reward/turn
    centroid               1.20            1.08         0.80
    sigma=1.0              1.48            1.24         0.85
    sigma=1.25             1.43            1.23         1.01
    sigma=1.5              1.33            1.17         1.01
    sigma=2.0              1.17            1.10         1.05
    sigma=2.5              1.07            1.04         0.98
    learned listener       1.65            1.55         1.50

These separate two things win rate conflates.

**Across the sigma ladder the three metrics disagree, and that is the finding.**
Own cards per turn falls monotonically with sigma (1.48 -> 1.07) because higher
sigma means smaller k. But reward per turn is flat at 0.98-1.05 for every sigma
from 1.25 up: the extra words a bold setting wins are almost exactly cancelled
by the opponent words and assassins it also hands over. sigma=1.0 is the one
clear loser on reward (0.85) despite the second-highest productivity, because
its 14% assassin rate is charged at -10.

So the ladder's win-rate differences (69/70/70/57/51) are NOT explained by
per-turn reward, which barely moves. They come from tempo: a turn taking 1.48
words finishes the board in fewer turns than one taking 1.07, and the race is
what decides the game.

**The learned listener is the only model that separates on all three.** 1.65
own per turn (the most productive of any model), 1.55 advantage, and 1.50
reward -- half again the best sigma setting on reward, where every sigma value
sits within 0.07 of every other. It is not trading productivity for safety; it
has more of both at once.

### The theory paper's results section, corrected

`docs/clue-selection-theory.tex` chose sigma = 2.5 from a per-turn reward
sweep over 100 positions, and that section is now followed by one reporting the
full-game result, which nearly reverses the ordering: 69/70/70/57/51 percent
for sigma = 1.0/1.25/1.5/2.0/2.5 against centroid over 100 games each.

The original sweep is kept rather than rewritten. It was not measured badly and
the paper is more useful with the disagreement visible than with the losing
answer quietly deleted. What the new section adds is why the two disagree:
per-turn reward is flat (0.98-1.05 for every sigma from 1.25 up) because a bold
setting's extra words are cancelled by the opponent words and assassins it
concedes, so it cannot discriminate, and the peak at 2.5 was inside that
flatness. Words per turn is what varies, 1.48 down to 1.07, and Codenames is a
race -- a per-turn proxy prices a turn in isolation and is blind to how many
turns the game will last.

The paper now takes sigma = 1.5: the cautious end of the win-rate plateau,
since 1.0 and 1.25 win no more often while hitting the assassin more than twice
as often.

**Not changed: `configs/spymasters.json` still sets sigma = 2.5 for the
`expected_words` entry.** Changing it would silently redefine what "the
baseline" means for every result already recorded against it, which
CLAUDE.md's iteration rules forbid doing quietly. It needs its own
`docs/versions/` entry, and that is a separate decision.

## Does the Plackett-Luce assumption actually hold for this model?

Shane's question: we compute logits once over the unrevealed board and drop
revealed rows from the softmax, but the GBT is a black box whose output can
move sharply when inputs change -- so a renormalised softmax is not the model's
true conditional. Correct, and now measured.

Seventeen of the 44 features change when a word is removed: all three
`gaptop_*`, `n_candidates`, `peak_z`, `lead_margin`, `p_max_sigma2`, both
`cohesion` features, `pmi_gaptop`, `swow_rev2_share`, `swow_asym`, and the four
rank columns. Removing the top-scoring word and comparing the renormalised
frozen logits against a genuine re-extraction, over 1,400 positions:

     words left     n  same top-1  Spearman       KL   max dp  logit shift
     4-6          175      84.6%    0.8868   0.0162   0.0577        0.492
     7-9          189      82.0%    0.9208   0.0157   0.0487        0.511
    10-13         271      85.6%    0.9477   0.0158   0.0425        0.513
    14-18         308      78.6%    0.9581   0.0161   0.0368        0.523
    19-24         387      78.0%    0.9672   0.0158   0.0337        0.512

So IIA fails: one removal flips the model's favourite word about a fifth of the
time, which a true Plackett-Luce model can never do, and logits move 0.51 on
average. The ordering is more disturbed on small boards (Spearman 0.967 ->
0.887) while KL stays flat.

**This does not mean the spymaster is stale.** It re-extracts features every
turn -- `_score_all_clues` builds candidates from the currently unrevealed
words -- so between turns its scores always describe the board in front of it.
The frozen assumption applies only to the k-step lookahead that prices a clue's
expected reward, and at k=1 it does not apply at all, since a single step needs
no lookahead. Late turns are almost entirely k=1 (mean k 1.19 by turn 5, 1.07
by turn 6), so the approximation bites least exactly where the board is
smallest, and most on turn-1 clues at k~3.

**The artifact is a different story, and this is where the observed
deterioration comes from.** It ships one scoring of the full 25-word board and
renormalises for the whole game, across turns as well as within them. Measuring
that directly -- full-board scores restricted to a shrunken board, against
scores computed for that board:

     words left     n  same top-1  Spearman       KL
     4-6          166      71.7%    0.7676   0.0520
     7-9          169      74.6%    0.8623   0.0414
    10-13         219      81.3%    0.9106   0.0367
    14-17         212      83.5%    0.9417   0.0217
    18-20         134      91.8%    0.9575   0.0148

Monotone, unlike the single-removal table: nearly exact at the opening, wrong
about the best word 28% of the time by the endgame. So the webpage's late clues
really are worse than the model's, and an earlier conclusion here -- that
refreshing would not help late -- was right about the deployed model and wrong
about the artifact.

**Some late-game decline is genuine and affects every spymaster.** Across
recorded games the best similarity available anywhere in the clue pool falls
12.47 -> 10.51 by turn 6, because the easily-clued words go first and the
remainder is harder to point at; and k is capped by own words remaining, which
is down to 1.4. Per-guess accuracy actually *rises* over a game (83.7% -> 100%
for learned_listener). What falls is productivity, not correctness.

`train_listener.py --refresh-features` re-extracts at every step rather than
slicing one full-board matrix, so the training-side version of this question
can be measured; that run is in progress.

## Local play server, and the k=1 rule in the artifact

`scripts/tools/play_server.py` serves one page and calls
`LearnedListenerSpymaster` directly, so there is no port and no fixed board
pool: boards are generated at random from the full word list and each clue is
the one the arena would see. Stdlib `http.server` rather than Flask -- three
endpoints, and no web framework is in the dependencies. Measured 0.5-1.1 s per
clue, 1.1 s to load the model.

State lives in the client, not the server: the page sends back the seed and the
revealed words, and `Board.generate(seed)` reconstructs the position exactly.
Refreshing mid-game loses nothing and two browsers can play the same seed.
`k1_max_similarity` is ON here (`--no-k1` disables it) and stays off in
`configs/spymasters.json`, where flipping it would redefine the baseline under
every recorded result.

The artifact got the same rule. It cannot be computed in the browser the way
Python computes it -- `_swap_k1` scans the whole 11,145-clue pool by raw
numberbatch cosine, not the Gaussian shortlist -- but it does not have to:
legality is fixed by the board, so the rule's answer is one clue word per board
word and `export_board_inputs.py` now resolves it exactly and ships it as `k1`.
Checked against Python on six forced-k=1 positions: all six identical
(shark/boundary/consecutive/buildings/hat/dying). The base clue differs on two
of the six because the browser's candidate pool is a subset, which is the
port's existing approximation, not a new one.

Shipping the k=1 table meant re-exporting, and the previous `--top` was not
recorded. At the default 260/side the file came to 24.7 MB, over the 16 MB
per-file artifact limit; `--top 145` gives ~250 clues per board and 13.2 MB,
a wider pool than the 208/board that was previously shipped.

## 2026-09-20 — role costs: the incumbent wins, and the sweep only looked one way

Eleven cost settings played the incumbent (neutral 0.2, opponent 1.0, assassin
10.0) head to head, 100 boards each both ways, gpt-oss-120b guesser, 1,826
games in 115 min for about $2. **Nothing beat it.** Ten of eleven lost:

    ass=5    56.9%  p=0.052      neu=0.7  43.3%  p=0.027
    neu=0.4  49.5%  p=1.000      neu=1    42.9%  p=0.023
    neu=0.1  49.4%  p=1.000      opp=2.5  42.7%  p=0.024
    opp=1.5  44.7%  p=0.052      opp=3    41.8%  p=0.009
    opp=2    44.0%  p=0.080      ass=20   36.8%  p=0.000
                                 ass=40   34.7%  p=0.000

The prior going in -- an opponent card is a two-card swing, so 1.0 underprices
it -- is not supported. Raising it is monotonically worse (44.7 -> 41.8 as the
cost goes 1.5 -> 3.0), and every penalty increase on every axis hurts. The
gradient points at a *more aggressive* model, not a more careful one.

The mechanism is doing what it should, which is the check that says the
parameter is wired up: mean k runs 2.31 at ass=5 down to 1.70 at ass=40, and
own/clue tracks it.

**The sweep bracketed opponent on one side only** -- 1.5/2.0/2.5/3.0, nothing
below 1.0, while neutral and assassin both straddle the incumbent. So the one
direction the data points to, on the axis the question was about, is the one
never tested. Next run: opp 0.5/0.75, ass 2/7.

Two reasons not to bank any of it yet. `ass=5` is p=0.052 and the best of
eleven comparisons, where ~0.6 false positives are expected. And the discard
question below is unresolved, worst exactly where the effects are largest.

### The guesser degenerates, and the retry does not help

gpt-oss discarded 6-32 boards per setting. It is not a refusal, a content
filter, or truncation -- finish_reason is `stop` and the JSON is well formed.
It repeats one token instead of ranking:

    clue 'gross', 20 words -> ["gross","gross","gross", ... ]  (19x)

`_query` retries once at double the token budget on the documented assumption
that truncation is the only failure. It isn't, and no temperature or seed is
set, so attempt two resamples identically -- visible in the errors, where both
attempts report the same count ("named only 1/19" twice). Doubling the budget
is a no-op for this failure.

Mostly transient: 'loan' and 'treasure' both returned full rankings when
retried later, though 'gross' reproduces.

**The censoring is arm-specific, which is the shape that biases a comparison.**
Of 61 boards lost in at least one arm, *zero* were lost in all seven then
complete -- if these were simply hard boards, every arm would lose the same
ones. And corr(boards lost, |win% - 50|) = +0.82 over 7 settings: the arms with
the most discards have the most extreme results. That is not proof of bias --
a setting that changes play a lot plausibly produces both more unusual clues
and a genuinely different win rate -- but n=7 cannot separate the two.

Cheap to settle: the cache key is (model, clue, candidates, number) and does
**not** include temperature, so a re-run after fixing the retry replays every
successful call from cache and only re-queries the ~10-20% that degenerated.

## 2026-09-20 — the k=1 rule becomes a tiebreak, and acronyms leave the pool

**The max-similarity rule was unsafe and is gone.** It took the
highest-raw-cosine legal clue for the word the model meant, which ignores the
rest of the board entirely. The failure is concrete: with CHICK and EAGLE both
up and EAGLE the last word needed, the most obvious clue for EAGLE in
isolation is BIRD, which hands CHICK to whoever owns it. Raw similarity cannot
see CHICK.

`k1_tiebreak` replaces it. Candidates must first come within
`k1_tie_tolerance` of optimal under the full board-aware expected reward --
which is computed over every remaining word and its role -- and similarity
only breaks ties among those. A clue that also points at CHICK is marked down
by exactly that much and leaves the tie set before similarity is consulted.
The tolerance is therefore not a tuning knob so much as a statement of how
much expected reward we will spend to be more obvious.

**Tolerance 0.1, after getting it wrong at 0.5.** Over 18 forced-k=1 positions:

    tol    changed   mean spend   max spend
    0.1      9/18       0.0302      0.0934
    0.25    10/18       0.0648      0.2190
    0.5     11/18       0.1146      0.3782
    1.0     11/18       0.1146      0.3782
    2.0     11/18       0.1146      0.3782

0.5 was the first default and it was wrong. The unit is own-words and the
best k=1 clue scores 0.96 on average (0.845-0.997), so a k=1 turn is worth
about one own word and the tolerance has to be read as a share of that:

    tol    fires    worst spend    as % of the turn
    0.1     9/18       0.093             9.5%
    0.25   10/18       0.219            23.2%
    0.5    11/18       0.378            37.9%

Clues 38% apart in expected reward are not tied, and swapping between them is
not a tiebreak -- it is choosing a materially worse clue because it reads
better. The argument originally given for 0.5 was that the effect saturates
there, 1.0 and 2.0 changing nothing more; that is an argument against going
higher and says nothing about 0.5 against 0.1. 0.1 still fires on half the
positions, so the rule does what it exists for, and caps the damage at a tenth
of a turn. A test now asserts the default stays at or under 0.15.

Larger tolerances also widen the door to the failure the rule was written to
avoid: on a contrived board with EAGLE against CHICK/HAWK/DUCK, off gives
BALD, 0.5 gives PATRIOT, and 2.0 reaches OWL. The regression test asserts that
progression.

A first attempt to measure the spend read the score off the swapped array,
where `_swap_k1` has inflated it by +1 to outrank the incumbent, so the
subtraction cancelled to 0.000 by construction. The real numbers come from an
independent un-inflated scoring pass.

### Acronyms

The model played `phd` 29 times, `uk` 29, `nasa` 21, `gm` 20, `rn` 14,
`hsbc` 8. All legal one-word clues, all bad at a table.
`scripts/data/build_acronym_mask.py` flags 9,128 of 111,440 vocabulary entries
(4.2% of the admissible pool after the rarity filter).

Detection reads WordNet's *casing*, which is the only surviving signal that a
word is conventionally written in caps: flag when every lemma spelling is ALL
CAPS or dotted (CIA, PhD, U.S.A.), plus a no-vowel test for `gm`/`hsbc` and a
short-and-absent test for `nba`/`wwe`. Three earlier attempts failed and are
worth recording. Requiring *any* caps spelling flagged `cat`, `pet`, `zip`,
`shape` and `led`, all of which carry an acronym sense beside the ordinary
word. Using WordNet membership alone flagged `australia`, `germany`, `limbs`
and `gods` -- proper nouns and plurals. Testing vowels without `y` flagged
`rhythm`, `sky`, `myth` and `fry`. The final rule leaves `laser`, `radar` and
`scuba` alone, lexicalised and lower-cased in WordNet, and its residual false
positives are short names WordNet lacks (`jill`, `joey`) plus `the` and
`when`, which are no loss.

**It is a pool restriction, not a rule of Codenames** -- applied like
`max_rarity` rather than in `is_legal_clue`, because legality is identical for
every model and every recorded result was produced under the current
definition. Moving this into the rules would silently redefine what those runs
measured. `exclude_acronyms` defaults to True, so this does change the shipped
model; it needs a docs/versions entry before it is treated as the baseline.

## 2026-09-22 — an absolute anchor: PASS is inert, decoys work

The listener's softmax normalises over board words only, so a clue is scored
on the ranking it induces and never on how confidently it induces it: z=1 over
four own words and z=3 over the same four are indistinguishable to it. Two
candidate anchors were measured before committing to a re-collection, on 40
boards x 3 clue grades, with the grades verified genuinely far apart (Gaussian
percentile 99.65 / 97.19 / 48.83, the last with *negative* expected reward).

**PASS -- asking the teacher where it stops recognising a connection -- is
dead.** Correlation with clue quality r = -0.068, p = 0.47. The damning
figure: on clues with negative expected reward gpt-oss claims to recognise 7.4
of 25 words; on strong clues, 8.5. It cannot separate its own best pick from a
random word by introspection, which was the stated risk of asking a model to
report its own uncertainty.

**Decoys work, but only at the top of the ranking.** Real board-vocabulary
words that are not on this board, mixed into the candidate list. At 5 decoys
the averages were flat (best-decoy rank r = +0.007) and only the rare event of
a decoy outranking every board word discriminated (0% / 8.6% / 18.4%,
p = 0.0122). At 15 decoys that becomes a strong signal:

    grade     decoy wins   in top3       (chance: 37.5%, 1.12)
    top            9.5%      0.62
    mid           25.0%      0.86
    random        56.1%      1.17

    decoy_wins      r = -0.305  p = 0.0014
    decoys_in_top3  r = -0.269  p = 0.0016
    top_decoy       r = -0.002  p = 0.985     <- the average, still inert

The flat average has a structural cause worth keeping: a decoy competes with
the ~21 *unrelated* board words, not with the intended ones, so its typical
rank tracks the size of that soup rather than the clue. Clue quality only
moves the top few positions. "An off-board word beat everything on the board"
is the outside-option event we wanted, reached behaviourally rather than by
introspection -- which is precisely why it survives where PASS did not.

Confound recorded: 56.1% sits *above* the 37.5% chance line, which a neutral
clue should not. `is_legal_clue` removes clues sharing a stem with any board
word, so a random *legal* clue is mildly anti-selected away from the board.
That inflates the random grade. The top grade (9.5% against 37.5%) is
unaffected and is the number to quote.

Not yet decided, and the reason nothing has been collected: how many decoys to
use at training versus inference time, and whether a decoy win should be
modelled as the turn ending at cost 0 or as the guesser picking on regardless
-- a human does not pass, they guess wrong, so cost 0 may understate it.

## 2026-09-22 — role costs, measured cleanly: the incumbent stands

The sweep re-run on the acronym-free model, with the guesser's retry fixed and
the arm-specific censoring gone -- discards fell from 6-32 boards per setting
to a mean of 2.6. 15 settings, 100 boards each both ways.

    ass=2    54.5%  p=0.188      opp=1.5  46.4%  p=0.248
    ass=5    54.0%  p=0.185      opp=2    45.5%  p=0.188
    opp=0.5  52.2%  p=0.608      neu=1    44.8%  p=0.099
    ass=7    50.5%  p=1.000      opp=2.5  44.4%  p=0.071
    neu=0.1  50.0%  p=1.000      neu=0.7  44.3%  p=0.061
    opp=0.75 49.0%  p=0.832      opp=3    43.4%  p=0.029
    neu=0.4  48.0%  p=0.503      ass=20   37.1%  p=0.000
                                 ass=40   36.6%  p=0.000

**Nothing beats the incumbent.** The best challenger is p=0.188 and is the
best of 15 comparisons, where ~0.8 false positives are expected.

The opponent axis now has both sides and is monotone across all six values:
52.2 / 49.0 / [50 by definition] / 46.4 / 45.5 / 44.4 / 43.4 as the cost goes
0.5 -> 3.0. The prior that an opponent card is underpriced at 1.0 is not just
unsupported, it is backwards -- and the other direction, which the first sweep
never tested, buys 2.2 points at p=0.61. There is nothing here.

Fixing the censoring shrank the effects, as the +0.82 discard/effect-size
correlation warned it would: ass=5 read 56.9% at p=0.052 under the biased run
and 54.0% at p=0.185 clean. Acting on the first run would have been acting on
the bias.

One result worth keeping for later. `ass=2` walks into the assassin **32 times
against the incumbent's 6** -- more than five times as often -- and still
finishes ahead on win rate. Cheap assassin insurance buys tempo that mostly
pays for the losses. It is a genuinely higher-variance strategy rather than a
straightforwardly better one, and win rate alone hides that.

The mechanism check holds throughout: mean k runs 2.50 at ass=2 down to 1.72
at ass=40, own/clue with it.

### The decoy anchor is incremental -- the case is made

Two further probes closed the question the three-grade result left open.

**Within the shortlist**, where the confound is gone by construction (80 boards
x 4 clues drawn from the listener's own top 50, 271 usable positions):

                        n   top pick OWN   ends turn   assassin
    decoy won          52          63.5%       36.5%       3.8%
    no decoy won      219          90.9%        9.1%       0.5%
                                        Fisher p = 4.6e-06

Among clues the deployed model would actually play, a decoy win costs 27
points of accuracy, quadruples the chance the turn ends, and multiplies the
assassin rate by 7.6.

This mattered because the pooled three-grade figure (34.3% vs 73.5%,
p = 0.00014) was partly measuring the grade: decoy wins are common on random
clues and random clues fail anyway. Broken out, the effect was absent within
`random` (27% vs 22%) and rested on n=4 within `top`.

**And the anchor is incremental, which is the whole case.** Against the
listener's own predicted P(pick is own):

    decoy won      model predicted 73.1%, actual 63.5%   (gap -9.7%)
    no decoy win   model predicted 86.0%, actual 90.9%   (gap +4.9%)

    within one band of model confidence:
      0.50-0.70   decoy won 47% actual  |  no decoy win 73%
      0.70-0.85   decoy won 69% actual  |  no decoy win 89%

The model is overconfident precisely where a decoy wins and slightly
underconfident elsewhere -- a ~15 point swing in calibration error that it
cannot currently see. Had decoy_wins merely tracked the model's own score,
there would be nothing to collect.

Thin cells worth naming: the 0.95+ band has n=3 on the decoy-won side (33%
actual, dramatic and unreliable) and the sub-0.5 band has n=5/6. The evidence
is the 0.50-0.85 range, where n is adequate.

Open, and the reason no collection has started: how many decoys at training
versus inference, and whether a decoy win should end the turn at cost 0 or
have the guesser pick on regardless. A human does not pass, they guess wrong
-- and the 36.5% turn-ending rate above suggests cost 0 understates it.

## The decoy cut-point was chance, and the signal is the first pick alone

The previous entry left decoy collection blocked on two questions. Answering
them retracted a headline number from it.

**Retraction: "mean cut 1.89 words" was not a finding.** It was read off
`probe_shortlist.json`, which is D=15. With 15 decoys among 25 board words, a
uniformly random ranking already puts 25/16 = 1.56 board words before the
first decoy. Observed: 1.63. Ratio 1.05 -- chance. The cut was then compared
against the model's mean announced k of 2.2-2.5 and read as "the model claims
more words than the guesser can find." That comparison put a D-dependent
statistic against a D-free one, and the D-dependence was essentially all of
it. Nothing about announced k was measured.

**What is real is confined to the first pick.** Per-position hazard against
the depletion-implied chance rate, 272 shortlisted boards at D=15:

    pos    P(decoy) obs     null    obs/null
      0            0.195    0.375      0.52      z = -6.1
      1            0.367    0.380      0.97
      2            0.354    0.380      0.93
      3            0.408    0.381      1.07
      4-7          ~0.39    ~0.38      0.95-1.14

And it tracks clue quality: the listener's own best shortlisted clue puts a
decoy first at 0.36 of chance, the worst at 0.58.

So gpt-oss at low effort identifies one word and is at chance thereafter.
That is consistent with the earlier `decoy_wins` results (r = -0.305;
63.5% vs 90.9% own) -- it just locates the effect, and narrows the claim: the
anchor informs *whether the top word is findable*, not *how many* are.

**D dilutes the signal, as predicted, and cannot be chosen from the data.**
Subsampling the 15 collected decoys down (valid under Luce/IIA):

    D'     1     2     3     5     8    10    15
    O/N  0.43  0.45  0.45  0.47  0.48  0.49  0.52
    gap  0.22  0.28  0.26  0.25  0.25  0.24  0.22

More decoys, more chances one relates to the clue by accident and outranks
the word the clue meant, so the deficit from chance erodes. But `obs/null` is
feature-blind and the fitted model is not -- a decoy that wins on genuine
relatedness has similarity features saying so -- and the discriminative gap is
flat from D=2 to D=10. The curve cannot settle D, and picking a compromise
would hide the assumption rather than test it.

**So D is randomised over {2,5,10}, which makes IIA measurable.** Under IIA
the fitted level is a property of the clue and must not move with D.
`scripts/tools/analyze_decoy_invariance.py` subsamples the collected D=10
positions down to 2 and 5 and compares against positions actually collected
at 2 and 5 -- same clue distribution, same prompt, differing only in whether
the decoys were absent when the teacher answered or removed afterwards. A
systematic gap is IIA failing, which would invalidate reading a level off
decoys at all.

**Collection design** (`scripts/data/collect_decoy_data.py`, 1,704 positions):
the clue number is GIVEN and randomised over 1-5, because `k` is already a
feature and -1 ("no number") is a regime the spymaster never operates in.
Rankings are truncated at and including the first decoy; below that line
own-rate is 33.2% against a 36% base rate, i.e. zero information. Decoys are
screened against the board via `is_legal_clue` but never against the clue --
screening on clue relatedness makes the reference distribution clue-dependent,
which destroys the fixed anchor and would have to be reproduced at inference
using the same embedding similarity the model is judged against.

It buys through the ordinary response cache under the real guesser prompt.
The cache key is (model, clue, candidates, number) and decoys are part of
`candidates`, so these cannot collide with existing rows. `probe_anchors.py`
had to bypass the cache because it asked a different question under the same
key; this asks the same question with a longer candidate list.

**One blocker dissolved.** Whether a decoy win means the guesser stops or
guesses wrong anyway affects only the reward computation -- the PL likelihood
needs nothing but the observed orderings. It does not block collection and is
revisable without re-collecting.

**Still open:** what the outside option *means* at inference, where there is
no off-board choice. Either a scale reference ("how strongly does this clue
point at anything"), under which a lucky decoy is legitimate evidence; or the
mixing weight on a "guesser is lost" component of the reward, under which it
is contamination. The two want different treatment of exactly the case above,
and the reward code has to pick one.

## The decoy level is invariant to D, so D can be chosen for efficiency

1,679 positions collected (25 guesser refusals, 1.5%, discarded).
`scripts/tools/analyze_decoy_invariance.py`:

    D    source        n     O/E    95% CI
    ---------------------------------------
    2    collected   588    0.50   [0.37,0.66]
    5    collected   535    0.50   [0.40,0.60]
   10    collected   556    0.50   [0.42,0.59]

    2    from D=10   556    0.48   gap vs collected -0.02
    5    from D=10   556    0.48   gap vs collected -0.02

O/E is observed decoy-first events over the sum of per-row chance rates, which
handles the varying board size; 1.00 is no information.

**Both halves came out clean.** The level is the same at every D -- 0.50, 0.50,
0.50 -- so mixing in more decoys does not move where the teacher's boundary
sits. And subsampling D=10 down reproduces the collected figures to within
0.02, far inside the intervals, so removing decoys after the fact gives the
same answer as never having shown them. Decoys do not interact.

**This settles how to pick D, which the earlier subsample curve could not.**
Contamination from accidentally-related decoys is a real mechanism -- the
probe's subsample showed observed/chance drifting 0.43 to 0.52 from D=1 to
D=15 -- but over D in [2,10] it does not bite: the level is flat and only the
precision changes, with the CI narrowing from 0.29 wide at D=2 to 0.17 at
D=10 for the same number of positions. So D=10 is both unbiased and the most
efficient of the three, and the choice costs nothing. This should NOT be
extrapolated past 10; the probe's drift was measured at 15 and the range
above 10 is untested here.

**Announced k does not move the boundary either.** Mean cut by k over the
first 444 rows: 3.46, 3.50, 3.46, 3.58, 3.65 for k=1..5, with the ratio to
chance flat at 1.13-1.24. Telling the teacher to name more words does not make
it name more real ones. Two consequences: the boundary is a property of the
clue and board rather than of the prompt, which is the k-analogue of the
invariance above; and the earlier justification for randomising k -- "so the
fit observes positions past where the teacher stops knowing" -- was wrong, as
the teacher stops in the same place regardless. Randomising k remains
necessary because `k` is a feature and the fit must not extrapolate at
inference, but it buys no observations past the boundary.

Overall cut 3.52 against a null of 2.99, ratio 1.18 -- genuinely above chance,
unlike the D=15 probe's 1.05. The difference is that these boards are
partially revealed and carry fewer decoys.

**Truncation is an adapted stopping time**, so dropping the tail after the
first decoy costs no validity, but it does keep more terms when the clue is
good (decoy late) than when it is bad. That is an implicit weighting toward
good clues in the likelihood; weighting rows by 1/terms would remove it if it
turns out to matter.

Next: fit the scorer with these rows in the PL denominator and check that the
fitted level, not just the nonparametric O/E, is invariant to D.

## The fitted level is invariant to D too, and the scale is ~3.4 nats

`scripts/tools/fit_decoy_level.py`, decoy positions only (1,176 train / 503
val, split by board seed; 5,286 / 2,291 PL groups truncated at the first
decoy):

      D     n    level (nats)      predicted   observed    gap
      2   169     -3.46 +-0.11        0.071      0.071  +0.000
      5   174     -3.52 +-0.10        0.141      0.144  -0.003
     10   160     -3.35 +-0.12        0.241      0.244  -0.003

    val McFadden R2 0.1901   top-1 0.315   best iter 388

`level` is the mean decoy score minus the best board score. Flat across D --
spread 0.17 nats against standard errors of ~0.11, and not even monotone --
so the fitted scale is a property of the clue and board rather than of how
many decoys were mixed in. `D` is not a feature, so nothing forces this.

The calibration columns are the stronger statement: one fitted function
reproduces the observed decoy-first rate at 2, 5 and 10 decoys to within
0.003, without being told which regime it is in. That is invariance in the
form the spymaster will actually consume.

**The number this was all for.** A word drawn at random from the vocabulary
sits about 3.4 nats below the board's best word -- roughly 3% of its rate in
the softmax. That quantity was previously unidentified: the board-normalised
softmax is shift-invariant, so nothing in the old training signal pinned it.
It is now estimated.

**Two corrections made on the way.** The first `level` statistic was
logsumexp(decoy scores) - best board score, which reported -2.60/-1.58/-0.67
and looked like a large trend. It is not a defect in the fit: the total decoy
mass grows like log(D) whatever the model does, and logsumexp is dominated by
the largest of D draws, so even dividing log(D) out leaves an extreme-value
term. A statistic carrying either cannot test invariance. The mean has no
such term. Separately, `n_candidates` is overwritten with the board count
before fitting -- extract() computes it over whatever list it is given, so
leaving it alone would let the model read D straight off a feature and make
the whole test vacuous.

R2 0.19 against the deployed model's ~0.35 is expected and not a regression:
this is fitted on 1,176 positions where the deployed model has ~9k, and on
truncated rankings. It is a measurement instrument, not a candidate model.

Next: retrain the deployed listener with these rows added to the existing
positions and check the level survives at full data, then decide how
`pl_reward` consumes it -- scale reference or lostness weight (previous
entry).

## Decoys at full data: the level holds at ~4.1 nats and boards do not suffer

`train_listener.py --decoys` now mixes the collected positions into the
distillation. Both runs below use the gpt-oss teacher and an IDENTICAL board
validation split (see the split note further down).

    run                     board acc (all steps)   step-1
    baseline, no decoys                    0.4766   0.6755
    + 1,679 decoy positions                0.4781   0.6713
    numberbatch z alone                    0.4340   0.6521

    R2 (decoy run)   boards 0.3559   decoys 0.2046   pooled 0.3147

Board accuracy is unchanged -- +0.0015, noise. That is the safety check the
whole integration needed: decoys buy a quantity the model could not previously
represent without costing anything on the task it already did.

Level invariance on the deployed-size model, validation only:

      D     n   level (nats)    predicted   observed
      2   142     -4.09 +-0.15       0.055      0.035
      5   118     -4.14 +-0.16       0.123      0.127
     10   159     -4.09 +-0.15       0.223      0.226

Flat to 0.05 nats against standard errors of 0.15. Calibration is near-exact
at D=5 and D=10; the D=2 arm over-predicts by 0.020, which is ~1.3 SE on about
5 events and is the low-D arm being noisy exactly as the CI widths predicted.

**The level is ~4.1 nats, not the 3.4 from the decoy-only fit.** A model that
also sees 8,869 board positions scores board words higher relative to a random
word, so the gap widens. 4.1 nats is about 1.7% of the best board word's rate.
The decoy-only figure was a measurement instrument on 1,176 positions; this is
the number that should be quoted.

**Three pipeline decisions worth naming**, all of which would have quietly
produced a wrong answer:

1. *The board and decoy splits are drawn separately*, so the board half is
   byte-identical whether or not `--decoys` is passed. Pooled, adding 1,679
   seeds changes the draw for all of them -- the first attempt at this
   comparison reported 0.4766 -> 0.4716 and that -0.005 was partly a different
   validation set. Drawn apart it is +0.0015.
2. *Decoy positions are exempt from the 0.75**j step decay.* The decay exists
   because the teacher's ranking tail is arbitrary; truncation already removes
   that tail, and the decoy term sits at the deepest kept step, so the decay
   would fall hardest on the one observation each position was collected for
   (mean cut 3.5 weights it 0.75**3.5 = 0.37).
3. *`n_candidates` is overwritten with the board count.* extract() computes it
   over whatever list it is given, so otherwise the model reads D straight off
   a feature and the invariance check above is vacuous.

Also noted: `DEFAULT_MODEL` in train_listener.py is `claude-sonnet-5`, but the
store holds 62k gpt-oss responses against 10k sonnet, and the arena guesser is
gpt-oss. Both runs here passed `--model deepinfra/openai/gpt-oss-120b+effort=low`
explicitly. Distilling sonnet while playing against gpt-oss would be modelling
the wrong guesser; the default looks stale and should be settled.

**Not deployed.** `cache/listener_gbt.txt` is untouched; both models are in the
scratchpad. Shipping this is a new model under the naming convention -- a
`docs/versions/` entry and a `configs/spymasters.json` entry -- and it has not
been played in the arena yet, which is the only test that matters. The open
question from the previous entry still decides the reward: scale reference or
lostness weight.

## The outside option in the reward: a non-team word that costs nothing

Decided (Shane): if the guesser hits the outside option its EV is zero --
equivalent to passing. That makes the implementation exact rather than
approximate, because `pl_reward` already has the right shape for it: the
outside option is a non-team word with cost 0. `gain_and_penalty(...,
s_out=)` appends it to `s_bad` with a zero cost, and omitting it is
bit-identical to every run made before it existed.

Two effects, both intended:

  * `big_lambda` rises, so the turn ends sooner and `gain` falls -- a vague
    clue no longer gets promised four words.
  * `cbar` falls, because some endings are now free -- so a vague clue is
    driven toward ZERO reward rather than toward a large penalty.

**The weak flank, recorded rather than hidden.** Because the outside option
competes with the assassin for the same probability mass, a clue that mostly
pointed at danger now mostly points at a harmless nothing, and its expected
reward goes UP. That is a direct consequence of costing it at zero, and it
runs against the measurement: in the probe, turns where a decoy won had 7.6x
the base assassin rate, so a guesser wandering off is more dangerous, not
less. The reward is therefore optimistic for exactly the clues it should fear
most. `tests/test_pl_reward.py::test_a_DANGEROUS_clue_is_worth_MORE_once_wandering_is_priced`
pins the behaviour so it cannot drift silently, and it is the argument for
eventually pricing the outside option above zero.

**`outside_n` is a free parameter, not a measurement.** The decoy training
identifies the LEVEL of an outside word (~4.1 nats below the best board word)
but not how many such alternatives a game contains -- a real game contains
none, since the guesser must pick from the board. So it joins `sigma` and the
role costs as something swept against play. It matters a lot:

    outside_n        0     1     5    10    25
    same clue as 0  12/12 11/12 10/12 9/12  5/12

At 1 the outside option takes under 1% of the rate and barely moves the
argmax; at 25 it changes seven clues in twelve and drops the announced number
from 3-4 to 2, which is the risk-aversion the whole exercise was after. The
default is 0, so nothing changes until a sweep says what to set.

Inference draws the outside words UNIFORMLY from the vocabulary, matching how
training sampled them -- a top-similarity draw would be cheaper but would
measure a different quantity, since the level was estimated against uniform
draws. The draw is seeded from a blake2b of the board's words rather than
`hash()`, whose string seed is randomised per process: the arena builds a
fresh spymaster in every worker, so a process-dependent draw would make the
same board score differently from one worker to the next.

**A caveat on how the arena result should be read** (Shane): this change may
do worse against an LLM guesser and better against humans. Humans stop being
able to order words by similarity after a few positions; gpt-oss keeps
producing a full ranking of 30-40 words whether or not it knows anything --
measured above, its decoy-vs-board hazard is at chance from the second pick
onward. The outside option exists to model exactly the behaviour gpt-oss does
not exhibit. So a flat or slightly negative head-to-head against gpt-oss is
not a verdict on the idea, and should not be treated as one.

## The outside option loses to gpt-oss, monotonically -- and the mechanism works

`outside_n` swept against the incumbent (`outside_n=0`), both arms on the
decoy-trained booster, 300 boards each played both ways, gpt-oss guesser,
paired sign test on decisive boards (2,820 games, 84 min):

    setting   win vs base    95% CI      swept    sign p   assassin   own/clue  mean k
    out=1        48.1%    [0.44,0.52]   20-31     0.161    38 v 37     1.60     2.00
    out=5        47.9%    [0.44,0.52]   40-52     0.251    34 v 38     1.58     1.93
    out=10       44.7%    [0.41,0.49]   28-58     0.002    30 v 37     1.52     1.84
    out=25       36.0%    [0.32,0.40]   19-96    <0.001    22 v 23     1.45     1.68
    out=50       31.3%    [0.28,0.35]   20-123   <0.001    15 v 28     1.35     1.54

(assassin = challenger's assassin losses v the base's.)

**The control behaved.** `out=1` takes under 1% of the softmax rate and lands
at 48.1%, p = 0.16 -- indistinguishable from the null.

**The mechanism does what it was built to do.** Announced k falls
monotonically, 2.00 -> 1.54, and at `out=50` the challenger loses to the
assassin 15 times against the base's 28. Pricing "the guesser has stopped
knowing" makes the spymaster claim fewer words and walk into the assassin
less. The pl_reward test that pinned the "dangerous clue becomes MORE
attractive" flank describes a real property of the reward, but in play the
drop in k dominates it.

**And against this guesser it costs more than it saves.** Own words per clue
fall 1.60 -> 1.35, and the win rate falls with them, monotonically, reaching
significance at 10 (p = 0.002, inside a Bonferroni threshold of 0.01 for five
comparisons) and collapsing to 31% at 50. The swept-board margins at 25 and
50 (19-96, 20-123) are far too large for the unbalanced discards (3 boards at
out=1, 18-25 elsewhere -- arm-specific censoring, since refusals depend on the
clue) to explain.

**This is the same shape as the role-cost sweep.** There, every price that
made the spymaster more cautious lost to gpt-oss -- opponent cost monotone
worse from 0.5 to 3.0, assassin 20 and 40 the two worst results. Two
independent levers now say the same thing: gpt-oss rewards tempo over caution.
It ranks every word whether or not it knows anything (its decoy-vs-board hazard
is at chance from the second pick on), so a clue that promises three words
gets three guesses, and conservatism simply leaves words on the table.

**What this does not settle** (raised by Shane before the run): whether a
human guesser rewards the caution. A human stops when they stop recognising a
connection; gpt-oss does not. The outside option models exactly the behaviour
this guesser lacks, so a monotone loss here is evidence about gpt-oss, not a
verdict on the idea. Its case now rests entirely on human play, which the
arena cannot provide.

`outside_n` stays at 0. The decoy-trained booster itself is not tested by this
sweep -- both arms used it -- and is not deployed.

**Infrastructure, found on the way:** the run used 6 processes x 16 threads
pulling from a shared queue (codenames/two_team_arena.py), after three wrong
diagnoses of why it was slow. Fixed chunks left five of six workers idle on
one worker's stragglers; an unguarded first stage peaked at 13 GB for two
processes; and the evidence first cited against threads compared new
llm_store rows during a cache replay, which counts nothing. Measured CPU-bound
at ~15.6 of 16 cores, versus an estimated 12 hours on the old 14-process
setup.

## A blind one-clue study, because the arena cannot answer the outside-option question

The outside_n sweep left the idea's case resting on human play: it models a
guesser who stops when they stop recognising a connection, and gpt-oss never
stops. So `scripts/tools/play_server.py` now serves `/eval`: one position, one
clue from one of two spymasters, the human guesses under real turn rules,
next. `scripts/tools/analyze_human_eval.py` compares the arms.

Design choices, each protecting the data rather than the page:

- **The arm and the key never reach the browser.** Each pick is a round trip
  that returns that one card's role. Checked over 40 served positions: every
  payload has exactly {token, words, clue, number, revealed, recorded}, no
  value names an arm or a model file, and mid-turn picks return only {role}.
- **Arms are assigned in shuffled blocks**, so they stay balanced to within one
  position (tested over 40; the driven run came out 26 v 26).
- **Stop is a recorded outcome, and there is no skip.** Stopping is the
  behaviour under test. A skip would let people drop the clues they dislike,
  and if they dislike one arm's clues more that is arm-specific censoring --
  the same worry the guesser-refusal discards raised in the sweeps.
- **Positions span a game**: 0-8 non-assassin cards pre-revealed, at least two
  own words left, as the listener's training positions were.

Default arms are `decoy` v `decoy_out25`: the same booster, differing only in
the outside option -- the exact comparison the arena made. The game page also
gained a spymaster menu (the incumbent and the decoy booster at every sweep
value); the decoy variants share one loaded booster, so six options cost 1.5
GB for the server.

From 52 scripted test turns (random clicking, not data): the four metrics
behave, and **reward is the noisiest** (sd ~2.1, dominated by -10 assassin
hits), so a pilot will resolve on first-pick-own and stopping rate sooner. The
plan is ~100 positions, then fix the full sample size from the observed gap
BEFORE collecting more -- choosing it after peeking would inflate the false
positive rate.

## The eval guesser is Sonnet, not Opus

Shane's call, on cost: the frozen eval suite (`configs/eval_suite.json`,
`holdout_v1`) now points at `configs/guesser_pool_llm_sonnet.json`, and
`llm_model` is `claude-sonnet-5`. Opus is kept for the very last stage, when
final numbers are produced, and not before.

This reverses the choice made in "Sonnet vs. Opus as the evaluation guesser"
above, which kept Opus for defensibility even though 25 paired positions
showed no detectable difference (-0.04 own/turn, 95% CI [-0.22, +0.14]) at
about a fifth of the price. That measurement is what makes the switch
reasonable: it rules out "a lot worse", though not "slightly worse".

Nothing is invalidated. `llm_model` is part of `EvalSuite.suite_id`, so the
switch does change the suite's identity -- but the suite has never been run
(zero suite-tagged rows in `game_records`), so there were no recorded games to
orphan.

**One cost worth naming.** Sonnet is also the listener that `expected_words`'
sigma was chosen against, so that baseline is now evaluated on the guesser it
was tuned for. `docs/versions/expected_words.md` says so. The learned listener
is distilled from gpt-oss-120b, so for it Sonnet is still a transfer test.

`scripts/tools/compare_guesser_models.py` still defaults to comparing Opus
and Sonnet. It is a diagnostic tool, not the suite, and was left alone.

## Cleanup: the spymasters that predate centroid are gone

Removed the three simple spymasters that predate `centroid` (code, registry
entries, config entries, tests), per Shane: nothing current compares against
them, and the worklog now starts at `centroid`. `centroid` was the only
remaining user of `spymasters/_util.py`, so its two helpers moved into
`centroid.py` and the module is gone. Arena tests that used the random
spymaster as a stand-in now use `CentroidSpymaster`.

This drops one argument `docs/design-decisions.md` used to make: that a
random spymaster catches harness bugs that would make every model look
equally good. `centroid` now plays that role, less sharply, since a harness
bug could plausibly flatter a real strategy more than a random one.

References to the earlier trained models were removed from every doc and
docstring, and this log's entries from before the `z_threshold` baseline
were cut (git history keeps them).

## Cleanup, part two: what was unused, what was wasted, and a runnable benchmark

Removed, because nothing current reads them: four synthetic guessers (blend,
rank-based, confidence-threshold, history-aware) and their three pool configs,
the single-team arena, the GPU two-team arena and the batch-scoring protocol
that existed only for it, the orphaned inspector page (its server was deleted
long ago), and four one-off probes. Kept at Shane's request: the scratch
notebooks, and `clue_number_arc.py`, which one of them plots.

**The history-aware guesser's bookkeeping was spending real money.** After
every turn, `game.py` called `Guesser.update_history`, which re-asked the
guesser to rank each earlier clue that had ended on a miss, against the words
still on the board -- so a guesser like `HistoryAwareGuesser` could decide
whether an earlier clue still "owed" a word. Only that guesser ever read the
answer. Every other guesser, the LLMs included, paid for the call and threw
the result away. Counted in the store: every response with no clue number is
exactly this shape (an earlier clue, a strict subset of its candidates), and
there are **3,354 of 10,176 Sonnet calls (33%, roughly $16) and 20,442 of
88,077 gpt-oss calls (23%)**. No game outcome was affected -- the answer was
never used -- but sweeps spent about a quarter of their guesser calls on it.
The mechanism is gone; sweeps now make only the calls a turn needs.

**The frozen suite is now runnable, and plays what the sweeps play.** It was
built on the GPU arena as self-play (one spymaster on both sides) and nothing
called it. It is now a head-to-head matchup on the held-out boards in both
seatings (`scripts/pipeline/run_eval_suite.py`), each game stored under its
seating, so a board counts as done only when both seatings are recorded and a
rerun pays only for what is missing. A spymaster's identity hashes its
parameters and the bytes of its booster (`Spymaster.model_files`).

**Guessers are one spec string**, e.g. `anthropic:claude-sonnet-5:medium` or
`deepinfra:openai/gpt-oss-120b`, replacing one JSON file per model. Checked
before switching: the specs reproduce the stored cache identities exactly
(`claude-sonnet-5+effort=medium`, `deepinfra/openai/gpt-oss-120b+effort=low`),
so every past game still replays from cache. The suite's identity is now its
name plus that spec; it had never been run, so nothing was orphaned.

Consolidated: `codenames/stats.py` (Wilson, sign test, Fisher, bootstrap,
permutation), `codenames/headtohead.py` (pairing recorded games by board),
and `codenames/listener_training.py` (everything above `main()` in
`train_listener.py`, which four tools imported by resetting `sys.argv` and
putting the repo on `sys.path`). Scripts now import only the package, through
the editable install. `DEFAULT_MODEL` for training is now the gpt-oss teacher
rather than Sonnet, which every real run had to override by hand.

**One bug fixed on the way:** a `--resume`d role-cost setting reported
own/clue and mean k pooled over *both* arms rather than the challenger's.
Only resumed rows were affected; freshly played rows were always right.

`docs/design-decisions.md` said "the LLM guesser never appears in training".
That stopped being true when the listener was distilled from gpt-oss; the
guard that survives, and is now stated, is that the *evaluation* guesser is
never the teacher.

## Qwen3-8B as the whole teacher: scoring set-up

**Expected:** a local model gives every candidate's probability at every
step, so a listener could learn from full distributions rather than one
sampled pick. **Decision (user):** Qwen is the *whole* teacher. Scoring
along gpt-oss's pick order would smuggle gpt-oss into steps 2+, and fitting
a temperature to gpt-oss or Sonnet picks would make them part of the
teacher. So step j conditions on Qwen's own argmax at steps 1..j-1
(`--source own`), labels are used at T=1, and Sonnet is used only to
evaluate. A second scoring pass along Sonnet's actual picks (`--source
<sonnet id>`) exists only to score raw Qwen as a predictor of Sonnet.

**Calibration, measured before deciding anything:** on 3,026 first picks,
Qwen's top word matched gpt-oss's 75% of the time when Qwen was 99%+ sure,
39% at 90–99%, 49% overall at a mean confidence of 0.86, and gave gpt-oss's
actual pick a median 0.002 when they disagreed. Overconfident as a model of
another guesser, which is the reason to distil rather than use it raw.

**FP8 numerics.** Batching prompts of different lengths needs a padding
mask, and the mask switches PyTorch to a different attention kernel. On this
FP8 checkpoint that moved candidate log-probs by ~1 nat, even for the
unpadded row. vLLM, with its own kernels, differed from HF by up to 0.6 in
probability. The HF backend now batches only prefixes of identical token
length (bit-identical to the same computation run one request at a time, and
2× faster than before);
vLLM works but is not used, since one dataset must come from one set of
kernels.

**Throughput.** `--max-rows` bounds candidates per forward pass; each copies
the prompt's key/value cache. At 80: 5.6 positions/s. At 120: 1.3/s, peak
15.9 of 16 GB -- under WSL the driver spills GPU memory into system RAM
instead of failing, so too large a batch is silently 4× slower rather than
an out-of-memory error. 80 it is.

## Qwen as the teacher: the distilled listener is worse, and the distributions add nothing

**Expected:** a listener trained on Qwen3-8B's full distributions would at
least match one trained on its argmax, and might approach the gpt-oss-taught
incumbent on Sonnet's picks, since neither teacher is Sonnet.

**Set-up.** Same 24,915 positions, same features, same split, same training
recipe as the incumbent (`cache/listener_gbt_oss_recipe.txt`, gpt-oss
one-hot). Only the labels change: `qwen_soft` learns Qwen's full distribution
at T=1, `qwen_hard` only Qwen's argmax, both along Qwen's own pick path.
Scored against **observed** picks (`scripts/tools/compare_listeners.py`),
McFadden R^2:

| Model | Sonnet pooled | Sonnet step 1 | Sonnet steps 2+ | gpt-oss holdout pooled | gpt-oss step 1 |
|---|---|---|---|---|---|
| incumbent (gpt-oss labels) | **0.553** | **0.735** | **0.300** | **0.342** | **0.553** |
| Qwen soft | 0.476 | 0.653 | 0.230 | 0.229 | 0.426 |
| Qwen hard | 0.476 | 0.654 | 0.231 | 0.226 | 0.427 |
| Qwen raw, T=1 | -0.262 | -0.016 | -0.601 | -- | -0.582 |

Sonnet: 6,318 positions, 10,771 choice events. gpt-oss holdout: 1,837
positions, 4,671 events (raw Qwen at step 1 only there: only own-path
distributions exist for it).

**What it says.**
- The Qwen-taught listener loses to the gpt-oss-taught one by ~0.08 R^2 on
  Sonnet, the guesser neither was trained on. A bigger teacher transfers
  better; the free local teacher is not a substitute.
- Soft and hard are indistinguishable (0.4755 vs 0.4761 on Sonnet; 0.3778 vs
  0.3757 on their own validation labels). Qwen's distributions are so peaked
  (mean top probability 0.86) that they carry almost nothing beyond the
  argmax, so the "we get the full distribution for free" advantage does not
  materialise with this model at T=1.
- Distillation is what makes Qwen usable at all: raw, it is worse than
  uniform against either guesser, because it is confidently wrong; the
  booster, which can only reach the answer through the 44 features, is
  smoothed into a reasonable predictor.
- Aside: the incumbent predicts Sonnet (0.553) better than it predicts its
  own teacher's held-out picks (0.342). Sonnet's picks are the more
  predictable of the two.

**A bug found on the way (fixed before the numbers above).** The first
`qwen_hard` scored R^2 0.01: 102 of 62,301 steps had a label that summed to
zero. Cause: Qwen's logits are bf16, which near 40 resolve only 0.25, so exact
ties happen (Horse vs Horseshoe, both -0.7334). The scorer broke a tie one
way, and the loader recomputed "the argmax" and broke it the other, so later
steps' labels sat on a word training had already removed. Targets and the T=0
label now follow the path read off the stored data (`scored_path`), with a
test. The soft run moved by 0.0003.

**Numerical noise, checked while chasing it.** FP8 Qwen's cached-prefix
computation and a plain full forward pass give slightly different logits
(0.75 on one token here). Over 40 positions, stored step-1 distributions vs
full-pass ones: median total variation 0.000, mean 0.004, max 0.124, argmax
agreement 39/40. Negligible for training.

## Does the prompt change the teacher's picks? Four ways of asking gpt-oss

**Question (from the professor).** Every listener is trained on one prompt:
clue and number, "rank ALL the words", with the ranking's j-th entry taken as
the step-j pick. Entry 2 is an exact sample of "what the model names second,
having already named entry 1" -- but that need not equal "what it guesses when
entry 1 is gone", which is what play asks. The hope is that it does not
matter.

**Set-up** (`scripts/data/collect_prompt_variants.py`, analysis
`scripts/tools/compare_prompt_methods.py`). 300 gpt-oss holdout positions,
8 samples per prompt at temperature 1.0:

- `ranked`: the training prompt.
- `ranked_nonumber`: the same without the number.
- `single`: name ONE word; step 2 re-asks with the first word silently removed.
- `single_feedback`: step 2 only; the first word removed and the prompt says it
  was guessed and was correct.

Step 2 is conditioned on the stored ranking's first word. For the ranked
prompts, only samples that began with it are kept (79%).

10,583 calls, none failed, 1.88M tokens in / 2.30M out. Stored in
`cache/prompt_variants.db`, never in `llm_store.db`.

**Found on the way: the training data is near-greedy.** With no temperature
in the request -- how all 88k gpt-oss rankings in the store were bought --
DeepInfra answers an identical prompt identically in runs (four sequential
calls: two identical pairs, same text, same token count). With
`temperature=1.0` every call differed. So the stored rankings are close to
the model's mode, not samples from its distribution. The listener is
therefore fitted to a sharper teacher than gpt-oss at T=1. It is also why
these samples set the temperature explicitly: otherwise 8 "samples" would be
far fewer independent draws, and the self-agreement noise floor would be
inflated.

**The measure.** Per position, `D = within(X) + within(Y) - 2 cross(X,Y)`,
built from the unbiased agreement estimators. It is an unbiased estimate of
||p_X - p_Y||^2, zero in expectation when two prompts share a distribution
however noisy the samples. For scale, disjoint distributions have D = 2.

| step | pair | agree | D | 95% CI |
|---|---|---|---|---|
| 1 | ranked vs ranked_nonumber | 0.770 | 0.011 | [-0.001, 0.023] |
| 1 | ranked vs single | 0.762 | 0.034 | [0.019, 0.052] |
| 1 | ranked_nonumber vs single | 0.759 | 0.038 | [0.022, 0.055] |
| 2 | ranked vs ranked_nonumber | 0.636 | 0.012 | [-0.011, 0.038] |
| 2 | ranked vs single | 0.542 | 0.127 | [0.088, 0.168] |
| 2 | ranked vs single_feedback | 0.545 | 0.121 | [0.084, 0.163] |
| 2 | single vs single_feedback | 0.543 | 0.061 | [0.038, 0.087] |

Self-agreement: step 1 is 0.78 for all three prompts. Step 2 is 0.63
(ranked), 0.65 (ranked_nonumber), 0.58 (single), 0.57 (single_feedback).

The incumbent listener's McFadden R^2 on each prompt's picks:

| step | ranked | ranked_nonumber | single | single_feedback |
|---|---|---|---|---|
| 1 | 0.613 | 0.603 | 0.607 | -- |
| 2 | 0.301 | 0.298 | 0.305 | 0.278 |

**Conclusions.**
- **Step 1 does not depend on the prompt.** The number changes nothing
  measurable. Asking for one word instead of a ranking moves the distribution
  by a small, significant D = 0.03, and the listener scores all three within
  0.01. Step 1 is 65% of what play reads.
- **Step 2 does.** Continuing a list and re-asking with the word removed
  differ by D ~0.12, about 5 points of agreement below self-agreement. Being
  told the first guess was correct moves it again (D = 0.06).
- **The re-ask is the more Plackett-Luce-like of the two.** With step 1's top
  word removed, the re-asked step-2 pick is step 1's runner-up 62% of the time
  (60% with feedback); the list continuation, only 51%. A list drifts toward
  words that go with the one it just wrote; a fresh question falls back to the
  next-best answer, which is what "remove it and renormalise" assumes.
- **For the listener it is roughly a wash.** Its R^2 on step-2 picks is 0.301
  (ranked, what it trained on) against 0.305 (single) and 0.278
  (single_feedback). The latter is 0.02 lower, the one hint that the most
  realistic step 2 is predicted slightly worse; no interval was computed.
- **Which step 2 is "right" depends on the guesser.** The arena's LLM guesser
  asks once per turn and reads its ranking top-down, so for arena evaluation
  the ranked continuation is exactly the matching data. A human guesser sees
  the first card turned over before guessing again, which `single_feedback`
  imitates.

**Caveat.** The kept ranked step-2 samples are selected: they are the 79% that
agreed with the stored first word, which may make them more concentrated
(self-agreement 0.63 vs 0.58).

**The distributions compared directly, with no listener involved.** Total
variation (TV) is the share of probability that has to move to turn one
distribution into the other. With 8 draws a side, two samples of the SAME
distribution already differ, so each pair's TV is set against a permutation
null (pool both prompts' draws, reshuffle, recompute). "Excess" is the part
that belongs to the prompts.

| step | pair | TV | noise | excess [95% CI] | same modal word |
|---|---|---|---|---|---|
| 1 | ranked vs ranked_nonumber | 0.129 | 0.120 | +0.009 [-0.002, +0.020] | 89% |
| 1 | ranked vs single | 0.148 | 0.123 | +0.025 [+0.013, +0.038] | 87% |
| 2 | ranked vs ranked_nonumber | 0.214 | 0.208 | +0.007 [-0.010, +0.024] | 78% |
| 2 | ranked vs single | 0.329 | 0.242 | +0.087 [+0.065, +0.110] | 66% |
| 2 | ranked vs single_feedback | 0.327 | 0.239 | +0.089 [+0.065, +0.112] | 67% |
| 2 | single vs single_feedback | 0.286 | 0.236 | +0.050 [+0.033, +0.069] | 70% |

Shape at step 2: the re-ask is flatter than the list continuation (entropy
0.93 vs 0.73 bits, 2.5 vs 2.1 distinct words in 8 draws, modal share 0.72 vs
0.76). At step 1 all three prompts share one shape (modal share 0.85,
0.46-0.47 bits).

Per position, the test has little power at 8 draws. The share of positions
individually significant at p < 0.05 is 1-3% at step 1, below the 5% a
permutation test gives when nothing differs (it is conservative on discrete
data), and 9-14% for the step-2 list-vs-re-ask pairs. The differences are
real in aggregate; per position, 8 draws cannot pin most of them down.

The largest step-2 gaps are sense changes, not reshuffles. For "exploring",
with Himalayas removed:
- list continuation: Africa 8/8;
- re-ask: Mercury 5, Saturn 2;
- told Himalayas was correct: Saturn 6, Mercury 2.

For "plaza", with London removed:
- list continuation: Embassy 5;
- re-ask: Stadium 8;
- told London was correct: Embassy 5, Stadium 3.

A list stays on the reading its first word committed to. A fresh question
can switch reading, and being told the first word was right sometimes pulls
it back (plaza, atlanta) and sometimes not (exploring). In aggregate the
feedback prompt is as far from the continuation as the silent re-ask is
(excess 0.089 vs 0.087).

## Association counts: an unnormalised target for the per-clue level

**Problem.** The board softmax is invariant to adding a constant to every
word's score for one clue. So a clue that points strongly at one word and
weakly at two (4, 2, 2) and a vague one (2, 0, 0) look identical. The
professor suggested a Bregman divergence. A loss on the normalised
distributions cannot help, because cross-entropy/KL already is one and they
all see the same probabilities. What can help is a Bregman divergence on
**unnormalised** targets, which needs a target that is not normalised over
the board.

**Target** (`scripts/data/collect_associations.py`). gpt-oss is asked for "the
25 words that *clue* most strongly makes you think of", with no board, at
temperature 1, 5 samples per clue. That is 9,494 clues and 47,444 lists
(136 failed on first pass; 66 persistently, for clues like "aa"), costing
5.0M tokens in and 5.7M out. A board word's count is how many lists name it,
matching plurals and ignoring case and punctuation. The base rate is 0.021 per
list per board word.

**Loss.** Row count y ~ Poisson(R · exp(a·s + b)), added to the board
softmax on step-1 rows only, so each (clue, word) enters once per position:

- The Poisson deviance is the generalized KL, the Bregman divergence of
  x log x.
- s is the listener's own score, so the term constrains the level the
  softmax leaves free.
- a = 0.87 is fixed from the incumbent: a Poisson GLM fit of its scores to
  the counts gives 0.865.
- b is the log base rate, so no trees are spent on a global shift.

Gradient checked against finite differences (`tests/test_association_term.py`).
`<out>.assoc.json` keeps a and b.

**Evaluation** (`scripts/tools/eval_association_level.py`). Each model gets its
own best (a, b) on training rows, so the incumbent is judged on what its
scores know, not on an arbitrary level. Validation is the incumbent's board
split; Sonnet and the gpt-oss holdout come from `compare_listeners.py`.

| Model | board R^2 | Sonnet R^2 | gpt-oss holdout R^2 | decoy R^2 | assoc D^2 | level ρ | unseen-clue D^2 | unseen-clue level ρ |
|---|---|---|---|---|---|---|---|---|
| incumbent | 0.3548 | 0.5528 | 0.3415 | 0.2129 | 0.696 | 0.577 | 0.600 | 0.414 |
| weight 0.3 | 0.3514 | 0.5542 | 0.3367 | 0.2070 | 0.785 | 0.710 | 0.669 | 0.513 |
| weight 1 | 0.3421 | 0.5472 | 0.3250 | 0.1969 | 0.807 | 0.746 | 0.673 | 0.542 |
| weight 3 | 0.3266 | 0.5346 | 0.3081 | 0.1801 | 0.811 | 0.765 | 0.655 | 0.556 |

- **Level ρ** is the Spearman correlation between a board's predicted total
  association mass, Σ_w R·exp(a·s_w + b), and the observed total, over 6,241
  validation boards. That is the per-clue number the softmax cannot learn.
- **Unseen clues** (902 validation positions) guard against a leak: the split
  is by board, clues repeat across boards, and counts depend only on
  (clue, word).

**Conclusions.**
- **The incumbent already knows part of the level.** Its ρ is 0.58 (0.41 on
  unseen clues). Its score is one function shared across all boards, and
  several features are raw rather than board-relative, so some absolute
  information leaks in despite the loss.
- **The Poisson term adds a lot of it, and at weight 0.3 almost for free.**
  Unseen-clue ρ goes 0.41 -> 0.51. Board-choice prediction is unchanged on
  Sonnet (+0.001) and slightly lower on gpt-oss (-0.003 val, -0.005 holdout).
  Higher weights buy more level (0.54, 0.56) at a growing board cost
  (Sonnet -0.006, -0.018).
- **Decoys cannot see this.** Decoy-board R^2 falls slightly as the weight
  rises. A decoy group is still a softmax within one clue, and a clue-level
  shift moves the random words' scores too, so the decoy data pin the level
  relative to "a random word under this clue", which is a different reference
  from an absolute association rate. Which reference "passing" should be
  priced against is a modelling choice for the reward, not settled by either
  dataset.
- **Not yet shown:** that a better level makes better clues. That needs the
  level in the reward, e.g. a fixed pass rate against exp(a·s + b) in place of
  the decoy-based outside option, and then games.

## The learned listener beats a Sonnet spymaster on the frozen suite

**Question (Shane's).** Can the learned listener beat Sonnet at the
spymaster's job, with Sonnet guessing for both sides?

**Set-up.**
- `sonnet_spymaster` (`codenames/spymasters/llm_spymaster.py`) is Claude
  Sonnet 5 at medium effort. It is shown the board with every card's team
  and asked for a one-word clue and a number from 1 to 4.
- It is told this arena's rules, not the tabletop ones: the guesser takes
  exactly N guesses, stops at the first miss, and cannot pass. The learned
  listener's reward assumes exactly that, so leaving Sonnet to assume the
  tabletop rules would handicap it.
- It uses the same legality rule as every spymaster, and an illegal answer
  is re-asked with the reason. None were needed: all 755 clues were legal
  on the first attempt.
- It may use any English word; the learned listener searches its ~11k-word
  clue pool.
- Played on `holdout_v1`: 100 held-out-vocabulary boards, each in both
  seatings, with the suite's Sonnet guesser (`anthropic:claude-sonnet-5:medium`).
- A 5-board pilot measured the cost first. Total: $7.75 for 200 games
  ($5.06 spymaster at 458 in / 635 out tokens per clue; about $2.69 guesser
  at 195 / 142 per ranking).

**Result.**

|  | win% | 95% CI | boards swept | assassin losses | mean k | own/clue | own% |
|---|---|---|---|---|---|---|---|
| learned_listener | **58.0%** | [0.51, 0.65] | **28** | **13** | 2.20 | 1.67 | 83.5% |
| sonnet_spymaster | 42.0% | [0.35, 0.49] | 12 | 31 | 2.24 | 1.60 | 80.2% |

Sign test on the 40 decisive boards (60 split): **p = 0.017**. Assassin losses:
Fisher p = 0.006.

**Where the margin comes from.**

| How the game ended | listener won | Sonnet won |
|---|---|---|
| other side's guesser hit the assassin | 31 | 13 |
| otherwise | 85 | 71 |

In the 156 games without an assassin, the listener wins 54.5%, an edge too
small to be significant on its own. Most of the difference is that Sonnet's
clues sent its own guesser into the assassin 31 times, against 13. That is
the thing the listener's reward prices explicitly (the assassin cost in
`pl_reward.py`) and a spymaster clueing by intuition does not. Per clue,
the two are close: 1.67 vs 1.60 own words, similar mean k.

**Why this is a strong result.** The setup favours Sonnet. It writes clues
for a copy of itself, while the listener was distilled from gpt-oss, so
predicting Sonnet is a transfer test for it.

**Caveats.**
- One effort level for Sonnet (medium); a higher-effort spymaster might
  weigh the assassin better.
- The 95% CI on the game win rate treats games as independent, but they are
  paired by board. The sign test is the paired comparison and the one to
  quote.

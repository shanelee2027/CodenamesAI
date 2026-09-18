# Working log

Record what was expected vs. what actually happened as work proceeds,
oldest first. The early sections are the original milestones (M0-M9);
the project has since moved to a sequence of named models, so later
entries are one per piece of work rather than per milestone. See
`docs/versions/` for the models and `docs/design-decisions.md` for
standing rationale.

## M0 — Corpus collection

**Expected:** manually click through each wiki's Special:Statistics page to
check/request dumps, per the raw SCOPE.md description.

**Actual:** automated the check/download/extract steps instead of the manual
click-through, with one deliberate exception. Findings:

- Fandom's wiki pages (`*.fandom.com/wiki/...`) are behind a Cloudflare JS
  challenge — plain scripted GET/HEAD requests get a 403 "Just a moment..."
  page, not real content.
- `*.fandom.com/api.php` (MediaWiki API) is *not* behind that challenge, and
  returns each wiki's `wikiid` (dbname) and page count as JSON.
- Actual dump files live on a public S3 bucket at a deterministic path
  derived from the dbname: `s3://wikia_xml_dumps/{c}/{cc}/{dbname}_pages_current.xml.7z`.
  This is also not behind Cloudflare, and HEAD requests return size,
  last-modified, storage class, and an MD5 checksum (`x-amz-meta-md5`) —
  enough to check status and verify downloads without ever touching the
  Cloudflare-protected pages.
- Requesting a *new* dump (for wikis with none) requires an
  autoconfirmed-or-higher Fandom account on that specific wiki — not
  something a script can do. Left as a manual step by design (see
  `scripts/check_fandom_dumps.py`'s docstring).
- Initial version of the storage-class check was wrong: it treated every
  class other than `STANDARD` as needing an S3 Glacier-style restore before
  download. Only `GLACIER`/`DEEP_ARCHIVE` actually require that —
  `STANDARD_IA` serves a normal ranged GET immediately. Caught this by
  testing a ranged GET directly against a `STANDARD_IA` object (got `206
  Partial Content`) before trusting the first pass. Fixed in both
  `check_fandom_dumps.py` and `download_fandom_dumps.py`.
- Built three scripts: `scripts/check_fandom_dumps.py` (status via API +
  S3 HEAD), `scripts/download_fandom_dumps.py` (resumable, checksum-verified
  download), `scripts/extract_fandom_corpus.py` (7z decompress +
  wikiextractor → plaintext, reports total GB against the 1-5GB target).
  Added `wikiextractor` to `pyproject.toml` dependencies (needed now, not a
  later milestone).
- Of the 45 candidate wikis in `scripts/fandom_wikis.txt`: 23 have a dump
  ready to download immediately (~1.93 GB compressed), 18 need a Glacier/Deep
  Archive restore first, 3 have no dump yet (need a manual request via
  Special:Statistics), 1 subdomain guess was wrong (`fullmetalalchemist`
  404s — needs the correct subdomain).
- `p7zip-full` is not installed and this environment has no passwordless
  `sudo`, so `extract_fandom_corpus.py` is untested end-to-end — install
  with `sudo apt-get install -y p7zip-full` to unblock it.
- Downloaded and checksum-verified all 23 ready-now dumps (~1.8 GB on disk).
  `p7zip-full` was then installed manually; running `extract_fandom_corpus.py`
  against the real data caught a real bug: it invoked `python` for
  wikiextractor, which doesn't exist on PATH here (only `python3` /
  the venv's `python`) — fixed by using `sys.executable`.
- Requested dumps for the 21 wikis needing one (18 archived + 3 missing) is
  blocked on the new Fandom account reaching the 4-day (96h) autoconfirmed
  threshold — Fandom's autoconfirmed rule is account age only, no edit
  requirement, and it's global across all wikis on one account.
- Ran `extract_fandom_corpus.py` against the real 23 dumps and caught a
  second real bug: 7z preserves whatever permission bits are stored in the
  archive verbatim, and the "disney" dump's stored bits were `000`
  (unreadable even by the owner) — decompressed fine, then wikiextractor
  hit `PermissionError` trying to open it. Fixed by chmod'ing the extracted
  XML to `rw-------` right after decompression, before handing it to
  wikiextractor.
- Result: all 23 wikis extracted to plaintext, **0.961 GB total** — just
  under SCOPE's 1-5GB target lower bound. Expected to grow well past it
  once the 21 pending wikis land.
- Pre-fetched the raw files for the other 3 embedding spaces (GloVe 6B,
  Numberbatch English 19.08, Wikipedia2Vec 300d text) via
  `scripts/download_embeddings.py`, ahead of M2/M4's actual milestone
  order — justified the same way as M0's early start (large, latency-bound,
  no code dependency), confirmed with the user first since SCOPE.md
  sequences these after M1. Only raw files staged in
  `data/embeddings/raw/`; no tensor-building code written.
- All 3 downloads completed and sha256-recorded (no official checksums are
  published by any of the three sources, so these sidecar hashes are only
  useful for detecting local corruption on re-run, not verifying against a
  canonical value): `glove.6B.zip` (862 MB, zip integrity checked with
  `zipfile.testzip()` — contains 50d/100d/200d/300d), `numberbatch-en-19.08.txt.gz`
  (325 MB), `enwiki_20180420_300d.txt.bz2` (2559 MB). Total ~3.7 GB raw.
  Throughput was much lower than the Fandom S3 downloads (~1.5 MB/s vs.
  ~21 MB/s) — bottlenecked by the source servers (Stanford, S3 buckets
  outside the fast one), not local bandwidth.

## M1 — Board and legality

**Expected:** straightforward per SCOPE.md's description — Board class,
`is_legal_clue()`, pytest coverage of the listed edge cases.

**Actual:** matched expectations, plus two decisions worth recording:

- Board vocabulary: no source was specified in SCOPE.md beyond "~800
  Codenames card words." Used the 400-word base-game list from
  [sagelga/codenames](https://github.com/sagelga/codenames)
  (`wordlist/en-EN/default`, itself sourced from two other open
  implementations) — verified as 400 unique words, no duplicates, and it
  naturally includes multi-word entries ("Ice cream", "New york", "Loch
  ness", "Scuba diver") which became real test fixtures rather than
  synthetic ones. Checked in at `codenames/assets/board_words.txt` (a
  package asset, not `data/` — that's for gitignored raw dumps). 400 words
  is half of SCOPE's "~800" figure; expansions would need to be added
  later if that matters, but wasn't blocking for M1.
- Legality's "morphological variants" rule: substring-containment already
  covers regular English suffixation for free (plural -s/-es, -ing, -ed
  are all concatenative, so "apple" ⊂ "apples" as strings). The only
  common gap is a stem-spelling change before a suffix -- y→i ("happy" /
  "happier", "city" / "cities") -- so `_stem_variants()` adds just that one
  extra form per word rather than pulling in a stemmer/lemmatizer library.
  Deliberately narrow: won't catch irregular forms (mouse/mice, go/went).
  Chosen over a real stemmer because SCOPE explicitly flags that legality
  bugs "silently inflate every downstream score" -- a small,
  fully-enumerable rule set that's easy to write precise tests against
  beats broader but less predictable coverage from a black-box stemmer.
- `codenames/board.py` + `tests/test_board.py`, 23 tests, all passing:
  board generation (determinism per seed, role counts, no duplicate
  words), reveal/role-query behavior, and legality (exact match, case
  insensitivity, plurals both directions, the y→i stem case, hyphenated
  board words, multi-word board entries).

## M2 — GloVe and similarity tensor

**Expected:** download script, clue vocab filter, mmap tensor build,
similarity.py loader, sanity check. Per SCOPE.md's own warning, treat the
sanity check as a gate, not a formality.

**Actual:** matched expectations. Notes:

- The "download script with resume and checksum verification" requirement
  was already satisfied by `scripts/download_embeddings.py` (built during
  the M0 session when GloVe/Numberbatch/Wikipedia2Vec raw files were
  pre-fetched) — no separate GloVe-specific downloader needed.
- "Frequency threshold (CLI arg)" is implemented as a rank cutoff, not an
  external frequency source. Verified empirically that GloVe 6B's vocab
  file is frequency-descending ordered (first entries: "the", ",", ".",
  "of" ...; last entries: junk/rare tokens) — so line index doubles as
  frequency rank for free. Filtering to purely lowercase-alphabetic tokens
  (317,756 of the 400,000 total) and taking the top `--vocab-size` (default
  250,000) of those lands exactly on SCOPE's ~250k target with zero new
  dependencies. Avoided pulling in an external word-frequency library
  (e.g. `wordfreq`) for this reason.
- Board vocabulary has 4 multi-word/hyphenated entries ("Ice cream", "Loch
  ness", "New york", "Scuba diver") that have no single GloVe token.
  Resolved via mean-pooling the constituent words' vectors (a standard
  technique for representing short phrases with single-word embeddings) —
  verified first that every constituent word of every multi-word entry
  exists in GloVe's vocab, so no silent fallback was needed. All other 396
  board words matched a GloVe token directly. This mean-pooling approach
  will need to be repeated for each additional space in M4 (each space's
  own vocab may have different multi-word gaps).
- `codenames/similarity.py`: mmap loader over a `(n_clues, n_board_words,
  n_spaces)` fp16 tensor — the `n_spaces` axis exists now (size 1, just
  "glove") specifically so M4 can append space-slices without changing
  this interface, per SCOPE.md §2's exact tensor shape ordering.
- Build run: 250,000 clues × 400 board words × 1 space, 200MB on disk,
  9.9s wall time end-to-end (8.8s just loading the raw 400k-row GloVe
  file), 0.5s for the actual GPU similarity compute, peak VRAM 0.12GB —
  trivial at this scale; expect this to matter more once M4 adds 3 more
  spaces and the board vocabulary grows.
- Ran `scripts/sanity_check_sims.py` and eyeballed it per SCOPE's
  instruction not to skip this. Results were clean and semantically
  correct across the board: King → queen/prince/monarch/throne; Shark →
  whale/dolphin/crocodile; Spy → espionage/CIA/KGB/Mossad; Egypt →
  egyptian/syria/cairo (all correctly self-matching at exactly 1.0000);
  New york (multi-word/mean-pooled) → york/new (its own parts, expected)
  then manhattan/jersey/brooklyn/boston — no signs of index misalignment.
  Caught one bug in the process: the script's own hardcoded sample word
  list used the wrong casing (lowercase "king" vs. the wordlist's actual
  "King") and silently reported false "NOT IN BOARD VOCABULARY" skips —
  bug was in the sanity script's sample list, not the tensor/loader; fixed
  by matching the asset file's actual Title Case.
- `tests/test_similarity.py`: 12 tests against a small synthetic
  hand-built tensor fixture (not the real GloVe-derived one) — keeps tests
  fast and independent of the large downloads, same principle as
  `test_board.py`'s determinism tests.
- Also caught: the package wasn't actually pip-installed (`pip install -e
  .` had never been run, despite being in the README setup steps), so
  `scripts/sanity_check_sims.py` failed on `import codenames` until fixed.
- Follow-up after the casing bug above: made board-word lookups fully
  case-insensitive everywhere, matching `is_legal_clue()` which already
  normalized case. Fixed in three places: `Board._card()` (backs
  `role_of`/`reveal`/`is_revealed`), `SimilarityTensor`'s `board_index`
  (backs `similarity`/`similarities_for_board`/`top_clues`), and
  `sanity_check_sims.py`'s own pre-check (which had been reaching into
  `board_index` directly instead of going through the case-insensitive
  API — same class of bug one layer up). This also surfaced a second
  latent bug while fixing the first: `Board.reveal()` was storing the
  *caller's* casing in the `revealed` set rather than the card's
  canonical casing, so `reveal("king")` followed by `is_revealed("King")`
  would have disagreed. Fixed by having `reveal()` store `card.word`.
  4 new regression tests added across `test_board.py`/`test_similarity.py`
  (39 total, all passing).

**Correction (found while working on M4):** the clue vocabulary as
originally built was structurally broken for this project's own stated
purpose. It was the top-250k GloVe-ranked alphabetic words only, which
means a word GloVe doesn't know (or ranks too low) could **never** appear
as a candidate clue, regardless of how well any other space knows it —
directly undercutting SCOPE.md's own motivating example: "GloVe has no
vector for [Technoblade] at all, and could not produce the clue at any
threshold," which implies another space *should* be able to supply it.
The user asked, while testing `scripts/check_clue.py`, whether a word
absent from GloVe but present in Wikipedia2Vec could be allowed — that
question is what surfaced this.

Fixed by rebuilding `clue_vocab.json` as a **union** across all three
downloaded spaces, not just GloVe's:
- GloVe and Wikipedia2Vec are both frequency-descending ordered in their
  raw files (verified empirically), so each contributes its own top-N
  alphabetic words via a rank cutoff (250k each, matching the original
  per-space target). Early-stopping once N are collected made this fast
  even for Wikipedia2Vec's 4.53M-line file — the first 250k qualifying
  tokens appear within the first 12% of the file (539,860 lines), so this
  script no longer needs the ~235s full scan it needed in M4's first
  pass just to establish the vocabulary.
- Numberbatch has no reliable frequency ordering (verified empirically —
  its first entries are junk tokens like "##") and a modest total
  alphabetic vocabulary (359,059 words), so all of it is included rather
  than attempting a rank cutoff.
- Result: **532,738** clue words (up from 250,000), of which 251,610 have
  no GloVe vector at all — these are exactly the words the original
  design would have permanently excluded. GloVe's own slice now has NaN
  gaps too (52.8% coverage of the new, larger vocab) using the same
  NaN-for-missing convention already established for the other spaces —
  a generalization that was needed anyway, since GloVe's slice could no
  longer assume 100% coverage of its own reference vocabulary once that
  vocabulary stopped being purely GloVe-derived.
- Every word in the union still gets checked against GloVe's *full*
  400k-word vocabulary (not just its top-250k) before being marked
  missing — a word freshly contributed by Numberbatch or Wikipedia2Vec
  might still exist further down GloVe's own list.
- Verified the fix directly: `fortnite` (not in GloVe's vocab at all —
  predates the word's 2017 gaming usage in GloVe's training corpus) now
  resolves via `check_clue.py`, showing `n/a` for GloVe and real values
  for Numberbatch/Wikipedia2Vec, instead of the hard "not in clue
  vocabulary" error it would have raised before.
- Extracted the loading/mean-pooling/NaN-masking logic that both
  `build_similarity_tensor.py` and `extend_similarity_tensor.py` now
  need into `scripts/_embedding_lib.py` — this duplication became real
  once GloVe's own slice needed the same NaN-aware machinery the other
  spaces already had, not speculative abstraction.
- One cosmetic side effect, not a bug: Numberbatch's top-clues for common
  board words now surface rare morphological derivatives it wasn't
  contributing before (e.g. "King" → nonking/kinging/kingless/unking
  ahead of queen/monarch) — a consequence of including its full
  vocabulary rather than only the slice that overlapped GloVe's top-250k.
  Harmless in practice: every one of these literally contains the board
  word as a substring, so `is_legal_clue()` already rejects them before
  they'd ever be scored as a real candidate.
- Full rebuild: 42.7s (GloVe + union vocab construction), 12.1s
  (Numberbatch), 241.2s (Wikipedia2Vec, dominated by the full scan needed
  to check membership against the new, larger wanted-set — the early-stop
  trick only applies to vocab *construction*, not to finding vectors for
  an already-fixed vocab). 41 tests still pass unchanged (they use a
  synthetic fixture, not the real tensor).

## M3 — Inspector

**Note:** initially skipped at the user's explicit request, out of SCOPE.md's
stated order (M3 before M4) — proceeded straight to M4's non-fastText half.
Built afterward, once M4 (partial) was done. `scripts/sanity_check_sims.py`
was used as a stand-in verification tool in the meantime.

**Expected:** CLI taking a board and a typed clue, printing per-space
similarity to all 25 words, per-space top-ranked words, what each guesser
would pick, and a baseline score.

**Actual:** built out of SCOPE.md's intended milestone order, so two of
the four pieces have real gaps that were flagged rather than papered
over:

- "What each guesser would pick" needs the guesser pool (M5), which
  doesn't exist yet. `scripts/inspector.py` prints an explicit
  placeholder line rather than silently omitting the section — so it's
  visibly incomplete each time it's run, not quietly missing.
- "A baseline score" — the real one (SCOPE.md §6, baseline 3) needs
  CMA-ES-tuned constants against the guesser pool, which also doesn't
  exist yet. Used an **untuned preview** instead: SCOPE.md's own stated
  example constants (own +1, opponent −1, neutral −0.3, assassin −10)
  applied directly, unweighted-averaged across whichever spaces have a
  value for a given word (NaN entries excluded from each role's mean,
  not treated as 0). Labeled clearly in the output as not the real
  tuned baseline.
- SCOPE.md's directory layout (§8) doesn't list an inspector module
  under `codenames/` at all — only `scripts/` and the "CLI (and
  optionally a small web UI)" phrasing — so this lives at
  `scripts/inspector.py`, not as a package module.
- Supports `--reveal WORD [WORD ...]` to simulate mid-game states (a
  word already picked no longer counts toward its role's baseline mean).
  Verified this works: revealing the assassin word makes its baseline
  contribution correctly drop to exactly 0, not skew from a stale value.
- Manually verified against a real board (seed 42): the illegal-clue
  path correctly caught a clue that was itself a board word ("England"),
  and separately surfaced that "England" would also be a *dangerous*
  clue regardless (0.55/0.32/0.53 similarity to the assassin word
  "Australia" across the three spaces) — exactly the kind of thing this
  tool is meant to catch. Also verified the NaN-handling path with
  "fortnite" (present in Numberbatch/Wikipedia2Vec, absent from GloVe
  per the M4 union-vocab fix) — GloVe's column correctly shows "n/a"
  throughout rather than crashing or defaulting to 0.
- `baseline_score()`'s formula (not just the script's I/O) got 6 unit
  tests in `tests/test_inspector.py`: role-mean correctness, revealed-word
  exclusion, a fully-revealed role contributing exactly 0, the weighted
  sum matching SCOPE's stated constants, and NaN exclusion from role
  means. 47 tests total, all passing.

## M4 — Remaining embedding spaces + fastText training (partial)

**Note:** the coverage numbers and vocab-size figures below (250,000
clues, 47.6%/97.6% coverage) describe the *first* pass and are superseded
by the vocab-union correction recorded under M2 above (532,738 clues,
67.4%/72.3% coverage after rebuilding). Left as-is below as a record of
what happened at the time rather than edited to match — the correction
entry explains why and what changed.

**Scope actually covered:** Numberbatch + Wikipedia2Vec extended onto the
existing tensor. fastText is NOT done — it trains on the Fandom corpus
(SCOPE.md §M4), which isn't fully collected yet (23/45 wikis; 21 pending
the Fandom account's autoconfirmed window, see M0). Explicitly deferred,
not forgotten.

**Expected:** straightforward per SCOPE.md — download (already done during
M0's pre-fetch), extend the tensor to 4 spaces, extend the inspector,
run the "technoblade" milestone test.

**Actual:**

- Both new spaces reuse the exact clue/board vocabulary fixed by M2's
  GloVe build (same 250,000 clue words, same 400 board words, same row/
  column indices) — required for the `n_spaces` axis to mean anything;
  each space is a slice appended to the same tensor, not a separate one.
- A clue or board word with no vector in a given space gets **NaN**, not
  zero or a silent drop. Zero would misleadingly read as "confirmed
  unrelated" rather than "no data" — a real design difference from the
  `-1` sentinel SCOPE uses elsewhere (§2) for masking revealed/missing
  board slots in the eventual feature vector; that's a different, later
  concern (M7) about a specific board's 25 slots, not about vocabulary
  coverage gaps in the tensor itself.
- Checked each source file's format empirically before writing loader
  code (this determined the design, not the other way around):
  - **Numberbatch** (`numberbatch-en-19.08.txt.gz`): plain word2vec-style
    text format, fully lowercase, and — unlike GloVe — multi-word concepts
    are already single underscore-joined tokens (`new_york`, `ice_cream`,
    `loch_ness`, `scuba_diver` all present directly). Result: **100%**
    board-word coverage with no mean-pooling needed at all. Clue coverage
    is lower, **47.6%** (119,026/250,000) — Numberbatch's ~517k-word
    vocab just doesn't overlap GloVe's frequency-ranked 250k as heavily
    as it does for the small, curated board vocabulary.
  - **Wikipedia2Vec** (`enwiki_20180420_300d.txt.bz2`): mixes plain word
    vectors with `ENTITY/Title_Case` Wikipedia-article vectors in the same
    file (4.53M total entries). Entity vectors are Wikipedia2Vec's actual
    differentiator per SCOPE's own table ("Encyclopedic entities, wiki
    link graph") but were **excluded** here — using them properly needs
    real Wikipedia title resolution (our board string "New york" isn't
    guaranteed to match the article title "New_York" in general, and
    guessing title-casing heuristically is fragile). Used plain word
    vectors only, with GloVe-style mean-pooling as the multi-word
    fallback (all 4 multi-word board entries needed it, same as GloVe).
    Result: 396/400 board words matched directly, 4 mean-pooled, **100%**
    board coverage; **97.6%** clue coverage (243,918/250,000) — much
    closer to GloVe's coverage than Numberbatch's, consistent with both
    being built from broad web/encyclopedia text rather than a curated
    concept graph. Using entity vectors properly is a reasonable follow-up
    enhancement, not abandoned by design — just out of scope for "the two
    embeddings we can do" as asked.
  - Wikipedia2Vec's file is large enough (4.53M lines) that a naive full
    parse would be slow/memory-heavy; `extend_similarity_tensor.py` only
    float-parses lines whose token is in the wanted set (250k clue words ∪
    400 board words ∪ their split parts), checked cheaply via string
    membership before paying for the `np.fromstring` conversion. Full run:
    236s wall time, dominated by the single pass over the compressed file
    (similarity compute itself was 0.3s).
  - Numberbatch: 6.6s wall time total (small file, no filtering bottleneck).
- `codenames/similarity.py`'s `top_clues()` needed a real fix once NaN
  entries existed in the tensor: `np.argpartition` mixed with NaN doesn't
  reliably sort NaN to one end the way a full sort does, so NaN rows are
  now filtered out *before* ranking rather than trusted to land somewhere
  sensible after. Caught by reasoning about it before running at scale
  (Numberbatch leaves ~52% of any clue column as NaN, so this wasn't a
  rare edge case) rather than by a failure.
- `scripts/sanity_check_sims.py` extended to loop over every space in the
  loaded tensor per board word, rather than only checking index 0. Ran it
  across all three spaces on King/New york/Egypt/Spy and eyeballed the
  results per SCOPE's instruction not to skip this: each space shows a
  genuinely different knowledge profile exactly as SCOPE's design predicts
  — Numberbatch surfaces commonsense/geographic specifics (Newburgh,
  Poughkeepsie, Masr, Hatshepsut for Egypt), Wikipedia2Vec skews
  encyclopedic (nyc, nubia, philby), GloVe stays generic co-occurrence
  (manhattan, jersey, boston). No signs of index misalignment across
  spaces.
- 2 new regression tests for the NaN-exclusion behavior in `top_clues()`.
  41 tests total, all passing.
- The "technoblade" milestone test from SCOPE.md's M4 section requires
  fastText (trained on the Fandom/pop-culture corpus) — GloVe/Numberbatch/
  Wikipedia2Vec are all general web/encyclopedia corpora and have no
  vector for "technoblade" at all. Confirmed by checking all three raw
  source files directly (not just the built vocab) — the word doesn't
  exist in any of them, not even as a Wikipedia2Vec entity vector. That
  test is meaningless until fastText exists; will run it properly once
  M0's corpus is complete enough to train on.

**Future improvement (not pursued now):** Wikipedia2Vec's only English
pretrained release is from April 2018 (confirmed via the project's own
pretrained-downloads page and general search — a 2020 paper trained the
tool on a January 2019 dump, but that was never published as a
downloadable pretrained file). A fresher Wikipedia snapshot might carry
entities/terms that postdate 2018. Not pursued: it would mean training
Wikipedia2Vec from scratch against a full current Wikipedia dump
(~20GB+ compressed, hours of training), which both diverges from
SCOPE.md's explicit "download, pretrained" design for this space (§1:
"Not a training-from-scratch embeddings project") and is redundant with
fastText-on-Fandom, which is already the project's designated mechanism
for recent pop-culture coverage. Left as a possible later stretch item,
not a current gap to close.

## M5 — Guesser pool

**Expected:** ~8 structurally different guessers per §3, a registry so
the arena can enumerate them, 2 held out from training, pool composition
in a config file.

**Actual:** matched expectations. Notes:

- The interface needed two methods, not one. A guesser can't just return
  a sorted word list: `NoisyGuesser` needs the underlying numeric scores
  to perturb, and `ConfidenceThresholdGuesser` needs to voluntarily
  return *fewer* candidates than it was given (early stop). So
  `Guesser.score_candidates()` is the abstract method every guesser type
  implements (this is where the actual knowledge/policy difference
  lives), and `rank_candidates()` has a default (sort by score) that only
  the threshold guesser overrides. This let `NoisyGuesser` and
  `ConfidenceThresholdGuesser` be generic *wrappers* around any other
  guesser instead of duplicating logic per base type.
- A candidate a guesser's knowledge source has no vector for scores
  `-inf`, not 0 -- consistent with the NaN-as-"unknown" convention
  established in M4, and important here specifically: 0 would compete
  with real low-but-nonzero similarity instead of always ranking last.
- 8 guessers in `configs/guesser_pool.json` (pool composition lives there,
  not in code, per SCOPE.md §3's explicit instruction): one per available
  space (`glove`, `numberbatch`, `wikipedia2vec`), two blends (`blend_uniform`,
  `blend_glove_heavy` -- SCOPE's "one or two"), one rank-based, one noisy
  (wraps `blend_uniform`), one confidence-threshold (wraps `glove`).
  **fastText has no guesser yet** -- it doesn't exist until M4's remaining
  half (needs the Fandom corpus). Noted directly in the config file's
  comment, not just here, so it's visible to whoever edits pool
  composition later: add a `single_space` entry once fastText exists,
  most likely training-visible given it's central to the project's own
  motivating case.
- Held out (2, matching SCOPE's requirement exactly): `numberbatch` and
  `rank_based` -- one single-space guesser and the structurally distinct
  rank-based one, so the held-out set tests generalization along two
  different axes (an unseen knowledge source, and an unseen decision
  policy), not just one.
- `RankBasedGuesser`'s "rank not score" property is real, not just
  labeled: unit-tested with a case where a raw-score blend and a
  rank-based aggregation genuinely disagree on the top candidate (one
  space has a huge outlier score that dominates any weighted average but
  loses on rank in both spaces) -- raw blend picks the outlier's word,
  rank-based correctly doesn't.
- `BlendGuesser` renormalizes over whichever weighted spaces actually
  have a vector for a given word, rather than treating a missing space
  as 0. Caught a flawed test while verifying this: two candidates each
  present in only *one* space can never show a weight-driven ranking
  change, since renormalizing by the single available space's own weight
  makes that weight cancel out algebraically -- had to rebuild the test
  with candidates present in *both* spaces to actually observe the
  effect. Not a bug in the guesser, just a bug in reasoning about what
  the test needed to construct.
- Registry (`codenames/guessers/registry.py`) builds guessers from the
  JSON config, resolving wrapper guessers' `base` references by name (an
  entry can only reference a base defined earlier in the list). Raises
  clear errors for an unresolvable base reference, a duplicate name, or
  an unknown guesser type, all tested.
- 23 new tests in `tests/test_guessers.py`. 70 tests total, all passing.
- Manually verified against the real tensor and a real board: all 8
  guessers produce visibly different rankings for the same clue+board
  (as they should -- that's the entire point of a diverse pool), and
  `cautious_glove`'s confidence-threshold cutoff landed exactly where
  hand-checking the raw GloVe scores said it should.
- Wired the pool into `scripts/inspector.py`'s "what each guesser would
  pick" section, closing the placeholder M3 had to leave there before
  guessers existed. Shows each guesser's own top-5 preference ranking
  over currently-unrevealed words (not a full simulated turn -- the
  number+1 attempt cap and turn-ending-on-a-miss rule are M6's job).
  Immediately useful for real: running it against a live board+clue
  showed `noisy_blend` picking the assassin word as its #2 guess while
  every other guesser avoided it entirely -- a concrete look at exactly
  the kind of risky-guess training signal the noisy guesser exists to
  produce.

## M6 — Arena

**Expected:** cross-play matrix, every spymaster x every guesser, over
fixed seeded boards. SQLite logging, one row per turn. Multiprocessing
across cores sharing the mmapped tensor, reporting per-worker RSS.
Metrics: win rate, mean turns, assassin rate, mean own-words per clue.

**Actual:** matched expectations, plus a mid-build correction on the
memory-sharing requirement. Notes:

- Built out of order relative to guessers-before-spymasters intuition:
  M8's learned scorer doesn't exist yet, so the arena needed *something*
  to pair against the pool. Built the three spymaster baselines SCOPE
  §6 lists that don't require the feature vector or a trained model:
  random legal clue, centroid, and the 8-constant linear scorer (§6
  items 1-3; items 4-5 need M7/M8). `codenames/spymasters/` mirrors
  `guessers/`'s shape: a `Spymaster` ABC with one abstract method,
  `give_clue(board, sims) -> (clue, number)`.
- **Centroid baseline without raw vectors.** SCOPE §6 says "clue nearest
  the mean of a random own-word subset," but build-time explicitly
  throws away the embedding models after the similarity tensor is built
  (§2) -- there's no vector to average. Standard proxy used instead: a
  candidate clue's mean cosine similarity to a set of points approximates
  its similarity to their mean, so "nearest the centroid" = highest mean
  similarity to the subset, flat across both the subset words and the
  available spaces (not nested mean-of-means, which would under-weight a
  word missing from one space relative to a word present everywhere).
- **"Number" isn't specified by SCOPE for the baselines**, since only the
  learned scorer's k-distribution (§2) is designed in detail. Random
  picks a random count. Centroid and the linear scorer use a shared
  `natural_number()` helper: rank unrevealed words by similarity to the
  clue, count how many own-words rank above the first non-own word. All
  three cap at `MAX_CLUE_NUMBER = 4`, matching the learned scorer's
  eventual k in 0..4 (§2), so every spymaster's outputs stay comparable
  once M8 exists.
- **Memory bug found via the arena's own RSS reporting.** First version
  of `LinearScorerSpymaster` cached `nanmean(tensor, axis=2)` once per
  instance (~850MB as float32) to avoid rescanning the full tensor every
  turn. Looked fine in isolated tests. Running the real arena with 2
  workers showed **9.4GB RSS per worker** -- SCOPE §7's memory design
  note is specifically about *not* duplicating large state across
  workers, and a per-instance derived-array cache is exactly that: the
  mmapped tensor's pages are shared by the OS across processes, but a
  materialized copy of it is not. Fixed by dropping the cache entirely
  and reading only the board-word columns actually needed (at most 24)
  directly off the memmap per call. Same tests, same arena output;
  worker RSS on the same real-data smoke run dropped to ~1.5GB, at the
  cost of a slower per-turn call (rescans the needed columns instead of
  reusing a cached array) -- correct tradeoff given SCOPE's explicit
  memory constraint. Documented in `linear_scorer.py`'s own docstring so
  the mistake and the reasoning aren't silently lost if someone is
  tempted to "optimize" it back to a cache later.
- **Off-diagonal by design, not as an afterthought.** The arena always
  plays the *full* guesser pool, held-out members included -- `held_out`
  is carried through into results and the SQLite rows purely as a label.
  It doesn't gate anything yet (none of the 3 baseline spymasters are
  data-driven, so "held out from training" has no referent for them),
  but it will matter the moment M8's learned spymaster exists, and
  getting the label plumbed through now means M8 doesn't need to touch
  the arena or DB schema to use it.
- Multiprocessing via `concurrent.futures.ProcessPoolExecutor` with a
  per-worker `initializer` that loads its own `SimilarityTensor` (own
  mmap handle onto the same file -- OS page cache shares the physical
  pages), the guesser pool, and the spymaster instances once, reused
  across every task that lands on that worker.
- SQLite schema is one row per turn (`codenames/arena.py::_init_db`):
  spymaster, guesser, guesser_held_out, board_seed, turn_index, clue,
  number, guesses (JSON), reward, ended_reason, game_outcome. Denormalized
  on purpose (game_outcome repeated on every turn row of a game) so a
  query never needs a join to filter by outcome.
- Real-data smoke run (`scripts/run_arena.py --n-boards 3`) surfaced a
  concrete, useful finding rather than just exercising the plumbing:
  `cautious_glove` (the confidence-threshold guesser, threshold 0.2)
  times out on every single game against every spymaster -- 0.0%
  win/assassin rate, 40 turns (the timeout cap), 0.000 own-words/clue.
  Its threshold is high enough that essentially no real clue clears it,
  so it always declines every guess. Not a bug -- SCOPE's guesser pool
  is deliberately supposed to include failure modes -- but worth flagging
  in case the threshold (chosen arbitrarily in M5) needs revisiting once
  M9's pool-sensitivity sweep happens.
- 19 new tests (`tests/test_spymasters.py`, `tests/test_game.py`,
  `tests/test_arena.py`), all using synthetic fixtures, no dependency on
  the real cache -- consistent with `test_guessers.py`/`test_similarity.py`.
  89 tests total, all passing. Arena tests use a tiny synthetic tensor but
  the *real* 400-word board vocabulary (`load_wordlist()`), since
  `Board.generate()` needs to actually find its sampled words in the
  fixture's board vocabulary.
- `scripts/run_arena.py`: CLI wrapper, `--n-boards` (seeds 0..n-1),
  `--max-workers`, `--max-turns` override, prints the win/assassin/turns/
  own-per-clue matrix plus per-worker peak RSS.

## M7 — Features and data generation

**Expected:** `features.py` per §2, with permutation-invariance and masking
tests written before anything is built on top of it. A data-generation
script sampling boards (including partial-reveal states), clues, and
guessers per SCOPE's 60/30/10 mix, producing appendable mmapped
`(features, k, reward)` output. Target 5-20M examples; report
examples/second.

**Actual:** matched expectations, with two non-obvious spec gaps resolved
explicitly (documented in `features.py`'s own docstring, not just here)
and a real throughput bottleneck found and partially fixed. Notes:

- **§2's feature-vector spec has two genuine ambiguities**, not just
  implementation details -- both are the kind of "non-obvious design
  choice" CLAUDE.md says to flag rather than silently pick:
  1. *Sorting is independent per space* (§2 step 3), which means "slot k"
     can be a different underlying board word in different spaces. A
     validity mask therefore can't be both space-specific *and* aligned
     to the independently-sorted positions without emitting one mask per
     space (§2's arithmetic -- "100, plus mask" -- only works out to
     ~115 if there's exactly one mask, not four). Resolved: one shared,
     space-independent mask that means "this role-position is a real
     unrevealed word," full stop -- not "this word has a vector in this
     specific space." A board word missing a vector in one space still
     gets that space's own -1 in its value slot; the mask doesn't flag
     it, since these are curated common-English board words where that's
     rare (unlike the clue vocabulary, where per-space gaps are the
     point of the whole union-vocab design from M2).
  2. *Sentinel is -1, not NaN* -- looks like it contradicts the
     established project convention (NaN for "no vector," never 0/-1,
     specifically so the model can't confuse "missing" with "confirmed
     value" -- see M2's log entry). It doesn't actually contradict it:
     that convention is for *stored, inspected* data (the tensor on
     disk, code that does `isnan()` checks); a feature vector about to
     hit a forward pass can't contain NaN at all (propagates and
     corrupts every downstream computation), so it needs a fixed
     placeholder no matter what -- which is exactly why §2 pairs the -1
     sentinel with an explicit mask rather than relying on the sentinel
     value alone to carry meaning.
- Feature width is `25*n_spaces + 25 + 3` -- currently 103 with 3 spaces
  (fastText doesn't exist yet), 128 once fastText joins. Neither is
  exactly SCOPE's worked "~115," which assumed 4 spaces and didn't specify
  mask width precisely enough to pin down a single number -- both are
  the right order of magnitude for what "~115" was gesturing at.
- `turn_index` and `score_differential` (the other two of the 3 scalars,
  alongside own-words-remaining) aren't specified anywhere in SCOPE beyond
  being named -- there's no 2-team turn structure in this single-team game
  (see game.py's module docstring), so "score differential" is computed
  directly from the board's own revealed counts (own-revealed minus
  opponent-revealed), not from simulating an opposing team. `turn_index`
  for *sampled* (as opposed to actually-played) states is approximated as
  the total number of revealed words at sampling time.
- 9 new tests in `tests/test_features.py`, covering exactly what SCOPE
  called out by name: permutation invariance (same word/role pairs in a
  different card order, and the same words revealed in a different
  order, produce bit-identical feature vectors) and masking (mask matches
  remaining-count exactly; padded slots hold the sentinel; a word missing
  a vector in one space doesn't corrupt the vector or wrongly flip the
  shared mask). Written and passing before `generate_training_data.py`
  was started, per SCOPE's explicit ordering.
- **Clue sampling and rollout simulation, not spymaster reuse.** M8's
  target label is "how many own-words would this guesser reveal for this
  clue" -- a property of (board, clue, guesser) alone, with no spymaster
  or chosen "number" involved. `simulate_natural_stop()` reads the
  guesser's own ranking and peeks at `board.role_of()` without ever
  calling `board.reveal()`, so many (clue, guesser) pairs can be rolled
  out against the identical sampled board state with no board-copying.
  Capped at `MAX_K` (reusing `spymasters.base.MAX_CLUE_NUMBER` -- same
  constant, same meaning, imported not redefined).
- Guessers are sampled from `training_pool()`, never `load_pool()` -- this
  is the concrete point where the held-out/training split (§3, built in
  M5, labeled-but-inert through all of M6) actually starts to matter:
  M8's model will never see a single example generated by `numberbatch`
  or `rank_based`.
- **Real throughput bottleneck found, partially fixed.** Initial
  real-data smoke run: ~21-24 examples/sec. Profiling isolated the cost
  entirely to `sample_clue()`'s scoring step (`mean_similarity_to_words`),
  not `build_features` or `simulate_natural_stop` (both sub-millisecond).
  First fix attempt (cache each board word's full-clue-vocabulary column
  once, since there are only ~400 possible board words repeated across
  millions of examples) roughly doubled throughput to ~49/sec -- real,
  but smaller than hoped, because a second profiling pass showed the
  *disk read* wasn't actually the dominant cost (a warm, cached column
  still cost the same on the next call): the remaining ~20ms/call is
  `nansum`/`isnan` over the full ~532k-length clue-vocabulary array,
  unavoidable at this per-example granularity without changing the
  algorithm (e.g. batching many examples' clue-searches into one
  vectorized pass, or approximate nearest-neighbor structures over the
  clue vocabulary). Deliberately not chased further in this milestone --
  flagged here as the concrete next lever, alongside the multiprocessing
  SCOPE §7 already anticipates ("data generation ... CPU-bound and
  parallel") for scaling toward the 5-20M target. At ~49/sec on one
  core, that target is a multi-day job as currently written; this is
  reported honestly rather than either quietly eating the cost or
  over-engineering a bigger redesign into this pass.
- Output is sharded `.npy` triples (`features_NNNNN.npy`,
  `k_NNNNN.npy`, `reward_NNNNN.npy`) under `cache/training_data/`,
  each independently mmap-loadable -- "appendable" concretely means
  re-running the script adds new shards after whatever's already there,
  rather than needing the eventual total size up front.
- Small refactor while wiring this up: pulled `top_legal_clue` (renamed
  from a spymaster-only helper) and the centroid "mean similarity to a
  word subset" logic out of `spymasters/_util.py` and `centroid.py` into
  a new top-level `codenames/clue_search.py`, since M7 needed the exact
  same "score every clue, find the best legal ones" logic and it isn't a
  spymaster concept. `mean_from_columns()` is split out from
  `mean_similarity_to_words()` specifically so the data-generation
  script's column cache (kept local to that script, not in the shared
  module) can reuse the math without re-reading from disk -- deliberately
  *not* added to the shared module itself, since the arena already had
  one cache-related RSS blowup fixed this milestone
  (`spymasters/linear_scorer.py`) and a shared cache there would
  reintroduce the same multi-worker multiplication risk for
  `CentroidSpymaster`.
- 12 new tests in `tests/test_generate_training_data.py` (board sampling
  never reveals the assassin and always leaves >=1 own word; clue mix
  fractions sum to 1; the rollout never mutates the board; k is correctly
  capped and the miss's reward is correctly folded in; shards have the
  right shapes/dtypes and k stays in [0, MAX_K]; held-out guessers are
  never sampled; re-running adds a new shard rather than overwriting).
  110 tests total, all passing.

## M8 — Scorer

**Expected:** MLP per §2 (input -> 256,256,128 -> 5 logits). Training
script with board-seed splitting, early stopping, checkpointing, training
curves, validation reliability diagrams. `spymasters/learned.py`
implementing play-time scoring with a runtime risk-aversion parameter.
Register with the arena.

**Actual:** matched expectations, after resolving one real gap in §2's own
formula (flagged to the user before building, per CLAUDE.md) and fixing
two circular-import bugs the new dependency direction exposed. Notes:

- **§2's play-time formula has a genuine hole**: the model only outputs
  P(k|clue) -- how many own-words get revealed before stopping -- with no
  information about *why* it stopped (neutral/opponent/assassin all
  collapse into "not own"). But `reward(k,n)` needs a penalty value, and
  the per-category table (0/-1/-10) can't be recovered from k alone.
  Raised this to the user directly (two options: a single
  runtime-adjustable "miss penalty" standing in for any stop, vs.
  expanding the model to also predict stop-category). Went with the
  former, which is also what SCOPE's own sentence -- "the assassin
  penalty is the risk-aversion parameter" -- literally says: one constant
  (default -10) charged on *any* stop, adjustable at play time with zero
  retraining, since P(k|clue) was never trained against a specific
  penalty value to begin with. Documented in full in `scorer.py`'s module
  docstring, including the resulting bias (a neutral miss gets charged as
  if it might have been the assassin) and a second smaller approximation
  (k=4 is a right-censored ">=4 or more" bucket in the training labels;
  treated as exactly 4 when computing reward(4, n), which only actually
  matters at n=4).
- **Two circular imports surfaced by the new dependency direction**
  (`spymasters/learned.py` -> `scorer.py`, the first time anything under
  `spymasters/` needed something outside it going the *other* way).
  Both fixed by relocating rather than restructuring: `MAX_CLUE_NUMBER`
  moved from `spymasters/base.py` to `board.py` (its natural home is
  arguably neither, but board.py has zero internal dependencies, so
  nothing importing it can ever cycle); `game.py`'s import of
  `Spymaster` moved behind `TYPE_CHECKING` (it was only ever used as a
  type hint, and `from __future__ import annotations` already makes every
  annotation in that file a lazy string at runtime). Both bugs were
  invisible to `pytest` -- test files happened to import submodules in an
  order that never triggered the cycle -- and only surfaced running
  `scripts/train_scorer.py` directly. Worth remembering: a clean test run
  is not proof an import graph is acyclic.
- **Full-vocabulary scoring needed a vectorized feature builder.** §2
  requires scoring all ~250k+ candidates via "one gather plus one small
  forward pass" -- calling `build_features()` in a Python loop over ~530k
  clues would dominate every turn's cost. Added
  `features.build_features_batch(board, sims, turn_index)`: the mask and
  scalar blocks don't depend on the clue at all, so they're computed once
  and broadcast; the per-space value blocks are built with one vectorized
  sort per (role, space) across the *entire* clue axis at once (NaN
  swapped to -inf before a descending sort so missing entries reliably
  land in the padding region, then swapped to the real -1 sentinel).
  Tested for exact numeric agreement against `build_features()` called
  once per clue, on a fixture with real missing-vector entries -- the two
  code paths computing bit-identical results is the actual correctness
  guarantee here, not just "it runs."
- `spymasters/learned.py`'s `give_clue()` needs `turn_index`, which isn't
  part of the `Spymaster` interface (no turn counter is threaded through
  `game.py`/`arena.py`). Uses the same proxy M7's data generation used to
  *label* training examples (count of currently-revealed words) --
  deliberately, since using a different proxy at play time than at
  training time would be a silent train/serve skew nobody would notice
  until the model behaved worse in the arena than in validation.
- `scripts/train_scorer.py`'s board-seed split is a deterministic hash
  (`seed % 1000 < val_fraction * 1000`), not a loaded-then-shuffled split
  -- every example from a given board lands in the same partition
  regardless of shard file, and the split is stable across re-runs or
  newly-added shards without needing the whole dataset in memory first.
  `ShardedTrainingData` keeps feature shards memory-mapped and only loads
  the small `k`/`seed` arrays fully, since filtering has to inspect every
  row's seed anyway.
- Added `matplotlib` as a new dependency (training curves + per-class
  reliability diagrams -- SCOPE explicitly asks for both, and nothing
  already in `pyproject.toml` plots).
- Real-data smoke test (3,000 examples generated, 8 epochs trained, then
  `run_arena.py --checkpoint ...` with `learned` added to the spymaster
  set): trained-and-scored end to end, including through the arena's
  multiprocessing path. Not a real result -- 3,000 examples and 8 epochs
  is nowhere near the 5-20M target M7 flagged as a multi-day job at
  current throughput -- purely a plumbing check. Worker RSS with the
  learned spymaster included rose to ~2.9GB (vs. ~1.5GB for M6's
  baselines-only run), from the model + the batched full-vocabulary
  feature matrix; still comfortably within SCOPE §7's budget at any
  reasonable worker count, so left as-is rather than optimized preemptively.
- 24 new tests: `tests/test_scorer.py` (reward-matrix cells checked
  against hand-derived values, including the k=4/n=4 censoring case;
  batched expected-reward math; risk-aversion actually changing the
  chosen number), `tests/test_train_scorer.py` (seed-based split
  correctness, sharded dataset filtering, and a full train() smoke run
  asserting the checkpoint and both diagnostic images exist),
  `tests/test_learned_spymaster.py` (legal-clue output, the full 0..4
  number range being reachable, the turn_index proxy, risk-aversion
  plumbing), plus 2 more in `tests/test_features.py` for
  `build_features_batch`. 134 tests total, all passing.

## Design revision (post-M8): restricted vocab, 3-guesser noisy pool, held-out board words

Not a milestone -- a deliberate revision of already-checked-off M2/M5 work,
made before starting M9, after reviewing early arena/inspector results.

**What prompted it:** anecdotally, numberbatch's per-clue rankings looked
most reasonable across a few manual checks. The first proposal was to make
numberbatch + noise *the* scoring metric, on the reasoning that a real
spymaster can't know exactly how a guesser thinks, so some noise should
stand in for that uncertainty.

**Why that specific proposal was rejected:** it's structurally identical to
the "GloVe alone" anti-pattern SCOPE §3 already warns against (a guesser
with no vector for "Technoblade" guesses badly, so the clue gets labeled
bad, so the model learns to avoid the exact clue the project wants to
enable) -- just with a different embedding as the single source of truth.
Gaussian noise only perturbs *that guesser's own* ranking; it can't
simulate a listener with genuinely different knowledge (e.g. one who knows
a proper noun numberbatch's graph doesn't cover). Noise models aleatoric
uncertainty in one decision process; it doesn't substitute for testing
against a structurally different one.

**How it was resolved**, across several rounds of back-and-forth:

1. **Restrict clue vocabulary to common words** (no pop-culture/proper-noun
   push for this first pass) to shrink -- not eliminate -- the
   different-embeddings-know-different-things problem. Landed on
   *intersection* rather than a frequency-based cutoff: every legal clue
   now has a real vector in every currently-built space, by construction.
   Rebuilt `scripts/build_similarity_tensor.py`'s vocabulary from a union
   (~532,738 words, the M2 fix) to an intersection: **111,440 words**, all
   three spaces at 100% coverage (down from GloVe's ~52.8% coverage of the
   old union). Real rebuild against the actual cached embedding files:
   GloVe+intersection ~47s, Numberbatch extend ~6s, Wikipedia2Vec extend
   ~256s (its source file has to be scanned for the candidate tokens, same
   cost structure as the original M4 build). A pleasant side effect,
   unplanned but not surprising: `generate_training_data.py`'s throughput
   roughly *tripled* (~45-50/sec -> ~126/sec) purely from the clue
   vocabulary shrinking ~4.8x, since the dominant per-example cost (§M7's
   log entry) scales with vocabulary size.
2. **Guesser pool: equal-weighted noisy version of each of the 3 currently-
   built embeddings, all training-visible.** This is genuine knowledge
   diversity (three different embedding types), not "one guesser + noise"
   -- the same principle SCOPE §3 states, just a smaller pool (3 members,
   not ~8) since this first pass doesn't need blends/rank-based/confidence-
   threshold variety to make its point. Required one small registry
   enhancement to get *exactly* 3 pool members rather than 3 wrappers plus
   3 redundant raw bases: `codenames/guessers/registry.py`'s wrapper `base`
   can now be an inline anonymous `{"type", "params"}` object, not just a
   string name referencing an earlier, separately-visible pool entry.
   Fully backward compatible (string bases still work). New
   `configs/guesser_pool.json`: `noisy_glove`, `noisy_numberbatch`,
   `noisy_wikipedia2vec`, `noise_std=0.15` (matching the old `noisy_blend`
   convention), distinct seeds, no `held_out` on any entry.
3. **Held-out guessers vs. held-out board words.** Proposed dropping
   held-out entirely once the pool shrank to 3 (holding 2 out would leave
   exactly one training-visible guesser -- the single-guesser problem
   again, just via the held-out mechanism instead of the pool design).
   Pushed back once: noise and held-out-ness address different failure
   modes (aleatoric label noise vs. overfitting to the specific decision
   functions trained against -- the latter is what M6's "off-diagonal
   results are what matter" framing and M10's human eval both lean on).
   Landed on a genuinely different mechanism instead of dropping the
   concept: hold out *board words*, not guessers. 60 of the 400 board
   words (`random.Random(42).sample(...)`, committed to
   `codenames/assets/board_words_holdout.txt`) are now excluded from all
   training data generation
   (`codenames/board.py::load_training_wordlist()`,
   wired into `scripts/generate_training_data.py::generate()`'s default).
   This checks a different, complementary thing than held-out guessers did
   (generalizing to unseen board *content* vs. unseen *listener type*) --
   not a replacement in the sense of testing the same property a different
   way, but a deliberate substitution given the pool is now too small to
   afford the original mechanism.

**Verification:** all 145 tests pass (11 new: registry inline-base +
held-out-mechanism tests decoupled from default-pool composition, since
the old tests conflated "does the mechanism work" with "does the current
config happen to have this shape" -- exactly what just broke when the
config changed; board holdout/training-wordlist partition tests).
Real-cache smoke tests: `scripts/inspector.py` shows exactly 3 guessers,
all tagged "training"; 200 sampled training boards, zero contained a
held-out word.

**Deferred, not forgotten:** fastText/Fandom corpus work (M0/M4's other
half) is lower priority under this simplification -- it exists
specifically to supply the pop-culture knowledge this first pass isn't
depending on. M9's ablations (space/sort/concatenation, linear baseline,
pool-sensitivity sweep) still make sense and pick up from here, now against
the revised vocabulary and pool.

## M9 — Evaluation and ablations

**Expected:** ablate each embedding space, ablate sorting, ablate
concatenation (average instead), a pool-sensitivity sweep, and a linear
model over the same features with coefficients reported (§6 baseline 4;
the gap between it and the MLP is the project's stated headline result).
Adapted the pool-sensitivity axis to the post-revamp 3-guesser pool
(sweep which embedding dominates the training mix) rather than SCOPE's
original glove/fastText/uniform/adversarial framing, which assumed the
old ~8-guesser pool. "Moderate real run" scale, per explicit user choice:
real numbers on an illustrative-scale dataset, not the 5-20M target.

**Actual:** matched expectations, plus a design simplification found while
planning that avoided a much heavier implementation, and a genuinely
interesting result tying back to the very start of this conversation.
Full numbers: `docs/m9_ablation_report.md`. Notes:

- **Key simplification: most ablations don't need new data at all.**
  `generate_training_data.py`'s sampling is driven entirely by a
  `random.Random(seed)` instance, and feature computation happens *after*
  a board/clue/guesser are already sampled -- it never itself consumes
  randomness. So: the drop-space and averaged-concatenation ablations are
  pure post-hoc array transforms on an *already-generated* dataset's
  feature vectors (new `codenames/ablation.py`, using `FeatureLayout` to
  know which columns are which) -- no regeneration, no new storage. The
  unsorted-similarity ablation needs a second feature-builder function
  (`build_features_unsorted`) but not a reconstruction pipeline -- just
  regenerating with the same seed reproduces the identical sampled
  boards/clues/guessers while computing different features for them. The
  pool-sensitivity sweep similarly needed only a `guesser_weights` sampling
  parameter (`rng.choices` instead of `rng.choice`), not per-example
  guesser-identity storage -- each composition is its own regeneration,
  and using the same seed across all 4 compositions means they share the
  identical underlying board/clue sample sequence, differing only in which
  guesser scored each one (holding the "exam questions" constant, varying
  only "who's grading," the cleanest version of this comparison). Net
  effect: zero changes to the M7 shard schema, despite covering every
  ablation axis SCOPE asks for.
- `scripts/train_scorer.py::train()` gained one parameter
  (`model_factory`, defaulting to `Scorer`) to support the linear baseline
  (new `LinearScorer` in scorer.py, a single `nn.Linear`) -- everything
  else (splitting, early stopping, checkpointing, curves, reliability
  diagrams) was already architecture-agnostic and needed no changes.
- Real run: 600k fresh examples generated (200k base, 200k unsorted, 4x50k
  pool-sensitivity) in ~37 minutes, plus 11 total model trainings
  (~20 epochs each with early stopping) in a few minutes combined --
  faster than the ~2-hour estimate in the plan, since throughput measured
  higher in practice than the M7/revamp estimate it was based on.
- **Every pre-registered sanity check confirmed in the expected direction**
  (val_loss, lower is better): MLP (0.9648) beats the linear baseline
  (0.9843); all three drop-space variants (0.9698-0.9731) underperform the
  full model; unsorted (0.9768) underperforms sorted; averaged (0.9668)
  underperforms full concatenation, though by a small margin at this
  scale -- reported honestly rather than oversold, and a candidate for a
  larger run to see if the gap widens with more data/epochs. The overall
  MLP-vs-linear gap is real but modest here too, for the same reason:
  illustrative scale, not the 5-20M target.
- **Pool-sensitivity result worth flagging on its own**: among the 4
  same-size (50k), same-underlying-samples pool compositions,
  numberbatch-heavy scored best (val_loss 0.9523, the best of *all* 11
  variants including the 200k-example full model) and wikipedia2vec-heavy
  scored worst (0.9878) -- directly echoing the anecdotal impression that
  kicked off this whole design-revision conversation ("numberbatch seems
  to have the most reasonable results"). Not proof of anything on its own
  (different guesser weightings change what the *label* rewards, so a
  model trained against a numberbatch-heavy mix doing well by that same
  mix's own standard is a softer claim than it first looks), but a
  concrete data point in favor of the earlier intuition, worth revisiting
  once a larger run is affordable.
- Linear baseline's top-weighted features (by L2 norm across the 5
  k-classes, labeled via new `FeatureLayout.describe()`) are almost
  entirely `*/own/rank0` and `*/own/rank1` across all three spaces, plus
  `*/assassin/rank0` -- i.e. the model's strongest learned signal is
  exactly the intuitive spymaster heuristic ("how close is this clue to
  my best own word, and how close is it to the assassin"), which is a
  clean, defensible, oral-defense-ready interpretability result (§9's
  explicit ask).
- 25 new tests (`build_features_unsorted`, `FeatureLayout.describe`,
  `codenames/ablation.py`'s two transforms, `LinearScorer`,
  `train_scorer.train()` with a custom `model_factory`). 162 tests total,
  all passing. `scripts/run_ablation_study.py` itself has no unit tests --
  like `run_arena.py`, it *is* the integration test, verified by actually
  running it (first at toy scale as a smoke test, then for real).

## Post-M9: noise reduction + a real numbering-convention bug found via the web UI

Using the web inspector's top-K clue browsing (previous entry) surfaced two
real findings, not just UI polish:

- **`learned:full` was picking `number 0` on nearly every board.** Measured
  the actual real-data similarity spread to check a hypothesis raised while
  discussing this: for a given clue, real per-space similarity across a
  board has std ~0.05-0.08 (measured directly against the rebuilt cache),
  but `noise_std=0.15` in `configs/guesser_pool.json` was 2-3x that --
  large enough to frequently overturn the *true* ranking rather than
  merely perturb it, not "some realistic uncertainty." Lowered to `0.03`,
  meaningfully below the smallest space's natural spread, for all three
  guessers. This only affects data generated from here forward -- the
  existing `cache/checkpoints/` and `cache/m9/checkpoints/*/` models were
  all trained under the old 0.15 and won't reflect this until retrained.
- **`OracleSpymaster` had a real off-by-one**, caught by explicitly
  confirming the numbering convention rather than assuming it: `number`
  is supposed to be the intended word count directly (every other
  spymaster already follows this via `spymasters/_util.py::natural_number`;
  `codenames.game.play_turn` then applies the standard "+1 bonus guess"
  itself, unchanged). Oracle was reporting `run_length - 1` instead of
  `run_length` -- silently under-announcing by one word relative to every
  other spymaster's convention. Fixed; `number` now equals the run length
  directly everywhere.

## Post-M9: noise sweep confirms the hypothesis, with a real gotcha along the way

Followed up the noise-reduction entry above with an actual controlled
sweep rather than a single before/after guess.

- **Parallelized first.** `run_ablation_study.py`'s 6 (later 11, once the
  noise sweep was added) dataset-generation calls are independent and
  CPU-bound, so they moved from sequential to a `ProcessPoolExecutor`
  (16 logical cores available) -- ~40min sequentially down to ~14min for
  the full 1.6M-example run. Also tuned `train_scorer.py`'s `DataLoader`
  (`num_workers=4`, `pin_memory`, default `batch_size` 512->2048): the
  MLP is tiny enough that small batches barely exercise the GPU per step.
- **Added an opt-in `--noise-levels` sweep**, structurally identical to
  the pool-sensitivity sweep: one dataset+model per requested `noise_std`,
  all sharing the same generation seed as `base` (identical underlying
  board/clue samples) and the same per-guesser seeds (1/2/3), so only the
  noise magnitude differs between levels -- a clean, directly-comparable
  sweep rather than independently-noisy runs.
- **First sweep run reused `cache/m9/`'s existing directories** (by
  design -- `_generate_if_needed` skips anything already generated) and
  produced a real gotcha: `full`'s val_loss was *identical* to
  `noise_0_15`, not `noise_0_03` as expected. Turned out `base/`,
  `unsorted/`, and the `pool_*` dirs were left over from the *original*
  M9 run, generated back when the pool config's default was still
  `noise_std=0.15` -- the noise-config-lowering commit only changed what
  *future* generation would use, not anything already on disk. Caught by
  literally reading the report table rather than assuming it was correct
  -- worth remembering as a pattern (resumable/skip-existing pipelines
  are exactly where a stale assumption hides silently, since nothing
  errors, it just quietly reuses old data).
- **Full clean rerun** (wiped `cache/m9/`, regenerated everything fresh)
  gave numbers that are actually comparable to each other. Two headline
  results:
  1. **The noise sweep is a clean monotonic curve**: val_accuracy 77.6%
     at `noise_std=0.0` down to 59.7% at the old `0.15` default -- an
     18-point swing from one hyperparameter, confirming the
     measured-std-based hypothesis (earlier entry) wasn't just plausible,
     it was the dominant effect.
  2. **SCOPE §6's headline comparison got dramatically clearer at the
     lower noise level**: full MLP val_loss 0.5935 vs. linear baseline
     0.8382 (were 0.9648 vs. 0.9843 in the original, noisier M9 run --
     barely distinguishable). Every other ablation direction held too
     (full beats every drop-space variant, beats averaged-concatenation,
     clearly beats unsorted) and the pool-sensitivity ranking replicated
     (numberbatch-heavy best, wikipedia2vec-heavy worst) -- same
     qualitative story as the first M9 run, just now with a much larger,
     more convincing margin now that noise isn't drowning out signal.
- `docs/m9_ablation_report.md` updated to the clean rerun's numbers.
  Noise remains at `0.03` in `configs/guesser_pool.json` (not lowered
  further to 0.0) -- pure zero noise makes every guesser perfectly
  deterministic given its embedding space, which was a deliberate
  first-pass design choice (§3's "diversity in knowledge, not noise"
  still wants *some* imperfection modeled); `0.0` is available any time
  via `--noise-levels` if that tradeoff is revisited.

## Post-M9: dropped the +1 bonus guess entirely (real rule change, not just naming)

While testing the web inspector, the user noticed a clue of "organism 1"
still let the guesser take 2 guesses. Earlier in the project (see the
numbering-convention entry above) the explicit decision was to keep the
standard Codenames `n+1`-attempts rule and only fix what the *announced*
number meant. This time, after thinking it over, the user asked for the
actual rule to change: a clue announcing `n` should grant exactly `n`
guesses, not `n+1` -- reasoning that in normal human play, the `+1` is a
bonus a team *chooses* to spend when it still feels confident, not a
default extra guess every clue gets, and this project's guessers have no
notion of "still feels confident" to make that judgment call with.

Changed:
- `codenames/game.py::play_turn`: `attempts = ranked[:number]` (was
  `ranked[: number + 1]`).
- `scripts/web_inspector.py::_simulate_turn`: same change, it's a
  read-only duplicate of the same logic for the UI's turn-simulation panel.
- `codenames/scorer.py::reward_matrix`: the play-time reward formula had
  the `n+1` budget baked directly into its math (`budget_exhausted =
  (n+1) * OWN_REWARD`, `natural_stop` applied when `k <= n`). Shifted to
  `budget_exhausted = n * OWN_REWARD`, `natural_stop` when `k < n`.
- A nice side effect: the module docstring's old caveat about the
  top k-bucket (`k=MAX_K`, right-censored "MAX_K or more") being a
  "second, smaller approximation" is now just gone. Under the old `n+1`
  rule, `n` could reach `MAX_K` while attempts could exceed what a
  censored `k=MAX_K` could represent, forcing an assumed miss at the
  bonus attempt. Since `n` never exceeds `MAX_K` and there's no bonus
  attempt anymore, a censored `k` always means the true `k >= n`, which
  always lands correctly in the budget-exhausted branch regardless of
  how far past `MAX_K` the true value actually is. `TestExpectedRewardAndBestN`'s
  tests were rewritten around this -- e.g. a clue certain to get `k=4`
  now correctly prefers `n=4` (it used to prefer `n=3`, to dodge the
  since-removed approximation).
- No training-data regeneration needed: `scripts/generate_training_data.py`'s
  `simulate_natural_stop` (the thing that produces the `k` labels) has
  never depended on the announced `number` at all -- it simulates a
  guesser's *natural* stopping point over an unlimited ranking, capped at
  `MAX_K`. Only the play-time scoring math (`reward_matrix`,
  `expected_reward_and_best_n`) and the actual game loop's attempt count
  needed to change, both pure functions of already-trained `P(k|clue)`
  outputs.
- Updated comments/docstrings/help-text referencing the old `n+1`
  convention across `codenames/guessers/base.py`,
  `codenames/spymasters/oracle.py`, `scripts/inspector.py`, and the web
  UI's simulation-panel placeholder text (`scripts/webui/inspector.html`).
- `docs/SCOPE.md`'s play-time-scoring section now documents this as an
  explicit divergence from standard Codenames rules.

All 169 tests pass after the change (several in `tests/test_scorer.py`
were rewritten, not just relabeled, since the actual reward values for a
given `(k, n)` pair changed).

## Post-M9: (k, cause) scorer redesign -- learn the difference between an opponent miss and an assassin miss

The user's next question, working through the model design: "isn't the
guesser unable to distinguish which specific board word is the
assassin?" Answer, worked out in conversation: not the guesser (that's
correct by design, it never sees roles), but the *scorer*. Its label was
`k` alone -- how many own-words a guesser gets right before *any* miss --
so a stop-on-neutral, a stop-on-opponent, and a stop-on-assassin all
collapsed into the same training signal, and at scoring time every
predicted miss got charged the same flat worst-case `miss_penalty`
(-10, the assassin value). The model could never learn "this clue risks
the opponent" as different from "this clue risks the assassin" -- both
just looked like "risk of an early stop."

Fix: widen the label from 5 classes (`k` in 0..4) to 13 classes, `(k,
cause)` jointly -- `codenames.scorer.outcome_class`/`decode_outcome_class`
pack/unpack `k*3 + cause_index` for `k` in 0..3 across
{neutral,opponent,assassin}, plus one class for `k=MAX_K` (censored,
cause undefined, no miss happened). `reward_matrix`/
`expected_reward_and_best_n` now take four independent reward
parameters -- `own_reward`, `neutral_reward`, `opponent_reward`,
`assassin_reward` -- instead of one `miss_penalty`, so a natural-stop
cell charges the *cause's own* value instead of a flattened one.
Confirmed with the user before implementing: `neutral_reward` defaults to
the true game's 0.0, not `spymasters/linear_scorer.py`'s baseline-3
constant of -0.3 (an untuned illustrative value for a different, hand-
coded spymaster, not something the actual scorer should be trained
against).

Also added, per the user's follow-up request: all four reward values are
now runtime-adjustable in the web UI (previously only the assassin one,
as "risk aversion"), for the same reason risk-aversion already was --
none of them are baked into training, only into the scoring formula
applied after inference. And a play-time noise-level dial on the turn-
simulation panel, picking which of the 5 already-trained noise-level
guesser pools (0.0/0.03/0.06/0.1/0.15) to simulate a clue against --
independent of which noise level the spymaster itself was *trained*
under, so a train/test noise mismatch can be explored directly
(`codenames/guessers/registry.py::load_pool` widened to accept an
in-memory config dict, not just a file path, so the 5 pools can be built
once at server startup without writing temp files).

Mechanically: `simulate_natural_stop` (scripts/generate_training_data.py)
now returns `(k, cause, reward)` instead of `(k, reward)` -- it already
computed the stopping role internally, just didn't return it. `generate()`
encodes via `outcome_class` and writes `outcome_*.npy` shards (renamed
from `k_*.npy`). No regeneration was needed for the *sampling* itself --
board/clue/guesser sampling never depended on the label scheme -- but the
*label* changed, so all training data needed fresh generation regardless
(the old `k_*.npy` shards don't carry cause information at all).

Also did the previously-agreed cleanup alongside this: the web UI's
spymaster dropdown is now exactly `random`, `centroid`,
`oracle:numberbatch`, plus 5 `learned:noise_*` entries (was 15+ stale M9
ablation-study checkpoints). `web_inspector.py::_discover_checkpoints`
now globs `noise_*/scorer_best.pt` specifically rather than every
subdirectory, so a future full ablation-study rerun (drop-space,
pool-sensitivity, etc. -- kept in `run_ablation_study.py` for research
use, just not permanent UI options) won't repopulate the dropdown with
those again. Added `--noise-only` to `run_ablation_study.py` so
refreshing just the 5 UI checkpoints doesn't pay for the other 6 axes.

**Retrain results** (`cache/m9/`, wiped and regenerated fresh under the
new 13-class label, `--noise-levels "0.0,0.03,0.06,0.1,0.15" --noise-only`,
200k examples each, ~754s generation for all 5 in parallel + ~1-2min
training each):

| noise_std | val_loss | val_accuracy |
|---|---|---|
| 0.0  | 0.8507 | 0.6810 |
| 0.03 | 0.9766 | 0.6350 |
| 0.06 | 1.1788 | 0.5704 |
| 0.1  | 1.4123 | 0.5024 |
| 0.15 | 1.6228 | 0.4332 |

Same clean monotonic noise relationship as the original 5-class sweep, as
expected -- noise magnitude is still the dominant effect, that finding
didn't depend on the label scheme. Both val_loss and val_accuracy are not
directly comparable to the old 5-class numbers (a 13-way classification
problem has a higher entropy floor and more ways to be "close but wrong"
than a 5-way one), so this isn't evidence of the new model doing worse --
it's a harder, more informative prediction target by construction.
`docs/m9_ablation_report.md` is deliberately left un-updated -- it's a
dated snapshot of the old architecture's full 11-variant study, not
reproduced today (out of scope for this change; would need a fresh full
run, not just the noise axis, to be a fair comparison).

All 184 tests pass (169 + 15 new: `outcome_class`/`decode_outcome_class`
round-trip and validation tests, `reward_matrix`'s per-cause behavior,
`load_pool` accepting a dict, `LearnedSpymaster`'s 3 new reward params).

## Post-M9: web UI clue-rarity filter, and a noise-level observation to revisit

Two things from playing with the retrained UI. First, an anecdotal
observation worth recording even though nothing was changed in response
to it yet: the user's impression, trying different `learned:noise_*`
spymasters in the UI, was that clue quality subjectively peaked around
`noise_std` 0.06-0.10, not 0.03 -- despite 0.03 having the better
val_accuracy of the two (63.5% vs. 57.0%/50.2%). Val_accuracy measures
"how often does the model's top-1 prediction match the exact (k, cause)
outcome," which isn't the same thing as "how good do this model's actual
clue choices feel" -- a model trained against a noisier pool may learn
more conservative, safer clue-giving that reads as better play even with
lower raw predictive accuracy. Not investigated further this session;
flagged here since it's a real tension between the metric being reported
and the thing that's actually supposed to matter, worth a closer look
before ever picking a single "default" model for actual play.

Second, a concrete UI feature: the user pointed out some given clues are
obscure (e.g. "confectionery") -- something few human players would
reliably know -- and asked for a way to filter those out. Added
`CLUE_RARITY_PERCENTILE` (scripts/web_inspector.py): GloVe's raw file is
frequency-descending ordered (already established and reused from
scripts/_embedding_lib.py/build_similarity_tensor.py -- verified
empirically, the file starts with "the"/"of"/"to"), and every clue word
is guaranteed to appear in it, since `build_similarity_tensor.py` only
ever admits a word into the clue vocabulary if it's among GloVe's own
top-N alphabetic words. So GloVe's own file position already *is* a
frequency rank, with no new dependency (no `wordfreq` package, no new
cache artifact) -- just a ~0.4s token-only scan (reusing
`_embedding_lib.ranked_alphabetic_words`) at server startup, converted to
a percentile *within the clue vocabulary itself* (0 = most common clue
word, 100 = rarest), not against the full 400k-word GloVe vocabulary --
the clue vocabulary already skews common by construction, so a
full-vocabulary percentile would make even a fairly obscure Codenames
clue look deceptively tame.

New `max_rarity` param on `/api/give_clue` (and a "max rarity %" UI
field, default blank = no filtering): excludes any candidate clue above
that percentile before truncating to `top_k`, via an over-fetch-then-
filter on the existing `top_k_clues()` mechanism (fetch up to 300
candidates instead of just `top_k`, filter, then truncate) -- no changes
needed to any spymaster class or `codenames/clue_search.py`, since the
forward pass that scores the whole vocabulary already happens regardless
of how many results are requested; asking for a bigger pool afterward is
close to free. Doesn't apply to `random` (nothing to rank). Each
candidate's rarity percentile is now also just displayed in the UI
alongside its score, even when no filter is active, for visibility.

No test suite changes needed beyond the manual verification above (this
is UI-facing plumbing over already-tested `top_k_clues`/percentile-derived
data, not new core logic) -- spot-checked end-to-end against the running
server: unfiltered top-5 for one seed included "stretching"/"overlooking"
(8.4th percentile), a `max_rarity=3` request against the same seed
correctly dropped both in favor of lower-percentile alternatives
("fort" 1.8, "area" 0.2). All 184 tests still pass (no regressions).

## Post-M9: an in-between noise level (0.08), and making run_ablation_study.py's training step actually cache-aware

Following up on the noise anecdote above, the user wanted a 6th noise
level in between the two they'd been comparing (0.06 felt too sharp, 0.1
too conservative) -- 0.08. Before adding it, checked whether re-running
`run_ablation_study.py --noise-levels "...,0.08,..." --noise-only` would
actually only train the one new model, since that's the entire point of
`_generate_if_needed`'s skip-if-exists mechanism. Turned out it half-did:
**generation** was already properly cache-aware (skips a variant whose
shards already exist on disk), but **training** was not --
`_train_variant` called `train()` unconditionally for every variant in
the run, so extending `--noise-levels` would have silently retrained all
5 existing models along with the new one every time, even though their
checkpoints were already sitting right there. Fixed by mirroring the
same skip-if-exists pattern onto `_train_variant`: if
`checkpoints/<name>/scorer_best.pt` and `training_curves.csv` already
exist, read the existing `training_curves.csv` for its best-epoch metrics
and skip straight to reporting them, instead of calling `train()` again.

Verified this actually works before trusting it: reran with
`--noise-levels "0.0,0.03,0.06,0.08,0.1,0.15"` and confirmed via
directory mtimes that only `cache/m9/noise_0_08/` was freshly touched,
the other 5 untouched. Log output confirmed the same at the training
step -- `[skip] noise_0_03 already trained (val_loss=0.9766
val_acc=0.6350)` etc. for the 5 existing ones, `[train] noise_0_08`
actually running (11s) for the new one alone. Total run time: seconds,
not the ~13 minutes a full 6-level regeneration would have cost.

Result: `noise_0_08` -> val_loss=1.3017, val_acc=0.5358 -- lands exactly
where expected between `noise_0_06` (1.1788/0.5704) and `noise_0_1`
(1.4123/0.5024) in the monotonic curve, consistent with everything else
in the sweep. (Per the entry above, these numbers describe how
predictable the 0.08-noise task is, not "how good" 0.08 is for actual
play -- that comparison still needs the arena, not val_accuracy.)

Also set new web UI defaults per the user's request: `max_rarity=90`,
default spymaster `learned:noise_0_08`, default simulation noise `0.08`
(`scripts/webui/inspector.html`'s `DEFAULT_SPYMASTER` constant and the
`maxRarity`/`simNoise` inputs' default values).

All 184 tests pass (`_train_variant`'s new skip path has no dedicated
unit test, same as the rest of run_ablation_study.py -- it's the
integration test, verified by actually running it, per its own module
docstring).

## Post-M9: project restructure -- versions instead of milestones

The initial build (what used to be tracked as M1-M9 in `docs/SCOPE.md` and
`CLAUDE.md`'s checklist) is done; the project is now in an iteration phase,
trying ideas to improve the model rather than building out a fixed spec.
The user asked to reflect that directly in the project structure rather
than keep the milestone framing: `docs/SCOPE.md` is retired (deleted, not
kept as a stale "source of truth" pointer), and the project is now
organized around **model versions** instead.

New layout:
- `README.md` rewritten as the actual entry point -- what the project is,
  how the three-stage pipeline works, how to train/test/inspect a model,
  the baseline ladder, directory layout, current status.
- `docs/design-decisions.md` (new) -- standing design rationale that isn't
  tied to any one model version (feature vector sorting/concatenation,
  guesser-pool diversity philosophy, the pool-as-unvalidatable-assumption
  mitigations, the first-pass simplifications, method decisions,
  environment notes, references). Salvaged from SCOPE.md's §1/§3/§4/§7/§10
  rather than deleted outright -- still load-bearing reasoning, just no
  longer framed as "the spec."
- `docs/versions/` (new) -- one doc per model version. `v1.md`: the
  original k-alone scorer (now historical/superseded, its ablation report
  moved here as `v1_ablation_report.md`). `v2.md`: the current (k, cause)
  scorer -- output shape, the four reward parameters, the six trained
  noise-level checkpoints' results table, what's built on top of it in the
  web UI, and an "open questions for a future version" section capturing
  the two brainstormed directions from this session (cross-turn clue
  memory, win-probability/score-aware risk) as the natural place a v3 would
  start from.
- `CLAUDE.md` rewritten: no more milestone checklist, no more "SCOPE.md is
  the source of truth" -- points to README.md + design-decisions.md +
  versions/ instead, conventions section made self-contained.

Deliberately did NOT do: a mechanical sweep of the ~33 files whose
docstrings cite "SCOPE.md §N" internally (board.py, features.py,
scorer.py, every guesser/spymaster, most scripts). Those citations are
inert historical breadcrumbs explaining *why* code is the way it is, not
functional file-path reads (confirmed via grep -- nothing actually opens
docs/SCOPE.md at runtime), and rewriting dozens of docstrings purely to
update a citation target is a much bigger, mostly-cosmetic, error-prone
undertaking than what was actually asked for. Flagged as a real but
optional follow-up if the user wants that broader cleanup too, rather than
either silently doing it or silently leaving it unmentioned.

## Post-M9: dropped the old model's docs entirely, renamed the current one to v1

Immediate follow-up to the versions-instead-of-milestones restructure
above. The user wanted the README to be stricter than "historical vs.
current": no reference anywhere to the old k-alone model at all (not even
as a labeled-superseded entry), and a real name for the model that's
actually live -- picking from "v1," "benchmark 1," or a descriptive name.

Deleted `docs/versions/v1.md` and `v1_ablation_report.md` outright (not
just unlinked from the README -- confirmed with the user first, since
that's a step further than what "in the readme" literally said). Renamed
what was `v2.md` to `v1.md` -- reusing "v1" for the (k, cause) scorer now
that there's nothing else it could be confused with -- and gave it an
actual name throughout ("the (k, cause) scorer") rather than leaving it
identified only by a version number. `linear_scorer` dropped from the
README's baseline list too (it's not registered in the web UI's dropdown
either, hasn't been for a while -- this just makes the README match what's
actually live rather than listing something dormant as if it were active).

Also restructured where the model-architecture explanation lives: it used
to be an early, unnamed "How the model works" section before any model had
been introduced. Moved the feature-vector/MLP-architecture/output-meaning
content into a section literally titled "Model 1: the (k, cause) scorer,"
positioned after the baselines list -- simple reference points first, then
the actual named model, matching how the project's own iteration story
("simplest first, improve from here") already reads.

No code changed, docs only. All 184 tests still pass.

## First real self-play evaluation, and a per-guess role breakdown for the arena

The user wanted the README's new "Model 1" section to have actual metrics,
not just an architecture description: run real full games with the
noise_0.08 spymaster against the noise_0.08 guesser pool, and report
average game length, assassin-hit rate, and how often each card type gets
hit.

`codenames/arena.py` already had win_rate/assassin_rate/mean_turns
per (spymaster, guesser) pair, but nothing broke guesses down by role.
Added four fields to `CrossPlayResult` -- `guess_own_rate`,
`guess_opponent_rate`, `guess_neutral_rate`, `guess_assassin_rate` -- the
fraction of every individual guess (across every turn of every game, not
per-game) landing on each role, tallied alongside the existing stats in
`run_arena()`'s single pass over results (no second DB query needed).
Deliberately named distinctly from the existing `assassin_rate` field,
which is a per-*game* rate (exactly one assassin card exists per board and
hitting it always ends the game immediately -- confirmed via
`play_game`'s logic, so `assassin_rate` already *was* "how often the
assassin is hit," no new field needed for that one).
`scripts/run_arena.py` prints this as a second table under the existing
one rather than widening the main table to 11 columns.

Ran it for real: `python scripts/run_arena.py --n-boards 300
--guesser-pool-config cache/m9/pool_configs/noise_0_08.json --checkpoint
cache/m9/checkpoints/noise_0_08/scorer_best.pt`. A 50-board timing probe
first (166s for 12 spymaster x guesser pairs with 4 workers) to size the
real run before committing to it; the full 300-board run (3600 games, 8
workers) took 859s (~14 min), landing inside the estimated window.
`LearnedSpymaster` runs its forward pass on CPU by default, which is
almost certainly the dominant per-turn cost here (~28 GFLOPs over the full
~111k-clue vocabulary every turn) -- noted as a real but nuanced
optimization target in conversation (GPU doesn't trivially help the
arena's process-parallel structure; the actual win would be batching
multiple games' turns into one large forward pass rather than many small
per-process ones), not acted on.

Results, `learned:noise_0_08` vs. that noise level's 3 guessers, 900 games:
97.1% win rate, 2.9% assassin-hit rate, 6.43 mean turns -- every game ended
in a win or an assassin hit, zero timeouts, so those two rates sum to
exactly 100% as expected. Per-guess breakdown: 85.2% own, 5.9% opponent,
8.7% neutral, 0.3% assassin. Same setup against the fixed baselines for
context: centroid 87.0%/13.0%, linear_scorer 71.4%/28.6%, random
11.1%/88.9% (win/assassin-hit). Both tables now live in
`docs/versions/v1.md` (full breakdown) and the README (summary).

Added a test asserting the four new per-guess rates are valid
probabilities summing to 1.0 (`tests/test_arena.py`); didn't assert exact
values since the fixture board's real (non-uniform) role distribution
combined with tied similarity scores makes the exact guess sequence
untestable without over-specifying guesser tie-breaking behavior. All 184
tests pass.

## GPU-for-arena benchmark, a game-setup invariant check, and splitting mean game length

Three follow-ups from discussing the self-play results above.

**GPU preliminary benchmark, not a change.** The user asked whether the
CPU-forward-pass observation from earlier ("could be a big optimization")
was actually worth pursuing, before committing to any rewrite. Benchmarked
directly rather than guessing: `codenames/scorer.py::Scorer`'s forward
pass over the full ~111k-clue vocabulary (one board) takes ~106ms on CPU
vs. ~1.85ms on GPU once batched (measured at 1x/4x/8x/16x/32x board
multiples stacked into one call -- GPU per-board marginal cost stayed
flat across all of them, meaning it's nowhere near saturated even at 32
boards batched together). That's a genuine 57x per-call speedup. But
`build_features_batch` (the numpy gather/sort producing that forward
pass's input) costs a separately-measured ~80ms/board, entirely CPU-bound
and untouched by moving just the model to GPU. So naively flipping
`LearnedSpymaster`'s device to "cuda" would cut per-turn cost from
~186ms to ~82ms -- a real but much smaller ~2.3x win, not 57x -- and that
estimate doesn't even account for whether 8 concurrent CPU worker
processes all sharing one GPU device would contend with each other
(not tested). The full order-of-magnitude win needs feature construction
batched *across multiple games* in one process too, replacing the arena's
current one-board-per-OS-process model -- a real rewrite. Conclusion:
confirmed real and quantifiable, not acted on -- worth it if/when bulk
simulation throughput actually becomes a bottleneck (it hasn't yet; the
300-board self-play run finished in an acceptable ~14 minutes).

**Verified the 9-card-team invariant.** The user wanted confirmation that
the team going first (per real Codenames rules, the team with 9 cards)
is the one actually being simulated. `codenames/board.py::ROLE_COUNTS`
hardcodes `Role.OWN: 9` -- not randomized, not a parameter -- and since
this project only ever simulates the "own" team's perspective (no second
team's turns exist at all, per `game.py`'s long-standing single-team
simplification), "own" is unconditionally the 9-card role by construction.
Verified empirically too, not just by reading the code: generated 2000
boards across seeds 0-1999 and confirmed the role counts are exactly
9/8/7/1 every single time, zero mismatches. Worth being precise about what
this does and doesn't confirm: there's no actual turn alternation between
two teams in this codebase to "go first" in, so the real content of the
check is "the simulated team's role always matches the standard starting
team's card count," which it does, unconditionally.

**Split mean game length into all-games vs. wins-only.** The user's point:
blending a game that ends abruptly on turn 1 via the assassin together
with a game that runs its natural course to a win answers neither "how
long do successful games take" nor "how early do doomed games end."
`codenames/arena.py::CrossPlayResult` gains `mean_turns_on_win` (`None`
when there were no wins to average, e.g. an all-loss or zero-game result)
alongside the existing `mean_turns` (now explicitly "all games, blended").
`scripts/run_arena.py`'s table now has both columns. New test
(`test_mean_turns_on_win_is_none_when_nothing_won`) uses `max_turns=0` to
force every game to time out with zero wins, checking the None path
directly rather than only the happy path. Reran the noise_0.08 self-play
evaluation with the same seeds (0..299, deterministic) to get the split --
win/assassin/per-guess numbers reproduced exactly, confirming determinism,
with the new turns-on-win figure added. All 185 tests pass.

Also, per the user's framing: win rate is being de-emphasized as the
headline metric going forward -- it's dominated by "how often the
assassin gets hit," which the per-guess assassin rate already answers
more directly, and win rate alone can't distinguish a model that plays it
extremely safe from one that's actually finding own words efficiently.
The per-guess role breakdown and the turns-on-win figure are the more
informative numbers; both READMEs/version doc updated to lead with those.

## Built the GPU-batched arena for real: 13x measured speedup, plus a real multiprocessing/CUDA bug found along the way

Follow-up to the GPU preliminary benchmark above. The user wanted to
actually pursue this, but asked for more testing on feasibility first
given it was flagged as a real rewrite, not a config flag -- prototyped
before committing to anything in the codebase.

**Prototype 1 (looped, one GPU call per board):** ported
`build_features_batch`'s gather/sort logic to torch, verified exact
numerical match against the numpy reference, timed it. Only ~25-33ms/board
-- barely better than numpy's ~80ms/board, because each board still costs
12 small GPU kernel launches (4 roles x 3 spaces) whose overhead dominates
actual compute.

**Prototype 2 (genuinely batched across boards):** padded every board's
per-role unrevealed-word index list to a common width with a validity
mask, so the gather+sort happens as one tensor op across *all* boards in
a batch, not one Python-loop iteration per board. Correctness verified
exactly again. Real payoff: ~42.8ms/board at batch=1 down to ~5.8ms/board
at batch=32, still improving, not yet flattened. Combined with the
forward pass's own ~1.85ms/board (from the earlier benchmark), that's
~7.7ms/board-turn end-to-end at batch=32 vs. ~186ms on CPU -- a ~24x raw
per-turn compute speedup. Told the user honestly that the *realistic
arena* speedup would be smaller than 24x, since run_arena already gets
real parallelism from 8 CPU worker processes -- the fair comparison is
"1 GPU process" against "8 CPU processes," which the numbers suggested
would land around 3x. Asked "how much faster" -- answered with that
distinction rather than just repeating the flashier 24x number.

**Built for real**, given the prototype validated cleanly:
- `codenames/gpu_features.py` (new): `build_features_batch_multi`, the
  production version of prototype 2, plus a GPU-tensor cache keyed by
  `id(sims)` (documented as relying on SimilarityTensor living for a
  whole process's lifetime in this usage, not a general-purpose cache).
  Kept out of `codenames/features.py` deliberately -- that module stays
  pure-numpy, no torch, since scripts/generate_training_data.py and
  others import it with no reason to pull torch in.
- `codenames/game.py::play_turn` gained an optional `clue_and_number`
  param -- skips calling `spymaster.give_clue()` when given, so the new
  batched runner can compute clues for many boards at once (off this
  function's hot path) and still reuse this exact, already-tested
  attempt/reveal/stop logic per board unchanged. Zero behavior change for
  every existing caller (default `None` preserves the old path).
- `codenames/arena.py` refactored to export `new_stats_accumulator`/
  `update_stats`/`finalize_result` -- the stats bookkeeping `run_arena`
  already did, extracted so the new GPU runner computes `CrossPlayResult`
  identically instead of maintaining a second copy that could quietly
  drift from the first.
- `codenames/gpu_arena.py` (new): `run_arena_gpu`, driving N games in
  lockstep per "round" (fixed batch groups, not a streaming refill queue
  -- simpler to get right, and games are short enough that the wasted
  compute on already-finished boards near a group's end is minor).
  Batches only the spymaster's clue *selection*; guessers and the
  turn-resolution logic are untouched, reused directly via `play_turn`.
  Only accelerates `LearnedSpymaster` -- every other spymaster already
  scores a handful of candidates, not the full vocabulary, so there's
  nothing for them to gain here.
- `scripts/run_arena.py` gained `--gpu-batch-size`: baselines still run
  through the normal `run_arena`, the learned spymaster routes through
  `run_arena_gpu` instead when this is set, results merged into one
  report.

**A real bug found while cross-validating, not by inspection.** Testing
"GPU spymaster first, then the CPU multiprocess path" to compare results
hung indefinitely -- traced to a genuine hazard: `ProcessPoolExecutor`
defaults to `fork` on Linux, and forking a worker process *after* CUDA has
been initialized in the parent hands the child a broken, unusable CUDA
context even though the child never touches the GPU itself. This was
latent in `codenames/arena.py` before today -- it just never mattered
until a single script could plausibly touch CUDA (via the new GPU path)
and then spin up `run_arena`'s worker pool in the same process, which
`scripts/run_arena.py --gpu-batch-size` now does routinely. Fixed by
switching `run_arena`'s `ProcessPoolExecutor` to `mp_context=
multiprocessing.get_context("spawn")` -- verified the exact
previously-hanging order (GPU then CPU) now completes cleanly. Also
surfaced (and fixed in the throwaway benchmark script, not the codebase)
the standard companion gotcha: a `spawn`-based script needs its
top-level code guarded by `if __name__ == "__main__":`, or workers
re-execute the whole module; `scripts/run_arena.py` already had this
guard, only the ad hoc comparison script didn't.

**Correctness, not just speed.** Built `tests/test_gpu_features.py` (exact
match against `build_features_batch`, including a real NaN/missing-vector
case and boards with different numbers of unrevealed words per role) and
`tests/test_gpu_arena.py` (exact match against `run_arena`'s CPU path
end-to-end, using a deterministic non-noisy guesser specifically so the
comparison isn't contaminated by `NoisyGuesser`'s stateful RNG depending
on task-to-worker scheduling -- see below; plus a test that batch size
itself doesn't change the result). Both skip automatically without CUDA.
All pass. Manually reran the exact-match comparison after the `spawn` fix
too, both call orders: `n_games`, `win_rate`, `assassin_rate`,
`mean_turns`, `mean_turns_on_win`, and all four `guess_*_rate` fields
matched exactly, not just approximately.

**Real end-to-end benchmark**, matching the earlier self-play evaluation's
exact setup (noise_0.08 spymaster, that noise level's 3 guessers, 300
boards each, 900 games): CPU path (8 workers) took 716.4s; GPU path
(batch_size=32) took 54.9s. **13.05x measured speedup** -- notably better
than the ~3x estimated beforehand, because the estimate assumed each CPU
worker keeps hitting its solo ~186ms/turn baseline under 8-way concurrency,
but 8 processes each internally using multi-threaded BLAS/numpy contend
for the same 16 cores in practice (observed >100% CPU per worker even
before this session's changes), degrading real per-worker throughput well
below the naive "just divide by 8" assumption. Win rates were close but
not identical between the two runs (e.g. 99.0% vs. 97.0% for one guesser)
-- expected, not a correctness concern: `NoisyGuesser`'s RNG is stateful
and shared across every task a given worker process happens to handle, so
its exact draw sequence depends on real-time task-to-worker scheduling,
which isn't guaranteed identical run to run even with fixed seeds (this
was already true before today, just not previously measured this
directly). The deterministic-guesser test above is what actually proves
correctness; this benchmark's job was throughput, not bit-for-bit
reproduction.

Also discovered along the way, worth its own note: **`docs/versions/v1.md`'s
earlier "run to run" comparison** (the mean_turns_on_win rerun a few
entries up) showed small differences from the original self-play numbers
for exactly this same stateful-RNG-plus-scheduling reason, not a bug in
that feature.

190 tests pass (185 + 5 new: 3 `test_gpu_features.py`, 2
`test_gpu_arena.py`).

**Made the GPU path the default, not opt-in**, per explicit user request
mid-session ("let's not run the cpu one anymore, just the gpu one, I
don't want to have to wait"): `scripts/run_arena.py --gpu-batch-size`
now defaults to `32` (was `None`/off); added `--no-gpu-batch` for the
rare case the old per-process path is wanted instead. Falls back to CPU
automatically inside `run_arena_gpu` if no CUDA device is present, just
without the speedup, so this default doesn't break anything on a
GPU-less machine.

## Clue-rarity filter was using the wrong notion of "rare"

The user hit this directly in the UI: `max_rarity=10%` still returned
"Frankfurt, Helsinki, Budapest, Zurich, Paris, Stuttgart, Munich, Vienna,
Warsaw, Istanbul" for a geography-themed board. Checked whether this was a
bug before assuming a fix was needed: it wasn't -- every one of those
words genuinely sat under the 10th percentile of `CLUE_RARITY_PERCENTILE`
as originally built (e.g. "stuttgart" at 7.8%, "helsinki" at 8.2%).

The real problem was the underlying frequency source. `CLUE_RARITY_PERCENTILE`
was derived from GloVe's own file order (frequency-descending in its raw
training corpus). That's a bad proxy for "would a person recognize this
word" specifically for proper nouns -- city names get mentioned constantly
in the news/web/Wikipedia text GloVe was trained on (finance, travel,
sports datelines) regardless of whether an average speaker actually knows
them, so major European capitals ranked as more "common" than plenty of
genuinely everyday words.

Discussed three ways to fix it (a curated common-word list, psycholinguistic
familiarity/AoA norms, or a differently-sourced frequency measure) and
picked the `wordfreq` package: it blends movie/TV subtitle and
conversational-text frequency in alongside web text specifically to
correct for this exact skew -- subtitle frequency is the standard
psycholinguistic fix for "recognizable word" vs. "frequently printed
word" (a screenplay says "Vienna" only when the story is actually set
there; a news wire says it constantly regardless). Offline after install,
same "no network calls at runtime" property the old GloVe-based approach
had.

Swapped `_build_clue_rarity_percentile` (scripts/web_inspector.py) to use
`wordfreq.zipf_frequency(word, "en")` instead of GloVe file position --
same percentile-within-the-clue-vocabulary logic, just a better-sourced
input. Unknown-to-wordfreq words naturally sort as rarest (zipf_frequency
returns 0.0 for them) without needing the old explicit fallback-rank
logic. Verified concretely: "stuttgart" moved from 7.8% to 19.7%,
"helsinki" from 8.2% to 16.0%, "budapest" from 8.3% to 12.3% -- all now
correctly excluded by a 10% filter, while genuinely common words stayed
low ("the" 0.0%, "dog" 0.7%, "castle" 2.9%, "paris" 1.4% -- Paris really
is used constantly in ordinary conversation, unlike the others). Removed
the now-unused `_embedding_lib` GloVe-file-scanning import from
web_inspector.py; that machinery is still used by
build_similarity_tensor.py/extend_similarity_tensor.py, just no longer by
the rarity filter. Added `wordfreq` to pyproject.toml.

All 190 tests pass (no test coverage changes needed -- this swaps an
internal data source behind an already-untested-directly helper function;
verified manually against the exact words the user reported instead).

## New guesser: a single weighted blend across all three spaces, and a model trained on it

The user asked for a new guesser -- weighted average of the three
spaces' cosine similarities (glove 0.3, numberbatch 0.5, wikipedia2vec
0.2), plus 0.08 Gaussian noise -- and a model trained against it. No new
guesser *code* was needed: `codenames/guessers/blend.py::BlendGuesser`
(weighted average across spaces) and `codenames/guessers/noisy.py::NoisyGuesser`
(adds noise to any base guesser) already existed and compose directly.
The "new guesser" is `configs/guesser_pool_blend.json`: a single pool
entry, `NoisyGuesser(base=BlendGuesser(weights=...), noise_std=0.08)`,
using the registry's inline-anonymous-base support.

Flagged before building it: this is a single-guesser pool, which departs
from docs/design-decisions.md's "diversity must be in knowledge, not
noise" principle -- there's no second, differently-knowledgeable listener
to check robustness against. Fine as a deliberate one-off exploratory
variant, not a replacement for the standard 3-guesser pool.

Generated 200k examples (`cache/blend_pool/data/`) and trained
(`cache/blend_pool/checkpoints/`, same (k, cause) architecture as every
other model): val_loss=1.1869, val_acc=0.5680 after 20 epochs.
`scripts/web_inspector.py::_discover_checkpoints` extended with an
explicit (not wildcard-matched, same reasoning as the noise_* rule)
check for `cache/blend_pool/checkpoints/scorer_best.pt`, registered as
`learned:blend`. New test asserts `guesser_pool_blend.json` loads to the
expected `NoisyGuesser(BlendGuesser(...))` structure with the exact
requested weights. 191 tests pass at this point.

## GPU-batched training-data generation: real, but a smaller win than the arena's

Follow-up to the profiling from the self-play/GPU-arena discussion:
`scripts/generate_training_data.py`'s dominant per-example cost (measured
~3ms of ~3.2ms total, ~96%) is `_cached_mean_similarity_to_words`'s
full-vocabulary (~111k-row) numpy mean, recomputed fresh per example
since each one samples a different random word subset. Asked to actually
test batching this on GPU before building it, given the earlier
GPU-arena work.

**Prototype first.** A naive "gather everything, batch=2000" version
actually got *worse* than the CPU baseline (8.46ms/sample) and threw a
CUDA OOM warning -- a `(batch, 111440, max_words, spaces)` tensor plus
several same-shaped derived tensors (NaN mask, validity mask, zeroed
values) blow past 16GB VRAM well before batch=2000. The real sweet spot
was much smaller: ~0.15ms/sample at batch=32-128, a genuine ~30x on this
one operation, chunked to stay in a safe memory range rather than left
for every caller to discover the hard way.

**Built for real**, given the prototype validated (exact match against
`mean_from_columns`, not just close):
- `codenames/gpu_clue_search.py` (new): `batched_mean_similarity`,
  chunking internally at `DEFAULT_CHUNK_SIZE=64` regardless of how large
  a batch a caller passes in.
- `scripts/generate_training_data.py::sample_clue` split into
  `_plan_clue` (the RNG-consuming board/subset selection, cheap) and
  `_resolve_scored_clue` (turns a precomputed score array into a legal
  clue) -- `sample_clue` itself is now a thin wrapper composing both
  unchanged, so its existing tests and its role as the CPU-fallback
  entrypoint both still hold. `generate()` gained `use_gpu_batch=True`
  (auto-off without CUDA): the new default path samples `PLAN_BATCH_SIZE`
  (4096) examples' board/clue-plans ahead of time, batch-scores all of
  them in one `batched_mean_similarity` call, then resolves and emits
  each one exactly as before. CLI gained `--no-gpu-batch`.
- Documented explicitly in `generate()`'s docstring: this reorders the
  RNG draw sequence relative to the old one-example-at-a-time loop, so a
  given seed's exact shard contents differ from what an older version of
  this function produced. Still fully deterministic for a *given*
  version, which is what `scripts/run_ablation_study.py`'s same-seed
  reuse across `feature_builder`/`guesser_weights` variants actually
  needs -- not byte-for-byte stability across code changes, which nothing
  ever promised.

**Real measured result: 3.06x, not 30x** -- 865 examples/sec vs. 283/sec
at real vocabulary scale (111,440 clues), generating 20k real examples
both ways. Honest reason for the gap: the isolated microbenchmark only
measured the scoring step in isolation. In the full pipeline,
`top_k_legal_clues` (turning a score array into an actual legal-clue
candidate list, ~0.49ms/call measured separately) was never optimized and
is now the new dominant cost, proportionally much more visible now that
scoring itself dropped by ~30x -- Amdahl's law: fixing the biggest
bottleneck reveals the next one, not a free 30x end to end. Not pursued
further this session (`top_k_legal_clues` is per-example legality string-
checking, not an obviously GPU-friendly operation the way full-vocabulary
scoring was).

Verified real output too, not just throughput: generated 20k real
examples, checked shapes/dtypes/value ranges (features float32 zero NaNs,
outcome int32 in `[0, 12]`, seeds present) -- all valid. 194 tests pass
(191 + 3 new `tests/test_gpu_clue_search.py`, exact-match correctness
against the numpy reference plus a chunking-doesn't-change-the-result
check).

## Web UI: the blend guesser is now a selectable option

Trained `learned:blend` last session but only wired it up as a
spymaster choice -- the guesser side (the "what each guesser would
pick" list, and the pool a simulated turn plays against) still only
built from `configs/guesser_pool.json`'s standard 3-guesser pool, so
`blend` never showed up there. Requested: add it as a choice in the UI.

`scripts/web_inspector.py::_pool_config_at_noise` now appends the
`blend` entry (loaded once from `configs/guesser_pool_blend.json`) to
every noise-level pool it builds, *after* the loop that overrides
`noise_std` on the standard `noisy_*` entries -- so `blend` keeps its
own fixed `noise_std=0.08` (the level it was designed and trained
against) regardless of which noise level the play-time selector is set
to, rather than being silently reinterpreted at whatever noise the user
picks. Verified `POOLS_BY_NOISE` now contains `blend` alongside the 3
standard guessers at all 6 noise levels. 194 tests still pass (no test
constrained the web UI's pool composition, so nothing needed updating).

## Model 1.1: self-play stats for the blend-guesser checkpoint

Requested: stats on the new blend-guesser model, in the README, labeled
as a subversion since only the guesser pool changed, not the scorer
architecture.

Ran `scripts/run_arena.py --n-boards 300 --checkpoint
cache/blend_pool/checkpoints/scorer_best.pt --guesser-pool-config
configs/guesser_pool_blend.json` (300 boards, 1 guesser in this pool ->
300 games per spymaster, GPU-batched path, 59.5s total for all 4
spymaster x guesser pairs). Results:

- learned: 3.7% assassin-hit, 5.28 turns (all) / 5.41 (wins only), 84.7%
  own-word rate.
- centroid: 8.3% assassin-hit, 7.38 / 7.72 turns, 87.5% own-word rate --
  note centroid's own-word rate actually beats model 1.1's here, unlike
  against the standard 3-guesser pool, but its assassin-hit rate is still
  more than double.
- linear_scorer: 15.3% assassin-hit. random: 91.0%.

Explicitly not a controlled model-1-vs-model-1.1 comparison -- different
guesser pools mean different game difficulty (every spymaster does
better here than against the standard pool, including random: 91.0% vs.
88.5% assassin-hit), so the two runs aren't apples-to-apples. Documented
as a new `docs/versions/v1.1.md` (mirroring v1.md's self-play section
structure) plus a "Model 1.1" subsection in the README right after Model
1, both explicit about the non-comparability caveat.

## Reward table revision: neutral is no longer 0.0

Requested: `ROLE_REWARD[Role.NEUTRAL]` (the true game reward, `codenames/
game.py`) changed from 0.0 to -0.2. Reasoning: a neutral guess still
burns a turn and produces no progress toward winning, so treating it as a
true no-op understated its cost -- it should be mildly penalized, just
less severely than an opponent guess (-1). This reverses the earlier
explicit decision (see this file's `(k, cause)` redesign entry above) to
default neutral to 0.0 specifically *because* it was "the true game
value" -- the true value itself is what changed here, not the modeling
choice to use it as the default.

Single source of truth: `ROLE_REWARD` in `codenames/game.py`, which
`codenames/scorer.py::reward_matrix`/`expected_reward_and_best_n` and
`codenames/spymasters/learned.py::LearnedSpymaster`'s `neutral_reward`
parameter both default from. Since none of the four reward values are
baked into training (see the `(k, cause)` redesign entry), no checkpoint
needed retraining -- this only changes default *scoring*-time behavior:
what `LearnedSpymaster` picks when `neutral_reward` isn't explicitly
overridden, and the reward value actually logged during real games
(`codenames/game.py::play_turn`).

**This does change the spymaster's actual clue choices**, though --
`LearnedSpymaster` uses this reward when picking the best `(clue, n)`,
so it's a real behavior change, not just a documentation update. The
self-play tables already in the README / `docs/versions/v1.md` /
`docs/versions/v1.1.md` were generated under the old `neutral_reward=0.0`
default and are now stale relative to current default behavior (flagged
to the user, not rerun automatically -- rerunning is cheap via
`scripts/run_arena.py` if wanted).

Updated: `codenames/game.py` (constant + docstring), `codenames/
scorer.py` (module docstring's reward-table description), `scripts/
webui/inspector.html` (neutral reward field's placeholder, 0 -> -0.2),
`docs/versions/v1.md` (defaults description). `tests/test_scorer.py`'s
two tests that asserted the *default* neutral reward is exactly 0.0
updated to -0.2 (tests that pass `neutral_reward` explicitly as a
parameter, e.g. `test_neutral_reward_only_affects_neutral_causes`, were
unaffected). 194 tests pass.

## Rerunning v1/v1.1 self-play under the new neutral reward

Requested after the neutral-reward change above: rerun the self-play
evals so the README/docs numbers reflect current default behavior
instead of the stale `neutral_reward=0.0` ones.

**Caught and fixed a real bug mid-run.** First v1 rerun used
`configs/guesser_pool_blend.json`... no -- used
`codenames/guessers/registry.py::DEFAULT_POOL_CONFIG`
(`configs/guesser_pool.json`) directly, which defaults to
`noise_std=0.03` (lowered post-M9, see this file's noise-reduction
entry) -- NOT the `noise_std=0.08` pool v1's checkpoint was actually
trained and evaluated against (`cache/m9/pool_configs/noise_0_08.json`,
a separate noise-specific copy `scripts/run_ablation_study.py` writes
per level). Every spymaster's assassin-hit rate dropped far more than a
reward-table tweak could plausibly explain (centroid 12.0%->6.4%,
linear_scorer 27.2%->11.7%) -- the tell that this was a guesser-pool
mismatch, not the intended change, since a lower-noise pool makes
guessing more accurate for every spymaster uniformly, reward table
irrelevant. Reran with the correct
`cache/m9/pool_configs/noise_0_08.json`; results now move by a
plausible, smaller amount.

Also hit a smaller self-inflicted issue: reused the same `--db` path
across two `run_arena.py` invocations, which appends rather than
replaces -- a query aggregating "all guessers combined" directly against
that db double-counted games sharing the same `(guesser, board_seed)`
key, corrupting turn/outcome grouping (guess-role ratios were still
right, since duplication scales all categories proportionally, but
`turns`/`win`/`loss` counts weren't). Fixed by deleting the db and
rerunning once per fresh path before querying.

Final numbers (`cache/arena_v1_rerun.db`, `cache/arena_blend_rerun.db`,
both gitignored): v1 assassin-hit rate 3.2% -> 4.0%, v1.1 3.7% -> 4.0% --
both moved by a similar small amount, in line with expectations (a
neutral guess is no longer free, so the model gives up a little pure
assassin-avoidance for it). Baselines that don't read the reward table
(`centroid`, `random`) only moved by ordinary `NoisyGuesser`-RNG
run-to-run noise; `linear_scorer` uses its own separate hardcoded -0.3
constant, also unaffected by this change. Updated README (Model 1 and
Model 1.1 sections) and `docs/versions/v1.md`/`v1.1.md` (including the
per-guesser breakdown table) with the new numbers, both explicit that
they supersede the pre-rerun figures rather than being a second data
point. 194 tests still pass (no test asserted these values).

## Cross-turn clue memory: a backlog-aware guesser with an earned bonus guess

Picked up `docs/versions/v1.md`'s open question #1: let a guesser use
misses from previous clues in its reasoning, guesser-side only, no
spymaster or training changes.

**The mechanism, settled through discussion before building anything:**
a miss ending a turn early (announced number `n`, only `k < n` correct
before the miss) leaves `n - k` own-words plausibly still unaccounted
for by that clue. A later turn's guesser can spend one earned bonus
guess -- real Codenames' standard `n+1` rule, deliberately dropped
earlier in this project (see the numbering-convention entries above)
specifically because guessers had no "still feels confident" signal to
justify it -- chasing whichever pending backlog word looks most
confident. Key design points, each explicitly worked through with the
user before implementing:
- **No stored cursor.** A backlog entry is just `(old_clue, owed_count)`.
  Re-ranking `old_clue` against whatever's still unrevealed always
  reproduces its best remaining candidate, since anything already
  resolved has already dropped out of the candidate pool -- no need to
  remember *which* word specifically.
- **Collision handling.** If this turn's own correct guesses include the
  word an old backlog clue would itself rank top, that word is assumed
  to satisfy *both* clues at once (counted fully for both, not split) --
  the owed count decrements. Without this, a coincidental double-match
  would leave a backlog entry believing a word is still owed that was
  never really missing, risking a later guess spent chasing a word that
  doesn't exist.
- **Bonus capped at exactly +1 per turn**, matching real Codenames,
  regardless of how many backlog entries are pending -- extra backlog
  just takes longer to clear, one per turn.
- **Only the new guesser uses it.** Every existing guesser's
  `bonus_guesses` stays at the `Guesser` base class default of 0, so
  their behavior is provably unchanged (verified: all 194 pre-existing
  tests pass unmodified against the widened interface). Framed as: the
  bonus was always technically available to any guesser (that's the real
  Codenames rule), they just never had a reason to use it before.

**The comparability problem -- investigated empirically before building,
at the user's request, rather than assumed.** The natural mechanism is a
merge: whichever candidate (this turn's own, or a pending backlog word)
has the higher similarity score goes first. That only works if scores
from *different* clues are on a comparable scale. Checked directly
against the real similarity tensor: sampled ~2000 clue words, computed
each one's mean similarity to the full board vocabulary, and correlated
that against `wordfreq` frequency. Result: r=0.84 for GloVe, r=0.32 for
Numberbatch, r=0.38 for Wikipedia2vec. This is the well-known "hubness"
effect in cosine-similarity embedding spaces -- frequent/central words
read as vaguely similar to almost everything, rare/technical ones read
as dissimilar to almost everything, regardless of real topical
relevance -- and the cross-clue swing in *baseline* similarity (~0.34
hottest-to-coldest in the sample) dwarfs the ~0.08 std that actually
separates a good candidate from a bad one within one clue's own ranking.
Concretely: "move"/"since"/"about" (common, generic) sat at the top of
the "similar to everything" list; "mirtazapine"/"lysenkoism" (rare,
technical) at the bottom -- nothing about actual topical relevance.
Fixed by z-scoring every clue's scores against that same clue's own
similarity distribution over the full board vocabulary before any
cross-clue comparison, chosen over a percentile/rank-based alternative
as the simpler first-pass version.

**Built:**
- `codenames/guessers/base.py`: `Guesser.rank_candidates`/new
  `bonus_guesses`/new `update_history` -- the last one is a *generic*,
  concrete method (not abstract), so any guesser gets correct backlog
  bookkeeping automatically even if it never reads `history` itself.
- `codenames/guessers/history_aware.py::HistoryAwareGuesser` -- wraps any
  base guesser, does the z-score merge, decides whether the best pending
  backlog candidate is competitive enough to be worth the one available
  bonus.
- `codenames/game.py::play_turn`/`play_game`: thread `history` through
  (`budget = number + guesser.bonus_guesses(...)`); `codenames/scorer.py`
  and `game.py`'s module docstrings updated to note the spymaster's own
  reward math still assumes exactly `n` and is unaware of the bonus
  (real play with a bonus-claiming guesser slightly outperforms what
  that math predicted -- one-directional, harmless).
- `codenames/gpu_arena.py::_play_batch_group`: mirrors the same
  history-threading per board (keyed by seed), so a `HistoryAwareGuesser`
  gets identical treatment whether evaluated via the GPU-batched path
  (learned spymasters) or the plain CPU arena.
- `configs/guesser_pool_history_aware.json`: wraps the exact same base as
  `configs/guesser_pool_blend.json` (only difference: history-awareness),
  specifically so an arena comparison between the two pools isolates the
  effect of this feature -- not yet run for real self-play numbers.
- Registry support (`type: "history_aware"`), `codenames/guessers/__init__.py`
  export.

**Tests**: `Guesser.update_history`'s generic bookkeeping (miss creates
correct-sized backlog, clean finish/assassin create none, wrong guesses
never touch existing owed counts, collision decrements/retires correctly)
and `HistoryAwareGuesser`'s z-score merge (hand-verified against real
z-score arithmetic: a competitive backlog word gets spliced into the
ranking and earns the bonus; a non-competitive one or one with no valid
score doesn't) -- 13 new tests, 207 total, all passing. Also smoke-tested
against the real similarity tensor via `scripts/run_arena.py
--guesser-pool-config configs/guesser_pool_history_aware.json` (10
boards): ran cleanly, `own/clue` exceeded 1.0 in every case, confirming
the bonus mechanism actually fires against real data, not just the
synthetic test fixtures.

## Web UI: a "Full Game" tab, playing both sides to completion

Requested: a new UI mode/tab to pick a spymaster and a guesser and play
a real game out to the end, rather than the existing tab's single-turn
peeks.

- `scripts/web_inspector.py::build_play_game_response` runs
  `codenames.game.play_game` for real (reveals words on a fresh `Board`,
  not a read-only peek like `build_simulate_response`), returns the full
  turn-by-turn history plus final outcome/total reward/board state.
  `GAME_GUESSERS` merges the 0.08-noise standard pool, the blend guesser,
  and both history-aware variants built earlier this session into one
  flat name -> guesser dict for a simple dropdown (independent of the
  main tab's noise dial, since this is a separate mode with its own
  controls). New `/api/guessers` and `/api/play_game` routes.
- `scripts/webui/inspector.html`: added tab buttons (`#tabInspector`/
  `#tabGame`) toggling visibility between the existing content (wrapped
  in `#inspectorTab`, otherwise untouched) and a new `#gameTab` -- seed/
  spymaster/guesser controls, the same 4 reward-override fields as the
  main tab, a board render (colors shown upfront like the rest of this
  dev tool, not hidden), a win/loss/timeout banner, and a turn-by-turn
  log showing each clue/number/guesses/reward/ended_reason, with a
  "bonus guess used" tag (`len(guesses) == number + 1`) so the earned-
  bonus mechanic from the history-aware work above is visible rather than
  a silent "why are there more guesses than the number" mystery.
- Verified end-to-end against a live server (not just unit tests): all
  new endpoints, an error case (unknown spymaster), a full game with
  `learned:noise_0_08` + `history_aware_noisy_glove` showing real bonus
  guesses firing, and the HTML page itself loading. 207 tests still pass
  (no new automated tests -- this is a thin UI layer over already-tested
  `play_game`/`load_pool`, matching how `build_simulate_response` and the
  rest of this file were handled).

## Real two-team play, without touching any spymaster, guesser, or the scorer

Requested: the "Full Game" tab should have the opponent actually play
with their own (red) cards, not sit as a pure distractor -- i.e. real
two-team Codenames.

**Corrected my own initial overstatement.** First response called this a
large scope expansion needing a `Role`/reward-semantics redesign,
matching the "genuine second team" framing already flagged as out-of-
scope in `docs/versions/v1.md`'s open questions. The user pushed back,
correctly: every spymaster, guesser, and the scorer only ever touch a
board through `role_of`/`words_by_role`/`remaining`/`is_revealed`/
`reveal`/`words` -- never `Card.role` directly (confirmed by grep, no
exceptions in `codenames/features.py` or any spymaster). So a thin
read-only *view* that swaps OWN/OPPONENT while sharing the same
underlying revealed-state is sufficient, and none of that code needs to
change at all.

**Built:**
- `codenames/board.py::OpponentBoardView` -- wraps a `Board`, swaps
  OWN/OPPONENT in `role_of`/`words_by_role`/`remaining`, delegates
  `words`/`is_revealed`/`reveal`/`seed`/`revealed` straight through to
  the same physical `Board` (one shared, mutable revealed-set, not a
  copy). Hit one real gap while wiring it up: `codenames/spymasters/
  _util.py::state_rng` and `LearnedSpymaster` both read `board.revealed`
  as a raw attribute rather than through a method -- added a `revealed`
  property to the view too (a straightforward miss from just grepping
  the public *method* surface, not every raw attribute access).
- `codenames/game.py::play_two_team_game` -- alternates `play_turn`
  calls between team A (the real `Board`, whichever side has 9 cards)
  and team B (the `OpponentBoardView`), each with its own independent
  `history` (a HistoryAwareGuesser on one side can't see the other's
  backlog). Team A always moves first (matches the already-verified
  real rule). Win the instant either side's own words hit zero --
  including via the *other* team's mistake, exactly like an opposing
  team's accidental reveal helps you in the real game, which falls out
  for free from checking both sides' remaining-OWN after every
  half-turn rather than just the mover's. Assassin ends the game
  immediately, other team wins.
- Web UI: `/api/play_game` now takes a spymaster+guesser pair *per
  team* (`_a`/`_b` suffixed params, including separate reward overrides
  per side -- noted that two sides picking the *same* learned spymaster
  share one instance, so their overrides can't actually differ). Turn
  log is team-labeled (left border colored by team, own team-relative
  turn counter) instead of one flat list.
- Tests: `OpponentBoardView`'s role-swap/shared-state/reveal-passthrough
  (7 tests, `tests/test_board.py`), `play_two_team_game`'s turn order,
  win-by-own-mistake, opponent's-mistake-wins, assassin ending, timeout,
  and per-team history independence (6 tests, `tests/test_game.py`). 221
  total, all passing. Verified live against a running server too: normal
  play, an error case, and same-spymaster-both-sides reward overrides.

Next natural step, not done yet: the arena/self-play evaluation
machinery (`codenames/arena.py`, `codenames/gpu_arena.py`) is still
single-team-only -- extending it to bulk-simulate real two-team games
(and deciding what the interesting aggregate stats even are, e.g. a real
win/loss rate instead of the current `1 - assassin-rate` proxy) is a
separate, larger piece of work than the game engine + UI built here.

## Two-team self-play arena

Follow-up to the two-team game engine: bulk-run it for real stats, not
just the one-off games the web UI plays. Discussed the metric question
first -- since self-play means the *same* spymaster+guesser pair on
both sides, a symmetric win rate isn't informative (it's mostly just the
9-vs-8 first-move edge, not a quality signal). Kept the same shape of
metric the single-team arena already reports instead: assassin-hit rate
and a turns split (all games vs. clean-finish games), now measured in a
real two-team game where the board depletes from both sides' actual
play, not the single-team framing's static distractors.

`codenames/two_team_arena.py::run_two_team_self_play` -- mirrors
`codenames/arena.py`'s process-parallel structure (same "spawn" worker
fix, same construct-inside-the-worker pattern) but for one symmetric
pair, not a spymaster x guesser cross-product. `scripts/
run_two_team_arena.py` is the CLI wrapper. No GPU-batched path (unlike
`scripts/run_arena.py`) -- `codenames/gpu_arena.py`'s batching drives
many *independent single-team* boards through one shared forward pass
per round, which doesn't carry over to a two-team game the same way
(each game is already two calls per round, tied to one board's specific,
shared state) -- not attempted, noted in the script's docstring in case
that's revisited.

4 new tests (`tests/test_two_team_arena.py`): stats bookkeeping (clean
finish vs. assassin ending counted separately, pooled per-guess rates
across both teams) plus a real small-scale end-to-end run. 225 tests
total, all passing. Smoke-tested against the real similarity tensor with
both a baseline (`centroid`) and a learned checkpoint.

## A real bug found by two-team play: LearnedSpymaster can't play the 8-card side

While actually running the two-team self-play comparison (same
`learned:noise_0_08` on both sides, matching the earlier single-team
methodology), hit a `RuntimeError: mat1 and mat2 shapes cannot be
multiplied (111440x106 and 103x256)` -- a real bug, not a fluke,
correcting the earlier claim that no spymaster/guesser/scorer code
needed to change for two-team play to work.

**Root cause**: `codenames/features.py`'s feature vector has a *fixed*
per-role slot layout (`ROLE_SLOT_RANGES`, built from `ROLE_COUNTS`): 9
slots for OWN, 8 for OPPONENT. That's true for any board queried from
its own real perspective (`Board.generate` always produces exactly that
split) -- but `OpponentBoardView`'s swapped perspective can violate it:
team B's "own" (the physical board's 8-member OPPONENT group) fits fine
in the 9-slot OWN allocation, but team B's "opponent" (the physical
board's 9-member OWN group) can overflow the fixed 8-slot OPPONENT
allocation whenever all 9 are still unrevealed. `_sorted_padded_values_batch`
had no bounds check for that case at all -- it silently returned a
9-wide block instead of the declared 8-wide one, so the bug never raised
where it happened; it surfaced 100+ lines away as a mystery matmul shape
mismatch (106 vs. 103 -- exactly 3 extra columns, one per embedding
space, from that single extra word). This is exactly the failure mode
this file's own module docstring warns about: "a bug here is silent and
poisons everything built on top of it."

This isn't fixable by retraining or a small tweak -- it's a genuine
structural fact about the current model: `LearnedSpymaster`'s trained
weights only ever saw "I am the 9-card team," never "I have 8 own words
and 9 opponent words." Reusing it as the 8-card side would mean
evaluating it well outside its trained distribution even if the input
shape were patched to fit.

**Fixed**: added `_check_capacity()` in `codenames/features.py`, called
from every padding function (`_sorted_padded_values`,
`_sorted_padded_values_batch`, `_unsorted_padded_values`, `_role_mask`)
-- raises a clear, immediate `ValueError` naming exactly what happened
and why, instead of a silent wrong-shaped array that only surfaces as a
confusing crash several layers downstream. 5 new tests
(`tests/test_features.py::TestRoleCapacityGuard`), 230 total passing.

**Practical consequence**: a `LearnedSpymaster` can only play as team A
(the 9-card side) in two-team mode; team B needs a baseline spymaster
(random/centroid/oracle/linear_scorer -- none of which build a fixed-
size feature vector, so none of them hit this). The two-team self-play
comparison originally planned (same learned spymaster on both sides)
isn't possible without retraining a symmetric-capacity model -- a real
scope question for a future version, not attempted here.

## Bypassing the 9-vs-8 capacity mismatch with a one-word bootstrap reveal

The previous entry's "practical consequence" (LearnedSpymaster can only
play team A) turned out to be avoidable, not fundamental -- user's
suggestion: since the actual overflow is a single, specific mismatch
(team B's OpponentBoardView-swapped "opponent" role is team A's real
9-word group, one over the fixed 8-slot allocation), fix the *count*
rather than the model. `play_two_team_game` now silently reveals one of
team A's 9 own words before any turn, crediting no reward to anyone --
it's a board-setup artifact, not a real guess by either guesser. That
single reveal drops team A's remaining-own count to 8 everywhere it's
looked at: A's own real perspective (8 ≤ 9 own slots, fine) and B's
swapped view of A as "opponent" (8 ≤ 8 opponent slots, now exact fit).
No other role pairing in either direction ever overflows, so one reveal
is both necessary and sufficient -- confirmed by re-deriving each of the
four (perspective, role) counts by hand before implementing, not just by
testing after the fact.

This required no changes to `codenames/features.py` (the `_check_capacity`
guard from the previous entry stays as defense-in-depth, now expected to
never actually fire in two-team play) and no retraining. Updated
`tests/test_game.py::TestPlayTwoTeamGame::test_each_teams_backlog_history_is_independent`,
which had hardcoded `"Board0"` as a scripted own-word guess for team A --
now pre-revealed by the bootstrap step, so it never appears in that
turn's candidates. Swapped for `"Board1"`. 230 tests still passing (no
new tests added: the existing `TestRoleCapacityGuard` suite already
exercises the guard directly against a bare `OpponentBoardView`, which
remains valid since that test bypasses `play_two_team_game`'s bootstrap
entirely).

**Consequence**: `LearnedSpymaster` can now legitimately play *either*
side of two-team mode, including the same checkpoint as both teams --
the originally-planned pure self-play comparison the previous entry said
was blocked is now possible after all.

Ran it for real: `learned:noise_0.08` + `noisy_glove` on both sides, 300
boards (`python scripts/run_two_team_arena.py --n-boards 300 --checkpoint
cache/m9/checkpoints/noise_0_08/scorer_best.pt --guesser noisy_glove`).
Assassin-hit rate 0.3%, 8.07 half-turns/game (8.08 clean-finish-only --
almost every game ends cleanly), per-guess breakdown own 95.4% /
opponent 0.5% / neutral 4.0% / assassin 0.0%. Substantially safer than
the earlier `centroid`-opponent smoke test's 10.0% assassin-hit rate --
expected, since a symmetric self-play matchup never has to contend with
an erratic opposing clue-giver flooding the board with confusing
reveals the way a noisier baseline spymaster does. Written up in
`docs/versions/v1.md`'s open question #2.

## Superseded: decoupling real board counts from feature-vector slot widths

User pushback (correctly) on the bootstrap-reveal entry above: it changed
a real game rule -- team A's true win condition silently shrank from
"find all 9" to "find all 8," since the pre-revealed word is gone from
play, not just hidden. That's a real cost the previous entry's "no reward
charged" framing undersold.

Root cause, restated precisely: `codenames/board.py::ROLE_COUNTS` was
being reused for two different jobs that only looked identical by
coincidence -- (1) how many real cards `Board.generate` deals per role,
and (2) how wide each role's padded block is in the feature vector
(`codenames/features.py` imported the same dict for both). Job (1) must
stay 9/8/7/1 to keep the real game faithful to actual Codenames rules.
Job (2) just needs to be wide enough to hold whatever real count could
ever land in that slot from *any* perspective -- and under
`OpponentBoardView`'s two-team swap, "opponent" can genuinely be team A's
real 9-word group, one over the old 8-wide slot.

**Fix**: split them. `codenames/features.py` now defines its own
`FEATURE_SLOT_COUNTS` (own=9, opponent=9, neutral=7, assassin=1) and
`FEATURE_BOARD_SIZE` (26, not 25), used everywhere in that file instead
of `ROLE_COUNTS`/`BOARD_SIZE`. `codenames/board.py::ROLE_COUNTS` is
untouched -- `Board.generate` still deals a real 9/8/7/1 board, so
neither team's true win condition changes. The extra opponent slot is
ordinary padding (mask=0) on every normal single-team turn, exactly like
any role that's lost words to reveals; it's only genuinely populated on
team B's turn in two-team play, which is exactly the case it exists for.
`play_two_team_game`'s bootstrap-reveal step is removed entirely -- no
longer needed, and no longer correct once the game rules must stay
faithful.

Also fixed in the same pass, found while touching this: `_compute_scalars`
computed `own_revealed`/`opponent_revealed` as `ROLE_COUNTS[role] -
board.remaining(role)` -- correct for a board's own real perspective (own
total is genuinely 9), but wrong under `OpponentBoardView` (that view's
"own" total is genuinely 8, not `ROLE_COUNTS[Role.OWN]`'s constant 9), so
this silently reported one extra "revealed" own word for team B from
turn one. Fixed by deriving each total from `len(board.words_by_role(role))`
(revealed + unrevealed together, correct from whichever perspective is
asking) instead of the constant. Mirrored the same two fixes in
`codenames/gpu_features.py` (the GPU-batched path), including replacing
its silent truncate-on-overflow (`sorted_desc[:, :, :pad_to]`) with an
explicit raise, matching `features.py`'s `_check_capacity`.

**Real cost, not swept under the rug**: this widens the model's input
dimension (107 vs. the old 103, for 3 spaces), so every existing
`LearnedSpymaster` checkpoint (5 noise-level ones plus the blend-pool
one) is now incompatible and must be retrained before `LearnedSpymaster`
works again *at all*, single-team included. No baseline spymaster
(random/centroid/oracle/linear_scorer) touches `features.py`, so none of
them are affected or need retraining. 230 tests passing (rewrote several
in `tests/test_features.py`/`tests/test_ablation.py` that hardcoded the
old 25-wide role-block layout).

Also ran real two-team self-play for every baseline spymaster (these
were never affected by the feature-vector issue -- only `LearnedSpymaster`
touches `codenames/features.py`), 200 boards each, `noisy_glove` on both
sides, real 9-vs-8 rules throughout:

| spymaster    | assassin-hit rate | half-turns (all/clean) | own/opp/neutral/assassin  |
|---------------|-------------------:|-------------------------:|-----------------------------|
| random        | 85.5%              | 9.71 / 15.48              | 33.6% / 32.0% / 27.7% / 6.7% |
| oracle        | 56.0%              | 5.89 / 7.57                | 58.7% / 17.8% / 19.3% / 4.2% |
| centroid      | 15.5%              | 10.20 / 11.04              | 90.0% / 3.8% / 5.1% / 1.0%   |
| linear_scorer | 4.5%               | 18.71 / 18.71              | 41.8% / 30.1% / 27.9% / 0.2% |

`oracle` (added to `scripts/run_two_team_arena.py`'s baseline choices in
this same pass -- it was simply missing from the CLI list, nothing
architectural blocked it) has perfect single-space foresight but no
noise-robustness, so it posts a surprisingly high assassin rate despite
a low turn count. `linear_scorer`'s low assassin rate comes from being
slow and overly conservative (18.71 half-turns/game, rarely finishing
cleanly), not from being skilled. Written up in `docs/versions/v1.md`'s
open question #2.

Retraining `LearnedSpymaster`'s checkpoints for the new feature width,
and rerunning its two-team self-play numbers under the real (non-pre-
revealed) 9-vs-8 rules, is the natural next step -- not done yet, since
it's real compute time and worth confirming before spending it.

## Retraining, round two: training data needed both perspectives too

User caught a real gap before the retrain above would have been wasted:
retraining under the new feature width only teaches the model to *accept*
the wider input, not to *interpret* it. Every training example was
generated by `sample_partial_board`/`Board.generate()` directly -- team
A's real perspective only, own<=9/opponent<=8 -- so the new 9th opponent
slot was mask=0/sentinel=-1 in literally every example. Retraining alone
would produce a checkpoint that runs without crashing on team B's turns
but has never once seen that slot's mask=1/real-value case, relying
entirely on untested generalization.

**Fix**: `scripts/generate_training_data.py::sample_partial_board_perspective`
-- a 50/50 coin flip (`SWAP_PERSPECTIVE_PROB`, "team A is just as
important as team B") between the real board and its `OpponentBoardView`-
swapped counterpart, before any clue sampling or feature building
happens. Every function downstream (clue sampling, guesser rollout,
`build_features`) only ever touches the shared role_of/words_by_role/
is_revealed/words/remaining interface, exactly like a spymaster or
guesser does, so this needed no other change. One edge case handled: if
the swap is drawn but team B's real own words (the physical OPPONENT
role) are already fully revealed, that's an already-over-for-B state --
falls back to team A's real perspective for that example instead,
mirroring `sample_partial_board`'s existing "always >=1 own word left"
rule. New tests confirm the widened slot is populated when forced
(`swap_prob=1.0`) and never populated when disabled (`swap_prob=0.0`).
235 tests passing.

Deleted and regenerated all 6 datasets/checkpoints a second time (the
first retrain's data was already stale, missing this mix entirely).
`--noise-levels` list corrected too: the first retrain attempt only
covered `0.0,0.03,0.06,0.1,0.15`, missing `0.08` -- the level actually
referenced everywhere else in this project (`configs/`, `docs/`, prior
arena runs) -- a real mistake, caught immediately after checking the
report and noticing it wasn't in the output. All 6 val_loss numbers
after the corrected retrain, comparable to or better than before the
whole features.py refactor started (`noise_0_08`: 1.2720 vs. the
original 1.3017): `noise_0_0`=0.8276, `noise_0_03`=0.9429,
`noise_0_06`=1.1494, `noise_0_08`=1.2720, `noise_0_1`=1.3879,
`noise_0_15`=1.5915; `blend`=1.1466 (vs. the original 1.1869 and the
first, swap-less retrain's 1.1868 -- the biggest improvement of any
variant, plausibly because a single-guesser pool benefits most from the
extra training signal a second perspective's board states provide).

Reran the two-team self-play comparison this was all in service of:
`learned:noise_0.08` + `noisy_glove`, 300 boards, real 9-vs-8 rules,
this retrained checkpoint. **Assassin-hit rate 0.0%** (down from the
now-superseded pre-reveal run's 0.3%), 9.03 half-turns/game (up from
8.07, expected -- team A genuinely needs all 9 words now, not the
pre-reveal hack's silently-shrunk 8), per-guess breakdown own 97.4% /
opponent 0.3% / neutral 2.3% / assassin 0.0% (own rate up from 95.4%).
Both safer and more accurate than any prior two-team number for this
model. Written up in `docs/versions/v1.md`'s open question #2, which now
also flags the superseded pre-reveal-era number as not a valid baseline
for future comparisons.

## Policy: two-team self-play is now the headline evaluation, not single-team

User direction: "we shouldn't have any single team stats at this point,
everything should be considered with both teams playing against each
other." Scope, per follow-up question: replace the numbers reported in
README.md and the version docs' headline sections with two-team
self-play numbers -- not a retirement of `codenames/arena.py`/
`codenames/gpu_arena.py`, which stay in use for fast iteration (the
6-checkpoint noise sweep's internal per-clue rollouts, ablation-study
comparisons) where two-team's current ~50x-slower non-batched cost would
make routine comparisons impractical.

Ran the full set of two-team comparisons needed to replace README's
existing single-team tables, 300 boards each:

Standard pool (`noisy_glove`, matches model 1's `noise_std=0.08`):
learned 0.0% assassin-hit / 9.03 half-turns / 97.4% own; centroid 13.3% /
10.33 / 90.0%; linear_scorer 5.0% / 18.69 / 40.5%; random 83.3% / 9.77 /
32.8% (random/centroid/linear_scorer rerun at 300 boards for consistency
with learned's count -- the earlier 200-board versions from the prior
entry are superseded by these).

Blend pool (matches model 1.1): learned:blend 5.0% assassin-hit / 8.18
half-turns / 86.9% own; centroid 17.0% / 10.64 / 86.0%; linear_scorer
8.7% / 18.97 / 39.3%; random 85.0% / 9.26 / 32.3%.

Notable finding, not yet explained: model 1.1's real two-team
assassin-hit rate (5.0%) is worse than model 1's (0.0%), the opposite of
their single-team ordering (model 1.1 was reported as comparable or
slightly safer than model 1 there). Plausible hypothesis: a single,
more-knowledgeable blended listener behaves more predictably than three
diverse noisy ones, which may let the spymaster get away with -- and
therefore learn -- slightly more aggressive clues that occasionally
backfire in real two-team play, where both sides are actually acting
rather than one side facing static distractors. Not investigated further
here; a real open question for the next version.

README.md's Model 1 and Model 1.1 sections rewritten with these numbers,
using the same pooled-stats methodology write-up as the two-team-arena
work above (assassin-hit rate + half-turns split, not a symmetric win
rate). `docs/versions/v1.md` and `v1.1.md`'s single-team self-play
sections are kept as historical record (not deleted -- they document a
real methodology this project used for most of its life) but flagged at
the top as superseded, pointing to the new numbers.

Also surfaced in the same conversation: the current two-team arena has
no GPU-batched path (`codenames/two_team_arena.py`'s own docstring
already flagged this), unlike single-team's `gpu_arena.py` -- not a
fundamental limitation, just an unbuilt optimization (different *games*
are independent of each other even though turns within one game aren't,
so the same per-round batching trick should apply, just batching each
half-turn's forward pass across many simultaneous games instead of each
full turn). Deferred until after this round of numbers; a real next step
now that two-team is the standing evaluation method, not an occasional
one-off.

## Two Full Game tab bugs: swapped turn colors, and a missing rarity filter

Two small UI bugs, both user-reported, unrelated to each other:

1. **Turn-log colors were backwards for team B.** Each turn's guess chips
   were colored by `_turn_payload`'s reported role, which is *team-
   relative* (team B's own words come back as Role.OWN via
   OpponentBoardView's swap) -- but the persistent board display always
   colors by the *physical* role (team A's real 9 words are always blue,
   team B's real 8 are always red, regardless of whose turn it is). So a
   correct guess on team B's turn showed blue when the same card shows
   red on the board. Fixed by having `_turn_payload` take the real,
   un-swapped `Board` and color every guess from `board.role_of(word)`
   directly, ignoring the team-relative role from `TurnResult.guesses`
   (still used correctly elsewhere for `t.reward`/`t.ended_reason`).

2. **No max-rarity filter for the Full Game tab, and the Inspector's own
   default was 90% (should be 50%).** The existing `max_rarity` filter
   only ever applied to the Inspector tab's candidate-*listing* endpoint
   (`build_give_clue_response`'s own post-hoc filter over an over-fetched
   pool) -- real gameplay (`play_two_team_game` -> `play_turn` ->
   `spymaster.give_clue()`) had no rarity knob at all. Fixed by giving
   `LearnedSpymaster` itself an optional `max_rarity`/`rarity_percentile`
   (class default `max_rarity=100.0` = no filtering, so arena/training
   scripts are unaffected unless they opt in) and a `_pick_legal_clue`
   helper mirroring the same over-fetch-then-filter pattern, falling back
   to the unfiltered pick if nothing in the pool clears the threshold
   (never let a UI-only setting make a real game unable to produce a
   clue). Moved the rarity-percentile computation itself
   (`clue_rarity_percentile`, was `_build_clue_rarity_percentile`) from
   `scripts/web_inspector.py` into `codenames/clue_search.py` so both the
   spymaster and the UI share one implementation. Added per-side
   `max rarity %` inputs to the Full Game tab, wired through the same
   `_apply_reward_overrides`-style per-request mechanism the 4 reward
   fields already use. Web UI's own `LearnedSpymaster` instances are
   constructed with `max_rarity=50.0` as their starting point (a UI-only
   default, not a change to the class default); both the Inspector's
   existing field and the new Full Game fields default to 50 in the HTML
   too. 235 tests passing (no behavior change for anything outside the
   web UI).

## Real bug: NoisyGuesser's stateful RNG broke the backlog/bonus mechanism

User report, watching a real Full Game sim: team B's clue for 2 got 1
correct then a neutral (creating a backlog of 1 owed word). Next turn's
clue for 2 correctly used the bonus to satisfy that backlog (3 correct
guesses total). The turn *after that* still claimed a bonus guess, with
no missed clue left to justify it.

**Root cause**: `codenames/guessers/base.py`'s entire backlog design
(see its own docstring) assumes "re-ranking the same clue against the
same candidates always reproduces the same answer" -- true for every
guesser in the pool except `NoisyGuesser`, whose noise came from one
continuously-advancing RNG stream (`self.rng.normal(...)`, consumed
further with every call). The same clue scored twice -- once for real
during play, once retrospectively in `update_history`'s "did this
backlog get satisfied" collision check -- could get different noisy
values and therefore disagree about which word was the clue's top pick,
so the collision-credit decrement silently failed to fire even when the
backlog genuinely had just been satisfied. A second, related bug from
the same root cause: `codenames/game.py::play_turn` calls
`bonus_guesses()` and `rank_candidates()` as two separate calls, each
independently re-deriving `HistoryAwareGuesser._merge` -- with a noisy
base guesser, the bonus-claim decision and the actual guess order used
in the same turn could already be based on inconsistent noise before
update_history ever ran.

**Fix**: `codenames/guessers/noisy.py` -- noise is now a deterministic
function of `(seed, clue, word)` (`zlib.crc32`-hashed into a fresh
per-call `np.random.default_rng` seed), not a draw from mutable
per-instance state. Same per-word "idiosyncratic misperception" semantics
as before, just now a pure, repeatable function like every other guesser
in the pool -- restoring the invariant `update_history` and
`HistoryAwareGuesser`'s baseline cache both already assumed held.
Bonus: this also makes `scripts/run_ablation_study.py`'s noise-level
sweep comparability guarantee *more* robust, not less -- "same seed at a
different noise_std reuses the same underlying draw, just scaled" no
longer depends on call order/count being identical across differently-
generated datasets, since it's now keyed by the literal (clue, word)
content instead of a stream position.

Updated three stale comments that had documented the old (buggy)
statefulness as an accepted fact rather than a bug: `base.py`'s own
docstring, `history_aware.py`'s baseline-cache comment, and
`run_ablation_study.py`'s noise-sweep comparability note.

New tests (`tests/test_guessers.py`): `TestNoisyGuesser` gains a
same-instance-repeated-call determinism check and a noise-std-
independence check (the property the ablation sweep relies on);
`TestHistoryAwareGuesser` gains an end-to-end regression test wrapping a
`NoisyGuesser` specifically (a deterministic base guesser could never
have exposed this) that reproduces the exact reported scenario: a
backlog satisfied via bonus guess must not resurface on a later turn.
238 tests passing.

**Not yet done, on purpose** (user: "don't rerun anything yet"): every
existing `NoisyGuesser`-based result -- every arena run, every trained
checkpoint's training data, every two-team self-play number in this
document and the README -- was generated under the old, buggy noise
semantics. The noise *values* themselves change completely under the
new deterministic scheme (same distribution, different actual numbers),
so none of it is bit-for-bit reproducible from before this fix, though
past directional findings likely still hold. A full regenerate-and-
rerun is the natural next step whenever asked for.

## GPU-batched two-team arena

`codenames/two_team_arena.py`'s own docstring had flagged this as "not
attempted" -- two-team play alternates turns within one game, which
can't be batched, but *different games* are still independent of each
other exactly like single-team play, so the same per-round batching
trick (`codenames/gpu_arena.py`) should apply per *half-turn* instead of
per turn. Confirmed the key invariant that makes this valid: every
active game advances by exactly one half-turn per outer-loop iteration
and only ever leaves the active set by finishing (never by desyncing),
so at any iteration every still-active game is on the *same* team's
turn -- meaning the correct board perspective (real `Board` for team A,
`OpponentBoardView` for team B) can be gathered across every active game
and scored in one batched forward pass.

New `codenames/two_team_gpu_arena.py::run_two_team_self_play_gpu` /
`_play_batch_group`, mirroring `gpu_arena.py`'s structure closely but
doubled for two sides (per-game, per-side backlog history instead of
one history per game). Reuses `two_team_arena.py`'s stats bookkeeping
unchanged. Wired into `scripts/run_two_team_arena.py` as the default
path for `--checkpoint` (mirrors `run_arena.py`'s `--gpu-batch-size`;
`--no-gpu-batch` reverts to the process-parallel path). Baseline
spymasters are unaffected -- nothing here to batch for a spymaster
that scores a handful of candidates instead of the whole vocabulary.

Verified correctness before trusting the speedup: a manual comparison
against the CPU path (deterministic guesser, so no cross-path RNG-order
sensitivity) matched **exactly**, bit-for-bit, across every stat, and
was invariant to `--gpu-batch-size` (1, 5, 20 all agreed). New tests in
`tests/test_two_team_gpu_arena.py` mirror `test_gpu_arena.py`'s two
tests (exact CPU match, batch-size invariance) -- both pass for real
against a genuine CUDA device in this environment (240 tests total).

Real benchmark, `learned:noise_0.08` + `noisy_glove`, 300 boards,
batch=32: **26.9s**, vs. the CPU path's earlier **338.4s** for the exact
same comparison -- a **~12.6x speedup**, with statistically consistent
results (0.0% assassin-hit, 97.4% own-rate, matching closely; small
differences from noisy_glove's RNG draw order versus the process-
parallel path, not a correctness concern). Two-team self-play is now
cheap enough to be the routine evaluation method going forward, not just
an occasional headline-number run.

Also surfaced, unrelated to this work: `tests/test_gpu_features.py`'s
existing `test_matches_numpy_reference_exactly_for_each_board` failed
once, flakily, when run as part of the full suite (passed in isolation
and on every rerun) -- root cause is almost certainly
`gpu_features.py::tensor_on_gpu`'s `_TENSOR_GPU_CACHE`, keyed by
`id(sims)` and explicitly documented as assuming "a SimilarityTensor is
loaded exactly once per process and lives for that process's whole
lifetime" (true in production, not true across many short-lived
SimilarityTensor fixtures in one pytest process, where `id()` reuse
after garbage collection is a real risk this test suite's growth just
started to expose). Not fixed here -- flagged as a known, pre-existing
flaky-test hazard, not something introduced by this change.

## HistoryAwareGuesser in real two-team play: the negative result holds up

With two-team self-play now cheap (previous entry), reran the
HistoryAwareGuesser comparison the earlier single-team-arena negative
finding was based on, this time in real two-team play and after the
NoisyGuesser determinism fix -- both changes could plausibly have
altered the picture. They didn't: 300 boards each, GPU-batched
(`scripts/run_two_team_arena.py --gpu-batch-size 32`).

- `learned:noise_0.08` + `noisy_glove`: 0.0% assassin-hit, 97.4% own.
- `learned:noise_0.08` + `history_aware_noisy_glove`: 7.3% assassin-hit,
  85.2% own (148.1s -- notably slower than the plain guesser's 26.9s,
  since HistoryAwareGuesser's backlog scoring adds real per-turn cost).
- `learned:blend` + `blend`: 5.0% assassin-hit, 86.9% own.
- `learned:blend` + `history_aware_blend`: 7.3% assassin-hit, 83.8% own
  (151.1s).

Confirms the earlier single-team finding rather than overturning it:
history-awareness is a real, reproducible negative for both models in
the setting that actually matters (real two-team play), not an artifact
of the older single-team-vs-static-distractors methodology or the since-
fixed NoisyGuesser bug. Written up in the README (new "History-aware
guessers: a negative result" section) and `docs/versions/v1.md`'s open
question #1, which previously read "not yet evaluated" -- now corrected
to describe the actual, confirmed negative result.

## Mixed-guesser evaluation and two new stats

User correction: model 1's README numbers evaluated against `noisy_glove`
alone, 300 games -- narrower than the 3-guesser, equally-weighted
distribution the model was actually trained against (`docs/design-
decisions.md`'s "guesser pool is 3 members ... equally weighted"). Added
`MIXED_GUESSER` to `codenames/two_team_arena.py`: each game independently
draws its guesser, uniformly, from the whole pool, matching
`generate_training_data.py`'s own `rng.choice(guesser_names)` sampling.
Implemented for both the CPU path (`random.Random(seed).choice(...)` per
task) and the GPU-batched path (a per-seed guesser dict -- the guesser
only matters after the batched spymaster forward pass, so different
games can use different guessers with no change to the batching itself).

Also added two requested stats to `TwoTeamSelfPlayResult`: mean announced
clue number and mean correct guesses per clue, both pooled across both
teams. 243 tests passing (new: mixed-guesser end-to-end on CPU,
mixed-guesser GPU-vs-CPU exact match, the two new stats' bookkeeping).

Reran model 1 and model 1.1's README numbers under this corrected
methodology, 300 boards each:

- Model 1 (mixed guesser): 0.7% assassin-hit (up from 0.0% under
  `noisy_glove` alone -- expected, a genuinely harder/more varied test),
  96.3% own, mean clue number 1.81, mean correct/clue 1.74. Baselines
  rerun the same way: centroid 14.0%/89.2% own, linear_scorer
  7.7%/41.2% own, random 89.7%/31.8% own.
- Model 1.1 (still `blend` only, by design -- a single guesser is the
  whole point of that subversion): 5.7% assassin-hit, 86.1% own, mean
  clue number 2.11, mean correct/clue 1.74. Baselines: centroid
  17.3%/86.0%, linear_scorer 24.3%/38.0%, random 86.7%/32.6%.

README and `docs/versions/v1.md` updated with these numbers superseding
the single-guesser ones from the previous entries.

## LLMGuesser: an external guesser for evaluation, not training

User's concern, raised while discussing self-play evaluation: scoring a
spymaster against a guesser pool it's also implicitly co-adapted to
(the guesser pool the whole training signal is defined by) can't
distinguish "this spymaster is genuinely good" from "this spymaster
and this guesser just happen to share the same blind spots." There's no
way to compare across spymaster/guesser pairs on that basis alone.
Proposed fix: use a real LLM as an *evaluation-only* guesser -- something
that was never part of the training loop, so a high score against it
means something closer to "would an actual, independent listener guess
this."

Added `codenames/guessers/llm.py::LLMGuesser` (Claude Haiku 4.5 by
default -- cheap enough for this: ranking ~20 remaining words against
one clue is a simple task, not one that needs a larger model). One API
call per (clue, candidate_words, number) turn, not one per candidate
word -- the model ranks the whole remaining board at once, matching the
shape of question a real guesser answers each turn. Responses are
cached per exact input key: cheap (no redundant spend on repeated
inputs) and, more importantly, this is what keeps it a pure function of
its inputs -- real LLM sampling isn't perfectly deterministic even at
temperature 0, and codenames/guessers/base.py's backlog mechanism (and
the just-fixed NoisyGuesser bug) both depend on that property holding
for every guesser that gets wrapped or re-scored.

Deliberately kept out of `training_pool()`'s sampling entirely --
`scripts/generate_training_data.py` samples millions of (board, clue,
guesser) triples, and a real API call per example would be far too slow
and expensive. This is an evaluation-only addition, wired in exactly
like any other guesser (`codenames/two_team_arena.py`,
`scripts/run_two_team_arena.py --guesser-pool-config
configs/guesser_pool_llm.json --guesser llm`).

Registered in `GUESSER_CLASSES` as `"llm"`. New
`tests/test_guessers.py::TestLLMGuesser` (6 tests) using a fake injected
client -- no real network calls in the test suite: ranking uses the
model's ordering, malformed/partial responses fall back to the board's
own order for anything not mentioned, repeated identical inputs are
cached (verified against call count), a different `number` is a cache
miss, `score_candidates` stays monotonic with the ranking, and the
registry builds one from a plain config. 250 tests passing. Real
API cost estimate (Haiku 4.5 pricing): roughly $3 for a 300-board
two-team run (~4,000 turns, ~350 input + ~80 output tokens each) --
cheap in dollars, though real wall-clock time (thousands of network
round-trips) and non-determinism (needs averaging across a few runs) are
the actual costs to weigh. Not yet run for real -- no API key configured
in this environment; a small trial run is the natural next step once one
is.

Ran a real 15-board two-team trial against Haiku after getting API
access working (identity-linked keys need an `anthropic-workspace-id`
header, unless the key is scoped to a workspace at creation time --
worked around by creating a workspace-scoped key rather than plumbing
the header). Assassin-hit rate came back dramatically worse than
against the embedding guesser pool (~40% vs. 0-5%), which is the
concerning signal this guesser was built to be able to surface -- a
spymaster scored only against guessers it was co-adapted with can't
be told apart from one that's actually good. Spot-checked one
suspicious assassin hit ("shores" picking `England` over `Port`) by
replaying the exact same input outside the cache: got real run-to-run
variance (a fresh Haiku sample ranked `Port` #1 instead), and three
fresh Sonnet 5 samples on the same input all kept the assassin out of
the top 2 -- suggestive that Sonnet is more consistent here, though 3
samples isn't enough to call it settled.

Added `codenames/llm_store.py`: a persistent SQLite file (default
`cache/llm_store.db`, gitignored) with two tables --
`LLMResponseCache` (write-through cache for `LLMGuesser`'s API calls,
keyed by the same `(model, clue, candidates, number)` tuple as its
in-memory cache, opt-in via `cache_path=`) and `GameRecordStore` (one
row per two-team game: board layout by role, snapshotted before any
reveal, plus the full turn sequence). The point of both is that once a
run has spent real API money, that spend and the resulting data should
survive a crash, a Ctrl-C, or an accidental rerun -- not just live in
one process's memory until it exits. WAL mode makes concurrent writes
from the arena's multiprocessing workers safe without any extra
locking. `scripts/run_two_team_arena.py --record-games PATH
[--run-label NAME]` wires this into both the CPU path
(`two_team_arena.py`'s per-process workers) and the GPU-batched path
(`two_team_gpu_arena.py`, single process); `scripts/dump_game_records.py
PATH [--label ...] [--seed ...]` prints the same
board+transcript format `scripts/scratch_llm_transcripts.py` printed
ad hoc, but from durable storage instead of a replay. Storage cost is
negligible either way -- a few KB/game, so even a 10,000-game run is
well under a few hundred MB.

Found and fixed a real parallelization gap ahead of a larger Sonnet 5
run: `codenames/two_team_gpu_arena.py::_play_batch_group` batches the
spymaster's GPU forward pass across every active game in a round, but
the guesser call right after it was a plain sequential `for` loop -- one
blocking network round-trip at a time. With a learned spymaster (the
only case that routes through this GPU path), that loop, not the
spymaster batching, was the real bottleneck whenever the guesser is
`LLMGuesser`. Fixed by running each round's per-game guesser calls on a
`ThreadPoolExecutor` (sized to the batch, created once per batch group
and reused across rounds rather than per round) instead of a sequential
loop -- threads, not another process pool, since the work is I/O-bound
network waiting, not CPU-bound. Required making `LLMGuesser` and
`LLMResponseCache` safe under concurrent calls from one shared instance
(a `threading.Lock` around each's own mutable state -- `_cache`/lazy
`_client` construction, and the sqlite connection respectively) without
serializing the network calls themselves, which would have defeated the
whole point. New tests prove the calls actually overlap (elapsed time
for N concurrent slow fake calls stays well under N * delay, at both the
`LLMGuesser` level and the `_play_batch_group` level) rather than just
checking correctness. The CPU path (`two_team_arena.py`, used for
baseline spymasters) was already parallel across worker processes and
didn't need this.

Ran the first real multi-checkpoint Sonnet 5 trial (`noise_0_08`, 15
boards): assassin-hit rate 20.0%, half-turns (all) 8.60, mean clue
number 1.91, mean correct per clue 1.43 -- meaningfully safer than the
earlier Haiku trial (~40% assassin-hit) but still far above the ~0-5%
this spymaster gets against its own training-time guesser pool,
consistent with the original motivation for building `LLMGuesser` at
all. Found mid-session that `LLMGuesser` was significantly more
expensive than estimated: Sonnet 5 runs adaptive thinking by default
even though nothing asked for it, and `_query()` never captured
`response.usage`, so a token-count-based cost estimate (from stored
prompt/response text alone) silently missed ~192 invisible thinking
tokens per call -- real output tokens were ~294/call, not the ~136
visible ones, roughly doubling real cost versus the estimate. A direct
A/B on the same "shores" scenario found no quality difference between
thinking on and off (identical top-2 picks across 3 trials each,
if anything *more* consistent with thinking off) while cutting output
tokens ~2.7x (294 -> 108) and latency (~5.7s -> ~3s/call) -- so
`_query()` now passes `thinking={"type": "disabled"}` explicitly.
Confirmed this is accepted by both Sonnet 5 and the default Haiku 4.5
model (older models predating adaptive thinking already default to no
thinking when the param is omitted, but explicit `disabled` was
verified to not error on either). A `noise_0_0` trial started before
this fix was killed partway through rather than let finish on the
unfixed, more expensive path, given a real, small ($2.29) remaining API
balance -- multi-checkpoint comparison resumes with the fixed guesser.

## Renamed "codemaster" to "spymaster" throughout

The clue-giving role had been called `codemaster` everywhere since the
first commit; the actual Codenames rules call it the *spymaster*. Renamed
it project-wide so the code matches the game's vocabulary: 715
occurrences across 38 tracked files, plus the `codenames/codemasters/`
package directory and two test modules (`test_codemasters.py`,
`test_learned_codemaster.py`). Casing turned out to be uniform
(`codemaster` / `Codemaster` / `CODEMASTER`, no `CodeMaster`), so three
literal substitutions covered every case with no regex subtleties.

Checked before starting that nothing serialized depends on the old
module path: `LearnedSpymaster` checkpoints store a bare `model_state`
state_dict of tensors, not a pickled class reference, so existing
checkpoints under `cache/` still load unchanged. Verified afterwards by
reconstructing every tracked file from its `HEAD` version with the same
substitution applied to both path and contents and diffing against the
working tree -- 96 files, zero mismatches, so the commit is provably
nothing but the rename. Full suite passes (263 tests).

Docs were renamed along with the code, including historical entries in
this log and in `docs/versions/`. That does mean earlier entries now read
as though the role was always called "spymaster", which is a small
rewriting of the record; the alternative -- leaving history alone -- would
have left the docs disagreeing with the code about what the role is
called, which seemed worse for a project that has to be read and
defended as a whole.

## Iteration architecture steps 1-3: TurnContext, spymaster registry, batched-scoring protocol

Implemented the first three steps of `docs/iteration-architecture.md`
(steps 4-7 deliberately left for later, per that doc's own scoping).

**Step 1 (TurnContext + top-k interface).** `Spymaster.give_clue(board,
sims)` became `Spymaster.top_clues(ctx, sims, k)` as the one abstract
method, with `give_clue` now a concrete one-line wrapper
(`top_clues(ctx, sims, 1)[0]`). All six spymasters and every caller
(`codenames/game.py::play_turn`, `codenames/gpu_arena.py`,
`codenames/two_team_gpu_arena.py`, `scripts/web_inspector.py`) updated.
`play_turn` builds the `TurnContext` (`turn_index=len(board.revealed)`,
the same proxy `LearnedSpymaster` used to reconstruct internally) so
`LearnedSpymaster` no longer recomputes it itself -- closes the
train/serve-skew risk its own docstring used to flag.

Judgment call: `RandomSpymaster` never had a `top_k_clues` method before
(there's no real "ranking" for a uniform random pick) -- `top_clues` now
returns k independent, distinct, legal random draws with score 0.0, and
the k=1 case reproduces the exact old `rng.choice`/`rng.randint` draw
sequence so `give_clue`'s behavior is byte-for-byte unchanged. One
consequence: `scripts/web_inspector.py`'s rarity filter, which previously
silently no-op'd for Random (it had no `top_k_clues` to hasattr-gate on),
now actually filters Random's draws too. This only affects the UI
exploration tool, not evaluation.

Judgment call: `LearnedSpymaster` used to have two independent
clue-selection paths -- `give_clue` (via `_pick_legal_clue`, which applies
a rarity filter when one is configured) and `top_k_clues` (via
`top_k_legal_clues`, which never filtered). Unifying them behind one
`top_clues` while preserving both existing behaviors exactly meant
branching on `k` inside `top_clues` (k==1 keeps the rarity-aware pick,
k>1 keeps the unfiltered top-k) rather than generalizing the filter to
every k -- generalizing would have either silently started returning
fewer-than-requested candidates in a new way or silently started padding
results to k, both of which are behavior changes a "pure refactor" step
shouldn't introduce on its own judgment.

**Step 2 (spymaster registry).** Added `codenames/spymasters/registry.py`
+ `configs/spymasters.json`, mirroring `codenames/guessers/registry.py`'s
structure (`SPYMASTER_CLASSES`, `load_spymasters`, a `SpymasterEntry` with
a picklable `.spec` and a `.build()`). Deleted the three duplicated
hardcoded baseline tables in `scripts/run_arena.py`,
`scripts/run_two_team_arena.py`, and `scripts/web_inspector.py`.

Judgment call: `run_arena.py`'s baseline set (`random`/`centroid`/
`linear_scorer`) and `run_two_team_arena.py`'s (those three plus
`oracle`) were never the same set, so a single shared config listing all
four baselines could not just be consumed wholesale by both scripts
without silently adding `oracle` to `run_arena.py`'s cross-play matrix (a
real output/behavior change, not just a refactor). Each script instead
keeps its own short list of *names* into the shared config
(`BASE_SPYMASTER_NAMES`), preserving each script's exact existing
baseline set while still eliminating the duplicated `(class, kwargs)`
literals. `learned` isn't buildable straight from the config (no fixed
checkpoint path belongs in a static file) -- `registry.spymaster_spec(name,
**overrides)` merges a caller-supplied `checkpoint_path` (and any reward
overrides) into that entry's params at call time.

**Step 3 (batched-scoring protocol).** Added `TurnContext`-based
`BatchScoringSpymaster` protocol (`score_batch`/`to_device`) to
`codenames/spymasters/base.py`. `codenames/gpu_arena.py` and
`codenames/two_team_gpu_arena.py` no longer import `LearnedSpymaster` or
touch `.model`/`.own_reward`/`.miss_penalty`/`expected_reward_and_best_n`
-- they gather `TurnContext`s and call `score_batch`/`to_device`, then
still do `clue_search.top_legal_clue` themselves (legality stays in the
arena, per the doc). `LearnedSpymaster.top_clues`/`give_clue` now route
through `score_batch` with a one-element context list, so there is one
scoring implementation, not two.

Judgment call (flagged rather than silently resolved): `score_batch`'s
feature-construction step branches on `self.device.type`. On CUDA it uses
`codenames/gpu_features.py`'s torch-batched gather (which materializes
the whole similarity tensor on-device once per process -- already true of
the GPU arena before this refactor). On CPU it instead loops
`codenames/features.py::build_features_batch` per context, which only
reads the handful of tensor columns a board actually needs off the mmap.
Using the GPU-oriented path unconditionally would have been simpler (one
literal code path instead of a branch), but `scripts/run_arena.py
--checkpoint --no-gpu-batch` constructs a `LearnedSpymaster` fresh inside
each of N spawned CPU worker processes (`codenames/arena.py`), and
materializing a private full-tensor copy in every one of them is exactly
the RSS blowup `docs/design-decisions.md`'s memory design note (and
`spymasters/linear_scorer.py`'s docstring) warns against. Verified by
running that `--no-gpu-batch` path (fake checkpoint, real similarity
tensor) and confirming per-worker peak RSS only grew ~240MB over the
baselines-only run, not by the tensor's full size -- and separately
confirmed the CPU and CUDA branches produce numerically identical
results (`run_arena.py --checkpoint` with and without `--no-gpu-batch`,
and `run_two_team_arena.py` likewise, all matched exactly; this machine
actually has CUDA available, so `tests/test_gpu_arena.py` and
`tests/test_two_team_gpu_arena.py` ran for real rather than being
skipped).

All 277 tests pass (263 original + 14 new: a spymaster-registry test
class, a `RandomSpymaster.top_clues` test, a `score_batch` batching test,
and an abstractness test for the new `top_clues` requirement). Steps 4-7
(rollout caching, frozen eval suite, eval store, naming/layout) are
untouched, per the assigned scope.

## Review of steps 1-3: spymasters selected by role, not by name

Steps 1-3 were implemented by a delegated agent against
`docs/iteration-architecture.md`; this entry records the review.

Steps 1 and 3 held up. The GPU arenas now reach the model only through
`score_batch`/`to_device`, there is a single scoring implementation per
model, and the device branch inside `LearnedSpymaster.score_batch` is
right for a non-obvious reason: using the multi-board GPU feature builder
on CPU would materialize the whole similarity tensor in every spawned
worker, which is exactly the RSS failure `design-decisions.md`'s memory
note warns about.

Step 2 was the miss, and it was a miss against the *goal* rather than the
letter of the spec. The registry and config landed correctly, but each
arena script kept its own hardcoded list of names
(`BASE_SPYMASTER_NAMES = ["random", "centroid", "linear_scorer", ...]`) --
so adding a spymaster would still have meant editing two scripts, which
is the exact hand-editing the registry existed to remove. The reasoning
behind it was sound in isolation: the two scripts genuinely never had the
same set (`run_two_team_arena.py` also offers `oracle`), and consuming
one shared config wholesale would have silently added `oracle` to
`run_arena.py`'s cross-play output, which would not have been a pure
refactor.

The fix keeps that distinction but moves it into the config as data: each
entry carries `roles`, and scripts ask for a role rather than a list of
names -- `spymaster_names("baseline")` for `run_arena.py`,
`spymaster_names("baseline", "exploration")` for
`run_two_team_arena.py`, `oracle` being the sole "exploration" entry.
Both lists reproduce the previous sets exactly, verified by comparison.
`roles` defaults to `("baseline",)` so an entry that omits the field
appears in the arenas rather than being silently invisible -- confirmed
by adding a roles-less entry to an in-memory config and checking it is
picked up. 277 tests pass.

The general lesson for delegating the remaining steps: a brief that
states the requirement ("no other file changes to make a model
runnable") gets checked against the letter of each numbered step, not
against the requirement. Worth stating the acceptance test, not just the
task.

## Step 4: rollouts stored separately from features

**Expected:** splitting generation into "simulate the guesser" (expensive,
model-independent) and "build features" (cheap, model-specific) would make
a feature-design change much cheaper, since features are now expected to
vary between models and re-simulating identical rollouts to get different
columns is pure waste. Guessed ~6x storage saving and did not predict the
time saving.

**What actually happened**, measured on a 500-example sample at seed 123:

- Generation runs at ~318 examples/sec; featurizing stored rollouts runs
  at ~11,161. So a feature-vector change now costs about **1/35th** of a
  regeneration. Bigger than expected, and it is the real payoff -- the
  storage saving is secondary.
- Storage is 4.1x smaller (54,134 bytes vs 222,512), not the ~6x guessed.
  A row is ~87 bytes, not the ~60 estimated: the 25 uint16 word indices
  plus 25 role bytes dominate, and packing roles to 2 bits was not worth
  the decode complexity. `docs/iteration-architecture.md` corrected.
- **Equivalence was verified, not assumed.** The old generator was run
  first at seed 123 to capture a baseline dataset, then the refactored
  generate -> featurize path was run at the same seed: `features`,
  `outcome`, `reward` and `seed` all compare exactly equal. This is why
  `emit()` keeps its guesser RNG draw in precisely its old position --
  moving it would have reordered the draw sequence and silently changed
  which boards and clues a given seed produces.

Two deliberate consequences:

- **Reward is no longer stored.** It is exactly `k * ROLE_REWARD[OWN] +
  ROLE_REWARD[cause]`, a pure function of (k, cause) and the four reward
  constants, and `docs/design-decisions.md` requires those to remain
  changeable at scoring time with no retraining -- a saved `reward`
  column quietly contradicted that. `rollouts.reward_for` derives it, so
  changing a reward constant now reprices an existing rollout set for
  free. The featurized output still contains the identical column, since
  it is recomputed there.
- **Perspective is stored resolved.** Rows sampled from
  `OpponentBoardView` store each word's role *as the acting side sees
  it*, so `board_from_row` returns a plain `Board` and nothing downstream
  needs a perspective flag. `swapped` is kept for diagnostics only.

The featurized output is byte-compatible with the old layout on purpose,
so `scripts/train_scorer.py` needed no changes whatsoever -- which is also
what made the equivalence check a direct array comparison.

Four tests in `test_generate_training_data.py` failed after the change,
all asserting the old `features_*.npy` output. Rather than relax them, the
two that test a genuine *feature* property (the widened 9th opponent slot
under perspective swapping) now run generate -> featurize end to end,
which is a better test than before. 292 pass (277 + 15 new in
`tests/test_rollouts.py`).

**Follow-up not taken:** `run_ablation_study.py`'s `unsorted` variant uses
the same seed as `base`, so its rollouts are now bit-identical to base's
and it could skip generation entirely and just re-featurize -- one fewer
generation pass out of six. Not done because those jobs run concurrently
in a process pool and would race on the shared directory; it needs the
job graph sequenced first.
## Iteration architecture steps 5-6: frozen eval suite, eval store

Implemented steps 5 and 6 of `docs/iteration-architecture.md` (step 7,
naming/layout, deliberately left for later per the assigned scope; steps
1-4's files -- `scripts/generate_training_data.py`,
`scripts/train_scorer.py`, `codenames/features.py`,
`codenames/gpu_features.py` -- were treated as read-only since another
workstream was editing them concurrently).

**Step 5 (frozen eval suite).**

Raised `codenames/assets/board_words_holdout.txt` from 60 to 150 words,
keeping the original 60 as an exact subset (per the doc's math: two
25-word boards drawn from an H=60 pool share ~10.4 words; H=150 brings
that down to ~4.2). Reproducing snippet (also in
`codenames/board.py::load_holdout_wordlist`'s docstring and exercised
byte-for-byte by `tests/test_board.py::test_holdout_selection_is_reproducible_from_the_recorded_snippet`):

```python
import random
from codenames.board import load_wordlist

vocab = load_wordlist()  # already alphabetically sorted, 400 words
original_60 = set(random.Random(42).sample(vocab, 60))  # the pre-existing holdout
remaining = [w for w in vocab if w not in original_60]  # 340 words
assert len(remaining) == 340
new_90 = random.Random(43).sample(remaining, 90)  # uniform, no stratification
holdout_150 = sorted(original_60 | set(new_90))
# written to codenames/assets/board_words_holdout.txt, one word per line,
# alphabetically sorted (matching the existing file's convention) so a
# diff against the old 60-word file stays readable.
```

Seed 43 was picked with no special reasoning beyond "a fresh, explicitly
recorded seed adjacent to the existing 42" -- the doc only requires that
it be recorded and reproducible, not that it be meaningful.

Added `configs/eval_suite.json`: a fixed list of 100 board seeds (0-99,
not just a count -- extending it later means appending more entries, not
changing the meaning of the existing ones), pointing at
`configs/guesser_pool_llm_sonnet.json` / `"llm"` as the fixed evaluation
guesser, with `"llm_model": "claude-sonnet-5"` recorded explicitly
alongside it (duplicating what the guesser config already implies)
because the doc calls out the LLM model id specifically as part of suite
identity -- see `EvalSuite.suite_id` below.

Judgment call: the doc's step-5 bullets talk about the *holdout word
list*, but design-decisions.md's "First-pass simplifications" section is
explicit that the entire point of that holdout mechanism is "a later
evaluation pass can build boards entirely from unseen words," and flags
that this "has never actually been run against a trained model." Read
together, the frozen eval suite is that pass -- so
`codenames/eval_suite.py::run_eval_suite` builds every eval board via
`Board.generate(seed=s, vocabulary=load_holdout_wordlist())`, not the
default full 400-word vocabulary. This required adding a `vocabulary`
parameter to `codenames/two_team_gpu_arena.py::run_two_team_self_play_gpu`
(threaded through to `Board.generate`, default `None` so every existing
caller is unaffected). Flagging this explicitly since it's an inference
from two docs read together, not a literal instruction in either one --
if held-out-vocabulary boards weren't intended for this suite, that one
parameter is the line to revert.

**Step 6 (eval store).**

`codenames/llm_store.py::GameRecordStore` gained `spymaster_id` and
`suite_id` columns (migrated onto an existing db file via `ALTER TABLE`
if missing, so a live `cache/llm_store.db` isn't invalidated) and a
`UNIQUE INDEX` on `(spymaster_id, suite_id, seed)`. `add_game` now takes
optional `spymaster_id`/`suite_id` kwargs and always issues `INSERT OR
REPLACE` -- SQLite treats NULL as distinct from every other value
(including another NULL) in a unique index, so every pre-existing caller
that only passes `label` (e.g. `scripts/run_two_team_arena.py`-style
training diagnostics) is completely unaffected and keeps appending rather
than colliding. Two new query methods, `recorded_seeds` and
`games_for_suite`, are what an eval runner checks *before* deciding to
simulate anything.

Added `codenames/eval_suite.py`:
- `EvalSuite`/`load_eval_suite`: parses `configs/eval_suite.json`.
  `suite_id` hashes `(name, llm_model, guesser_name, guesser_pool_config's
  file *content*)` -- deliberately **not** `board_seeds`, so extending a
  suite from 100 to 300 boards doesn't invalidate the 100 games already
  paid for. Hashing the guesser config's content rather than its path
  also means an edit to that file (a different noise seed, a changed
  prompt parameter) invalidates the suite instead of silently reusing
  stale results under an unchanged path -- a stricter reading of "the LLM
  model id is part of the suite identity" than the doc technically asks
  for, but consistent with its reasoning.
- `spymaster_identity`/`checkpoint_content_hash`: a trained model's
  identity is `f"{name}:{sha256(checkpoint_bytes)[:16]}"`, not its path,
  directly per the doc's `scorer_best.pt`-gets-overwritten warning.
  Baselines (no checkpoint) are identified by name alone.
- `run_eval_suite`: filters `suite.board_seeds` down to what
  `store.recorded_seeds(spymaster_id, suite.suite_id)` doesn't already
  have *before* calling `run_two_team_self_play_gpu` at all -- so a fully
  cached rerun never enters the simulation/LLM-call path, not just
  cache-hits inside it.

Judgment call: reused `run_two_team_self_play_gpu` (GPU-batched two-team
self-play, same spymaster on both sides) as the simulation engine rather
than writing a new one, since it already does exactly what evaluation
needs (batches an LLM guesser's network calls across simultaneous games
on a thread pool) and duplicating that logic would violate "one scoring/
simulation implementation" as much as step 3's original motivation. Only
`two_team_gpu_arena.py` was touched, not the CPU-parallel
`two_team_arena.py::run_two_team_self_play` -- the eval suite doesn't
need a CPU fallback path today, and adding the same parameters there
unused would be speculative.

Judgment call: 100 board seeds for `configs/eval_suite.json`'s default is
an arbitrary but explicit choice, not derived from anything in either
doc -- picked as a plausible frozen-suite size to start from
(considerably fewer than the "100 to 300" example the doc uses in
passing). This number should be revisited before the suite is actually
run for real money.

Tests added: `tests/test_board.py` (3 new: exact count, original-60
subset, byte-for-byte reproducibility), `tests/test_eval_suite.py` (15
new: suite-id stability/dependence on each identity component,
checkpoint-hash-based spymaster identity including the same-path
different-checkpoint case, `GameRecordStore` key scoping and
idempotency, and `run_eval_suite`'s skip-if-cached and
extend-only-simulates-new-seeds behavior via a monkeypatched simulation
function so no real LLM calls or CUDA are needed to test the caching
logic itself). `tests/test_llm_store.py` and
`tests/test_two_team_gpu_arena.py` (pre-existing) still pass unchanged,
confirming the new columns/parameters are additive. 295 tests pass
total (280 pre-existing + 15 new; the pre-existing count itself grew
from 277 to 280 as the concurrent step-4 workstream landed its own
tests).

Not done, flagged rather than silently skipped: no script was added
under `scripts/` to actually invoke `run_eval_suite` end-to-end against
a real trained checkpoint and real Sonnet-5 API calls -- that would spend
real money and touches territory close to step 7's "naming and layout"
reorganization, which was explicitly out of scope for this pass. The
plumbing (`codenames/eval_suite.py`) is complete and unit-tested, but has
not been run against a live LLM.

## Step 7: scripts grouped, naming conventions written down

`scripts/` was 22 files with no grouping — a one-time corpus download sat
beside a per-iteration training run. Now grouped by *when you run it*:
`data/` (build time, once), `pipeline/` (the iteration loop), `tools/`
(interactive), with a `scripts/README.md` index.

The grouping wasn't free to choose. Four scripts import a sibling through
`sys.path` (`build_similarity_tensor`/`extend_similarity_tensor` ->
`_embedding_lib`, `run_ablation_study` -> the three pipeline scripts,
`web_inspector` -> `inspector`), so any split that separated one of those
pairs would have broken the import. All four pairs happened to fall inside
the same natural group, so `Path(__file__).parent` inserts still resolve
and nothing needed rewriting — worth recording, because had they not, the
right fix would have been to move those functions into `codenames/` and
leave thin CLI wrappers behind, a much bigger change.

Two path bugs the move introduced, both caught by checking rather than by
tests:

- `featurize_rollouts.py` computed `PROJECT_ROOT` as
  `Path(__file__).resolve().parent.parent`, which silently became
  `scripts/` instead of the repo root once the file moved a level deeper.
  Now `parents[2]` with a comment saying why. The test suite did *not*
  catch this (tests import the module directly rather than running it as a
  script), which is exactly the kind of gap a 310-test suite can still
  have.
- Three test files inserted `scripts/` into `sys.path`; they now insert
  the specific group they import from.

`scratch_llm_transcripts.py` fails a `--help` smoke test, but it did
before the move too: it has no argparse and calls the API at import, so it
dies on a missing `ANTHROPIC_API_KEY`. Not a regression; noted so the next
person doesn't chase it.

Deliberately *not* done: renaming existing `cache/` artifacts (`cache/m9/`,
`cache/arena_blend.db`, `cache/sanity_check2.db`). They are gitignored
local data — renaming breaks nothing and proves nothing, while touching
the directory that holds `cache/llm_store.db`, the only record of paid LLM
responses and the only reason past evals are reproducible. The naming
convention in `CLAUDE.md` applies to new artifacts.

Also deliberately not done: rewriting the ~79 references to old script
paths in this log. Those entries describe where files were when they were
written. Every current-facing document (README, design-decisions,
iteration-architecture, the code's own docstrings — 35 of those) was
updated.

`CLAUDE.md` gained the descriptive-model-naming rule (a name should say
what changed; `k_cause_mlp`, not `v2`), the cache-layout convention, a
pointer to `docs/iteration-architecture.md`, and an explicit warning that
`v1`/`v1.1` are not comparable against models trained under the 150-word
holdout. 310 tests pass.

## Clean slate before the next model

Cleared out what the restructuring had left behind, before starting a new
baseline.

**The big one: `SCOPE.md` did not exist.** It was the project's original
spec, was never tracked in git, and yet 99 references to it survived in
docstrings and docs, along with 76 milestone tags (M0-M9) -- despite
`CLAUDE.md` stating the project is no longer organized around milestones.
Every citation pointed at nothing, which for an orally-defended project is
a liability ("what's SCOPE.md?" / "it doesn't exist"). All of them are now
gone outside `docs/log.md`, each rewritten to keep whatever information it
carried rather than blank-deleted, and repointed at documents that do exist
where one applied.

`docs/log.md` was deliberately excluded: it records what was true when each
entry was written. Rewriting it would falsify the record, which is the
opposite of what a log is for.

Verification worth keeping: after the sweep, every modified `.py` file's
AST -- with docstrings stripped -- was compared against the previous commit.
Four files showed real differences, all of them display strings that
themselves contained SCOPE references (a `print`, a report header, a `note=`
field). Nothing executable changed. That check is the only reason I can
state that confidently across ~25 files.

**Retired:** `docs/versions/v1.md` and `v1.1.md`, the two superseded LLM
pool configs (Haiku and Sonnet, replaced by Opus), and
`scratch_llm_transcripts.py`. The README's per-model result tables are gone
too; no model holds published results now, since the earlier ones were
trained against the 60-word holdout and aren't comparable with anything
trained under the 150-word split.

**Kept, against a broader reset, with reasons:**

- The ablation machinery. `docs/design-decisions.md` cites it in three
  places as standing rationale, and `run_ablation_study.py` is the only
  generator for the `noise_*` checkpoints `web_inspector.py` offers.
  Deleting it would have orphaned three citations -- the same failure being
  cleaned up here.
- The blend and history-aware pool configs. `web_inspector.py` loads them
  at import and tests assert on them; rewiring them inline instead would
  contradict `design-decisions.md`'s rule that pool composition lives in a
  config file, not code.
- **All 7 trained checkpoints.** This one nearly went wrong. The plan was
  to delete `cache/m9/`, estimated at ~25MB of superseded run data. It is
  actually 514MB, and it contains the six `noise_*` checkpoints plus
  `blend_pool`'s -- the only trained models in the project, loaded by the
  web UI. Worse, they can no longer be reproduced: the board holdout went
  from 60 to 150 words, so the training vocabulary is different and a rerun
  would not recreate these weights. Deleting 3.5MB of irreplaceable
  artifacts to reclaim disk, while breaking a working tool, would have been
  a bad trade. Deleted the regenerable bulk instead -- training-data shards
  and superseded arena/sanity databases -- freeing 617MB (880MB -> 263MB)
  and keeping every checkpoint.

`cache/llm_store.db` was backed up first, to
`~/CodenamesAI-backups/`, and verified on both copies: integrity ok, 1230
cached responses, 105 game records. It is the only record of paid LLM calls
and, since LLM output isn't deterministic even at temperature 0, the only
thing making past evaluations reproducible.

**Process note.** Two delegated agents ran concurrently on the same working
tree by mistake, and both were killed mid-run (a machine sleep, then a rate
limit). One of them, while recovering, restored the deleted files from a
pre-deletion commit -- so the deletions had to be redone. The lesson is the
one already known and ignored here: concurrent agents belong in separate
worktrees. Nothing was lost, because the partial work had been committed
first, but the check that caught it was reading `git status` rather than
trusting the agents' reports.

314 tests pass.

## Fourth embedding space (fastText on Fandom) dropped from scope

The plan was four embedding spaces; three were built (GloVe, Numberbatch,
Wikipedia2Vec) and the fourth -- a fastText model trained locally on a
Fandom wiki corpus, meant to supply pop-culture and proper-noun knowledge
-- never was. Declared out of scope, so the corpus-collection machinery
and every "coming later" reference are gone: four scripts deleted
(`download_fandom_dumps.py`, `check_fandom_dumps.py`,
`extract_fandom_corpus.py`, `fandom_wikis.txt`, 513 lines) and 12
references rewritten across 8 files. `docs/log.md` untouched, as with the
SCOPE sweep.

Two things fell out of this that were not the point of the change:

- **A pre-existing documentation bug in `codenames/features.py`.** Its
  layout docstring described the per-space block as 25 slots and the total
  as `25*n + 25 + 3`. The real per-space block has been **26** since
  `FEATURE_SLOT_COUNTS[OPPONENT]` was widened from 8 to 9 for
  `OpponentBoardView`'s swapped perspective -- so the documented width has
  been wrong ever since, and would still have been wrong if only the
  "4 planned spaces" had been changed to 3. Verified against the code:
  9+9+7+1 = 26, and 26*3 + 26 + 3 = 107 = `feature_dim(3)`, matching the
  107-wide feature rows generated in practice. The docstring now points at
  `feature_dim()` as the source of truth rather than restating a number
  that can drift again.
- **The concatenate-don't-average rationale had to be re-illustrated.**
  `docs/design-decisions.md` argued for concatenating spaces using "high
  fastText similarity, near-zero GloVe similarity". With fastText gone the
  example named a space that does not exist, so it now contrasts
  Numberbatch against GloVe: a clue driven by a structured commonsense
  relation ConceptNet encodes explicitly, which GloVe's co-occurrence
  statistics never surface. The argument is unchanged; only the pair of
  spaces illustrating it moved.

**Open, and deliberately not resolved here:** the motivating problem at the
top of `design-decisions.md` still uses the Technoblade example -- a clue
GloVe cannot represent at any threshold. The fourth space was what would
have supplied it. The multi-space argument survives intact (three spaces do
carry genuinely different knowledge), but the headline example now
describes a gap nothing in the project can close. Whether to re-anchor it
on a gap the built spaces actually exhibit is a framing decision about how
the project is presented, not a cleanup task.

Raw corpus data (`data/fandom_dumps`, `data/fandom_text`, ~2.7GB) is left
on disk pending a decision; only `cache/fandom_dump_status.json`, written
by one of the deleted scripts, was removed. 314 tests pass.

## Human evaluation (not started)

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

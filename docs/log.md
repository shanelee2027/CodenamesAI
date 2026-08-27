# Working log

Record what was expected vs. what actually happened as work proceeds. One
section per milestone. See `docs/SCOPE.md` for the milestone definitions.

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

## M7 — Features and data generation

## M8 — Scorer

## M9 — Evaluation and ablations

## M10 — Human evaluation

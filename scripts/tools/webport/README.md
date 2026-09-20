# The browser port of the distilled listener

The source of the published artifact (a Codenames game playable on a phone,
with the model running client-side). Kept here because it was previously only
on claude.ai: a WSL reset wiped the scratchpad it was written in, and the
published copy was briefly the only copy.

| file | what it is |
|---|---|
| `index.html` | the page: board, game loop, clue search, the k=1 rule |
| `features.js` | port of `codenames/listener_features.py::extract` |
| `gbt.js` | LightGBM tree walker, missing-value rules included |
| `reward.js` | port of `codenames/pl_reward.py` |

The two large data files are **generated, not authored**, and stay out of git:

    .venv/bin/python scripts/tools/export_listener_js.py --out <dir>/model.json
    .venv/bin/python scripts/tools/export_board_inputs.py --out <dir>/inputs.json \
        --n 20 --top 145

`--top 145` is not arbitrary: at the default 260/side the file reaches 24.7 MB,
past the artifact service's 16 MB per-file limit. 145 gives ~250 clues per
board at 13.2 MB.

**This is not the model.** It is a port over ~20 pre-exported boards, because a
browser cannot hold the 11,145 x 25 x 3 similarity tensor. To play the real
thing against freshly generated boards, run `scripts/tools/play_server.py`
instead -- that calls `LearnedListenerSpymaster` directly.

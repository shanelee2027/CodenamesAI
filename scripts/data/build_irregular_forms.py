"""Build codenames/assets/irregular_forms.tsv: irregular inflections and their base forms, from
WordNet 3.0's exception lists (noun.exc, verb.exc, adj.exc, adv.exc).

codenames/board.py::is_legal_clue catches regular forms by suffix stripping, which cannot reach
irregular ones (knives/Knife, drew/Drawing, mice/Mouse, led/Lead). Measured on 2026-11-11, 1.6% of
the Gaussian spymaster's clues on boards of the 800 online words were such forms, and gpt-oss
guessed the board word every time. WordNet's exception lists are exactly the irregular cases, so
the legality rule uses them as a fixed table: no lemmatizer runs at game time and no nltk import
is needed, which keeps the rule predictable and testable.

Reads the project's own copy of WordNet (data/nltk_data, gitignored) directly from the zip:

    python scripts/data/build_irregular_forms.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORDNET_ZIP = ROOT / "data" / "nltk_data" / "corpora" / "wordnet.zip"
OUT = ROOT / "codenames" / "assets" / "irregular_forms.tsv"

HEADER = """\
# Irregular inflection -> base forms, from the exception lists of WordNet 3.0
# (noun.exc, verb.exc, adj.exc, adv.exc); single alphabetic words only.
# Built by scripts/data/build_irregular_forms.py; used by codenames/board.py::is_legal_clue.
# WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved.
# WordNet is distributed under the WordNet license (https://wordnet.princeton.edu/license-and-commercial-use),
# which permits use, copying and distribution provided this copyright notice is kept.
"""


def main() -> None:
    forms: dict[str, set[str]] = {}
    with zipfile.ZipFile(WORDNET_ZIP) as z:
        for pos in ("noun", "verb", "adj", "adv"):
            for line in z.read(f"wordnet/{pos}.exc").decode().splitlines():
                form, *bases = line.split()
                if not re.fullmatch(r"[a-z]+", form):
                    continue
                bases = [b for b in bases if re.fullmatch(r"[a-z]+", b) and b != form]
                if bases:
                    forms.setdefault(form, set()).update(bases)
    lines = [f"{form}\t{','.join(sorted(bases))}" for form, bases in sorted(forms.items())]
    OUT.write_text(HEADER + "\n".join(lines) + "\n")
    print(f"{len(lines)} forms -> {OUT}")


if __name__ == "__main__":
    main()

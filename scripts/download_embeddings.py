"""Pre-fetch the raw pretrained embedding files for M2/M4 (SCOPE.md §2, §5).

Downloads GloVe, ConceptNet Numberbatch, and Wikipedia2Vec into
data/embeddings/raw/. This only stages the raw files -- it does not build
the similarity tensor (that's M2/M4 code, built after M1 per SCOPE.md's
milestone order). Staging now is justified the same way M0 is: these are
large, latency-bound downloads with no code dependency, not a shortcut
around "one module at a time."

None of these hosts publish an official checksum, so "checksum
verification" here means: record a sha256 after every successful download
into a .sha256 sidecar file, and use expected Content-Length (from a HEAD
request) to detect a truncated / corrupt local copy on re-run, not to
verify against a canonical hash.

fastText (the fourth embedding space) is trained locally on the Fandom
corpus per SCOPE.md §M4 and is not downloaded here.

Usage:
    python scripts/download_embeddings.py
    python scripts/download_embeddings.py --only glove numberbatch
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "CodenamesAI-embedding-fetch/0.1 (senior CS project, research use)"
CHUNK_SIZE = 1024 * 1024


@dataclass
class EmbeddingSource:
    name: str
    url: str
    filename: str
    note: str


SOURCES = [
    EmbeddingSource(
        name="glove",
        url="https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
        filename="glove.6B.zip",
        note="6B-300d per SCOPE.md M2 (fast iteration first; 840B tested later)",
    ),
    EmbeddingSource(
        name="numberbatch",
        url="https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz",
        filename="numberbatch-en-19.08.txt.gz",
        note="English-only v19.08 -- board/clue vocab is English, no need for the ~78-language file",
    ),
    EmbeddingSource(
        name="wikipedia2vec",
        url="http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2",
        filename="enwiki_20180420_300d.txt.bz2",
        note="300d text format, matches GloVe's dimensionality; text format avoids needing the wikipedia2vec package just to load vectors",
    ),
]


def head_content_length(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
        return int(length) if length else None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, expected_size: int | None, timeout: float) -> None:
    existing = dest.stat().st_size if dest.exists() else 0
    if expected_size is not None and existing == expected_size:
        return

    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            return
        if e.code == 200 and mode == "ab":
            # server ignored Range and is sending the full file again
            mode, existing = "wb", 0
            resp = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)
        else:
            raise

    downloaded = existing
    with dest.open(mode) as f:
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if expected_size:
                pct = 100 * downloaded / expected_size
                line = f"  {dest.name}: {downloaded/1e6:8.1f} / {expected_size/1e6:.1f} MB ({pct:5.1f}%)"
                print(f"\r{line:<80}", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", type=Path, default=Path("data/embeddings/raw"))
    parser.add_argument("--only", nargs="*", choices=[s.name for s in SOURCES], help="restrict to these sources")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    sources = [s for s in SOURCES if args.only is None or s.name in args.only]
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    for src in sources:
        dest = args.dest_dir / src.filename
        print(f"{src.name}: {src.note}")
        expected_size = head_content_length(src.url, args.timeout)
        print(f"{src.name}: downloading -> {dest} ({(expected_size or 0)/1e6:.1f} MB)")
        download_with_resume(src.url, dest, expected_size, args.timeout)

        digest = sha256_of(dest)
        (dest.with_suffix(dest.suffix + ".sha256")).write_text(f"{digest}  {dest.name}\n")
        print(f"{src.name}: done, {dest.stat().st_size/1e6:.1f} MB, sha256={digest}\n")


if __name__ == "__main__":
    main()

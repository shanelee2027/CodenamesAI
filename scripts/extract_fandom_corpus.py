"""Extract plaintext from downloaded Fandom dumps (SCOPE.md §M0).

For each <wiki>_pages_current.xml.7z in data/fandom_dumps/:
  1. Decompress with `7z` (requires p7zip-full: `sudo apt-get install p7zip-full`).
  2. Run wikiextractor over the resulting MediaWiki XML to strip markup down
     to plaintext, written to data/fandom_text/<wiki>/.
  3. Delete the intermediate .xml (kept only long enough for step 2) to avoid
     holding both the compressed and uncompressed copies on disk at once.

Reports total plaintext size across all wikis at the end, to track against
the 1-5GB target in SCOPE.md §M0.

Usage:
    python scripts/extract_fandom_corpus.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def find_7z() -> str:
    for candidate in ("7z", "7za", "7zr"):
        if shutil.which(candidate):
            return candidate
    raise SystemExit(
        "No 7z binary found. Install p7zip-full first:\n"
        "    sudo apt-get install -y p7zip-full"
    )


def extract_7z(sevenzip: str, archive: Path, out_dir: Path) -> Path:
    subprocess.run([sevenzip, "x", "-y", f"-o{out_dir}", str(archive)], check=True)
    xml_files = list(out_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"no .xml found after extracting {archive}")
    xml_path = xml_files[0]
    # These dumps sometimes carry stored permission bits of 000 (observed on
    # the "disney" dump), which 7z preserves verbatim on extraction.
    os.chmod(xml_path, stat.S_IRUSR | stat.S_IWUSR)
    return xml_path


def run_wikiextractor(xml_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "wikiextractor.WikiExtractor", str(xml_path), "-o", str(out_dir), "--no-templates"],
        check=True,
    )


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps-dir", type=Path, default=Path("data/fandom_dumps"))
    parser.add_argument("--xml-scratch-dir", type=Path, default=Path("data/fandom_xml_scratch"))
    parser.add_argument("--text-dir", type=Path, default=Path("data/fandom_text"))
    parser.add_argument("--keep-xml", action="store_true", help="don't delete the intermediate .xml")
    args = parser.parse_args()

    sevenzip = find_7z()
    archives = sorted(args.dumps_dir.glob("*_pages_current.xml.7z"))
    if not archives:
        raise SystemExit(f"no dumps found in {args.dumps_dir} -- run download_fandom_dumps.py first.")

    args.xml_scratch_dir.mkdir(parents=True, exist_ok=True)
    args.text_dir.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        wiki = archive.name.removesuffix("_pages_current.xml.7z")
        wiki_text_dir = args.text_dir / wiki
        if wiki_text_dir.exists() and any(wiki_text_dir.rglob("*")):
            print(f"{wiki}: already extracted, skipping")
            continue

        print(f"{wiki}: decompressing...")
        xml_path = extract_7z(sevenzip, archive, args.xml_scratch_dir)

        print(f"{wiki}: running wikiextractor...")
        run_wikiextractor(xml_path, wiki_text_dir)

        if not args.keep_xml:
            xml_path.unlink()

        size_mb = dir_size_bytes(wiki_text_dir) / 1e6
        print(f"{wiki}: {size_mb:.1f} MB of plaintext")

    total_gb = dir_size_bytes(args.text_dir) / 1e9
    print(f"\nTotal extracted plaintext: {total_gb:.3f} GB (target: 1-5 GB per SCOPE.md §M0)")


if __name__ == "__main__":
    main()

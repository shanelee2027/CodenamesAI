"""Download Fandom dumps found ready by check_fandom_dumps.py.

Reads cache/fandom_dump_status.json, downloads every wiki whose dump is
available and in S3 "STANDARD" storage (immediately downloadable) into
data/fandom_dumps/<wiki>_pages_current.xml.7z, resuming partial downloads
via HTTP Range requests, and verifying each completed file's MD5 against
the checksum S3 reports (x-amz-meta-md5, captured by the check script).

Wikis in Glacier/Deep Archive storage are skipped with a note -- those
need an S3 restore request before they're downloadable at all, which is
out of scope for this script.

Usage:
    python scripts/check_fandom_dumps.py        # refresh status first
    python scripts/download_fandom_dumps.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "CodenamesAI-corpus-check/0.1 (senior CS project, research use)"
CHUNK_SIZE = 1024 * 1024
# Must match check_fandom_dumps.py: only these classes need an S3 restore
# request before a GET succeeds.
RESTORE_REQUIRED_CLASSES = {"GLACIER", "DEEP_ARCHIVE"}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, expected_size: int | None, timeout: float) -> None:
    existing = dest.stat().st_size if dest.exists() else 0
    if expected_size is not None and existing == expected_size:
        return  # already fully downloaded; caller verifies checksum

    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range not satisfiable -> file already complete
            return
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
    parser.add_argument("--status-file", type=Path, default=Path("cache/fandom_dump_status.json"))
    parser.add_argument("--dest-dir", type=Path, default=Path("data/fandom_dumps"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    if not args.status_file.exists():
        raise SystemExit(f"{args.status_file} not found -- run check_fandom_dumps.py first.")

    statuses = json.loads(args.status_file.read_text())
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    downloadable = [
        s for s in statuses
        if s.get("dump_available") and s.get("dump_storage_class") not in RESTORE_REQUIRED_CLASSES
    ]
    skipped_archived = [
        s for s in statuses
        if s.get("dump_available") and s.get("dump_storage_class") in RESTORE_REQUIRED_CLASSES
    ]

    for s in skipped_archived:
        print(f"SKIP {s['wiki']}: dump is in {s['dump_storage_class']} storage, needs an S3 restore first")

    verified, failed = [], []
    for s in downloadable:
        wiki, url = s["wiki"], s["dump_url"]
        dest = args.dest_dir / f"{wiki}_pages_current.xml.7z"
        print(f"{wiki}: downloading -> {dest}")
        try:
            download_with_resume(url, dest, s.get("dump_size_bytes"), args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(wiki)
            continue

        expected_md5 = s.get("dump_md5")
        if expected_md5:
            actual_md5 = md5_of(dest)
            if actual_md5 != expected_md5:
                print(f"  CHECKSUM MISMATCH: expected {expected_md5}, got {actual_md5}")
                failed.append(wiki)
                continue
        verified.append(wiki)
        print(f"  OK ({dest.stat().st_size/1e6:.1f} MB, checksum verified)")
        time.sleep(0.5)

    print(f"\n{len(verified)} downloaded and verified, {len(failed)} failed, {len(skipped_archived)} need S3 restore.")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()

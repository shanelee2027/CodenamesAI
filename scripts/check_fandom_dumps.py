"""Check M0 corpus-collection status for a list of Fandom wikis.

For each wiki subdomain, queries the wiki's MediaWiki API for its page count
and internal dbname (the API is not behind Fandom's Cloudflare bot check,
unlike the wiki pages themselves), then HEAD-checks whether a "current
pages" XML dump already exists on Fandom's public S3 bucket at the
deterministic path derived from that dbname.

This only *checks* status -- it never requests a new dump. Requesting a
dump requires an autoconfirmed-or-higher Fandom account on that specific
wiki (see SCOPE.md §M0), which isn't something a script can do on your
behalf. For any wiki reported as "no dump", request one by logging into
Fandom and visiting:

    https://<wiki>.fandom.com/wiki/Special:Statistics

Usage:
    python scripts/check_fandom_dumps.py
    python scripts/check_fandom_dumps.py --wikis-file scripts/fandom_wikis.txt
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

USER_AGENT = "CodenamesAI-corpus-check/0.1 (senior CS project, research use)"
DUMP_BUCKET = "https://s3.amazonaws.com/wikia_xml_dumps"
REQUEST_DELAY_SECONDS = 1.0
# GLACIER / DEEP_ARCHIVE objects 404 on a plain GET until an S3 restore
# request completes (hours to days). Every other class -- including
# STANDARD_IA -- serves a normal GET immediately, just at a different
# storage cost tier.
RESTORE_REQUIRED_CLASSES = {"GLACIER", "DEEP_ARCHIVE"}


@dataclass
class WikiStatus:
    wiki: str
    reachable: bool
    dbname: str | None = None
    pages: int | None = None
    dump_available: bool = False
    dump_url: str | None = None
    dump_size_bytes: int | None = None
    dump_last_modified: str | None = None
    dump_storage_class: str | None = None
    dump_md5: str | None = None
    error: str | None = None


def read_wiki_list(path: Path) -> list[str]:
    wikis = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            wikis.append(line)
    return wikis


def fetch_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def head(url: str, timeout: float) -> urllib.request.addinfourl | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


def dump_url_for(dbname: str) -> str:
    key = dbname.lower()
    return f"{DUMP_BUCKET}/{key[0]}/{key[:2]}/{key}_pages_current.xml.7z"


def check_wiki(wiki: str, timeout: float) -> WikiStatus:
    api_url = (
        f"https://{wiki}.fandom.com/api.php"
        "?action=query&meta=siteinfo&siprop=general%7Cstatistics&format=json"
    )
    try:
        data = fetch_json(api_url, timeout)
    except Exception as e:  # noqa: BLE001 - report any failure, don't crash the batch
        return WikiStatus(wiki=wiki, reachable=False, error=f"{type(e).__name__}: {e}")

    general = data.get("query", {}).get("general", {})
    stats = data.get("query", {}).get("statistics", {})
    dbname = general.get("wikiid")
    pages = stats.get("pages")
    if not dbname:
        return WikiStatus(wiki=wiki, reachable=True, error="no wikiid in siteinfo response")

    url = dump_url_for(dbname)
    try:
        resp = head(url, timeout)
    except Exception as e:  # noqa: BLE001
        return WikiStatus(
            wiki=wiki, reachable=True, dbname=dbname, pages=pages,
            error=f"dump HEAD failed: {type(e).__name__}: {e}",
        )

    if resp is None:
        return WikiStatus(wiki=wiki, reachable=True, dbname=dbname, pages=pages, dump_available=False)

    headers = resp.headers
    return WikiStatus(
        wiki=wiki,
        reachable=True,
        dbname=dbname,
        pages=pages,
        dump_available=True,
        dump_url=url,
        dump_size_bytes=int(headers.get("Content-Length", 0)) or None,
        dump_last_modified=headers.get("Last-Modified"),
        dump_storage_class=headers.get("x-amz-storage-class", "STANDARD"),
        dump_md5=headers.get("x-amz-meta-md5"),
    )


def format_row(s: WikiStatus) -> str:
    if not s.reachable:
        return f"{s.wiki:20s}  UNREACHABLE  ({s.error})"
    if s.error:
        return f"{s.wiki:20s}  pages={s.pages!s:>8}  ERROR: {s.error}"
    if not s.dump_available:
        return f"{s.wiki:20s}  pages={s.pages!s:>8}  dump=NO   -- request via Special:Statistics"
    size_mb = (s.dump_size_bytes or 0) / 1e6
    needs_restore = s.dump_storage_class in RESTORE_REQUIRED_CLASSES
    note = f"  [{s.dump_storage_class}, needs restore before download]" if needs_restore else ""
    return (
        f"{s.wiki:20s}  pages={s.pages!s:>8}  dump=YES  "
        f"{size_mb:8.1f} MB  modified={s.dump_last_modified}{note}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikis-file", type=Path, default=Path("scripts/fandom_wikis.txt"))
    parser.add_argument("--output", type=Path, default=Path("cache/fandom_dump_status.json"))
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    wikis = read_wiki_list(args.wikis_file)
    print(f"Checking {len(wikis)} wikis from {args.wikis_file}...\n")

    results: list[WikiStatus] = []
    for i, wiki in enumerate(wikis):
        status = check_wiki(wiki, args.timeout)
        results.append(status)
        print(format_row(status))
        if i < len(wikis) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    ready = [r for r in results if r.dump_available and r.dump_storage_class not in RESTORE_REQUIRED_CLASSES]
    archived = [r for r in results if r.dump_available and r.dump_storage_class in RESTORE_REQUIRED_CLASSES]
    missing = [r for r in results if r.reachable and not r.dump_available and not r.error]
    unreachable = [r for r in results if not r.reachable or r.error]

    total_ready_gb = sum(r.dump_size_bytes or 0 for r in ready) / 1e9
    print(
        f"\n{len(ready)} ready to download now (~{total_ready_gb:.2f} GB compressed), "
        f"{len(archived)} need an S3 restore first, "
        f"{len(missing)} have no dump (need a manual request), "
        f"{len(unreachable)} unreachable/errored."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nFull status written to {args.output}")


if __name__ == "__main__":
    main()

import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# passages from new english wikipedia articles that ran in "did you know" in 2026. dyk articles are
# at most 7 days old when they run, have 1,500+ characters of prose, and get reviewed by a person
API = "https://en.wikipedia.org/w/api.php"
HEAD = {"User-Agent": "liltemp research bot (https://navthings.github.io; navneet.dagdiya@gmail.com)"}
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs2")
NEED = 1000
PROMPT_WORDS, CONT_WORDS = 40, 80
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September"]
CREATED_AFTER = "2025-11-01"


def get(params: dict) -> dict:
    for attempt in range(8):
        r = requests.get(API, params={**params, "format": "json", "formatversion": 2}, headers=HEAD, timeout=30)
        if r.status_code == 200:
            time.sleep(0.7)
            return r.json()
        wait = int(r.headers.get("Retry-After", 0)) or 5 * 2 ** attempt
        print(f"  {r.status_code}, waiting {wait}s", flush=True)
        time.sleep(wait)
    r.raise_for_status()


def dyk_titles() -> list[str]:
    titles = []
    for m in MONTHS:
        j = get({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                 "titles": f"Wikipedia:Did you know archive/2026/{m}"})
        page = j["query"]["pages"][0]
        if "revisions" not in page:
            print(f"{m}: no archive page", flush=True)
            continue
        text = page["revisions"][0]["slots"]["main"]["content"]
        # the new article in each hook is the bold link
        got = re.findall(r"'''+\[\[([^\]|#]+)", text)
        titles += [t.strip() for t in got]
        print(f"{m}: {len(got)} dyk articles", flush=True)
    return list(dict.fromkeys(titles))


def prose(text: str) -> str | None:
    # body text with headings and short lines (captions, list items) dropped, read straight through
    keep = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("=") or len(line.split()) < 12 or line.count("|") > 1:
            continue
        keep.append(line)
    words = " ".join(keep).split()
    return " ".join(words) if len(words) >= PROMPT_WORDS + CONT_WORDS else None


def main():
    OUT.mkdir(exist_ok=True)
    cache = OUT / "titles.json"
    titles = json.loads(cache.read_text()) if cache.exists() and json.loads(cache.read_text()) else None
    if not titles:
        titles = dyk_titles()
        random.Random(0).shuffle(titles)
        cache.write_text(json.dumps(titles))
    print(f"{len(titles)} titles", flush=True)

    def fetch(t: str):
        # full-page extracts only come back one page per request, so 4 requests run at once
        j = get({"action": "query", "prop": "extracts|revisions", "explaintext": 1, "exsectionformat": "wiki",
                 "rvprop": "timestamp", "rvdir": "newer", "rvlimit": 1, "titles": t, "redirects": 1})
        page = j["query"]["pages"][0]
        # some dyk hooks are expansions of old articles, keep only pages created after every model's cutoff
        created = page.get("revisions", [{}])[0].get("timestamp", "0")
        para = prose(page.get("extract", "")) if not page.get("missing") and created >= CREATED_AFTER else None
        if not para:
            return None
        w = para.split()
        return [" ".join(w[:PROMPT_WORDS]), " " + " ".join(w[PROMPT_WORDS:PROMPT_WORDS + CONT_WORDS]), page["title"]]

    rows = []
    with ThreadPoolExecutor(4) as pool:
        for start in range(0, len(titles), 100):
            if len(rows) >= NEED:
                break
            rows += [r for r in pool.map(fetch, titles[start:start + 100]) if r]
            (OUT / "prompts.json").write_text(json.dumps(rows[:NEED]))
            print(f"  {len(rows)} passages from {start + 100} titles", flush=True)

    (OUT / "prompts.json").write_text(json.dumps(rows[:NEED]))
    print(f"saved {len(rows[:NEED])} passages to {OUT / 'prompts.json'}")


if __name__ == "__main__":
    main()

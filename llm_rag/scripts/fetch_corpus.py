"""Fetch a curated slice of the pytest documentation as the RAG corpus.

Pulls raw reStructuredText files from the pytest GitHub repo. Stored
as `.rst` under `data/corpus/`. The chunker (see `build_index.py`)
strips obvious rst directives at index time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE = "https://raw.githubusercontent.com/pytest-dev/pytest/main/doc/en"

PAGES = [
    "getting-started",
    "how-to/fixtures",
    "how-to/parametrize",
    "how-to/mark",
    "how-to/monkeypatch",
    "how-to/tmp_path",
    "how-to/capture-stdout-stderr",
    "how-to/skipping",
    "how-to/assert",
    "how-to/doctest",
    "how-to/usage",
    "how-to/cache",
]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


def fetch(client: httpx.Client, page: str) -> str:
    url = f"{BASE}/{page}.rst"
    r = client.get(url, timeout=30.0)
    r.raise_for_status()
    return r.text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        for page in PAGES:
            text = fetch(client, page)
            name = page.replace("/", "__") + ".rst"
            (OUT_DIR / name).write_text(text)
            logger.info("fetched %s (%d chars)", name, len(text))
    logger.info("done, %d pages in %s", len(PAGES), OUT_DIR)


if __name__ == "__main__":
    main()

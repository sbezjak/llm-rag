"""Chunk the corpus and persist to a Chroma collection.

Chunking strategy: section-aware. Split each rst file on its top-level
and sub-level headers (underlined `===` and `---` lines). Each section
becomes one chunk. Sections larger than `MAX_CHARS` are split further
on paragraph boundaries. Section headers are kept as part of the chunk
text so semantic embeddings have the topic signal.

Chunk ids are `<file-stem>::<section-slug>::<part>` so retrieval tests
can reference them precisely. The chunk's source file and section
title are stored as metadata for later citation correctness checks.

Trade-off: this is a deliberately simple chunker. Production RAG often
uses recursive character splitters with semantic boundaries and
overlap. We're keeping it readable so the failures we observe (chunk
too big, chunk straddles topics, header lost) are diagnosable by eye.
A heavier chunker can be swapped in later if a finding warrants it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = ROOT / ".chroma"
COLLECTION = "pytest_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

MAX_CHARS = 1200
MIN_CHARS = 80

# Directive/comment lines: `.. note:: ...` (double colon) and single-colon
# meta lines like `.. regendoc: wipe`. The `::?` matches one or two colons.
RST_DIRECTIVE = re.compile(r"^\.\. [a-zA-Z_-]+::?.*$", re.MULTILINE)
# Explicit hyperlink targets: `.. _name:`, phrase targets `.. _`a b c`:`, and
# external targets with a trailing URL `.. _name: https://...`. Non-greedy up
# to the first colon so an https:// in the trailing URL is not the delimiter.
RST_TARGET = re.compile(r"^\.\. _.+?:.*$", re.MULTILINE)
# Inline interpreted-text roles: :role:`text`, :domain:role:`text`, optionally
# with an explicit `title <target>`. We keep the human-readable title and drop
# the role tag and any <target>, so `:py:meth:`x <Y.x>`` becomes `x`.
RST_ROLE = re.compile(r":(?:[\w.+-]+:)+`([^`]+)`")
# Inline literal markup: ``code`` -> code.
RST_INLINE_LITERAL = re.compile(r"``([^`]+)``")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str


def slugify(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower()).strip("-")
    return s or "untitled"


def _role_sub(m: re.Match) -> str:
    inner = m.group(1)
    # `title <target>` keeps just the title; a bare `text` is kept as-is.
    lt = inner.find("<")
    if lt != -1 and inner.rstrip().endswith(">"):
        return inner[:lt].strip()
    return inner.strip()


def clean_rst(text: str) -> str:
    text = RST_TARGET.sub("", text)
    text = RST_DIRECTIVE.sub("", text)
    # Strip inline roles and literal markup so rst syntax does not leak into
    # chunk text (and from there into generated answers).
    text = RST_ROLE.sub(_role_sub, text)
    text = RST_INLINE_LITERAL.sub(r"\1", text)
    return text


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_title, section_body) tuples.

    Detects rst headers: a line of text followed by a line of repeated
    `=`, `-`, `~`, `^`, or `"` of equal-or-greater length.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("(intro)", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line.strip()
            and nxt
            and len(set(nxt.strip())) == 1
            and nxt.strip()[0] in '=-~^"'
            and len(nxt.strip()) >= len(line.strip())
        ):
            title = line.strip()
            sections.append((title, []))
            i += 2
            continue
        sections[-1][1].append(line)
        i += 1
    return [(t, "\n".join(body).strip()) for t, body in sections if "\n".join(body).strip()]


def split_oversized(title: str, body: str) -> list[str]:
    if len(body) <= MAX_CHARS:
        return [body]
    paras = re.split(r"\n\s*\n", body)
    parts: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 > MAX_CHARS and buf:
            parts.append(buf.strip())
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        parts.append(buf.strip())
    return parts


def chunk_file(path: Path) -> list[Chunk]:
    raw = clean_rst(path.read_text())
    stem = path.stem
    chunks: list[Chunk] = []
    for title, body in split_sections(raw):
        for idx, part in enumerate(split_oversized(title, body)):
            if len(part) < MIN_CHARS:
                continue
            slug = slugify(title)
            chunk_id = f"{stem}::{slug}::{idx}"
            text = f"# {title}\n\n{part}"
            chunks.append(Chunk(id=chunk_id, text=text, source=path.name, section=title))
    return chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.glob("*.rst")):
        raise SystemExit(
            f"no corpus at {CORPUS_DIR}, run `uv run python -m llm_rag.scripts.fetch_corpus`"
        )

    all_chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.rst")):
        cs = chunk_file(path)
        logger.info("%s: %d chunks", path.name, len(cs))
        all_chunks.extend(cs)
    logger.info("total chunks: %d", len(all_chunks))

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    coll = client.create_collection(name=COLLECTION, embedding_function=ef)

    coll.add(
        ids=[c.id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[{"source": c.source, "section": c.section} for c in all_chunks],
    )
    logger.info(
        "indexed %d chunks into collection %r at %s", len(all_chunks), COLLECTION, CHROMA_DIR
    )


if __name__ == "__main__":
    main()

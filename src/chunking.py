"""Markdown reading and sentence-aware text chunking."""

import glob
import os
from pathlib import Path


def read_markdown_files(directory: str) -> list[dict]:
    """Read all .md files from a directory, return list of {title, content, source}."""
    docs = []
    md_patterns = [os.path.join(directory, "*.md"), os.path.join(directory, "**/*.md")]

    seen = set()
    for pattern in md_patterns:
        for filepath in glob.glob(pattern, recursive=True):
            filepath = os.path.abspath(filepath)
            if filepath in seen:
                continue
            seen.add(filepath)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"  ⚠ Skipping {filepath}: {e}")
                continue

            if not content.strip():
                continue

            title = Path(filepath).stem
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break
                if line.startswith("Title:"):
                    title = line.split(":", 1)[1].strip()
                    break

            docs.append({
                "title": title,
                "content": content,
                "source": filepath,
            })

    return docs


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "。", "! ", "? "]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        new_start = end - overlap
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks

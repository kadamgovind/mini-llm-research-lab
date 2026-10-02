from pathlib import Path


def load_corpus(path: str | Path) -> str:
    """Load a UTF-8 text corpus from disk."""
    corpus_path = Path(path)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    return corpus_path.read_text(encoding="utf-8")
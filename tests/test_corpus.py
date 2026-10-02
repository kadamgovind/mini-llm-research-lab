from pathlib import Path

from mini_llm.data.corpus import load_corpus
from mini_llm.data.preprocessing import prepare_corpus
from mini_llm.tokenization.tokenizer import BasicTokenizer


def test_load_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("I love AI.", encoding="utf-8")

    text = load_corpus(corpus_path)

    assert text == "I love AI."


def test_prepare_corpus() -> None:
    tokenizer = BasicTokenizer()

    sequences = prepare_corpus(
        "I love AI. I study ML.",
        tokenizer,
    )

    assert sequences == [
        ["<BOS>", "I", "love", "AI", ".", "<EOS>"],
        ["<BOS>", "I", "study", "ML", ".", "<EOS>"],
    ]
from mini_llm.tokenization.tokenizer import BasicTokenizer


def test_tokenizer() -> None:
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize("I love AI.")

    assert tokens == ["I", "love", "AI", "."]
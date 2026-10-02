from mini_llm.language_modeling.bigram import BigramLanguageModel


def test_bigram_probability() -> None:
    tokens = ["I", "love", "AI", "I", "love", "ML"]

    model = BigramLanguageModel()
    model.fit(tokens)

    assert model.probability("I", "love") == 1.0
    assert model.probability("love", "AI") == 0.5
    assert model.probability("love", "ML") == 0.5


def test_unknown_context_probability() -> None:
    model = BigramLanguageModel()

    assert model.probability("unknown", "token") == 0.0
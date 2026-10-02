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

def test_fit_multiple_sequences() -> None:
    model = BigramLanguageModel()

    sequences = [
        ["<BOS>", "I", "love", "AI", "<EOS>"],
        ["<BOS>", "I", "study", "AI", "<EOS>"],
    ]

    model.fit_sequences(sequences)

    assert model.probability("<BOS>", "I") == 1.0
    assert model.probability("I", "love") == 0.5
    assert model.probability("I", "study") == 0.5


def test_next_token_probabilities() -> None:
    model = BigramLanguageModel()

    model.fit(["I", "love", "AI", "I", "love", "ML"])

    probabilities = model.next_token_probabilities("love")

    assert probabilities == {
        "AI": 0.5,
        "ML": 0.5,
    }
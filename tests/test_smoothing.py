from collections import Counter

import pytest

from mini_llm.language_modeling.bigram import BigramLanguageModel
from mini_llm.language_modeling.smoothing import AdditiveSmoothing


def build_model() -> BigramLanguageModel:
    """Build a small deterministic bigram model for testing."""
    model = BigramLanguageModel()

    model.fit(
        [
            "the",
            "dog",
            "the",
            "cat",
            "the",
            "dog",
        ]
    )

    return model


def test_unseen_bigram_gets_positive_probability() -> None:
    """Smoothing must eliminate zero probability for unseen bigrams."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    smoothing = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=1.0,
    )

    probability = smoothing.probability("the", "sat")

    assert probability > 0.0


def test_seen_bigram_probability_is_smoothed() -> None:
    """A seen bigram should receive a smoothed probability."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    smoothing = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=1.0,
    )

    probability = smoothing.probability("the", "dog")

    expected = (2 + 1) / (3 + 4)

    assert probability == pytest.approx(expected)


def test_smoothed_distribution_sums_to_one() -> None:
    """The complete smoothed conditional distribution must normalize."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    smoothing = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=1.0,
    )

    probabilities = smoothing.next_token_probabilities("the")

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_all_vocabulary_tokens_receive_probability() -> None:
    """Every vocabulary token must receive a probability."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    smoothing = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=1.0,
    )

    probabilities = smoothing.next_token_probabilities("the")

    assert set(probabilities) == set(vocabulary)

    assert all(
        probability > 0
        for probability in probabilities.values()
    )


def test_unknown_context_gets_uniform_distribution() -> None:
    """An unseen context should produce a uniform smoothed distribution."""
    counts: dict[str, Counter[str]] = {}

    vocabulary = ["the", "dog", "cat", "sat"]

    smoothing = AdditiveSmoothing(
        counts,
        vocabulary,
        alpha=1.0,
    )

    probabilities = smoothing.next_token_probabilities("unknown")

    expected = 1 / len(vocabulary)

    assert all(
        probability == pytest.approx(expected)
        for probability in probabilities.values()
    )


def test_alpha_changes_probability() -> None:
    """Increasing alpha should change the probability estimate."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    smoothing_small = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=0.1,
    )

    smoothing_large = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=2.0,
    )

    probability_small = smoothing_small.probability(
        "the",
        "sat",
    )

    probability_large = smoothing_large.probability(
        "the",
        "sat",
    )

    assert probability_small != probability_large


def test_zero_alpha_is_rejected() -> None:
    """Alpha must be strictly positive."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    with pytest.raises(ValueError, match="alpha"):
        AdditiveSmoothing(
            model.bigram_counts,
            vocabulary,
            alpha=0.0,
        )


def test_negative_alpha_is_rejected() -> None:
    """Negative smoothing strength must be rejected."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    with pytest.raises(ValueError, match="alpha"):
        AdditiveSmoothing(
            model.bigram_counts,
            vocabulary,
            alpha=-1.0,
        )


def test_empty_vocabulary_is_rejected() -> None:
    """Smoothing requires at least one possible next token."""
    model = build_model()

    with pytest.raises(ValueError, match="vocabulary"):
        AdditiveSmoothing(
            model.bigram_counts,
            [],
            alpha=1.0,
        )


def test_formula_is_correct_for_unseen_bigram() -> None:
    """Verify the additive smoothing formula directly."""
    model = build_model()

    vocabulary = ["the", "dog", "cat", "sat"]

    smoothing = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=1.0,
    )

    probability = smoothing.probability("the", "sat")

    assert probability == pytest.approx(1 / 7)
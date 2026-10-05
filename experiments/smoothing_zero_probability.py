from mini_llm.language_modeling.bigram import BigramLanguageModel
from mini_llm.language_modeling.smoothing import AdditiveSmoothing


def main() -> None:
    """Demonstrate how additive smoothing solves zero probability."""
    tokens = [
        "the",
        "dog",
        "the",
        "cat",
        "the",
        "dog",
    ]

    vocabulary = [
        "the",
        "dog",
        "cat",
        "sat",
    ]

    model = BigramLanguageModel()
    model.fit(tokens)

    smoothing = AdditiveSmoothing(
        model.bigram_counts,
        vocabulary,
        alpha=1.0,
    )

    mle_probability = model.probability("the", "sat")

    smoothed_probability = smoothing.probability(
        "the",
        "sat",
    )

    probabilities = smoothing.next_token_probabilities("the")

    print("Mini-LLM: Zero-Probability Experiment")
    print("======================================")

    print("\nTraining sequence:")
    print(tokens)

    print("\nVocabulary:")
    print(vocabulary)

    print("\nTarget transition:")
    print("P('sat' | 'the')")

    print("\nMaximum-Likelihood Estimate (MLE)")
    print("--------------------------------")
    print(
        f"P('sat' | 'the') = "
        f"{mle_probability:.6f}"
    )

    print("\nAdd-one / Laplace Smoothing")
    print("---------------------------")
    print(
        f"P('sat' | 'the') = "
        f"{smoothed_probability:.6f}"
    )

    print("\nComplete smoothed distribution")
    print("------------------------------")

    for token, probability in probabilities.items():
        print(
            f"P('{token}' | 'the') = "
            f"{probability:.6f}"
        )

    total_probability = sum(probabilities.values())

    print("\nNormalization check")
    print("-------------------")
    print(
        "Sum of smoothed probabilities = "
        f"{total_probability:.6f}"
    )

    print("\nConclusion")
    print("----------")

    if mle_probability == 0.0 and smoothed_probability > 0.0:
        print(
            "Smoothing successfully removes the "
            "zero-probability problem."
        )
    else:
        print(
            "Unexpected result: "
            "zero-probability condition was not demonstrated."
        )


if __name__ == "__main__":
    main()
from collections import Counter
from collections.abc import Mapping


class AdditiveSmoothing:
    """Additive smoothing for a count-based language model.

    The smoothed conditional probability is:

        P(next | current) =
            (count(current, next) + alpha)
            / (count(current) + alpha * vocabulary_size)

    When alpha=1, this is Laplace (add-one) smoothing.
    """

    def __init__(
        self,
        bigram_counts: Mapping[str, Counter[str]],
        vocabulary: list[str] | set[str],
        alpha: float = 1.0,
    ) -> None:
        """Initialize an additive smoothing model.

        Args:
            bigram_counts: Mapping from current tokens to next-token counts.
            vocabulary: Complete vocabulary over possible next tokens.
            alpha: Positive smoothing strength.

        Raises:
            ValueError: If alpha is not positive.
            ValueError: If vocabulary is empty.
        """
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0.")

        if not vocabulary:
            raise ValueError("vocabulary must not be empty.")

        self.bigram_counts = bigram_counts
        self.vocabulary = tuple(vocabulary)
        self.vocabulary_size = len(self.vocabulary)
        self.alpha = alpha

    def probability(
        self,
        current_token: str,
        next_token: str,
    ) -> float:
        """Return the smoothed P(next_token | current_token)."""
        counts = self.bigram_counts.get(current_token, Counter())
        context_count = sum(counts.values())
        bigram_count = counts.get(next_token, 0)

        numerator = bigram_count + self.alpha
        denominator = context_count + (
            self.alpha * self.vocabulary_size
        )

        return numerator / denominator

    def next_token_probabilities(
        self,
        current_token: str,
    ) -> dict[str, float]:
        """Return the complete smoothed next-token distribution."""
        return {
            token: self.probability(current_token, token)
            for token in self.vocabulary
        }
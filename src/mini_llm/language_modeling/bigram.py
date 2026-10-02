from collections import Counter, defaultdict


class BigramLanguageModel:
    """A count-based bigram language model."""

    def __init__(self) -> None:
        self.bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, tokens: list[str]) -> None:
        """Learn bigram counts from one token sequence."""
        for current_token, next_token in zip(tokens, tokens[1:]):
            self.bigram_counts[current_token][next_token] += 1

    def fit_sequences(self, sequences: list[list[str]]) -> None:
        """Learn bigram counts from multiple token sequences."""
        for sequence in sequences:
            self.fit(sequence)

    def probability(
        self,
        current_token: str,
        next_token: str,
    ) -> float:
        """Return P(next_token | current_token)."""
        counts = self.bigram_counts[current_token]
        total = sum(counts.values())

        if total == 0:
            return 0.0

        return counts[next_token] / total

    def next_token_probabilities(
        self,
        current_token: str,
    ) -> dict[str, float]:
        """Return next-token probability distribution."""
        counts = self.bigram_counts[current_token]
        total = sum(counts.values())

        if total == 0:
            return {}

        return {
            token: count / total
            for token, count in counts.items()
        }
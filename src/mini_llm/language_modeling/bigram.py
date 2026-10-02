from collections import Counter, defaultdict


class BigramLanguageModel:
    """A simple count-based bigram language model."""

    def __init__(self) -> None:
        self.bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, tokens: list[str]) -> None:
        """Learn bigram counts from a token sequence."""
        for current_token, next_token in zip(tokens, tokens[1:]):
            self.bigram_counts[current_token][next_token] += 1

    def probability(self, current_token: str, next_token: str) -> float:
        """Return P(next_token | current_token)."""
        counts = self.bigram_counts[current_token]
        total = sum(counts.values())

        if total == 0:
            return 0.0

        return counts[next_token] / total
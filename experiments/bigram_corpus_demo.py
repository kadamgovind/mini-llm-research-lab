from pathlib import Path

from mini_llm.data.corpus import load_corpus
from mini_llm.data.preprocessing import prepare_corpus
from mini_llm.language_modeling.bigram import BigramLanguageModel
from mini_llm.tokenization.tokenizer import BasicTokenizer
from mini_llm.tokenization.vocabulary import Vocabulary


def main() -> None:
    corpus_path = Path("data/raw/corpus.txt")

    text = load_corpus(corpus_path)

    tokenizer = BasicTokenizer()
    sequences = prepare_corpus(text, tokenizer)

    tokens = [
        token
        for sequence in sequences
        for token in sequence
    ]

    vocabulary = Vocabulary()
    vocabulary.build(tokens)

    model = BigramLanguageModel()
    model.fit_sequences(sequences)

    print("Mini-LLM Bigram Corpus Experiment")
    print("--------------------------------")
    print(f"Sentences: {len(sequences)}")
    print(f"Vocabulary size: {len(vocabulary)}")

    print("\nNext-token probabilities after 'I':")

    probabilities = model.next_token_probabilities("I")

    for token, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{token:15} {probability:.4f}")


if __name__ == "__main__":
    main()
from mini_llm.data.preprocessing import prepare_corpus
from mini_llm.language_modeling.bigram import BigramLanguageModel
from mini_llm.tokenization.tokenizer import BasicTokenizer
from mini_llm.tokenization.vocabulary import Vocabulary


def test_language_model_pipeline() -> None:
    text = (
        "I love artificial intelligence. "
        "I love machine learning. "
        "I study artificial intelligence."
    )

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

    assert len(vocabulary) > 3

    assert model.probability("<BOS>", "I") == 1.0

    assert model.probability("I", "love") == 2 / 3
    assert model.probability("I", "study") == 1 / 3
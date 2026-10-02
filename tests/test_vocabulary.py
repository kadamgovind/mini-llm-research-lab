from mini_llm.tokenization.vocabulary import Vocabulary


def test_vocabulary_encode_decode() -> None:
    vocabulary = Vocabulary()

    vocabulary.build(["I", "love", "AI"])

    encoded = vocabulary.encode(["I", "love", "AI"])

    assert encoded == [3, 4, 5]

    decoded = vocabulary.decode(encoded)

    assert decoded == ["I", "love", "AI"]


def test_unknown_token() -> None:
    vocabulary = Vocabulary()

    vocabulary.build(["I", "love", "AI"])

    encoded = vocabulary.encode(["quantum"])

    assert encoded == [0]


def test_special_tokens() -> None:
    vocabulary = Vocabulary()

    vocabulary.build(["I", "love", "AI"])

    assert vocabulary.token_to_id["<UNK>"] == 0
    assert vocabulary.token_to_id["<BOS>"] == 1
    assert vocabulary.token_to_id["<EOS>"] == 2
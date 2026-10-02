import re

from mini_llm.tokenization.tokenizer import BasicTokenizer
from mini_llm.tokenization.vocabulary import Vocabulary


def split_sentences(text: str) -> list[str]:
    """Split basic text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def prepare_sentence(
    sentence: str,
    tokenizer: BasicTokenizer,
) -> list[str]:
    """Tokenize a sentence and add boundary tokens."""
    tokens = tokenizer.tokenize(sentence)

    return [
        Vocabulary.BOS_TOKEN,
        *tokens,
        Vocabulary.EOS_TOKEN,
    ]


def prepare_corpus(
    text: str,
    tokenizer: BasicTokenizer,
) -> list[list[str]]:
    """Convert raw corpus text into tokenized sentences."""
    sentences = split_sentences(text)

    return [
        prepare_sentence(sentence, tokenizer)
        for sentence in sentences
    ]
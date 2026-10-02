import re


class BasicTokenizer:
    """A simple whitespace and punctuation tokenizer."""

    def tokenize(self, text: str) -> list[str]:
        """Convert text into a list of tokens."""
        return re.findall(r"\w+|[^\w\s]", text)
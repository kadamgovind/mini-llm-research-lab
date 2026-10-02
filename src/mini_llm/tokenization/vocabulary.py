class Vocabulary:
    """Maps tokens to integer IDs and IDs back to tokens."""

    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def build(self, tokens: list[str]) -> None:
        """Build vocabulary from tokens."""
        special_tokens = [
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        unique_tokens = list(dict.fromkeys(tokens))

        all_tokens = special_tokens + [
            token for token in unique_tokens if token not in special_tokens
        ]

        self.token_to_id = {
            token: index for index, token in enumerate(all_tokens)
        }

        self.id_to_token = {
            index: token for token, index in self.token_to_id.items()
        }

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens into integer IDs."""
        unk_id = self.token_to_id[self.UNK_TOKEN]

        return [
            self.token_to_id.get(token, unk_id)
            for token in tokens
        ]

    def decode(self, token_ids: list[int]) -> list[str]:
        """Convert integer IDs back into tokens."""
        return [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

    def __len__(self) -> int:
        return len(self.token_to_id)
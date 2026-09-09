"""Deterministic tokenizers implemented with only the Python standard library."""
import re


class BasicTokenizer:
    """Tokenizer that preserves words, numbers, punctuation, and special tokens."""
    def __init__(self, special_tokens=None):
        self.special_tokens = list(special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        self.vocabulary = {token: i for i, token in enumerate(self.special_tokens)}

    def tokenize(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return re.findall(r"[A-Za-z_]+(?:'[A-Za-z_]+)?|\d+(?:\.\d+)?|[^\w\s]", text)

    def fit(self, texts, min_frequency=1):
        counts = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        for token in sorted(counts):
            if counts[token] >= min_frequency and token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = self.tokenize(text)
        ids = []
        if add_bos:
            ids.append(self.vocabulary["<BOS>"])
        unknown = self.vocabulary["<UNK>"]
        ids.extend(self.vocabulary.get(token, unknown) for token in tokens)
        if add_eos:
            ids.append(self.vocabulary["<EOS>"])
        return ids

    def decode(self, ids):
        inverse = {index: token for token, index in self.vocabulary.items()}
        tokens = [inverse.get(int(index), "<UNK>") for index in ids]
        text = ""
        for token in tokens:
            if not text:
                text = token
            elif re.match(r"^[A-Za-z_0-9]", token) and re.search(r"[A-Za-z_0-9]$", text):
                text += " " + token
            else:
                text += token
        return text

    def __len__(self):
        return len(self.vocabulary)


class ByteTokenizer:
    """UTF-8 byte-level tokenizer with stable special-token IDs.

    Every Unicode string is encoded to UTF-8 and represented by byte tokens.
    This gives deterministic coverage for Persian, English, emoji, and arbitrary
    Unicode text without requiring a learned vocabulary or external package.
    """
    def __init__(self, special_tokens=None):
        self.special_tokens = list(special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")
        self.token_to_id = {token: i for i, token in enumerate(self.special_tokens)}
        self.byte_offset = len(self.special_tokens)
        self.vocabulary_size = self.byte_offset + 256

    def tokenize(self, text):
        """Return raw UTF-8 byte values for the supplied text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def fit(self, texts, min_frequency=1):
        """Keep a compatible fit API; byte vocabulary is fixed and needs no fitting."""
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("all texts must be strings")
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = []
        if add_bos:
            ids.append(self.token_to_id["<BOS>"])
        ids.extend(self.byte_offset + byte for byte in text.encode("utf-8"))
        if add_eos:
            ids.append(self.token_to_id["<EOS>"])
        return ids

    def decode(self, ids):
        byte_values = bytearray()
        text_parts = []
        for raw_id in ids:
            token_id = int(raw_id)
            if self.byte_offset <= token_id < self.vocabulary_size:
                byte_values.append(token_id - self.byte_offset)
            else:
                if byte_values:
                    text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
                    byte_values.clear()
                special = next((token for token, index in self.token_to_id.items() if index == token_id), "<UNK>")
                text_parts.append(special)
        if byte_values:
            text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
        return "".join(text_parts)

    def to_dict(self):
        """Serialize tokenizer configuration without serializing learned state."""
        return {
            "type": "byte",
            "special_tokens": list(self.special_tokens),
            "byte_offset": self.byte_offset,
            "vocabulary_size": self.vocabulary_size,
        }

    @classmethod
    def from_dict(cls, payload):
        """Restore a byte tokenizer from a serialized configuration."""
        if not isinstance(payload, dict):
            raise TypeError("tokenizer configuration must be a dictionary")
        if payload.get("type") != "byte":
            raise ValueError("unsupported tokenizer type")
        tokenizer = cls(payload.get("special_tokens"))
        if int(payload.get("byte_offset", tokenizer.byte_offset)) != tokenizer.byte_offset:
            raise ValueError("tokenizer byte offset does not match special tokens")
        if int(payload.get("vocabulary_size", tokenizer.vocabulary_size)) != tokenizer.vocabulary_size:
            raise ValueError("tokenizer vocabulary size does not match configuration")
        return tokenizer

    def __len__(self):
        return self.vocabulary_size

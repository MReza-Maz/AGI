"""Simple deterministic word/subword-ready tokenizer using only the standard library."""
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

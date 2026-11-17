import re
import json
from collections import Counter
from pathlib import Path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2


def simple_tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)


class WordTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.itos = {i: w for w, i in vocab.items()}
        self.vocab_size = len(vocab)

    @classmethod
    def build(cls, texts: list[str], max_vocab: int = 10000) -> "WordTokenizer":
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))

        # Reserve special tokens first
        vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, CLS_TOKEN: CLS_ID}
        for word, _ in counter.most_common(max_vocab - len(vocab)):
            if word not in vocab:
                vocab[word] = len(vocab)

        return cls(vocab)

    def encode(self, text: str, max_len: int = 256, add_cls: bool = True) -> list[int]:
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(t, UNK_ID) for t in tokens]

        # Reserve one slot for [CLS]
        max_body = max_len - 1 if add_cls else max_len
        ids = ids[:max_body]

        if add_cls:
            ids = [CLS_ID] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos.get(i, UNK_TOKEN) for i in ids)

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.vocab, indent=2))

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        vocab = json.loads(Path(path).read_text())
        return cls(vocab)
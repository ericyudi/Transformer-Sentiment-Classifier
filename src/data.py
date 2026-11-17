"""
IMDB dataset loading from a single CSV (Kaggle version).
"""
import re
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import WordTokenizer, PAD_ID


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CSV_PATH = RAW_DIR / "IMDB_Dataset.csv"


def load_csv() -> tuple[list[str], list[int]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}.\n"
            f"Download it from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews "
            f"and place it at {CSV_PATH}"
        )

    texts, labels = [], []
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = re.sub(r"<br\s*/?>", " ", row["review"])
            texts.append(text)
            labels.append(1 if row["sentiment"].strip() == "positive" else 0)

    return texts, labels


def train_test_split(texts, labels, test_size=0.2, seed=42):
    import random
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(len(texts) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return (
        [texts[i] for i in train_idx], [labels[i] for i in train_idx],
        [texts[i] for i in test_idx],  [labels[i] for i in test_idx],
    )


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx], max_len=self.max_len)
        return ids, self.labels[idx]


def collate_fn(batch):
    ids_list, labels = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)

    padded = torch.full((len(ids_list), max_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(ids_list), max_len), dtype=torch.bool)

    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, :len(ids)] = True

    return {
        "input_ids": padded,
        "attention_mask": mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def get_dataloaders(batch_size=32, max_len=256, max_vocab=10000, num_workers=0):
    texts, labels = load_csv()
    print(f"Total samples: {len(texts)}")

    train_texts, train_labels, test_texts, test_labels = train_test_split(
        texts, labels, test_size=0.2
    )
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples:  {len(test_texts)}")

    tokenizer = WordTokenizer.build(train_texts, max_vocab=max_vocab)
    print(f"Vocab size:    {tokenizer.vocab_size}")

    train_ds = IMDBDataset(train_texts, train_labels, tokenizer, max_len)
    test_ds = IMDBDataset(test_texts, test_labels, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, test_loader, tokenizer


if __name__ == "__main__":
    train_loader, test_loader, tok = get_dataloaders(batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"labels: {batch['labels'].tolist()}")
    print(f"first sequence: {tok.decode(batch['input_ids'][0].tolist())[:120]}...")
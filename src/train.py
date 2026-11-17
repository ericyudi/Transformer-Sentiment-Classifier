import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import get_dataloaders
from src.model import TransformerClassifier


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_vocab=args.max_vocab,
        num_workers=0,
    )

    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        n_classes=2,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    Path("checkpoints").mkdir(exist_ok=True)

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            logits, _ = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(input_ids, mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch}/{args.epochs} | "
              f"loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} | "
              f"lr={lr:.2e} | "
              f"{elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab": tokenizer.vocab,
                "args": vars(args),
                "val_acc": val_acc,
            }, "checkpoints/best.pt")
            print(f"  -> New best ({val_acc:.4f}), saved.")

    print(f"\nBest val acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train(args)
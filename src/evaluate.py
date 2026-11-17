import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from src.data import get_dataloaders
from src.predict import load_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_texts = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits, _ = model(input_ids, mask)
        probs = torch.softmax(logits, dim=-1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_probs, all_labels


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curve(labels, probs, output_path):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path} (AUC = {roc_auc:.4f})")


def show_misclassified(texts, labels, preds, probs, n=5):
    print(f"\nTop {n} most confident mistakes:")
    errors = np.where(preds != labels)[0]
    if len(errors) == 0:
        print("No errors found.")
        return

    error_conf = np.abs(probs[errors, preds[errors]] - 0.5)
    sorted_errors = errors[np.argsort(-error_conf)]

    for i, idx in enumerate(sorted_errors[:n]):
        true_label = "POSITIVE" if labels[idx] == 1 else "NEGATIVE"
        pred_label = "POSITIVE" if preds[idx] == 1 else "NEGATIVE"
        conf = probs[idx, preds[idx]]
        print(f"\n  {i+1}. True: {true_label} | Predicted: {pred_label} ({conf:.2%})")
        print(f"     Text: {texts[idx][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, tokenizer = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}\n")

    _, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_vocab=10000,
        num_workers=0,
    )

    print("Running inference on test set...")
    probs, labels = get_predictions(model, test_loader, device)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("=" * 50)

    print("\nClassification report:")
    print(classification_report(labels, preds,
                                 target_names=["Negative", "Positive"]))

    Path("results").mkdir(exist_ok=True)

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, "results/confusion_matrix.png")

    plot_roc_curve(labels, probs, "results/roc_curve.png")

    test_texts = test_loader.dataset.texts
    show_misclassified(test_texts, labels, preds, probs, n=5)
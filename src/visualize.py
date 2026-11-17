import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.predict import load_model, predict


@torch.no_grad()
def get_attention(text: str, model, tokenizer, device="cuda", max_len=64):
    """Returns tokens, attention weights per layer/head, and prediction."""
    ids = tokenizer.encode(text, max_len=max_len)
    tokens = [tokenizer.itos.get(i, "<UNK>") for i in ids]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones(1, len(ids), dtype=torch.bool, device=device)

    logits, all_attn = model(input_ids, mask)
    probs = torch.softmax(logits, dim=-1).cpu()[0]

    # Stack attention: list of (1, n_heads, seq, seq) -> (n_layers, n_heads, seq, seq)
    attn = torch.stack([a[0].cpu() for a in all_attn])

    return tokens, attn, probs


def plot_attention(tokens: list[str], attn: torch.Tensor, probs: torch.Tensor,
                   layer: int = -1, output: str = "results/attention.png"):
    """
    attn: (n_layers, n_heads, seq, seq)
    """
    n_layers, n_heads, seq, _ = attn.shape

    if layer < 0:
        layer = n_layers + layer  # negative indexing

    fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 4, 4.5))
    if n_heads == 1:
        axes = [axes]

    label = "POSITIVE" if probs[1] > probs[0] else "NEGATIVE"
    conf = max(probs[0].item(), probs[1].item())

    for h in range(n_heads):
        ax = axes[h]
        mat = attn[layer, h].numpy()

        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(seq))
        ax.set_yticks(range(seq))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"Layer {layer + 1} · Head {h + 1}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Attention weights — Prediction: {label} ({conf:.1%})",
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_all_layers(tokens: list[str], attn: torch.Tensor, probs: torch.Tensor,
                    output: str = "results/attention_all.png"):
    n_layers, n_heads, seq, _ = attn.shape

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(n_heads * 3.5, n_layers * 3.5))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_heads == 1:
        axes = axes.reshape(-1, 1)

    label = "POSITIVE" if probs[1] > probs[0] else "NEGATIVE"
    conf = max(probs[0].item(), probs[1].item())

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            mat = attn[l, h].numpy()
            ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_xticks(range(seq))
            ax.set_yticks(range(seq))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)
            ax.set_title(f"L{l + 1} · H{h + 1}", fontsize=9)

    fig.suptitle(
        f"All attention heads — Prediction: {label} ({conf:.1%})",
        fontsize=13, y=1.00
    )
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--text", type=str,
                        default="The cinematography was stunning, but the plot was predictable.")
    parser.add_argument("--max_len", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, device)

    tokens, attn, probs = get_attention(
        args.text, model, tokenizer, device, max_len=args.max_len
    )

    print(f"\nText: {args.text}")
    print(f"Tokens: {tokens}")
    print(f"P(negative)={probs[0]:.4f}  P(positive)={probs[1]:.4f}\n")

    # One layer per file (default = last layer)
    plot_attention(tokens, attn, probs, layer=-1,
                   output="results/attention.png")

    # All layers
    plot_all_layers(tokens, attn, probs,
                    output="results/attention_all.png")
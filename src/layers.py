"""
LayerNorm, Feed-Forward and Transformer Encoder Block. COOLEST PART.
"""
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """Layer normalization (Ba et al., 2016)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize across the last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    """Position-wise feed-forward: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # First time trying GELU lol
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm encoder block: attention + FFN, both with residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) boolean
        """
        # Attention sub-layer with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask)
        x = x + self.dropout(attn_out)

        # FFN sub-layer with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        # Trying to avoid vanishing gradient  with residuals...
        return x, attn_weights


if __name__ == "__main__":
    batch, seq_len, d_model, n_heads, d_ff = 2, 8, 32, 4, 128
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    mask[:, 5:] = False

    # LayerNorm
    ln = LayerNorm(d_model)
    out_ln = ln(x)
    print(f"LayerNorm:")
    print(f"mean per token: {out_ln[0, 0].mean().item():+.6f}  (should be ~0)")
    print(f"std  per token: {out_ln[0, 0].std().item():.6f}  (should be ~1)\n")

    # FeedForward
    ff = FeedForward(d_model, d_ff)
    out_ff = ff(x)
    print(f"FeedForward: {tuple(x.shape)} -> {tuple(out_ff.shape)}\n")

    # TransformerBlock
    block = TransformerBlock(d_model, n_heads, d_ff)
    out, attn = block(x, mask)
    print(f"TransformerBlock:")
    print(f"output shape: {tuple(out.shape)}")
    print(f"attention shape: {tuple(attn.shape)}")
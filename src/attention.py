"""
Scaled Dot-Product Attention and Multi-Head Attention.
"""
import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V: (batch, n_heads, seq_len, d_k)
    mask:    (batch, 1, 1, seq_len) boolean, True = keep, False = mask out

    Returns:
        output:     (batch, n_heads, seq_len, d_k)
        attn_weights: (batch, n_heads, seq_len, seq_len)
    I'm doomed...
    """
    d_k = Q.size(-1)

    # Similarity: (batch, n_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Masking (padding tokens)
    if mask is not None:
        # mask shape (batch, 1, 1, seq_len) -> broadcast to (batch, n_heads, seq_len, seq_len)
        scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax over the last dimension (the keys)
    attn_weights = torch.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) boolean, True = real token, False = padding
        """
        batch, seq_len, _ = x.shape

        # Project and reshape into heads: (batch, n_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Reshape mask for broadcasting: (batch, 1, 1, seq_len)
        if mask is not None:
            mask = mask.view(batch, 1, 1, seq_len)

        out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # Final projection
        out = self.W_o(out)

        return out, attn_weights


if __name__ == "__main__":
    batch, seq_len, d_model, n_heads = 2, 8, 32, 4
    x = torch.randn(batch, seq_len, d_model)

    # Fake mask: first 5 tokens real, last 3 padding
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, :5] = True

    mha = MultiHeadAttention(d_model, n_heads)
    out, attn = mha(x, mask)

    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Attention: {tuple(attn.shape)}")
    print(f"Attn row sum: {attn[0, 0, 0].sum().item():.4f}  (should be ~1.0)")
    print(f"Attn on pad: {attn[0, 0, 0, 5:].tolist()}  (should be all 0)")

    """
    prints:
    Input:          (2, 8, 32)
    Output:         (2, 8, 32)
    Attention:      (2, 4, 8, 8)
    Attn row sum:   1.0000  (should be ~1.0)
    Attn on pad:    [0.0, 0.0, 0.0]
    """
import torch
import torch.nn as nn

from src.layers import LayerNorm, TransformerBlock
from src.tokenizer import PAD_ID


class PositionalEmbedding(nn.Module):
    """Learned positional embedding (BERT-style)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.embedding(positions)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 256,
        n_classes: int = 2,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = LayerNorm(d_model)

        # Encoder stack
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head on [CLS] token
        self.classifier = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) boolean, True = real token

        Returns:
            logits: (batch, n_classes)
            all_attn: list of attention weights, one per layer
        """
        # Token + positional embeddings
        x = self.token_emb(input_ids) + self.pos_emb(input_ids)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        # Pass through the encoder stack
        all_attn = []
        for block in self.blocks:
            x, attn = block(x, attention_mask)
            all_attn.append(attn)

        # Take [CLS] representation and classify
        cls_repr = x[:, 0]  # (batch, d_model)
        logits = self.classifier(cls_repr)

        return logits, all_attn


if __name__ == "__main__":
    batch, seq_len, vocab_size = 4, 16, 100
    model = TransformerClassifier(vocab_size=vocab_size)

    input_ids = torch.randint(3, vocab_size, (batch, seq_len))
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    mask[:, 12:] = False

    logits, all_attn = model(input_ids, mask)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input ids: {tuple(input_ids.shape)}")
    print(f"Logits: {tuple(logits.shape)}  (should be (4, 2))")
    print(f"Num layers: {len(all_attn)}")
    print(f"Attn shapes: {tuple(all_attn[0].shape)}")
    print(f"Parameters: {n_params:,}")
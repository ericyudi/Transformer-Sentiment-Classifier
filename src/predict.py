"""
Interactive inference: classify a review and show probabilities.
"""
import argparse
import torch

from src.model import TransformerClassifier
from src.tokenizer import WordTokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]

    tokenizer = WordTokenizer(checkpoint["vocab"])
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        d_ff=args["d_ff"],
        max_len=args["max_len"],
        n_classes=2,
        dropout=args["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict(texts: list[str], model, tokenizer, device="cuda", max_len=256):
    """Classify a list of texts. Returns (probs, labels)."""
    encoded = [tokenizer.encode(t, max_len=max_len) for t in texts]
    max_seq = max(len(ids) for ids in encoded)

    input_ids = torch.full((len(texts), max_seq), 0, dtype=torch.long)
    mask = torch.zeros((len(texts), max_seq), dtype=torch.bool)

    for i, ids in enumerate(encoded):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = True

    input_ids = input_ids.to(device)
    mask = mask.to(device)

    logits, _ = model(input_ids, mask)
    probs = torch.softmax(logits, dim=-1)
    return probs.cpu(), logits.argmax(dim=-1).cpu()


def interactive_loop(model, tokenizer, device="cuda"):
    print("\nType a review (or 'quit' to exit).\n")
    while True:
        text = input("> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        probs, _ = predict([text], model, tokenizer, device)
        p_neg, p_pos = probs[0, 0].item(), probs[0, 1].item()
        label = "POSITIVE" if p_pos > p_neg else "NEGATIVE"
        confidence = max(p_neg, p_pos)

        print(f"  Prediction: {label}  (confidence: {confidence:.2%})")
        print(f"  P(negative)={p_neg:.4f}  P(positive)={p_pos:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--text", type=str, default=None,
                        help="Classify a single text and exit (non-interactive)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint} on {device}")

    if args.text:
        probs, _ = predict([args.text], model, tokenizer, device)
        p_neg, p_pos = probs[0, 0].item(), probs[0, 1].item()
        print(f"P(negative)={p_neg:.4f}  P(positive)={p_pos:.4f}")
    else:
        interactive_loop(model, tokenizer, device)
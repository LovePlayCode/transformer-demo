"""A tiny one-file entry point for the Transformer language model demo."""

from __future__ import annotations

import torch

from transformer_core import CharacterTokenizer, TinyTransformerLM, TransformerConfig, get_lm_batch


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = (
        "transformer models learn context with attention. "
        "attention lets every token look back at earlier tokens. "
        "small demos make big ideas easier to understand. "
    ) * 80

    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_embd=64, n_layer=2)
    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch_size = 32
    train_steps = 300

    model.train()
    for step in range(train_steps):
        xb, yb = get_lm_batch(data, config.block_size, batch_size, device)
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == train_steps - 1:
            print(f"step {step:03d} | loss {loss.item():.4f}")

    start = torch.tensor([[tokenizer.stoi["t"]]], dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=160)[0].tolist()
    print("\nGenerated text:")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()

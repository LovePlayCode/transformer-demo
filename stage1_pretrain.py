"""Stage 1: pretrain a tiny base language model with next-token prediction."""

from __future__ import annotations

from pathlib import Path

import torch

from toy_data import PRETRAIN_TEXT, all_training_text
from transformer_core import CharacterTokenizer, TinyTransformerLM, TransformerConfig, get_lm_batch, save_checkpoint


ARTIFACT_DIR = Path("artifacts")
PRETRAIN_CKPT = ARTIFACT_DIR / "stage1_pretrained.pt"


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CharacterTokenizer(all_training_text())
    # 预训练数据
    data = torch.tensor(tokenizer.encode(PRETRAIN_TEXT), dtype=torch.long)

    config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=160)
    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch_size = 32
    train_steps = 350

    model.train()
    for step in range(train_steps):
        xb, yb = get_lm_batch(data, config.block_size, batch_size, device)
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == train_steps - 1:
            print(f"pretrain step {step:03d} | loss {loss.item():.4f}")

    save_checkpoint(PRETRAIN_CKPT, model, tokenizer, extra={"stage": "pretrain"})
    print(f"\nsaved base model to {PRETRAIN_CKPT}")

    start = torch.tensor([[tokenizer.stoi["t"]]], dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=180)[0].tolist()
    print("\nBase model sample:")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()

"""Stage 2: supervised fine-tuning on prompt-answer examples."""

from __future__ import annotations

from pathlib import Path

import torch

from stage1_pretrain import PRETRAIN_CKPT
from toy_data import SFT_EXAMPLES
from transformer_core import encode_supervised_example, load_checkpoint, save_checkpoint


ARTIFACT_DIR = Path("artifacts")
SFT_CKPT = ARTIFACT_DIR / "stage2_sft.pt"


def main() -> None:
    torch.manual_seed(43)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not PRETRAIN_CKPT.exists():
        raise FileNotFoundError("run `python3 stage1_pretrain.py` before SFT")

    model, tokenizer, _ = load_checkpoint(PRETRAIN_CKPT, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_steps = 220

    model.train()
    for step in range(train_steps):
        example = SFT_EXAMPLES[step % len(SFT_EXAMPLES)]
        xb, yb = encode_supervised_example(
            tokenizer=tokenizer,
            prompt=example["prompt"],
            answer=example["answer"],
            block_size=model.config.block_size,
            device=device,
        )
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 40 == 0 or step == train_steps - 1:
            print(f"sft step {step:03d} | loss {loss.item():.4f}")

    save_checkpoint(SFT_CKPT, model, tokenizer, extra={"stage": "sft"})
    print(f"\nsaved instruction model to {SFT_CKPT}")

    prompt = "User: What is pretraining?\nAssistant:"
    start = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=140, temperature=0.7)[0].tolist()
    print("\nSFT model sample:")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()

"""Chat with the tiny instruction-tuned Transformer from the terminal."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from stage1_pretrain import PRETRAIN_CKPT
from stage2_sft import SFT_CKPT
from stage3_dpo import DPO_CKPT
from transformer_core import CharacterTokenizer, TinyTransformerLM, load_checkpoint


def pick_checkpoint() -> Path:
    """优先加载最新 stage 的 checkpoint（DPO > SFT > Pretrain）。"""
    for path in (DPO_CKPT, SFT_CKPT, PRETRAIN_CKPT):
        if path.exists():
            return path
    raise FileNotFoundError("run `python3 run_full_pipeline.py` before chatting")


def unsupported_chars(text: str, tokenizer: CharacterTokenizer) -> list[str]:
    """找出不在词表 stoi 中的字符（如中文），避免 encode 时 KeyError。"""
    return sorted({ch for ch in text if ch not in tokenizer.stoi})


@torch.no_grad()
def generate_answer(
    model: TinyTransformerLM,
    tokenizer: CharacterTokenizer,
    question: str,
    device: str,
    max_new_tokens: int = 180,
    temperature: float = 0.7,
) -> str:
    prompt = f"User: {question}\nAssistant:"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        # temperature 大 → 分布更平 → 更随机；小 → 更确定
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

        next_char = tokenizer.decode([int(next_idx.item())])
        if next_char == "\n":
            break

    generated = tokenizer.decode(idx[0].tolist())
    return generated[len(prompt) :].strip()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = pick_checkpoint()
    model, tokenizer, extra = load_checkpoint(checkpoint, device)

    stage = extra.get("stage", "unknown")
    print(f"Loaded {checkpoint} ({stage})")
    print("Type an English question, or `exit` to quit.")
    print("This is a tiny teaching model, so answers are only for demonstrating the pipeline.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        bad_chars = unsupported_chars(question, tokenizer)
        if bad_chars:
            print(f"AI: I cannot tokenize these characters yet: {bad_chars}")
            print("    This toy tokenizer was trained on a tiny English character set.\n")
            continue

        answer = generate_answer(model, tokenizer, question, device)
        print(f"AI: {answer}\n")


if __name__ == "__main__":
    main()

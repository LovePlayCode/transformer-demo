"""Stage 3: preference tuning with Direct Preference Optimization (DPO)."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.nn import functional as F

from stage2_sft import SFT_CKPT
from toy_data import PREFERENCE_EXAMPLES
from transformer_core import load_checkpoint, save_checkpoint, sequence_logprob


ARTIFACT_DIR = Path("artifacts")
DPO_CKPT = ARTIFACT_DIR / "stage3_dpo.pt"


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO loss：让 policy 相对 reference 更偏好 chosen、更不偏好 rejected。

    loss = -log σ(β * ((log π(chosen)-log π(rejected)) - (log π_ref(chosen)-log π_ref(rejected))))
    """

    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    return -F.logsigmoid(beta * (policy_logratio - ref_logratio))


def preference_accuracy(policy_model, tokenizer, device: str) -> tuple[int, int]:
    """评估 toy 偏好对：chosen 的平均 log-prob 是否高于 rejected。"""
    wins = 0
    for example in PREFERENCE_EXAMPLES:
        chosen = sequence_logprob(policy_model, tokenizer, example["prompt"], example["chosen"], device, normalize=True)
        rejected = sequence_logprob(policy_model, tokenizer, example["prompt"], example["rejected"], device, normalize=True)
        wins += int(chosen.item() > rejected.item())
    return wins, len(PREFERENCE_EXAMPLES)


def main() -> None:
    torch.manual_seed(44)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not SFT_CKPT.exists():
        raise FileNotFoundError("run `python3 stage2_sft.py` before DPO")

    policy_model, tokenizer, _ = load_checkpoint(SFT_CKPT, device)
    # reference model 冻结为 SFT  checkpoint 的副本，防止偏好优化偏离太远
    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-5)
    beta = 0.1  # 偏好强度；越大越激进地拉开 chosen/rejected
    train_steps = 140

    policy_model.train()
    for step in range(train_steps):
        example = PREFERENCE_EXAMPLES[step % len(PREFERENCE_EXAMPLES)]

        policy_chosen = sequence_logprob(policy_model, tokenizer, example["prompt"], example["chosen"], device, normalize=True)
        policy_rejected = sequence_logprob(policy_model, tokenizer, example["prompt"], example["rejected"], device, normalize=True)
        with torch.no_grad():
            ref_chosen = sequence_logprob(reference_model, tokenizer, example["prompt"], example["chosen"], device, normalize=True)
            ref_rejected = sequence_logprob(reference_model, tokenizer, example["prompt"], example["rejected"], device, normalize=True)

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == train_steps - 1:
            margin = (policy_chosen - policy_rejected).item()
            print(f"dpo step {step:03d} | loss {loss.item():.4f} | normalized margin {margin:.2f}")

    wins, total = preference_accuracy(policy_model, tokenizer, device)
    save_checkpoint(DPO_CKPT, policy_model, tokenizer, extra={"stage": "dpo", "beta": beta})
    print(f"\nsaved preference-tuned model to {DPO_CKPT}")
    print(f"toy preference accuracy: {wins}/{total}")

    prompt = "User: What does DPO optimize?\nAssistant:"
    start = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = policy_model.generate(start, max_new_tokens=140, temperature=0.7)[0].tolist()
    print("\nDPO model sample:")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()

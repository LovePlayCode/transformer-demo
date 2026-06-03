"""Run the full educational LLM pipeline end to end.

Pretrain → SFT → DPO，每阶段 checkpoint 写入 artifacts/。
"""

from __future__ import annotations

import stage1_pretrain
import stage2_sft
import stage3_dpo


def main() -> None:
    print("\n=== Stage 1: Pretraining ===")
    stage1_pretrain.main()

    print("\n=== Stage 2: Supervised Fine-Tuning ===")
    stage2_sft.main()

    print("\n=== Stage 3: DPO Preference Tuning ===")
    stage3_dpo.main()


if __name__ == "__main__":
    main()

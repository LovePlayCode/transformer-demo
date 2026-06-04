"""Shared building blocks for the educational Transformer pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    """模型超参数。vocab_size 由 CharacterTokenizer 决定，其余有默认值。"""

    vocab_size: int  # 词表大小 = 字符种类数，决定 embedding 和 lm_head 的输出维
    block_size: int = 64  # 上下文窗口：一次最多处理多少个 token
    n_embd: int = 96  # 每个 token 的 hidden 向量维度（也是 attention 里的 C）
    n_head: int = 4  # 注意力头数；必须整除 n_embd，每头维度 head_dim = n_embd // n_head
    n_layer: int = 3  # TransformerBlock 堆叠层数
    dropout: float = 0.1


class MultiHeadSelfAttention(nn.Module):
    """Masked multi-head self-attention for decoder-only language modeling.

    完整数据流（以 x shape=(B,T,C) 为例）：
        x → qkv Linear → split 分离 Q/K/V → split_heads 拆多头
        → scores = QK^T/√D → causal mask → softmax → out = weights @ V
        → 合并多头 → proj → 输出 (B,T,C)

    其中 B=batch_size, T=seq_len, C=n_embd, H=n_head, D=head_dim=C/H。
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # 合并 Q/K/V 投影：等价于三个 Linear(C,C)，但一次 matmul 更快。
        # 输出最后一维为 3C，布局为 [Q | K | V]，每个占 C 维。
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 多头 attention 输出合并后的线性层，混合各 head 的信息。
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码（下三角）：位置 i 只能 attend 到 j<=i，防止偷看未来 token。
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — B=批次, T=序列长度, C=n_embd
        batch_size, seq_len, n_embd = x.shape

        # Step 1: 生成 QKV — (B,T,C) → (B,T,3C)
        qkv = self.qkv(x)
        # Step 2: 在最后一维(dim=2)分离三种角色，各 (B,T,C)
        # dim=0 是 batch，dim=1 是 token 位置，dim=2 是特征维（Q/K/V 拼在这里）
        q, k, v = qkv.split(n_embd, dim=2)

        # Step 3: 拆多头 — (B,T,C) → (B,H,T,D)，H 个头各自独立做 attention
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Step 4: 各 head 并行计算 attention 分数 — (B,H,T,T)
        # scores[b,h,i,j] = token i 的 Query 与 token j 的 Key 的匹配度
        # PyTorch 的 @ 会对 H 维自动批量处理，无需显式 for 循环
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        # Step 5: 每行 softmax 得到 attention 权重，再对 V 加权求和 — (B,H,T,D)
        # scores 决定“看谁”，softmax 决定“看多少”，weights @ v 得到“看完之后汇总出来的信息”。
        weights = F.softmax(scores, dim=-1)
        # 对注意力权重做dropout,训练时会随机丢掉一部分注意力连接，防止模型过度依赖某些固定token，
        # 提高模型的泛化能力。
        # 在推理时，dropout 被禁用，所以不会影响结果。
        weights = self.attn_dropout(weights)
        """
        用注意力权重对 v 做加权求和。也就是说，每个 token 根据自己对其他 token 的关注程度，把其他 token 的内容信息混合进来。
        """
        out = weights @ v

        # Step 6: 合并多头 — (B,H,T,D) → (B,T,C)，再经 proj 混合各 head
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        # 混合不同head的信息 
        """
        前面每个 head 是独立计算 attention 的，合并后只是简单拼接。proj 会让模型学习如何组合这些 head 的输出，比如某些 head 更关注语法关系，某些 head 更关注当前位置附近的词，线性层负责把这些信息重新融合成一个新的 token 表示。
        """
        out = self.proj(out)
        return self.resid_dropout(out)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 (B,T,C)  reshape 为 (B,H,T,D)，便于各 head 独立计算 attention。"""
        batch_size, seq_len, n_embd = x.shape
        return x.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)


class FeedForward(nn.Module):
    """Position-wise MLP used after attention in each Transformer block.

    对每个 token 位置独立做相同的两层 MLP（不跨 token 混合信息）。
    中间层扩到 4×n_embd 是 GPT 类模型的常见做法。
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One pre-norm decoder-only Transformer block.

    Pre-Norm 结构：先 LayerNorm 再子层，最后残差相加。
    x → x + Attention(LN(x)) → x + FFN(LN(x))
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class TinyTransformerLM(nn.Module):
    """A small GPT-style decoder-only Transformer language model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # token id → n_embd 维向量；vocab_size 个可查表的 embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置 0..block_size-1 → n_embd 维向量；attention 本身不感知顺序，需额外注入
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 每个位置输出 vocab_size 维 logits，预测下一个 token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) — 整型 token id 序列
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")

        positions = torch.arange(seq_len, device=idx.device)
        # (B,T,n_embd)：token 语义 + 位置信息相加
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 标准 next-token CE loss：logits[t] 预测 targets[t]（即 idx[t+1]）
            loss = F.cross_entropy(logits.reshape(batch_size * seq_len, -1), targets.reshape(batch_size * seq_len))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        """自回归生成：每次取最后一个位置的 logits 采样下一个 token 并拼到序列末尾。"""
        self.eval()
        for _ in range(max_new_tokens):
            # 只保留最近 block_size 个 token，避免超出上下文窗口
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


class CharacterTokenizer:
    """字符级 tokenizer：每个不重复字符对应一个 id，无 BPE/子词。

    词表生成：sorted(set(text)) → {字符: 0,1,2,...}
    Stage 1 用 all_training_text() 建词表，后续阶段从 checkpoint 加载，不再重建。
    """

    def __init__(self, text: str | None = None, stoi: dict[str, int] | None = None) -> None:
        if stoi is not None:
            # 从 checkpoint 恢复词表
            self.stoi = dict(stoi)
        elif text is not None:
            # 从语料统计所有不重复字符，排序保证 id 稳定
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
        else:
            raise ValueError("provide either text or stoi")
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        # 字符不在 stoi 中会 KeyError；chat_cli 会提前检查
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict[str, int]:
        return dict(self.stoi)


def get_lm_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """预训练 batch 采样：随机截取 block_size 长度的连续片段做 next-token 预测。

    data: 一维 token id 序列（整段语料 encode 后的结果）
    返回 xb (B,T), yb (B,T)，其中 yb 是 xb 右移一位（预测下一个 token）
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def encode_supervised_example(
    tokenizer: CharacterTokenizer,
    prompt: str,
    answer: str,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造一条 SFT 样本，prompt 部分的 label 设为 -100 以跳过 loss 计算。

    PyTorch cross_entropy 遇到 target=-100 会自动 ignore，从而只训练 answer 部分。
    """

    full_text = prompt + answer
    ids = tokenizer.encode(full_text)
    if len(ids) > block_size + 1:
        ids = ids[: block_size + 1]

    prompt_len = min(len(tokenizer.encode(prompt)), len(ids) - 1)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    # prompt 对应的 target 位置置 -100，不参与 loss
    y[:, : max(prompt_len - 1, 0)] = -100
    return x, y


def sequence_logprob(
    model: TinyTransformerLM,
    tokenizer: CharacterTokenizer,
    prompt: str,
    answer: str,
    device: str,
    normalize: bool = False,
) -> torch.Tensor:
    """计算 answer 部分 token 的条件 log 概率（DPO 用）。

    normalize=True 时对 answer token 取平均 log-prob，减轻 chosen/rejected 长度差异的影响。
    """

    full_text = prompt + answer
    ids = tokenizer.encode(full_text)
    if len(ids) > model.config.block_size + 1:
        ids = ids[: model.config.block_size + 1]

    prompt_len = min(len(tokenizer.encode(prompt)), len(ids) - 1)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)

    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

    answer_mask = torch.arange(y.size(1), device=device) >= max(prompt_len - 1, 0)
    answer_log_probs = token_log_probs[:, answer_mask]
    if normalize:
        return answer_log_probs.mean()
    return answer_log_probs.sum()


def save_checkpoint(
    path: Path,
    model: TinyTransformerLM,
    tokenizer: CharacterTokenizer,
    extra: dict[str, Any] | None = None,
) -> None:
    """保存模型权重、config、词表 stoi，供后续 stage 或 chat_cli 加载。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "tokenizer": tokenizer.to_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str) -> tuple[TinyTransformerLM, CharacterTokenizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = TransformerConfig(**payload["config"])
    tokenizer = CharacterTokenizer(stoi=payload["tokenizer"])
    model = TinyTransformerLM(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model, tokenizer, payload.get("extra", {})


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

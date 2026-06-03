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
    vocab_size: int
    block_size: int = 64
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1


class MultiHeadSelfAttention(nn.Module):
    """Masked multi-head self-attention for decoder-only language modeling."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(n_embd, dim=2)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        out = self.proj(out)
        return self.resid_dropout(out)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        return x.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2)


class FeedForward(nn.Module):
    """Position-wise MLP used after attention in each Transformer block."""

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
    """One pre-norm decoder-only Transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # masked multi-head self-attention
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # feed-forward
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
        # 词向量化
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(batch_size * seq_len, -1), targets.reshape(batch_size * seq_len))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


class CharacterTokenizer:
    """Minimal character-level tokenizer for small educational corpora."""

    def __init__(self, text: str | None = None, stoi: dict[str, int] | None = None) -> None:
        if stoi is not None:
            self.stoi = dict(stoi)
        elif text is not None:
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
        else:
            raise ValueError("provide either text or stoi")
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict[str, int]:
        return dict(self.stoi)


def get_lm_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
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
    """Create one SFT training example and mask prompt tokens from the loss."""

    full_text = prompt + answer
    ids = tokenizer.encode(full_text)
    if len(ids) > block_size + 1:
        ids = ids[: block_size + 1]

    prompt_len = min(len(tokenizer.encode(prompt)), len(ids) - 1)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
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
    """Log-probability of answer tokens conditioned on prompt tokens."""

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

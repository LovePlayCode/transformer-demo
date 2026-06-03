# Tiny Transformer / LLM Training Demo

这是一个用于学习 Transformer 和现代 LLM 训练流程的极简 PyTorch 项目。它实现的是 GPT 风格的 **decoder-only Transformer**，并额外提供了一个教学版完整流程：

```text
Stage 1: Pretraining  ->  Stage 2: SFT  ->  Stage 3: DPO
```

项目目标不是训练出强大的模型，而是让你看清楚 Transformer、预训练、监督微调和偏好对齐之间的关系。

## 运行方式

建议先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 simple_transformer.py
```

运行后你会看到一个最小 next-token 训练 demo，最后会基于起始字符生成一小段文本。

## 完整流程

一键运行三个阶段：

```bash
python3 run_full_pipeline.py
```

也可以分阶段运行：

```bash
python3 stage1_pretrain.py
python3 stage2_sft.py
python3 stage3_dpo.py
```

每个阶段会把 checkpoint 保存到 `artifacts/`：

- `artifacts/stage1_pretrained.pt`：预训练后的 base model。
- `artifacts/stage2_sft.pt`：指令微调后的 instruction model。
- `artifacts/stage3_dpo.pt`：DPO 偏好对齐后的模型。

## 交互式提问

跑完完整流程后，可以启动一个命令行聊天入口：

```bash
python3 chat_cli.py
```

然后输入英文问题，例如：

```text
You: What is pretraining?
AI: ...
```

注意：当前模型非常小，训练数据也是 toy data，所以它主要用于演示“输入问题 -> 模型生成回答”的链路。它不是一个真正有知识量的大模型。当前字符 tokenizer 也只覆盖 toy 数据里的英文字符，直接输入中文会提示无法 tokenize。

## 代码结构

核心模块：

- `transformer_core.py`：Transformer 模型、字符 tokenizer、batch 构造、SFT loss mask、DPO log-prob 工具、checkpoint 工具。
- `toy_data.py`：小型预训练语料、SFT prompt-answer 数据、DPO chosen/rejected 偏好数据。
- `simple_transformer.py`：最小 Transformer 训练入口，适合先理解模型结构。
- `stage1_pretrain.py`：预训练阶段。
- `stage2_sft.py`：监督微调阶段。
- `stage3_dpo.py`：DPO 偏好对齐阶段。
- `run_full_pipeline.py`：按顺序运行三个阶段。
- `chat_cli.py`：加载最新 checkpoint，提供终端问答入口。

`transformer_core.py` 中的关键类：

- `TransformerConfig`：集中管理模型超参数，例如上下文长度、embedding 维度、注意力头数和层数。
- `MultiHeadSelfAttention`：实现 masked multi-head self-attention，保证当前位置只能看到自己和之前的 token。
- `FeedForward`：每个 Transformer block 中的逐位置 MLP。
- `TransformerBlock`：组合 LayerNorm、Self-Attention、FeedForward 和残差连接。
- `TinyTransformerLM`：完整的 decoder-only 语言模型。
- `CharacterTokenizer`：简单字符级 tokenizer，方便 demo 不依赖外部数据集。

## 学习顺序

建议按这个顺序阅读：

1. 先看 `simple_transformer.py`，理解一个最小语言模型如何训练。
2. 再看 `TinyTransformerLM.forward()`，理解 token embedding、position embedding、block、输出 logits 的流程。
3. 然后看 `TransformerBlock.forward()`，理解残差连接：

   ```python
   x = x + self.attn(self.ln_1(x))
   x = x + self.ffwd(self.ln_2(x))
   ```

4. 最后重点看 `MultiHeadSelfAttention.forward()`，这里是 Transformer 的核心。
5. 看 `stage1_pretrain.py`，理解 base model 如何通过 next-token prediction 学语言模式。
6. 看 `stage2_sft.py`，理解为什么 SFT 只在答案部分计算 loss。
7. 看 `stage3_dpo.py`，理解 chosen/rejected 偏好对如何变成 DPO loss。

## 三个训练阶段

### 1. Pretraining

预训练使用无标注文本做 next-token prediction：

```text
输入: transformer models learn context with attentio
目标: ransformer models learn context with attention
```

这个阶段让模型学习语言模式，但它还不一定会“按用户指令回答”。

### 2. SFT

SFT 使用 prompt-answer 数据：

```text
User: What is pretraining?
Assistant: Pretraining teaches a language model...
```

代码里通过 `encode_supervised_example()` 把 prompt 部分的 label 设成 `-100`，这样交叉熵只训练 answer 部分。这对应真实 LLM 里的 instruction tuning。

### 3. DPO

DPO 使用偏好对：

```text
prompt:   User: What does DPO optimize?
chosen:   DPO optimizes the model so preferred responses...
rejected: DPO only changes the tokenizer...
```

`stage3_dpo.py` 会保留一个冻结的 reference model，然后训练 policy model 让 chosen 相对 rejected 的 log-prob 更高：

```python
loss = -logsigmoid(beta * (policy_logratio - ref_logratio))
```

这比传统 PPO 版 RLHF 更适合教学，因为不需要额外训练 reward model。

为了让 toy 数据更稳定，代码里对 answer token 使用平均 log-prob，减少 chosen/rejected 回答长度不同带来的偏差。真实 DPO 训练通常会更严格地控制数据格式、长度分布和 batch 构造。

## 关键概念

### 为什么需要 causal mask？

语言模型训练的是“根据前面的 token 预测下一个 token”。如果第 3 个 token 在训练时能看到第 4 个 token，就会发生信息泄漏。因此代码中使用了下三角 mask：

```python
mask = torch.tril(torch.ones(config.block_size, config.block_size))
```

它让每个位置只能关注当前位置和之前的位置。

### 为什么要多头注意力？

一个注意力头只能从一种表示子空间中学习关系。多头注意力把 embedding 拆成多个 head，让模型可以同时学习多种关系，例如局部搭配、长距离依赖、重复模式等。

### 为什么要位置编码？

Self-attention 本身不天然知道 token 的顺序。所以模型把 token embedding 和 position embedding 相加：

```python
x = self.token_embedding(idx) + self.position_embedding(positions)
```

这样模型才能区分“第 1 个字符”和“第 10 个字符”。

## 可以尝试的修改

- 把 `stage1_pretrain.py` 里的 `train_steps` 改大，观察 base model 生成效果。
- 把 `stage2_sft.py` 里的 `SFT_EXAMPLES` 扩充，观察模型是否更像对话助手。
- 把 `stage3_dpo.py` 里的 `PREFERENCE_EXAMPLES` 扩充，观察 chosen/rejected margin 的变化。
- 把 `n_layer` 从 `2` 改成 `4`，观察训练速度和 loss。
- 把 `block_size` 改大，让模型看到更长上下文。
- 把 `toy_data.py` 替换成你自己的小语料和偏好数据。

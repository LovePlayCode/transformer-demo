"""Tiny datasets used by the educational training pipeline.

Real LLM projects use web-scale corpora, curated instruction data, and large
human preference datasets. These examples are deliberately small so the whole
pipeline can run on a laptop and the data format is easy to inspect.
"""

from __future__ import annotations


PRETRAIN_TEXT = (
    "transformer models learn context with attention. "
    "attention lets every token look back at earlier tokens. "
    "a language model predicts the next token from previous tokens. "
    "pretraining teaches broad patterns from raw text. "
    "fine tuning teaches the model how to follow instructions. "
    "preference tuning teaches the model which answer style is better. "
    "small demos make big ideas easier to understand. "
) * 80


SFT_EXAMPLES = [
    {
        "prompt": "User: What is a transformer?\nAssistant:",
        "answer": " A transformer is a neural network that uses attention to model relationships between tokens.\n",
    },
    {
        "prompt": "User: What is pretraining?\nAssistant:",
        "answer": " Pretraining teaches a language model to predict the next token on large amounts of raw text.\n",
    },
    {
        "prompt": "User: What is supervised fine tuning?\nAssistant:",
        "answer": " Supervised fine tuning trains the model on prompt and answer examples so it follows instructions.\n",
    },
    {
        "prompt": "User: What is DPO?\nAssistant:",
        "answer": " DPO is a preference optimization method that makes chosen answers more likely than rejected answers.\n",
    },
    {
        "prompt": "User: Why do transformers use a causal mask?\nAssistant:",
        "answer": " The causal mask prevents a token from seeing future tokens during next-token prediction.\n",
    },
]


PREFERENCE_EXAMPLES = [
    {
        "prompt": "User: What is pretraining?\nAssistant:",
        "chosen": " Pretraining learns general language patterns by predicting the next token on raw text.\n",
        "rejected": " Pretraining is just memorizing one fixed answer without learning token probabilities.\n",
    },
    {
        "prompt": "User: Explain SFT in one sentence.\nAssistant:",
        "chosen": " SFT trains a pretrained model on high-quality prompt-answer pairs to follow instructions.\n",
        "rejected": " SFT means the model ignores prompts and only generates random text.\n",
    },
    {
        "prompt": "User: What does DPO optimize?\nAssistant:",
        "chosen": " DPO optimizes the model so preferred responses receive higher probability than rejected responses.\n",
        "rejected": " DPO only changes the tokenizer and does not train the model weights.\n",
    },
    {
        "prompt": "User: Why keep a reference model during DPO?\nAssistant:",
        "chosen": " The reference model anchors training, so preference updates do not drift too far from the SFT model.\n",
        "rejected": " The reference model is used to delete all learned knowledge before alignment.\n",
    },
]


def all_training_text() -> str:
    """Return every character that may appear in the toy pipeline."""

    pieces = [PRETRAIN_TEXT]
    for example in SFT_EXAMPLES:
        pieces.append(example["prompt"])
        pieces.append(example["answer"])
    for example in PREFERENCE_EXAMPLES:
        pieces.append(example["prompt"])
        pieces.append(example["chosen"])
        pieces.append(example["rejected"])
    return "".join(pieces)

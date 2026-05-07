#!/usr/bin/env python3
"""Fine-tune roberta-babe-ft on BASIL's training split.

Addresses the BABE→BASIL domain mismatch: the base checkpoint knows lexical
bias patterns but has never seen BASIL labels. Fine-tuning on the 80% train
split directly adapts it to BASIL's annotation scheme.

Outputs
-------
outputs/roberta-babe-basil-ft/   saved model + tokenizer (use as AUDITOR_MODEL_ID)
outputs/finetune_eval.json        metrics on the test split after training

Usage
-----
    python scripts/finetune_basil.py

Env vars
--------
    BASIL_DATA_DIR      path to BASIL *.json articles
    AUDITOR_MODEL_ID    base checkpoint to fine-tune (default: roberta-babe-ft)
    FT_EPOCHS           number of training epochs (default: 3)
    FT_BATCH_SIZE       per-device batch size (default: 16)
    FT_LR               learning rate (default: 2e-5)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir

BASE_MODEL = os.environ.get("AUDITOR_MODEL_ID", "mediabiasgroup/roberta-babe-ft").strip()
EPOCHS = int(os.environ.get("FT_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("FT_BATCH_SIZE", "16"))
LR = float(os.environ.get("FT_LR", "2e-5"))
MAX_LENGTH = 256
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "roberta-babe-basil-ft"


class BasilDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.encodings.items()} | {"labels": self.labels[idx]}


class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to handle the 20% positive imbalance."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def main() -> None:
    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading BASIL from {data_dir} ...")
    frame = load_basil_sentences(data_dir)
    train_df, test_df = split_sentence_frame(frame, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)} sentences (pos rate: {train_df['label'].mean():.3f})")
    print(f"Test:  {len(test_df)} sentences  (pos rate: {test_df['label'].mean():.3f})")

    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_texts = train_df["sentence_text"].astype(str).tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["sentence_text"].astype(str).tolist()
    test_labels = test_df["label"].tolist()

    print("Tokenizing ...")
    train_dataset = BasilDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = BasilDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    # Class weights inversely proportional to frequency
    n_neg = sum(1 for l in train_labels if l == 0)
    n_pos = sum(1 for l in train_labels if l == 1)
    total = n_neg + n_pos
    w_neg = total / (2 * n_neg)
    w_pos = total / (2 * n_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    print(f"Class weights: neg={w_neg:.3f}, pos={w_pos:.3f}")

    OUTPUTS.mkdir(exist_ok=True)
    SAVE_DIR.mkdir(exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=False,
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nFine-tuning for {EPOCHS} epochs (lr={LR}, batch={BATCH_SIZE}, device={device}) ...")
    trainer.train()

    print("\nEvaluating on test split ...")
    results = trainer.evaluate(test_dataset)
    print(json.dumps({k: round(v, 4) for k, v in results.items()}, indent=2))

    eval_path = OUTPUTS / "finetune_eval.json"
    eval_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {eval_path}")

    print(f"\nSaving fine-tuned model to {SAVE_DIR} ...")
    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    print("Done.")

    print(f"\nTo use the fine-tuned model:")
    print(f"  export AUDITOR_MODEL_ID='{SAVE_DIR}'")
    print(f"  python scripts/threshold_sweep.py")


if __name__ == "__main__":
    main()

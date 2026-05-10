#!/usr/bin/env python3
"""Fine-tune the BABE-fitted RoBERTa auditor on BASIL's training split (domain adaptation).

The public ``mediabiasgroup/roberta-babe-ft`` checkpoint is trained on BABE, not
BASIL. This script performs supervised fine-tuning on BASIL sentence labels
(event-grouped train split) and writes a local directory you can point
``AUDITOR_MODEL_ID`` at for evaluation / MCP / threshold sweeps.

Outputs
-------
outputs/roberta-babe-basil-ft/     tokenizer + model weights
outputs/basil_finetune_metrics.json   quick dev-set metrics after training

Usage
-----
    BASIL_DATA_DIR=... python scripts/finetune_basil_auditor.py

Env vars
--------
    BASIL_DATA_DIR          BASIL article JSON folder
    BASIL_FINETUNE_EPOCHS   default 2
    BASIL_FINETUNE_CAP      optional max train rows for a fast dry-run (0 = all)
    AUDITOR_BASE_MODEL      required — HF id or path of 2-label classifier to fine-tune
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from mcp_server.basil_dataset import load_basil_sentences, split_sentence_frame
from mcp_server.basil_paths import resolve_basil_data_dir

OUTPUTS = ROOT / "outputs"
OUT_DIR = OUTPUTS / "roberta-babe-basil-ft"
EPOCHS = int(os.environ.get("BASIL_FINETUNE_EPOCHS", "2"))
CAP = int(os.environ.get("BASIL_FINETUNE_CAP", "0"))
BASE_MODEL = os.environ.get("AUDITOR_BASE_MODEL", "").strip()


class SentenceLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
        )
        enc["labels"] = int(self.labels[i])
        return enc


def _eval_split(model, tokenizer, texts: list[str], labels: list[int], device: torch.device, bs: int = 32) -> dict:
    model.eval()
    preds: list[int] = []
    for start in range(0, len(texts), bs):
        chunk = texts[start : start + bs]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        preds.extend(int(x) for x in pred)
    y = np.array(labels, dtype=np.int64)
    p = np.array(preds, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
    }


def main() -> None:
    if not BASE_MODEL:
        print(
            "ERROR: set AUDITOR_BASE_MODEL (base checkpoint to fine-tune; no default in code).",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        data_dir = resolve_basil_data_dir()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading BASIL from {data_dir} …")
    frame = load_basil_sentences(data_dir)
    train, test = split_sentence_frame(frame, test_size=0.2, random_state=42)
    if CAP > 0:
        train = train.iloc[:CAP].reset_index(drop=True)
        print(f"Train cap active: using {len(train)} train rows.")

    texts_tr = train["sentence_text"].astype(str).tolist()
    labels_tr = train["label"].astype(int).tolist()
    texts_te = test["sentence_text"].astype(str).tolist()[:2000]
    labels_te = test["label"].astype(int).tolist()[:2000]

    print(f"Train sentences: {len(texts_tr)}  |  Dev sample: {len(texts_te)}")
    print(f"Base checkpoint : {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    if model.config.num_labels != 2:
        raise ValueError("Expected a 2-label sequence classifier.")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_ds = SentenceLabelDataset(texts_tr, labels_tr, tokenizer)
    OUT_DIR.parent.mkdir(exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=100,
        report_to=[],
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    print("Training …")
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"Saved model to {OUT_DIR}")

    device = next(model.parameters()).device
    metrics = _eval_split(model, tokenizer, texts_te, labels_te, device)
    metrics_path = OUTPUTS / "basil_finetune_metrics.json"
    metrics_path.write_text(json.dumps({"dev_eval_sample": metrics, "out_dir": str(OUT_DIR)}, indent=2))
    print(f"Quick dev metrics (first {len(texts_te)} test rows): {metrics}")
    print(f"Wrote {metrics_path}")
    print("\nNext steps:")
    print(f"  set AUDITOR_MODEL_ID={OUT_DIR.resolve()}")
    print("  python scripts/threshold_sweep.py")
    print("  python scripts/run_full_eval.py")


if __name__ == "__main__":
    main()

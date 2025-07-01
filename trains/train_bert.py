#!/usr/bin/env python3
"""
Fine-tune DistilBERT on ATIS intents (no optional Trainer features).

Usage
-----
hatch run python train_bert.py \
  --train data/atis/train.tsv --test data/atis/test.tsv \
  --out-dir models/bert --epochs 3
"""

from pathlib import Path
import argparse, json

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

# Patch Trainer._wrap_model to skip accelerate.unwrap_model
def _legacy_wrap_model(self, model_wrapped):
    # Bypass accelerate.unwrap_model(model, keep_torch_compile=False)
    return model_wrapped

Trainer._wrap_model = _legacy_wrap_model


def main(a):
    # 1. Load TSVs
    ds = load_dataset(
        "csv",
        data_files={"train": a.train, "test": a.test},
        sep="\t",
        column_names=["text", "label"],
    )

    # 2. Build label map from both splits
    labels = sorted(set(ds["train"]["label"]) | set(ds["test"]["label"]))
    lab2idx = {lab: i for i, lab in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def encode(batch):
        x = tok(batch["text"], truncation=True, padding="max_length", max_length=128)
        x["label"] = [lab2idx[l] for l in batch["label"]]
        return x

    ds_tok = ds.map(encode, batched=True, remove_columns=["text"])

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(labels)
    )

    # 4. Trainer
    args = TrainingArguments(
        output_dir="runs/bert",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=a.epochs,
        logging_steps=50,
        save_strategy="no",       # disable mid-training checkpoints
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
    )
    trainer.train()

    # 5. Save artefacts
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (1) weights
    torch.save(model.state_dict(), out / "pytorch_model.bin")

    # (2) config & tokenizer
    model.config.to_json_file(out / "config.json")
    tok.save_pretrained(out)

    # (3) label map + script params
    (out / "labels.json").write_text(json.dumps(lab2idx, indent=2))
    (out / "run_config.json").write_text(json.dumps(vars(a), indent=2))
    print(f"Saved BERT artefacts to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/atis/train.tsv")
    ap.add_argument("--test",  default="data/atis/test.tsv")
    ap.add_argument("--out-dir", default="models/bert")
    ap.add_argument("--epochs", type=int, default=3)
    main(ap.parse_args())

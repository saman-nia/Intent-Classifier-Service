from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split

# reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True      # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False         # type: ignore[attr-defined]

# tokeniser
class Tokeniser:
    """Whitespace splitter – swap for SentencePiece for multi-lingual bonus."""
    def encode(self, text: str) -> List[str]:
        return text.lower().split()

# dataset
class ATISDataset(Dataset):
    """
    If build_vocab=False, rows whose label is unknown in `label2idx`
    are dropped so that tensor batches stay rectangular.
    """
    def __init__(
        self,
        path: str | Path,
        tok: Tokeniser,
        vocab: dict[str, int] | None = None,
        label2idx: dict[str, int] | None = None,
        build_vocab: bool = False,
    ):
        sent_tok, labels = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                txt, lab = line.rstrip("\n").split("\t")
                sent_tok.append(tok.encode(txt))
                labels.append(lab)

        if build_vocab:                                   # TRAIN split
            cnt = Counter(tok for sent in sent_tok for tok in sent)
            self.vocab = {"<pad>": 0, "<unk>": 1}
            self.vocab.update({t: i + 2 for i, t in enumerate(sorted(cnt))})

            uniq = sorted(set(labels))
            self.label2idx = {lab: i for i, lab in enumerate(uniq)}
        else:                                             # VAL / TEST
            assert vocab is not None and label2idx is not None
            self.vocab, self.label2idx = vocab, label2idx

        unk = self.vocab["<unk>"]
        self.samples: List[Tuple[List[int], int]] = []
        for sent, lab in zip(sent_tok, labels):
            if lab not in self.label2idx:                 # skip unseen intent
                continue
            ids = [self.vocab.get(t, unk) for t in sent] or [unk]
            self.samples.append((ids, self.label2idx[lab]))

    def __len__(self) -> int:  return len(self.samples)
    def __getitem__(self, i):  return self.samples[i]

def collate_batch(batch: Sequence[Tuple[List[int], int]]):
    labels, flat, offs = [], [], [0]
    for toks, lab in batch:
        labels.append(lab)
        flat.extend(toks)
        offs.append(offs[-1] + len(toks))
    text     = torch.tensor(flat, dtype=torch.long)
    offsets  = torch.tensor(offs[:-1], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return text, offsets, labels_t

# model
class TextClassifier(nn.Module):
    def __init__(self, vocab_sz: int, embed_dim: int, n_classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, embed_dim, mode="mean")
        self.fc        = nn.Linear(embed_dim, n_classes)
        self._init()

    def _init(self):
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))

# training loop
def train(args):
    set_seed(args.seed)
    tok = Tokeniser()

    train_ds = ATISDataset(args.train, tok, build_vocab=True)
    vocab, label2idx = train_ds.vocab, train_ds.label2idx

    val_len = int(0.1 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_len, val_len])

    test_ds = ATISDataset(args.test, tok, vocab=vocab, label2idx=label2idx)

    # class-imbalance weights
    freq = defaultdict(int)
    for _, lab in train_ds: freq[lab] += 1
    weights = torch.tensor([1 / np.sqrt(freq[i]) for i in range(len(label2idx))],
                           dtype=torch.float)

    dl_train = DataLoader(train_ds, args.bs, True,  collate_fn=collate_batch)
    dl_val   = DataLoader(val_ds,   args.bs, False, collate_fn=collate_batch)
    dl_test  = DataLoader(test_ds,  args.bs, False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TextClassifier(len(vocab), args.embed_dim, len(label2idx)).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit  = nn.CrossEntropyLoss(weight=weights.to(device))

    best_state, best_val, patience = None, 0.0, args.patience

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        corr = tot = 0
        for text, offs, labs in dl_train:
            text, offs, labs = text.to(device), offs.to(device), labs.to(device)
            optim.zero_grad()
            out  = model(text, offs)
            loss = crit(out, labs)
            loss.backward()
            optim.step()

            corr += (out.argmax(1) == labs).sum().item()
            tot  += labs.size(0)
        tr_acc = corr / tot

        # validate
        val_acc = evaluate_accuracy(model, dl_val, device)
        print(f"[{epoch:02d}] train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val + 1e-4:
            best_val, best_state, patience = val_acc, model.state_dict(), args.patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test metrics
    test_acc, y_true, y_pred = evaluate(model, dl_test, device)
    print(f"Test accuracy: {test_acc:.4f}")

    present = sorted(set(y_true) | set(y_pred))
    idx2lbl = {v: k for k, v in label2idx.items()}
    names   = [idx2lbl[i] for i in present]
    print(classification_report(
        y_true, y_pred,
        labels=present,
        target_names=names,
        digits=3,
        zero_division=0
    ))

    # artefacts
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pth")
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out / "labels.json").write_text(json.dumps(label2idx, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"⭐  artefacts saved to {out.resolve()}")

# helpers
def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for txt, off, lab in loader:
            pred = model(txt.to(device), off.to(device)).argmax(1).cpu()
            corr += (pred == lab).sum().item()
            tot  += lab.size(0)
    return corr / tot if tot else 0.0

def evaluate(model, loader, device):
    model.eval()
    y_t, y_p = [], []
    with torch.no_grad():
        for txt, off, lab in loader:
            pred = model(txt.to(device), off.to(device)).argmax(1).cpu()
            y_t.extend(lab.tolist());  y_p.extend(pred.tolist())
    acc = sum(p == t for p, t in zip(y_p, y_t)) / len(y_t)
    return acc, y_t, y_p

# CLI
def parse_args():
    p = argparse.ArgumentParser("Train an ATIS intent classifier.")
    p.add_argument("--train", default="data/atis/train.tsv")
    p.add_argument("--test",  default="data/atis/test.tsv")
    p.add_argument("--out-dir",    default="models")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--bs",         type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--embed-dim",  type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=3)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    train(args)
    print(f"Finished in {(time.time() - t0):.1f}s")

# app/services/intent_classifier.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from app.services.base_classifier import BaseClassifier


# model definition
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_class)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.5
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embedding(text, offsets))


# classifier wrapper
class IntentClassifier(BaseClassifier):
    """EmbeddingBag → Linear intent classifier (mirror of train.py)."""

    def __init__(self) -> None:
        self._ready: bool = False
        self.model: nn.Module | None = None
        self.vocab: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config: Dict[str, Any] = {}

    # lifecycle
    def load(self, model_path: str, **kwargs: Any) -> None:
        """Load weights, vocabulary and metadata produced by `train.py`."""
        p = Path(model_path)
        with open(p / "vocab.json", encoding="utf-8") as fh:
            self.vocab = json.load(fh)
        with open(p / "labels.json", encoding="utf-8") as fh:
            label2idx = json.load(fh)
        self.idx2label = {int(idx): lab for lab, idx in label2idx.items()}

        # Optional hyper-param file written by the new train.py
        cfg_file = p / "config.json"
        if cfg_file.exists():
            self._config = json.loads(cfg_file.read_text())
            embed_dim = self._config.get("embed_dim", 100)
        else:
            embed_dim = 100

        self.model = TextClassificationModel(
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            num_class=len(self.idx2label),
        ).to(self.device)

        state_dict = torch.load(p / "model.pth", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._ready = True

    def close(self) -> None:
        """Release memory (especially GPU) when hot-reloading."""
        self.model = None
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    # inference
    @torch.inference_mode()
    def predict(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self._ready or self.model is None:
            raise RuntimeError("Model not loaded")

        tokens: Sequence[str] = text.lower().split()
        unk_id = self.vocab.get("<unk>", 1)
        indices = [self.vocab.get(tok, unk_id) for tok in tokens] or [unk_id]

        t_text = torch.tensor(indices, dtype=torch.long, device=self.device)
        t_offsets = torch.tensor([0], dtype=torch.long, device=self.device)

        logits = self.model(t_text, t_offsets).squeeze(0)
        probs = F.softmax(logits, dim=0)

        top = torch.topk(probs, k=min(k, probs.size(0)))
        return [
            (self.idx2label[idx], float(score))
            for idx, score in zip(top.indices.tolist(), top.values.tolist())
        ]

    def predict_batch(
        self, texts: Sequence[str], k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        return [self.predict(t, k=k) for t in texts]

    # metadata
    def labels(self) -> Sequence[str]:
        return [self.idx2label[i] for i in range(len(self.idx2label))]

    def config(self) -> Dict[str, Any]:
        return self._config

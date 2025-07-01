# app/services/bert_classifier.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from app.services.base_classifier import BaseClassifier


class BertIntentClassifier(BaseClassifier):

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tok = None
        self.idx2lab: Dict[int, str] = {}
        self._ready = False

    # lifecycle
    def load(self, model_path: str, **_: Any) -> None:
        p = Path(model_path)
        lbl_file = p / "labels.json"
        if not lbl_file.exists():
            raise FileNotFoundError("labels.json not found in %s" % p)

        # label map
        self.idx2lab = {int(v): k for k, v in json.loads(lbl_file.read_text()).items()}

        # HF objects
        cfg = AutoConfig.from_pretrained(p / "config.json")
        self.tok = AutoTokenizer.from_pretrained(p)

        self.model = AutoModelForSequenceClassification.from_config(cfg).to(self.device)
        state = torch.load(p / "pytorch_model.bin", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self._ready = True

    def is_ready(self) -> bool: return self._ready
    def close(self) -> None:     self.model = None; self._ready = False

    # inference
    @torch.inference_mode()
    def predict(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self._ready:
            raise RuntimeError("BERT model not loaded")

        enc = self.tok(text, truncation=True, max_length=128,
                       return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits.squeeze(0)          # [C]
        probs  = torch.softmax(logits, dim=0)
        top    = torch.topk(probs, k=min(k, probs.size(0)))
        return [(self.idx2lab[i], float(s))
                for i, s in zip(top.indices.tolist(), top.values.tolist())]

    def predict_batch(
        self, texts: Sequence[str], k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        return [self.predict(t, k) for t in texts]

    def labels(self) -> Sequence[str]:
        return [self.idx2lab[i] for i in range(len(self.idx2lab))]

    def config(self) -> Dict[str, Any]:
        return {"backend": "bert", "ready": self._ready}

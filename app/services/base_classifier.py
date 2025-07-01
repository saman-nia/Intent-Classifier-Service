from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple


class BaseClassifier(ABC):

    # lifecycle
    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> None:
        """Load weights, vocabulary and metadata from *model_path*."""
        raise NotImplementedError()

    @abstractmethod
    def is_ready(self) -> bool:
        """Return *True* once the model is fully initialised and usable."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Free resources (GPU memory, file handles, network sockets, …)."""
        raise NotImplementedError()

    # inference
    @abstractmethod
    def predict(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return the top-*k* `(label, confidence)` pairs for *text*."""
        raise NotImplementedError()

    # Optional batch helper; default falls back to per-item predict
    def predict_batch(
        self, texts: Sequence[str], k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        return [self.predict(t, k=k) for t in texts]

    # metadata
    def labels(self) -> Sequence[str]:
        """Full, ordered list of intent labels recognised by the model."""
        return []

    def config(self) -> Dict[str, Any]:
        """Return model hyper-parameters or other metadata for monitoring."""
        return {}

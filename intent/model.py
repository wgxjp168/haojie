"""Transformer-based intent classifier (optional backend).

HuggingFace ``transformers`` and ``torch`` are *optional* dependencies — this
module must import without them. When they are missing, ``TransformerIntentModel``
raises a clearly-typed ``ModelNotReadyError`` on first use, which the
``IntentClassifier`` then catches and falls back to the rule-based backend.

Design notes
------------
* We never download a fresh model at import time. Loading happens in
  :meth:`TransformerIntentModel.ensure_loaded`, which is driven either by
  the first request or by the ``preload_model`` setting.
* The model head is frozen to the fixed taxonomy from ``intent.taxonomy``;
  we don't attempt to re-train on boot. In practice a Part-3 deployment ships
  a fine-tuned checkpoint whose ``id2label`` matches our taxonomy.
* If the checkpoint's ``id2label`` doesn't line up with our taxonomy we
  refuse to load and log a clear error.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from intent.core.exceptions import ModelFailedError, ModelNotReadyError
from intent.core.logging import get_logger
from intent.core.metrics import INTENT_MODEL_LOAD_EVENTS
from intent.taxonomy import INTENT_IDS

log = get_logger(__name__)


@dataclass
class TransformerPrediction:
    scores: List[Tuple[str, float]]  # [(intent_id, prob), ...] descending

    @property
    def top(self) -> Tuple[str, float]:
        return self.scores[0]


class TransformerIntentModel:
    """Thin wrapper around a HuggingFace sequence-classification checkpoint."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 128,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.device_spec = device
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[Exception] = None
        self._loaded_at: Optional[float] = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self._id2label: Dict[int, str] = {}
        self._label2id: Dict[str, int] = {}

    # ---- public api ----

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[Exception]:
        return self._load_error

    def ensure_loaded(self) -> None:
        """Load model + tokenizer if not already loaded. Raises on failure."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            INTENT_MODEL_LOAD_EVENTS.labels(event="attempt").inc()
            try:
                self._load()
                self._loaded = True
                self._load_error = None
                self._loaded_at = time.time()
                INTENT_MODEL_LOAD_EVENTS.labels(event="success").inc()
                log.info(
                    "intent.model.loaded",
                    model=self.model_name_or_path,
                    device=str(self._device),
                    num_labels=len(self._id2label),
                )
            except Exception as exc:  # noqa: BLE001
                self._load_error = exc
                INTENT_MODEL_LOAD_EVENTS.labels(event="failure").inc()
                log.error(
                    "intent.model.load_failed",
                    model=self.model_name_or_path,
                    error=str(exc),
                )
                raise ModelNotReadyError(
                    f"failed to load transformer model: {exc}",
                    details={"model": self.model_name_or_path},
                ) from exc

    def predict(
        self,
        texts: Sequence[str],
        *,
        top_k: int = 3,
    ) -> List[TransformerPrediction]:
        if not texts:
            return []
        self.ensure_loaded()
        try:
            return self._predict(list(texts), top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            log.error("intent.model.predict_failed", error=str(exc))
            raise ModelFailedError(f"transformer inference failed: {exc}") from exc

    # ---- internals ----

    def _load(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers/torch are not installed; "
                "install with `pip install torch transformers` to enable the transformer backend"
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(torch)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            use_fast=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
        )
        model.to(self._device)
        model.eval()

        id2label_raw = getattr(model.config, "id2label", None) or {}
        id2label = {int(k): str(v) for k, v in id2label_raw.items()}
        if not id2label:
            raise RuntimeError(
                "model config has no id2label; finetune the checkpoint against the intent taxonomy"
            )

        taxonomy = set(INTENT_IDS)
        label_vals = set(id2label.values())
        if not label_vals.issubset(taxonomy):
            unknown = sorted(label_vals - taxonomy)
            raise RuntimeError(
                f"model labels are not a subset of the taxonomy; unknown: {unknown[:5]}"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._label2id = {v: k for k, v in id2label.items()}

    def _resolve_device(self, torch):  # type: ignore[override]
        spec = (self.device_spec or "auto").lower()
        if spec == "cpu":
            return torch.device("cpu")
        if spec.startswith("cuda"):
            return torch.device(spec if ":" in spec else "cuda")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _predict(self, texts: List[str], *, top_k: int) -> List[TransformerPrediction]:
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        torch = self._torch
        all_preds: List[TransformerPrediction] = []
        bs = max(1, self.batch_size)
        top_k = max(1, min(top_k, len(self._id2label)))

        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = texts[start : start + bs]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                for row in probs:
                    # Sort descending, keep top_k.
                    topv, topi = torch.topk(row, top_k)
                    scores: List[Tuple[str, float]] = []
                    for prob, idx in zip(topv.tolist(), topi.tolist()):
                        label = self._id2label.get(int(idx))
                        if not label:
                            continue
                        scores.append((label, float(prob)))
                    all_preds.append(TransformerPrediction(scores=scores))
        return all_preds

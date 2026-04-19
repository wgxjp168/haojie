"""Strategy abstract base class.

Every strategy consumes a ``FeatureVector`` and emits a ``StrategyOutcome``
— a strategy-agnostic envelope of ranked candidate actions plus metadata.
The orchestrator (``decision.engine.DecisionEngine``) is the only thing
that knows about HTTP, caching, or circuit breakers; strategies are pure
inference units, which keeps them testable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from decision.catalog import FALLBACK_ACTION
from decision.features import FeatureVector


@dataclass
class StrategyOutcome:
    """The strategy-agnostic result returned by every strategy."""

    candidates: List[Tuple[str, float]]  # [(action_id, score), ...] descending
    rationale: str
    strategy: str
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)

    @property
    def top_action(self) -> str:
        if not self.candidates:
            return FALLBACK_ACTION
        return self.candidates[0][0]

    @property
    def top_confidence(self) -> float:
        if not self.candidates:
            return 0.0
        return float(self.candidates[0][1])

    def head(self, n: int) -> List[Tuple[str, float]]:
        return self.candidates[: max(0, n)]


class BaseStrategy(ABC):
    """Abstract decision strategy.

    Strategies must be:
    * **synchronous** — async I/O belongs in the orchestrator.
    * **idempotent** — same input must produce the same output.
    * **side-effect-free** — no logging of PII, no metric writes (those
      live in the orchestrator).
    """

    name: str = "base"

    @property
    def ready(self) -> bool:
        """Whether the strategy is initialised and able to serve requests."""
        return True

    def warmup(self) -> None:
        """Optional eager initialisation (e.g. model load)."""
        return None

    @abstractmethod
    def evaluate(self, features: FeatureVector) -> StrategyOutcome:
        """Evaluate one feature vector."""

    def evaluate_batch(self, batch: Sequence[FeatureVector]) -> List[StrategyOutcome]:
        """Default batch implementation falls back to one-by-one evaluation."""
        return [self.evaluate(f) for f in batch]

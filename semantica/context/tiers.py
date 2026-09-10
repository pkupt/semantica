"""
Trust Tiers for Graph Facts

Retrieval results carry a relevance score but no evidence-quality signal, so a
fact corroborated by three ingested sources and a singleton extracted at low
confidence arrive at the agent looking identical. This module grades a fact
into a discrete trust tier using signals that already exist inside semantica:

    - Corroboration count: how many sources back the entity, as reported by
      ProvenanceManager.get_all_sources.
    - Extraction confidence: the value graph_builder persisted on the node via
      getattr(item, "confidence", 1.0).

Tiers are derived on demand at retrieval time and never persisted, so they stay
accurate as source sets evolve and existing storage layouts are untouched.

Algorithms Used:

    - Primary signal: corroboration count. Confidence alone cannot separate
      "0.9 because it was earned" from "0.9 because that is the schema
      default" -- extraction schemas default to 0.9 and graph_builder falls
      back to 1.0, so the stored value flatters facts with no measured
      confidence.
    - Secondary signal: extraction confidence, used to keep or downgrade the
      tier produced by corroboration.
    - Missing confidence degrades the tier instead of defaulting high, so
      absent data is never trusted as if it had been measured.

Tiers:

    - gold: multiple independent sources plus usable confidence.
    - silver: partial support, a single source or absent/low confidence.
    - bronze: thin support, a single weak piece of evidence.
    - quarantine: no corroboration and no usable confidence; needs review.

Example:
    >>> from semantica.context.tiers import TierCalculator, TrustTier
    >>> calculator = TierCalculator()
    >>> calculator.calculate(3, 0.9) is TrustTier.GOLD
    True
    >>> calculator.calculate(0, None) is TrustTier.QUARANTINE
    True
"""

from enum import Enum
from typing import Optional


class TrustTier(Enum):
    """
    Evidence-quality tier for a retrieved graph fact.

    Members are ordered from least to most trustworthy. Values are stable
    strings so a tier can be written straight into RetrievedContext.metadata
    for downstream consumers, while ordering is exposed through rank and meets
    rather than through the value type.
    """

    QUARANTINE = "quarantine"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

    @property
    def rank(self) -> int:
        """
        Return the position of this tier in the trust ordering.

        Returns:
            Integer rank: 0 for quarantine through 3 for gold.
        """
        return _TIER_RANKS[self]

    def meets(self, minimum: "TrustTier") -> bool:
        """
        Return whether this tier is at least as trustworthy as minimum.

        Args:
            minimum: Lowest acceptable tier.

        Returns:
            True when this tier ranks at or above minimum.

        Raises:
            TypeError: If minimum is not a TrustTier.

        Example:
            >>> TrustTier.GOLD.meets(TrustTier.SILVER)
            True
            >>> TrustTier.BRONZE.meets(TrustTier.SILVER)
            False
        """
        if not isinstance(minimum, TrustTier):
            raise TypeError(
                f"minimum must be a TrustTier, got {type(minimum).__name__}"
            )
        return self.rank >= minimum.rank

    def downgrade(self) -> "TrustTier":
        """
        Return the next lower tier, flooring at quarantine.

        Used when a secondary signal (extraction confidence) is missing or
        unusable, so the corroboration-based tier cannot be kept as it stands.

        Returns:
            The tier one step less trustworthy, or quarantine at the floor.

        Example:
            >>> TrustTier.GOLD.downgrade()
            <TrustTier.SILVER: 'silver'>
            >>> TrustTier.QUARANTINE.downgrade() is TrustTier.QUARANTINE
            True
        """
        if self is TrustTier.GOLD:
            return TrustTier.SILVER
        if self is TrustTier.SILVER:
            return TrustTier.BRONZE
        return TrustTier.QUARANTINE


_TIER_RANKS = {
    TrustTier.QUARANTINE: 0,
    TrustTier.BRONZE: 1,
    TrustTier.SILVER: 2,
    TrustTier.GOLD: 3,
}


class TierCalculator:
    """
    Derive a trust tier from corroboration count and extraction confidence.

    Args:
        min_corroboration_for_gold: Number of corroborating sources required
            before a fact can reach gold (default: 2).
        min_confidence: Confidence at or above which the corroboration-based
            tier is kept; below it the tier is downgraded (default: 0.7).
        missing_confidence_degrades: Downgrade one tier when no confidence
            signal is available, instead of treating absent data as high
            confidence (default: True).

    Raises:
        ValueError: If min_corroboration_for_gold is below 1, or
            min_confidence is outside [0.0, 1.0].

    Example:
        >>> TierCalculator().calculate(2, 0.8) is TrustTier.GOLD
        True
        >>> TierCalculator().calculate(2, None) is TrustTier.SILVER
        True
    """

    def __init__(
        self,
        min_corroboration_for_gold: int = 2,
        min_confidence: float = 0.7,
        missing_confidence_degrades: bool = True,
    ) -> None:
        if min_corroboration_for_gold < 1:
            raise ValueError("min_corroboration_for_gold must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0.0, 1.0]")

        self.min_corroboration_for_gold = int(min_corroboration_for_gold)
        self.min_confidence = float(min_confidence)
        self.missing_confidence_degrades = bool(missing_confidence_degrades)

    def calculate(
        self,
        corroboration_count: Optional[int],
        confidence: Optional[float] = None,
    ) -> TrustTier:
        """
        Calculate the trust tier for a single fact.

        Corroboration count sets the tier, confidence keeps or downgrades it,
        and a complete absence of both signals yields quarantine.

        Args:
            corroboration_count: Number of independent sources backing the
                fact. None, negatives and unusable values count as zero.
            confidence: Extraction confidence recorded for the fact, or None
                when no confidence signal is available.

        Returns:
            The TrustTier for the fact.

        Example:
            >>> calculator = TierCalculator()
            >>> calculator.calculate(3, 0.9) is TrustTier.GOLD
            True
            >>> calculator.calculate(1, 0.9) is TrustTier.SILVER
            True
            >>> calculator.calculate(3, None) is TrustTier.SILVER
            True
            >>> calculator.calculate(0, None) is TrustTier.QUARANTINE
            True
        """
        count = self._normalize_count(corroboration_count)
        has_usable_confidence = self._is_usable_confidence(confidence)

        if count >= self.min_corroboration_for_gold:
            tier = TrustTier.GOLD
        elif count >= 1:
            tier = TrustTier.SILVER
        elif has_usable_confidence:
            tier = TrustTier.BRONZE
        else:
            tier = TrustTier.QUARANTINE

        if confidence is None:
            if self.missing_confidence_degrades:
                return tier.downgrade()
            return tier

        if has_usable_confidence:
            return tier

        return tier.downgrade()

    @staticmethod
    def _normalize_count(corroboration_count: Optional[int]) -> int:
        """Coerce a corroboration count into a non-negative integer."""
        if corroboration_count is None:
            return 0
        try:
            count = int(corroboration_count)
        except (TypeError, ValueError):
            return 0
        return max(0, count)

    def _is_usable_confidence(self, confidence: Optional[float]) -> bool:
        """Return whether confidence is numeric and at least min_confidence."""
        if confidence is None:
            return False
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return False
        return value >= self.min_confidence

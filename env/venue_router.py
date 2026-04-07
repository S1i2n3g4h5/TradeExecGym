<<<<<<< HEAD
import random
from typing import Tuple
=======
from typing import Optional, Tuple
import numpy as np
>>>>>>> gh/feature/planning-docs

class VenueRouter:
    """Advanced Venue Router with Toxic Flow detection.

    Simulates the complex relationship between Dark Pools and Lit Venues:
<<<<<<< HEAD
    * **Dark Pool Fills:** High fill probability but fluctuates with volatility.
    * **Toxic Flow:** Over-reliance on Dark Pools (high fractions) can lead to 
      information leakage, causing litigation venue spreads to widen.
    * **Liquidity Withdrawal:** During high volatility, dark fills drop significantly
      as market makers pull their quotes.
=======
    * **Dark Pool Fills:** High fill probability (~40% base) but degrades with
      volatility as market makers pull quotes (VIX-style penalty).
    * **Toxic Flow:** Routing > 50% of an order to dark pools on a single name
      triggers information leakage — adversaries see the IOIs and front-run
      the lit leg, incurring an 8.5 bps penalty.
    * **Liquidity Withdrawal:** During high volatility (sigma > 0.02), the
      effective dark fill probability drops linearly.

    Determinism: Seeded via ``seed()`` at episode reset so that a fixed seed
    produces identical dark-pool outcomes across runs, matching the README
    claim that "all results are deterministic".
>>>>>>> gh/feature/planning-docs
    """

    def __init__(self, dark_fill_prob: float = 0.40):
        self.base_dark_fill_prob = dark_fill_prob
<<<<<<< HEAD
=======
        # Use a dedicated numpy RNG for full reproducibility — seeded per episode
        self._rng = np.random.default_rng(seed=None)

    def seed(self, seed: Optional[int]) -> None:
        """Seed the venue router RNG for deterministic dark-pool outcomes.

        Must be called during episode reset (after PriceModel.reset()) so that
        a fixed episode seed produces the same dark-pool fill sequence every run.

        Args:
            seed: Integer seed, or None for non-deterministic behaviour.
        """
        self._rng = np.random.default_rng(seed=seed)
>>>>>>> gh/feature/planning-docs

    def route_order(
        self,
        use_dark_pool: bool,
        dark_pool_fraction: float,
        shares_to_fill: int,
        current_price: float,
        volatility: float = 0.02
    ) -> Tuple[int, int, float, float, float]:
<<<<<<< HEAD
        """Route an order and return fill details."""
=======
        """Route an order across dark pool and lit venues and return fill details.

        Args:
            use_dark_pool: Whether to attempt a dark pool IOI for this order.
            dark_pool_fraction: Fraction of ``shares_to_fill`` to route dark (0–1).
            shares_to_fill: Total shares to execute this step.
            current_price: Current mid-price used as the dark fill price.
            volatility: Current GBM sigma; higher vol reduces dark fill probability.

        Returns:
            Tuple of (dark_filled, lit_filled, dark_price, lit_price, additional_slippage_bps).
              - dark_filled: Shares filled via dark pool at mid-price (zero impact).
              - lit_filled: Shares filled on NASDAQ lit venue (subject to temp impact).
              - dark_price: Dark fill price (current mid-price).
              - lit_price: Lit fill price before temp impact is applied.
              - additional_slippage_bps: Extra slippage from toxic-flow leakage (0 or 8.5 bps).
        """
>>>>>>> gh/feature/planning-docs
        # Type safety and missing value handling
        use_dark_pool = bool(use_dark_pool)
        dark_pool_fraction = float(dark_pool_fraction or 0.0)
        shares_to_fill = int(shares_to_fill or 0)
        current_price = float(current_price or 150.0)
        volatility = float(volatility or 0.02)

        dark_filled = 0
        lit_filled = shares_to_fill
        dark_price = current_price
        additional_slippage = 0.0

        if not use_dark_pool or shares_to_fill <= 0:
            return 0, shares_to_fill, current_price, current_price, 0.0

<<<<<<< HEAD
        # 1. Dynamic Fill Prob: Vix-style penalty
        vol_penalty = max(0.0, (volatility - 0.02) * 10.0) 
        effective_prob = max(0.05, self.base_dark_fill_prob - vol_penalty)

        # 2. Toxic Flow Detection
        # Routing > 50% to dark pools on a single name is often 'toxic'
        # Adversaries see the IOIs and front-run the lit leg.
        if dark_pool_fraction > 0.50:
            leakage_prob = (dark_pool_fraction - 0.5) * 0.5
            if random.random() < leakage_prob:
                additional_slippage = 8.5 # 8.5 bps penalty for information leakage

        # 3. Execution
        dark_target = int(shares_to_fill * min(1.0, dark_pool_fraction))
        if random.random() < effective_prob:
            dark_filled = dark_target
            lit_filled = shares_to_fill - dark_filled
        else:
            # Dark pool missed (unfilled IOI)
=======
        # 1. Dynamic Fill Prob: VIX-style penalty — dark liquidity evaporates during vol spikes
        vol_penalty = max(0.0, (volatility - 0.02) * 10.0)
        effective_prob = max(0.05, self.base_dark_fill_prob - vol_penalty)

        # 2. Toxic Flow Detection
        # Routing > 50% to dark pools on a single name is often 'toxic':
        # Adversaries observe the IOIs and front-run the lit leg.
        if dark_pool_fraction > 0.50:
            leakage_prob = (dark_pool_fraction - 0.5) * 0.5
            if self._rng.random() < leakage_prob:
                additional_slippage = 8.5  # 8.5 bps penalty for information leakage

        # 3. Execution — probabilistic dark pool fill (IOI accepted or missed)
        dark_target = int(shares_to_fill * min(1.0, dark_pool_fraction))
        if self._rng.random() < effective_prob:
            dark_filled = dark_target
            lit_filled = shares_to_fill - dark_filled
        else:
            # Dark pool missed — unfilled IOI falls back to lit venue
>>>>>>> gh/feature/planning-docs
            dark_filled = 0
            lit_filled = shares_to_fill

        return dark_filled, lit_filled, dark_price, current_price, additional_slippage

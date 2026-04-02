import random
from typing import Tuple

class VenueRouter:
    """Advanced Venue Router with Toxic Flow detection.

    Simulates the complex relationship between Dark Pools and Lit Venues:
    * **Dark Pool Fills:** High fill probability but fluctuates with volatility.
    * **Toxic Flow:** Over-reliance on Dark Pools (high fractions) can lead to 
      information leakage, causing litigation venue spreads to widen.
    * **Liquidity Withdrawal:** During high volatility, dark fills drop significantly
      as market makers pull their quotes.
    """

    def __init__(self, dark_fill_prob: float = 0.40):
        self.base_dark_fill_prob = dark_fill_prob

    def route_order(
        self,
        use_dark_pool: bool,
        dark_pool_fraction: float,
        shares_to_fill: int,
        current_price: float,
        volatility: float = 0.02
    ) -> Tuple[int, int, float, float, float]:
        """Route an order and return fill details."""
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
            dark_filled = 0
            lit_filled = shares_to_fill

        return dark_filled, lit_filled, dark_price, current_price, additional_slippage

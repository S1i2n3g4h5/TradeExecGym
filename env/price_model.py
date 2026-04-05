import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketState:
    price: float  # The persistent mid-market price
    volatility: float
    cumulative_participation: float
    time: float  # normalized [0,1]
    last_temp_impact_bps: float = 0.0  # Observed only for the current fill
    last_perm_impact_bps: float = 0.0  # Persistent shift added this step

class PriceModel:
    """Almgren‑Chriss price dynamics (GBM + impact).

    Parameters are set to typical default values (see project README).
    """

    def __init__(
        self,
        sigma: float = 0.02,
        lam: float = 1e-4,
        eta: float = 0.1,
        gamma: float = 0.01,
        dt: float = 1 / 780,  # one 30‑sec step in a 6.5‑hour day (780 steps)
    ):
        self.sigma = sigma
        self.lam = lam
        self.eta = eta
        self.gamma = gamma
        self.dt = dt
        self.rng: Optional[np.random.Generator] = None
        self.state: Optional[MarketState] = None

    def reset(self, initial_price: float = 150.0, seed: Optional[int] = None) -> MarketState:
        """Initialize the price process.

        Args:
            initial_price: Starting mid‑price.
            seed: Optional RNG seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.state = MarketState(
            price=initial_price,
            volatility=self.sigma,
            cumulative_participation=0.0,
            time=0.0,
        )
        return self.state

    def step(self, participation_rate: float) -> MarketState:
        """Advance one time step.

        Args:
            participation_rate: Fraction of market volume executed this step (0‑0.25).
        Returns:
            Updated MarketState.
        """
        if self.state is None:
            raise RuntimeError("PriceModel must be reset before stepping.")
            
        # 1. GBM drift (μ is set to 0 for a martingale price)
        drift = 0.0
        z = self.rng.standard_normal()
        
        # 2. Random price movement (GBM)
        # S_k = S_{k-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
        growth_factor = np.exp(
            (drift - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * z
        )
        mid_price = self.state.price * growth_factor

        # 3. Permanent impact (Almgren-Chriss)
        # This impact shifts the mid-price permanently.
        # Reference: Almgren & Chriss (2000), Equation 7 (Linear Permanent Impact)
        perm_impact_bps = self.gamma * participation_rate
        mid_price *= (1.0 + perm_impact_bps / 10_000.0)

        # 4. Temporary impact (Almgren-Chriss)
        # This impact affects only the FILL price of this step, not the mid-price.
        # Reference: Almgren & Chriss (2000), Equation 8 (Linear Temporary Impact)
        temp_impact_bps = self.eta * participation_rate

        # Update cumulative participation
        cum_part = self.state.cumulative_participation + participation_rate
        
        # Advance time
        new_time = min(1.0, self.state.time + self.dt)
        
        self.state = MarketState(
            price=mid_price,
            volatility=self.sigma,
            cumulative_participation=cum_part,
            time=new_time,
            last_temp_impact_bps=temp_impact_bps,
            last_perm_impact_bps=perm_impact_bps,
        )
        return self.state

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketState:
    price: float
    volatility: float
    cumulative_participation: float
    time: float  # normalized [0,1]
    temporary_impact: float = 0.0
    permanent_impact: float = 0.0

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
        # GBM drift (μ is set to 0 for a martingale price)
        drift = 0.0
        # Random shock
        z = self.rng.standard_normal()
        price = self.state.price * np.exp(
            (drift - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * z
        )
        # Temporary impact (linear in participation)
        tmp_impact = self.eta * participation_rate
        # Permanent impact (proportional to cumulative participation)
        perm_impact = self.gamma * self.state.cumulative_participation
        # Apply impacts to price (additive in basis points)
        price *= 1.0 + (tmp_impact + perm_impact) / 10_000
        # Update cumulative participation
        cum_part = self.state.cumulative_participation + participation_rate
        # Advance time
        new_time = min(1.0, self.state.time + self.dt)
        self.state = MarketState(
            price=price,
            volatility=self.sigma,
            cumulative_participation=cum_part,
            time=new_time,
            temporary_impact=tmp_impact,
            permanent_impact=perm_impact,
        )
        return self.state

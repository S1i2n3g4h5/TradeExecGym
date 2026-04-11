"""
Market Regime Generator for TradeExecGym.

Procedurally generates market scenarios that create infinite training variety.
Inspired by Agentic Traffic's procedural city network generation.

6 Regimes:
1. NORMAL      — baseline (GBM, typical sigma)
2. FLASH_CRASH — sudden 20% price drop, sigma spikes 5x for 3-5 steps
3. MOMENTUM    — persistent drift (trend), good for TWAP timing
4. MEAN_REVERT — price oscillates around arrival price (high-freq opportunities)
5. NEWS_SHOCK  — sudden arrival price revaluation +/- 50-150 bps mid-episode
6. LIQUIDITY_CRISIS — dark pool disabled, spreads 5x, max fill 30% of normal

Regime transitions can happen MID-EPISODE for advanced task types.
Reference: Cont (2001) "Empirical properties of asset returns: stylized facts and statistical issues"
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class Regime(str, Enum):
    NORMAL = "NORMAL"
    FLASH_CRASH = "FLASH_CRASH"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERT = "MEAN_REVERT"
    NEWS_SHOCK = "NEWS_SHOCK"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"

@dataclass
class RegimeState:
    regime: Regime
    sigma_multiplier: float = 1.0       # Volatility multiplier
    drift_bps_per_step: float = 0.0     # Directional drift
    dark_pool_disabled: bool = False    # Liquidity crisis
    spread_multiplier: float = 1.0      # Bid-ask spread expansion
    fill_rate_penalty: float = 0.0      # Dark fill probability reduction
    news_shock_bps: float = 0.0         # One-time arrival price shock
    steps_remaining: int = 999          # Steps until regime expires

    def to_market_text(self) -> str:
        """Human-readable regime alert for LLM observation."""
        if self.regime == Regime.NORMAL:
            return "Market Regime: NORMAL — Standard conditions."
        elif self.regime == Regime.FLASH_CRASH:
            return f"[!] REGIME: FLASH CRASH — Volatility {self.sigma_multiplier:.0f}x normal! Spreads wide. Darks offline."
        elif self.regime == Regime.MOMENTUM:
            d = "UP" if self.drift_bps_per_step > 0 else "DOWN"
            return f"Market Regime: MOMENTUM {d} — Price trending {abs(self.drift_bps_per_step):.1f} bps/step. Trade WITH the trend."
        elif self.regime == Regime.MEAN_REVERT:
            return "Market Regime: MEAN REVERSION — Price oscillating. Best to accumulate at dips."
        elif self.regime == Regime.NEWS_SHOCK:
            return f"[!] REGIME: NEWS SHOCK — Arrival price revaluation: {self.news_shock_bps:+.1f} bps. Benchmark reset."  
        elif self.regime == Regime.LIQUIDITY_CRISIS:
            return f"[!] REGIME: LIQUIDITY CRISIS — Dark pool OFFLINE. Spreads {self.spread_multiplier:.0f}x. Reduce aggression!"
        return f"Regime: {self.regime}"


class MarketRegimeGenerator:
    """
    Generates and manages market regime transitions.
    Called by TradeExecEnvironment.step() to apply regime effects.
    """
    
    REGIME_PROBABILITIES = {
        Regime.NORMAL: 0.50,
        Regime.MOMENTUM: 0.20,
        Regime.MEAN_REVERT: 0.12,
        Regime.FLASH_CRASH: 0.06,
        Regime.LIQUIDITY_CRISIS: 0.07,
        Regime.NEWS_SHOCK: 0.05,
    }
    
    def __init__(self, allow_regimes: bool = True):
        self.allow_regimes = allow_regimes
        self._rng: Optional[np.random.Generator] = None
        self.current_regime = RegimeState(regime=Regime.NORMAL)
        self._regime_history: List[Regime] = []
    
    def seed(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)
        self.current_regime = RegimeState(regime=Regime.NORMAL)
        self._regime_history = []
    
    def step(self, step_count: int, max_steps: int) -> RegimeState:
        """
        Called each step. May transition to a new regime.
        Returns the current (possibly new) regime state.
        """
        if not self.allow_regimes:
            return self.current_regime
        
        rng = self._rng or np.random.default_rng()
        
        # Decrement current regime lifetime
        self.current_regime.steps_remaining -= 1
        
        # Regime transition probability increases when current regime expires
        transition_prob = 0.0
        if self.current_regime.steps_remaining <= 0:
            transition_prob = 0.80  # High chance to switch after expiry
        elif step_count > max_steps * 0.4:
            transition_prob = 0.04  # Small chance mid-episode
        
        if rng.random() < transition_prob:
            self.current_regime = self._sample_regime(rng, step_count, max_steps)
            self._regime_history.append(self.current_regime.regime)
        
        return self.current_regime
    
    def _sample_regime(self, rng, step_count: int, max_steps: int) -> RegimeState:
        """Sample a new regime state."""
        # Don't trigger news shock in last 20% of episode
        probs = dict(self.REGIME_PROBABILITIES)
        if step_count > max_steps * 0.80:
            probs[Regime.NEWS_SHOCK] = 0.0
            probs[Regime.FLASH_CRASH] = 0.0  # Too late for crash
            # Normalize
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
        
        regimes = [r.value for r in probs.keys()]
        weights = list(probs.values())
        chosen_val = rng.choice(regimes, p=weights)
        chosen = Regime(chosen_val)
        
        if chosen == Regime.NORMAL:
            return RegimeState(regime=Regime.NORMAL, steps_remaining=rng.integers(5, 20))
        
        elif chosen == Regime.FLASH_CRASH:
            duration = int(rng.integers(3, 7))
            return RegimeState(
                regime=Regime.FLASH_CRASH,
                sigma_multiplier=rng.uniform(3.0, 6.0),
                dark_pool_disabled=True,
                spread_multiplier=rng.uniform(3.0, 8.0),
                steps_remaining=duration,
            )
        
        elif chosen == Regime.MOMENTUM:
            direction = rng.choice([-1, 1])
            return RegimeState(
                regime=Regime.MOMENTUM,
                drift_bps_per_step=direction * rng.uniform(0.5, 3.0),
                sigma_multiplier=0.8,  # Lower vol in trending markets
                steps_remaining=rng.integers(8, 20),
            )
        
        elif chosen == Regime.MEAN_REVERT:
            return RegimeState(
                regime=Regime.MEAN_REVERT,
                sigma_multiplier=1.2,
                drift_bps_per_step=0.0,  # Oscillates (handled in price model)
                steps_remaining=rng.integers(6, 15),
            )
        
        elif chosen == Regime.NEWS_SHOCK:
            shock = rng.uniform(-150, 150) * rng.choice([-1, 1])
            return RegimeState(
                regime=Regime.NEWS_SHOCK,
                news_shock_bps=shock,
                sigma_multiplier=rng.uniform(1.5, 3.0),
                steps_remaining=1,  # One-shot event
            )
        
        elif chosen == Regime.LIQUIDITY_CRISIS:
            return RegimeState(
                regime=Regime.LIQUIDITY_CRISIS,
                dark_pool_disabled=True,
                spread_multiplier=rng.uniform(4.0, 10.0),
                fill_rate_penalty=0.8,  # Only 20% of normal fills
                sigma_multiplier=rng.uniform(1.5, 3.0),
                steps_remaining=rng.integers(4, 12),
            )
        
        return RegimeState(regime=Regime.NORMAL)
    
    @property
    def regime_history_text(self) -> str:
        if not self._regime_history:
            return "No regime transitions this episode."
        counts = {}
        for r in self._regime_history:
            counts[r.value] = counts.get(r.value, 0) + 1
        return "Regime events: " + ", ".join(f"{r}×{c}" for r, c in counts.items())

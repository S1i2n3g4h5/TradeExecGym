"""
Synthetic L2 Order Book for TradeExecGym.

Simulates a realistic limit order book with:
- 10 levels of bid/ask quotes
- Dynamic spread based on volatility
- Queue depth and hidden liquidity
- Iceberg order detection
- Partial fill probability per level

This adds a fundamentally new dimension to the observation space:
Instead of just price + participation rate, the LLM can now reason about
WHERE in the book its order sits and HOW FAST it will fill.

Reference: Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class OrderBookLevel:
    price: float      # Price of this level
    size: int         # Number of shares at this level
    is_iceberg: bool  # True if hidden iceberg order behind this quote

@dataclass
class OrderBookSnapshot:
    """Full L2 snapshot — 10 bid + 10 ask levels."""
    bids: List[OrderBookLevel]   # Sorted descending (best bid first)
    asks: List[OrderBookLevel]   # Sorted ascending (best ask first)
    mid_price: float
    spread_bps: float
    bid_depth: int  # Total shares in top 5 bid levels
    ask_depth: int  # Total shares in top 5 ask levels
    imbalance: float  # (bid_depth - ask_depth) / (bid_depth + ask_depth), [-1, 1]
    
    def to_text(self) -> str:
        """Render L2 book as readable text for LLM consumption."""
        lines = ["ORDER BOOK (L2) — Top 5 Levels"]
        lines.append(f"  Spread: {self.spread_bps:.1f} bps | Imbalance: {self.imbalance:+.2f}")
        lines.append(f"  {'ASK':<10} {'SIZE':>10}")
        for level in reversed(self.asks[:5]):
            lines.append(f"  ${level.price:<9.4f} {level.size:>10,}{'  [ICE]' if level.is_iceberg else ''}")
        lines.append(f"  {'---MID---'} ${self.mid_price:.4f}")
        for level in self.bids[:5]:
            lines.append(f"  ${level.price:<9.4f} {level.size:>10,}{'  [ICE]' if level.is_iceberg else ''}")
        lines.append(f"  BID DEPTH: {self.bid_depth:,} | ASK DEPTH: {self.ask_depth:,}")
        return "\n".join(lines)
    
    def estimate_fill_cost(self, shares: int, side: str = "buy") -> Tuple[float, float]:
        """
        Walk the book: estimate execution price and market impact for given shares.
        Returns (avg_price, impact_bps).
        """
        levels = self.asks if side == "buy" else self.bids
        remaining = shares
        total_cost = 0.0
        
        for level in levels:
            if remaining <= 0:
                break
            fill_at_level = min(remaining, level.size)
            total_cost += fill_at_level * level.price
            remaining -= fill_at_level
        
        if shares > 0 and total_cost > 0:
            avg_price = total_cost / (shares - remaining)
            impact_bps = abs(avg_price - self.mid_price) / self.mid_price * 10_000
            return avg_price, impact_bps
        return self.mid_price, 0.0


class OrderBookSimulator:
    """
    Generates realistic L2 snapshots based on current market conditions.
    
    Parameters are calibrated to typical NASDAQ equity microstructure.
    """
    
    def __init__(self, levels: int = 10, tick_size: float = 0.01):
        self.levels = levels
        self.tick_size = tick_size
        self._rng: Optional[np.random.Generator] = None
    
    def seed(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)
    
    def generate(
        self,
        mid_price: float,
        volatility: float,
        participation_rate: float = 0.05,
        volume_ratio: float = 1.0,
        session: str = "midday",
    ) -> OrderBookSnapshot:
        """Generate a synthetic L2 snapshot for current market conditions."""
        rng = self._rng or np.random.default_rng()
        
        # 1. Dynamic spread (wider in volatile/low-volume markets)
        base_spread_bps = 5.0
        vol_spread = volatility / 0.02 * 2.0  # Volatility markup
        time_spread = 1.0 / volume_ratio       # Inverse vol ratio
        spread_bps = base_spread_bps + vol_spread + rng.uniform(-0.5, 0.5)
        spread_bps = min(50.0, max(2.0, spread_bps))
        half_spread = (mid_price * spread_bps / 10_000) / 2
        
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # 2. Generate bid levels (decreasing price, varying sizes)
        bids = []
        for i in range(self.levels):
            price = round(best_bid - i * self.tick_size, 4)
            # Size: larger near best price, smaller further out
            base_size = int(rng.lognormal(mean=7.5, sigma=0.8))  # ~1800 shares avg
            depth_decay = max(0.2, 1.0 - i * 0.08)
            size = int(base_size * depth_decay * volume_ratio)
            # 15% chance of iceberg order at key levels
            is_iceberg = rng.random() < 0.15 and i < 4
            if is_iceberg:
                size = int(size * rng.uniform(3, 8))  # Hidden reserve
            bids.append(OrderBookLevel(price=price, size=max(1, size), is_iceberg=is_iceberg))
        
        # 3. Generate ask levels (increasing price)
        asks = []
        for i in range(self.levels):
            price = round(best_ask + i * self.tick_size, 4)
            base_size = int(rng.lognormal(mean=7.5, sigma=0.8))
            depth_decay = max(0.2, 1.0 - i * 0.08)
            size = int(base_size * depth_decay * volume_ratio)
            is_iceberg = rng.random() < 0.15 and i < 4
            if is_iceberg:
                size = int(size * rng.uniform(3, 8))
            asks.append(OrderBookLevel(price=price, size=max(1, size), is_iceberg=is_iceberg))
        
        # 4. Queue imbalance (bullish vs bearish pressure)
        bid_depth = sum(l.size for l in bids[:5])
        ask_depth = sum(l.size for l in asks[:5])
        imbalance = (bid_depth - ask_depth) / max(1, bid_depth + ask_depth)
        
        return OrderBookSnapshot(
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread_bps=spread_bps,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance=imbalance,
        )

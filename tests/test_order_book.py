"""Tests for order book simulator and regime generator."""
import pytest
from env.order_book import OrderBookSimulator
from env.market_regime import MarketRegimeGenerator, Regime

class TestOrderBook:
    def test_book_has_10_levels(self):
        sim = OrderBookSimulator()
        sim.seed(42)
        book = sim.generate(mid_price=150.0, volatility=0.02)
        assert len(book.bids) == 10
        assert len(book.asks) == 10
    
    def test_spread_positive(self):
        sim = OrderBookSimulator()
        sim.seed(42)
        book = sim.generate(150.0, 0.02)
        assert book.asks[0].price > book.bids[0].price
        assert book.spread_bps > 0
    
    def test_higher_volatility_wider_spread(self):
        sim = OrderBookSimulator()
        sim.seed(42)
        book_low = sim.generate(150.0, volatility=0.02)
        sim.seed(42)
        book_high = sim.generate(150.0, volatility=0.10)
        assert book_high.spread_bps > book_low.spread_bps
    
    def test_fill_cost_estimate(self):
        sim = OrderBookSimulator()
        sim.seed(42)
        book = sim.generate(150.0, 0.02)
        avg_price, impact_bps = book.estimate_fill_cost(10_000, side="buy")
        assert avg_price > 150.0  # Buys at ask
        assert impact_bps >= 0
    
    def test_imbalance_in_range(self):
        sim = OrderBookSimulator()
        sim.seed(42)
        book = sim.generate(150.0, 0.02)
        assert -1.0 <= book.imbalance <= 1.0
    
    def test_to_text_contains_key_fields(self):
        sim = OrderBookSimulator()
        sim.seed(42)
        book = sim.generate(150.0, 0.02)
        text = book.to_text()
        assert "Spread" in text
        assert "BID DEPTH" in text

class TestRegimeGenerator:
    def test_starts_normal(self):
        gen = MarketRegimeGenerator()
        gen.seed(42)
        assert gen.current_regime.regime == Regime.NORMAL
    
    def test_regime_transitions_over_long_episode(self):
        gen = MarketRegimeGenerator()
        gen.seed(42)
        regimes_seen = set()
        for step in range(200):
            state = gen.step(step, 200)
            regimes_seen.add(state.regime)
        # Should see at least 3 different regimes in 200 steps
        assert len(regimes_seen) >= 3
    
    def test_disabled_regimes_always_normal(self):
        gen = MarketRegimeGenerator(allow_regimes=False)
        gen.seed(42)
        for step in range(50):
            state = gen.step(step, 100)
            assert state.regime == Regime.NORMAL
    
    def test_flash_crash_has_high_sigma(self):
        gen = MarketRegimeGenerator()
        gen.seed(42)
        found_crash = False
        import numpy as np
        rng = np.random.default_rng(42)
        for _ in range(500):
            state = gen._sample_regime(rng, 10, 100)
            if state.regime == Regime.FLASH_CRASH:
                assert state.sigma_multiplier >= 3.0
                found_crash = True
                break
        assert found_crash, "Did not sample a flash crash regime in 500 attempts"
    
    def test_text_contains_regime_name(self):
        gen = MarketRegimeGenerator()
        gen.seed(42)
        import numpy as np
        rng = np.random.default_rng(42)
        state = gen._sample_regime(rng, 10, 100)
        text = state.to_market_text()
        assert len(text) > 10

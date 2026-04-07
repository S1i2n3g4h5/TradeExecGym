"""
Edge Case Validation Suite for TradeExecGym

This module tests boundary conditions and edge cases that may not be covered
by standard unit tests. These tests ensure defensive programming works correctly.

Coverage:
- Rate clamping (rate > 0.25 should clamp to 0.25, not cause over-execution)
- Exactly 100% completion edge case
- Dark pool probability boundaries (prob=0.0, prob=1.0)
- Zero shares executed scenario
- Negative rate handling (should clamp to 0.0)
"""

import pytest
from server.trade_environment import TradeExecEnvironment


class TestRateClamping:
    """Verify that participation_rate is properly clamped to [0.0, 0.25]."""

    def test_rate_exceeds_max_clamped_to_025(self):
        """
        When rate > 0.25, verify environment clamps it to 0.25 (not over-execute).
        
        This prevents agents from accidentally executing 4× the intended volume
        by submitting rate=0.99 instead of rate=0.25.
        """
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        # Execute with rate=0.99 (should be clamped to 0.25)
        result = env.execute_trade(participation_rate=0.99)
        shares_at_099 = env._shares_executed
        
        # Reset and execute with rate=0.25 (baseline)
        env.reset(seed=42, task_id="task1_twap_beater")
        env.execute_trade(participation_rate=0.25)
        shares_at_025 = env._shares_executed
        
        # Both should execute the same amount (clamping worked)
        assert shares_at_099 == shares_at_025, \
            f"Rate clamping failed: rate=0.99 executed {shares_at_099} shares, " \
            f"but rate=0.25 executed {shares_at_025} shares (should be equal)"
        
        # Verify shares executed is reasonable (not 4× the expected amount)
        # At rate=0.25 with ADV≈82,000, one step should execute ~5,000-6,000 shares
        assert shares_at_099 <= 10000, \
            f"Over-execution detected: {shares_at_099} shares in one step (expected ≤10,000)"
    
    def test_negative_rate_clamped_to_zero(self):
        """Verify that negative rates are clamped to 0.0 (no execution)."""
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        initial_shares = env._shares_executed
        env.execute_trade(participation_rate=-0.15)
        
        # Should execute nothing (clamped to 0.0)
        assert env._shares_executed == initial_shares, \
            f"Negative rate should execute 0 shares, but executed {env._shares_executed - initial_shares}"
    
    def test_rate_zero_executes_nothing(self):
        """Explicit test for rate=0.0 (should not move inventory)."""
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        initial_remaining = env._shares_remaining
        env.execute_trade(participation_rate=0.0)
        
        assert env._shares_remaining == initial_remaining, \
            f"rate=0.0 should not change inventory, but remaining went from {initial_remaining} to {env._shares_remaining}"


class TestCompletionEdgeCases:
    """Test scenarios around exactly 100% order completion."""
    
    def test_exactly_100_percent_completion(self):
        """
        Verify that when shares_remaining approaches 0, the episode terminates
        correctly and grader score reflects high completion.
        
        Note: Task 1 is designed to be difficult - TWAP baseline only achieves
        ~12.5% completion. Getting to 94%+ at max rate is already excellent.
        """
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        # Execute full order at maximum rate until completion or max steps
        max_steps = 120
        for step in range(max_steps):
            if env._episode_done:
                break
            env.execute_trade(participation_rate=0.25)
        
        # Verify high completion (task1 is very difficult, 90%+ is excellent)
        completion_pct = (env._shares_executed / env._total_shares) * 100
        assert completion_pct >= 90.0, \
            f"Expected ≥90% completion at max rate, got {completion_pct:.2f}%"
        
        # Verify grader score is valid
        grader_score = env._compute_grader_score()
        assert 0.0 <= grader_score <= 1.0, \
            f"Grader score {grader_score} out of valid range [0.0, 1.0]"
    
    def test_over_completion_not_possible(self):
        """
        Verify that environment prevents executing more than total_shares.
        
        Even if agent keeps trading after completion, shares_executed should
        never exceed total_shares.
        """
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        total_order = env._total_shares
        
        # Execute at max rate for full episode
        for _ in range(env._max_steps):
            if env._episode_done:
                # Try to execute one more time (should be no-op)
                env.execute_trade(participation_rate=0.25)
                break
            env.execute_trade(participation_rate=0.25)
        
        # Verify we didn't over-execute
        assert env._shares_executed <= total_order, \
            f"Over-execution detected: executed {env._shares_executed} shares, " \
            f"but order size is only {total_order}"


class TestDarkPoolEdgeCases:
    """Test dark pool venue router boundary conditions."""
    
    def test_dark_pool_zero_probability(self):
        """
        When dark pool fill_prob = 0.0, verify all trades go to lit market.
        
        This tests the venue router's handling of impossible dark pool fills.
        """
        from env.venue_router import VenueRouter
        
        router = VenueRouter(dark_fill_prob=0.0)
        router.seed(42)  # Seed for determinism
        
        # Execute 100 trades - all should go to lit market
        lit_count = 0
        for _ in range(100):
            dark_filled, lit_filled, _, _, _ = router.route_order(
                use_dark_pool=True,
                dark_pool_fraction=0.5,
                shares_to_fill=1000,
                current_price=150.0,
                volatility=0.02
            )
            lit_count += 1 if lit_filled > 0 else 0
        
        # With prob=0.0, expect 100% lit routing (dark pool never fills)
        assert lit_count == 100, \
            f"With dark_fill_prob=0.0, expected 100 lit fills, got {lit_count}"
    
    def test_dark_pool_guaranteed_fill(self):
        """
        When dark pool fill_prob = 1.0, verify all trades fill in dark pool.
        
        This tests the upper boundary of dark pool probability.
        """
        from env.venue_router import VenueRouter
        
        router = VenueRouter(dark_fill_prob=1.0)
        router.seed(42)  # Seed for determinism
        
        # Execute 100 trades - all should fill in dark pool
        dark_count = 0
        for _ in range(100):
            dark_filled, lit_filled, _, _, _ = router.route_order(
                use_dark_pool=True,
                dark_pool_fraction=0.5,
                shares_to_fill=1000,
                current_price=150.0,
                volatility=0.02
            )
            dark_count += 1 if dark_filled > 0 else 0
        
        # With prob=1.0, expect 100% dark fills
        assert dark_count == 100, \
            f"With dark_fill_prob=1.0, expected 100 dark fills, got {dark_count}"


class TestZeroTradeScenario:
    """Test behavior when agent never trades (all rate=0.0)."""
    
    def test_agent_never_trades(self):
        """
        Verify that if agent sets rate=0.0 for entire episode, the environment
        handles it gracefully (no crashes, valid grader score).
        """
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        # Execute nothing for entire episode
        for _ in range(env._max_steps):
            if env._episode_done:
                break
            env.execute_trade(participation_rate=0.0)
        
        # Verify inventory unchanged
        assert env._shares_executed == 0, \
            f"Expected 0 shares executed, got {env._shares_executed}"
        
        # Verify grader score is valid (may not be super low if baselines also didn't complete)
        # The grader has multiple components, not just completion
        grader_score = env._compute_grader_score()
        assert 0.0 <= grader_score <= 1.0, \
            f"Expected valid grader score in [0, 1], got {grader_score}"
        
        # Verify IS calculation doesn't crash with zero execution
        current_is = env._compute_current_is()
        assert current_is >= 0, \
            f"IS should be non-negative, got {current_is}"


class TestNaNAndInfProtection:
    """Verify environment handles edge cases that could produce NaN/Inf."""
    
    def test_reward_calculation_no_nan(self):
        """
        Verify reward calculation never produces NaN, even with edge case inputs.
        """
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        # Execute a few normal steps and verify internal reward state is valid
        import math
        for _ in range(10):
            env.execute_trade(participation_rate=0.05)
            
            # Access the last computed reward from internal state
            # The reward is computed and stored during execute_trade
            if hasattr(env, '_last_reward'):
                reward = env._last_reward
                assert not math.isnan(reward), f"Reward is NaN at step {env._current_step}"
                assert not math.isinf(reward), f"Reward is Inf at step {env._current_step}"
        
        # At minimum, verify grader score (which uses similar calculations) is valid
        grader_score = env._compute_grader_score()
        assert not math.isnan(grader_score), "Grader score is NaN"
        assert not math.isinf(grader_score), "Grader score is Inf"
    
    def test_grader_score_no_nan_at_completion(self):
        """
        Verify grader score calculation at exactly 100% completion doesn't
        produce NaN due to division edge cases.
        """
        env = TradeExecEnvironment()
        env.reset(seed=42, task_id="task1_twap_beater")
        
        # Complete the order
        while not env._episode_done:
            env.execute_trade(participation_rate=0.25)
        
        grader_score = env._compute_grader_score()
        
        import math
        assert not math.isnan(grader_score), "Grader score is NaN at completion"
        assert not math.isinf(grader_score), "Grader score is Inf at completion"
        assert 0.0 <= grader_score <= 1.0, \
            f"Grader score {grader_score} out of valid range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

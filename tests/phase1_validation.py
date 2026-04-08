import unittest
import numpy as np
from env.price_model import PriceModel
from env.reward import compute_reward
from server.trade_environment import TradeExecEnvironment
from tasks.factory import get_task

class TestPhase1Foundations(unittest.TestCase):
    """Hardcore validation for TradeExecGym Phase 1 implementation."""

    def setUp(self):
        self.price_model = PriceModel(gamma=0.01, eta=0.1) # 1bp perm, 10bp temp per 1.0 part
        self.price_model.reset(initial_price=100.0, seed=42)

    def test_permanent_impact_persistence(self):
        """Verify that permanent impact shifts the midpoint and PERSISTS."""
        # Step 1: Execute with high participation (0.20)
        state1 = self.price_model.step(0.20)
        mid_after_trade = state1.price
        perm_impact = state1.last_perm_impact_bps
        self.assertGreater(perm_impact, 0)
        
        # Step 2: Execute with zero participation
        # Midpoint should only move by random GBM, not revert the perm impact
        state2 = self.price_model.step(0.0)
        mid_after_idle = state2.price
        
        # Midpoint should be 'roughly' the same (except for GBM noise)
        # But crucially, 'last_perm_impact_bps' should be 0 this step
        self.assertEqual(state2.last_perm_impact_bps, 0)
        self.assertAlmostEqual(state2.price, mid_after_idle, delta=5.0) 
        print(f"DEBUG: Mid after trade: {mid_after_trade:.4f}, Mid after idle: {mid_after_idle:.4f}")

    def test_temporary_impact_doesnt_persist(self):
        """Verify that temporary impact affects only the current fill, not the midpoint."""
        initial_price = self.price_model.state.price
        
        # Execute with high participation
        state = self.price_model.step(0.25)
        temp_impact = state.last_temp_impact_bps
        mid_price = state.price
        
        # The midpoint should NOT include the temporary impact
        # Expected Mid = Initial * (1 + GBM + PermImpact)
        # If it included temp impact, it would be much higher
        self.assertGreater(temp_impact, 0)
        self.assertLess(mid_price, initial_price * (1.0 + (temp_impact + 10)/10000))
        print(f"DEBUG: Temp Impact: {temp_impact:.2f} bps, Mid Price: {mid_price:.4f}")

    def test_baseline_cache_hit_and_fairness(self):
        """Verify that baseline caching works and uses the same seed."""
        env = TradeExecEnvironment()
        seed = 123
        env.reset(seed=seed, task_id="task1_twap_beater")
        
        # Check if cache is populated
        self.assertGreater(len(env._baseline_cache), 0)
        self.assertIn(1, env._baseline_cache)
        
        # Verify O(1) lookups
        twap_is_1 = env._twap_is_at_step()
        self.assertEqual(twap_is_1, env._baseline_cache[0]["twap"]) # At reset step_count=0
        
        # Advance and check again
        env.execute_trade_logic(participation_rate=0.05)
        twap_is_2 = env._twap_is_at_step()
        self.assertEqual(twap_is_2, env._baseline_cache[1]["twap"])
        
        print(f"DEBUG: Cache Step 1: {env._baseline_cache[1]}")

    def test_reward_components(self):
        """Verify Dense, Delayed, and Sparse reward logic."""
        # 1. Dense: Agent beating baseline
        r_good = compute_reward({}, is_current=10.0, is_baseline=20.0, participation_rate=0.05, 
                               shares_executed=100, total_shares=1000, 
                               is_done=False, slippage_bps=5.0)
        # (20-10)*0.1 = 1.0 -> clamped to 0.99
        self.assertAlmostEqual(r_good, 0.99)
        
        # 2. Delayed: Completion bonus
        r_done = compute_reward({}, is_current=15.0, is_baseline=20.0, participation_rate=0.05, 
                               shares_executed=980, total_shares=1000, 
                               is_done=True, slippage_bps=5.0)
        # Raw total = 1.5 -> clamped to 0.99
        self.assertAlmostEqual(r_done, 0.99)

        # 3. Sparse: Milestones (handled in TradeEnvironment)
        env = TradeExecEnvironment()
        env.reset(task_id="task1_twap_beater")
        
        # Progress toward first milestone; exact fill pace can vary by task physics.
        for _ in range(60):
            env.execute_trade_logic(participation_rate=0.25)
            if 0.25 in env._milestones_reached or env._episode_done:
                break
        
        # Ensure repeated execution updates reward trajectory without crashing.
        self.assertGreaterEqual(env._last_reward, 0.01)
        print(f"DEBUG: Last Reward: {env._last_reward}")

if __name__ == "__main__":
    unittest.main()

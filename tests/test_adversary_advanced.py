import unittest
import numpy as np
from tasks.task4_adversary import TaskAdversary

class TestAdversaryAdvanced(unittest.TestCase):
    def setUp(self):
        self.task = TaskAdversary()
        self.task._episode_seed = 123

    def test_uniform_penalty(self):
        """Test that constant rates trigger the uniformity detector."""
        penalty = 0
        for i in range(1, 10):
            penalty = self.task.on_trade_step(i, 0.10, 150.0, 0, 600000)
            if i >= 5:
                # Penalty should be roughly 15 +/- 5 bps
                self.assertGreater(penalty, 9.0)
                self.assertLess(penalty, 21.0)

    def test_alternating_penalty(self):
        """Test that alternating rates trigger the periodicity (autocorrelation) detector."""
        penalty = 0
        # Rates: [0.05, 0.15, 0.05, 0.15, 0.05]
        # This has high std_dev (~0.05) but perfect negative autocorrelation
        rates = [0.05, 0.15, 0.05, 0.15, 0.05, 0.15]
        for i, rate in enumerate(rates):
            penalty = self.task.on_trade_step(i+1, rate, 150.0, 0, 600000)
            if i+1 >= 5:
                self.assertGreater(penalty, 9.0, f"Failed to detect periodicity at step {i+1}")

    def test_random_jitter_evasion(self):
        """Test that random rates evade both detectors."""
        # Use a high-entropy sequence
        rates = [0.05, 0.12, 0.08, 0.21, 0.14, 0.06, 0.19, 0.11]
        for i, rate in enumerate(rates):
            penalty = self.task.on_trade_step(i+1, rate, 150.0, 0, 600000)
            # Should have no penalty if jitter is high enough
            self.assertEqual(penalty, 0.0, f"False positive detection at step {i+1} with rate {rate}")

    def test_reset_isolation(self):
        """Test that history is cleared between episodes."""
        # Step 1-5: Uniform
        for i in range(1, 6):
            self.task.on_trade_step(i, 0.10, 150.0, 0, 600000)
        
        self.task.reset()
        self.assertEqual(len(self.task.participation_history), 0)
        
        # Step 1 of new episode: Should NOT have penalty even if rate is same
        penalty = self.task.on_trade_step(1, 0.10, 150.0, 0, 600000)
        self.assertEqual(penalty, 0.0)

    def test_deterministic_penalty(self):
        """Test that the same seed + step results in the same penalty."""
        # Run 1
        self.task.reset()
        self.task._episode_seed = 777
        for i in range(1, 6):
            p1 = self.task.on_trade_step(i, 0.10, 150.0, 0, 600000)
        
        # Run 2
        task2 = TaskAdversary()
        task2._episode_seed = 777
        for i in range(1, 6):
            p2 = task2.on_trade_step(i, 0.10, 150.0, 0, 600000)
            
        self.assertEqual(p1, p2, "Penalties with same seed+step should be identical")

if __name__ == "__main__":
    unittest.main()

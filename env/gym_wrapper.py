import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
from client import TradeExecClient

class TradeExecGymEnv(gym.Env):
    """
    Gymnasium wrapper for TradeExecGym MCP environment.
    Connects to the server via TradeExecClient.
    """
    def __init__(self, base_url="http://localhost:7860", task_id="task1_twap_beater"):
        super().__init__()
        self.client = TradeExecClient(base_url=base_url)
        self.task_id = task_id
        
        # Action space: Participation rate [0, 0.25]
        self.action_space = spaces.Box(low=0.0, high=0.25, shape=(1,), dtype=np.float32)
        
        # Observation space: [Price, Progress%, Remaining%, Volatility]
        # We normalize these for better convergence.
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(4,), dtype=np.float32
        )
        
        self.loop = asyncio.get_event_loop()
        self._last_obs = None

    def _get_obs_vector(self, obs):
        """Extract and normalize observation vector from server output."""
        # This is a bit brittle as it depends on the prompt/metadata.
        # In Phase 1 we ensured metadata contains total_shares and max_steps.
        meta = obs.metadata if hasattr(obs, 'metadata') else {}
        
        price_norm = 1.0 # Reference price
        progress = 0.0
        remaining = 1.0
        vol = meta.get("sigma", 0.02) * 10.0 # scale sig for obs
        
        # We need to reach into the environment state properly.
        # Assuming the server returns these in metadata in later steps.
        # For now, let's use placeholders if not present.
        return np.array([price_norm, progress, remaining, vol], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start new episode
        obs = self.loop.run_until_complete(self.client.reset(task_id=self.task_id))
        self._last_obs = obs
        
        info = {}
        return self._get_obs_vector(obs), info

    def step(self, action):
        rate = float(action[0])
        
        # Execute trade
        result = self.loop.run_until_complete(self.client.execute_trade(
            participation_rate=rate,
            use_dark_pool=False,
            dark_pool_fraction=0.0
        ))
        
        # Fetch per-step reward
        reward = self.loop.run_until_complete(self.client.get_reward())
        
        # Check termination
        done = "EPISODE COMPLETE" in result
        truncated = False
        
        # Generate new obs vector
        # (Since execute_trade returns a string, we might need a better 
        # way to get the latest state. We'll call get_market_state if needed.)
        # For now, we'll keep it simple.
        obs_vec = self._get_obs_vector(self._last_obs) 
        
        return obs_vec, reward, done, truncated, {}

    def close(self):
        pass

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
    def __init__(self, base_url="http://localhost:7865", task_id="task1_twap_beater"):
        super().__init__()
        self.client = TradeExecClient(base_url=base_url)
        self.task_id = task_id
        
        # Action space: Participation rate [0, 0.25]
        self.action_space = spaces.Box(low=0.0, high=0.25, shape=(1,), dtype=np.float32)
        
        # Observation space: [PriceNorm, Progress%, Remaining%, Vol/VolRatio, CurrentIS]
        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(5,), dtype=np.float32
        )
        
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def _get_obs_vector(self, result_text):
        """Extract and normalize observation vector from server output."""
        # Default: [Price, Progress, Remaining, Vol, IS]
        obs = np.array([1.0, 0.0, 1.0, 0.5, 0.0], dtype=np.float32)
        
        if not result_text or not isinstance(result_text, str):
            return obs

        try:
            # 1. Implementation Shortfall (bps)
            if "Your IS:" in result_text:
                val = result_text.split("Your IS:")[1].split("bps")[0].strip()
                obs[4] = float(val) / 50.0 # 50 bps = 1.0
            
            # 2. Inventory Progress
            # 2. Inventory Metrics
            if "Executed:" in result_text and "/" in result_text:
                parts = result_text.split("Executed:")[1].split("\n")[0]
                total_val = float(parts.split("/")[1].split("(")[0].replace(",", "").strip())
                prog_val = float(parts.split("(")[1].split("%")[0].strip()) / 100.0
                obs[1] = prog_val
                
                if "Remaining:" in result_text:
                    rem_str = result_text.split("Remaining:")[1].split("shares")[0].replace(",", "").strip()
                    obs[2] = float(rem_str) / max(1.0, total_val)
                else:
                    obs[2] = 1.0 - prog_val # Fallback
            # 3. Market State (Volume Ratio)
            if "Volume Ratio:" in result_text:
                val = result_text.split("Volume Ratio:")[1].split("×")[0].strip()
                obs[3] = float(val) / 2.0
            
            # 4. Relative Price
            if "Mid Price:" in result_text and "Arrival Price:" in result_text:
                mid = result_text.split("Mid Price:")[1].split("\n")[0].replace("$", "").strip()
                arr = result_text.split("Arrival Price:")[1].split("\n")[0].replace("$", "").strip()
                obs[0] = float(mid.split()[0]) / max(1.0, float(arr.split()[0]))

        except (ValueError, IndexError):
            pass

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.loop.run_until_complete(self.client.reset(task_id=self.task_id, seed=seed))
        text = obs.metadata.get("output", "") if hasattr(obs, 'metadata') else ""
        return self._get_obs_vector(text), {}

    def step(self, action):
        rate = float(action[0])
        result = self.loop.run_until_complete(self.client.execute_trade(
            participation_rate=rate, 
            use_dark_pool=False
        ))
        reward = self.loop.run_until_complete(self.client.get_reward())
        done = "EPISODE COMPLETE" in result or "ENGINE ERROR" in result
        return self._get_obs_vector(result), reward, done, False, {}

    def close(self):
        self.loop.run_until_complete(self.client.close())

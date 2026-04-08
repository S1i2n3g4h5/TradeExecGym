# Copyright (c) 2026 TradeExecGym Contributors
# BSD 3-Clause License

"""
TradeExecGym Client — Standardized HTTP Client.
Aligned with FarmSimulation project structure.
"""

import requests
import json
import re
from typing import Optional, Dict, Any

class TradeExecClient:
    """Standard HTTP Client for TradeExecGym Smart Order Router."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    async def __aenter__(self): 
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb): 
        pass

    def _unwrap(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenEnv response to flat dict with reward and done."""
        if "observation" in raw:
            flat = dict(raw["observation"])
            flat["reward"] = raw.get("reward")
            flat["done"] = raw.get("done", False)
            if "info" in raw["observation"] and isinstance(raw["observation"]["info"], dict):
                 flat.update(raw["observation"]["info"])
            return flat
        return raw

    def _reset_sync(self, task_id: str = "task_1", seed: Optional[int] = None) -> Dict[str, Any]:
        """Synchronous implementation of reset."""
        payload = {"task_id": task_id, "seed": seed}
        r = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())

    async def reset(self, task_id: str = "task_1", seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        return self._reset_sync(task_id=task_id, seed=seed)

    def _step_sync(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous implementation of step."""
        r = self._session.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one simulation step."""
        return self._step_sync(action=action)

    async def execute_trade(
        self,
        participation_rate: float,
        use_dark_pool: bool = False,
        dark_pool_fraction: float = 0.0
    ) -> str:
        """Domain-specific helper for the primary trading action."""
        action = {
            "participation_rate": float(participation_rate),
            "use_dark_pool": bool(use_dark_pool),
            "dark_pool_fraction": float(dark_pool_fraction)
        }
        obs = await self.step(action)
        # Return the output text from info if available, otherwise generic
        return obs.get("info", {}).get("output", "Trade executed.")

    def _get_state_sync(self) -> Dict[str, Any]:
        """Synchronous state fetch."""
        r = self._session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()

    async def get_market_state(self) -> str:
        """Fetch human-readable market narrative."""
        data = self._get_state_sync()
        return data.get("text_summary") or data.get("output") or "No state available."

    async def get_reward(self) -> float:
        """Retrieve the current grader score / reward."""
        data = self._get_state_sync()
        return float(data.get("reward", 0.0))

    async def get_grader_score(self) -> float:
        """Alias for get_reward."""
        return await self.get_reward()

    async def close(self):
        """Close the underlying HTTP session."""
        self._session.close()

    def sync(self):
        """Returns a synchronous wrapper for this client."""
        return SyncTradeEnv(self)

class SyncTradeEnv:
    """Synchronous wrapper for TradeExecClient matching YourRlEnv interface."""
    def __init__(self, client: TradeExecClient):
        self.client = client

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Frees any session resources."""
        self.client._session.close()

    def reset(self, **kwargs):
        res = self.client._reset_sync(**kwargs)
        from models import YourRlObservation
        try:
            obs = YourRlObservation(**res)
        except Exception:
            # Fallback to empty observation if validation still fails for some reason
            obs = YourRlObservation()
            
        from collections import namedtuple
        Result = namedtuple("Result", ["observation", "done", "reward"])
        return Result(observation=obs, done=False, reward=0.0)

    def step(self, action):
        from models import YourRlAction
        if isinstance(action, YourRlAction):
            # Convert command to participation_rate if needed
            import re
            nums = re.findall(r"0\.\d+", action.command)
            if nums:
                action.participation_rate = float(nums[0])
            p_rate = action.participation_rate if action.participation_rate is not None else 0.05
        else:
            p_rate = 0.05

        p_rate = max(0.0, min(0.25, float(p_rate)))
        res = self.client._step_sync({"participation_rate": p_rate})

        from models import YourRlObservation
        try:
            obs = YourRlObservation(**res)
        except Exception:
            obs = YourRlObservation()
            
        # Ensure reward and done are picked up from the response
        reward_val = res.get("reward", 0.0)
        done_val = res.get("done", False)
        
        from collections import namedtuple
        Result = namedtuple("Result", ["observation", "done", "reward"])
        return Result(observation=obs, done=done_val, reward=reward_val)

YourRlEnv = TradeExecClient

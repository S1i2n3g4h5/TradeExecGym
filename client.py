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

    async def reset(self, task_id: str = "task_1", seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        payload = {"task_id": task_id, "seed": seed}
        r = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one simulation step."""
        r = self._session.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())

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

    async def get_market_state(self) -> str:
        """Fetch human-readable market narrative."""
        r = self._session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        # In a standard Environment, state() returns the State pydantic model, 
        # but create_app often includes a natural language summary.
        data = r.json()
        return data.get("text_summary") or data.get("output") or "No state available."

    async def get_reward(self) -> float:
        """Retrieve the current grader score / reward."""
        r = self._session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return float(r.json().get("reward", 0.0))

    async def get_grader_score(self) -> float:
        """Alias for get_reward."""
        return await self.get_reward()

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
        pass

    def reset(self, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.client.reset(**kwargs))
        from models import YourRlObservation
        obs = YourRlObservation(**res)
        from collections import namedtuple
        Result = namedtuple("Result", ["observation", "done", "reward"])
        return Result(observation=obs, done=False, reward=0.0)

    def step(self, action):
        import asyncio
        from models import YourRlAction
        if isinstance(action, YourRlAction):
            # Convert command to participation_rate if needed
            import re
            nums = re.findall(r"0\.\d+", action.command)
            if nums:
                 action.participation_rate = float(nums[0])
            
            p_rate = action.participation_rate
        else:
            p_rate = 0.05

        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.client.step({"participation_rate": p_rate}))
        
        from models import YourRlObservation
        obs = YourRlObservation(**res)
        obs.task_achieved = obs.reward > 0.8
        
        from collections import namedtuple
        Result = namedtuple("Result", ["observation", "done", "reward"])
        return Result(observation=obs, done=obs.done, reward=obs.reward)

YourRlEnv = TradeExecClient

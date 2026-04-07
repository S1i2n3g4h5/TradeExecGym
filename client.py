# Copyright (c) 2026 TradeExecGym Contributors
# BSD 3-Clause License

"""
TradeExecGym Client — Custom MCPToolClient interface.

Extends the standard OpenEnv MCPToolClient with convenience async
helpers tailored for the trading domain.
"""

from typing import Optional
from openenv.core.mcp_client import MCPToolClient

class TradeExecClient(MCPToolClient):
    """
    Client for TradeExecGym Smart Order Router Environment.

    Provides domain-specific async helpers over the raw call_tool API.
    All methods here must be async since MCPToolClient is inherently async.
    """

    async def get_market_state(self) -> str:
        """Read the current market state."""
        return await self.call_tool("get_market_state")

    async def get_baseline_comparison(self) -> str:
        """Compare your execution vs TWAP/VWAP/AC Optimal."""
        return await self.call_tool("get_baseline_comparison")

    async def execute_trade(
        self,
        participation_rate: float,
        use_dark_pool: bool = False,
        dark_pool_fraction: float = 0.0,
        order_type: str = "MARKET",
        limit_offset_bps: float = 0.0,
    ) -> str:
        """Execute block shares at the specific rate."""
        return await self.call_tool(
            "execute_trade",
            participation_rate=participation_rate,
            use_dark_pool=use_dark_pool,
            dark_pool_fraction=dark_pool_fraction,
            order_type=order_type,
            limit_offset_bps=limit_offset_bps,
        )

    async def get_reward(self) -> float:
        """Fetch the latest per‑step reward via MCP tool."""
        return await self.call_tool("get_reward")

    async def run_twap_episode(
        self,
        task_id: str = "task1_twap_beater",
        n_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run a full TWAP (uniform slicing) baseline episode directly via client.
        """
        obs = await self.reset(task_id=task_id)
        if verbose:
            metadata = obs.observation.get('metadata', {}) if hasattr(obs, 'observation') else getattr(obs, 'metadata', {})
            print(metadata.get("output", ""))

        max_steps = n_steps or 30
        results = {"task_id": task_id, "steps_taken": 0, "final_is_bps": None, "grader_score": None}

        for step in range(max_steps):
            result = await self.execute_trade(participation_rate=0.05)
            results["steps_taken"] += 1

            if verbose:
                print(f"Step {step+1}: {result[:120]}...")

            if "EPISODE COMPLETE" in result:
                for line in result.split("\n"):
                    line = line.strip()
                    if "Final IS:" in line:
                        try:
                            raw = line.split("Final IS:")[1].strip()
                            results["final_is_bps"] = float(raw.lower().replace("bps", "").strip().split()[0])
                        except Exception:
                            pass
                    if "Grader Score:" in line:
                        try:
                            raw = line.split("Grader Score:")[1].strip()
                            results["grader_score"] = float(raw.split("/")[0].strip())
                        except Exception:
                            pass
                break

        return results

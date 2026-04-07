# Copyright (c) 2026 TradeExecGym Contributors
# BSD 3-Clause License

"""
TradeExecGym Client — Custom MCPToolClient interface.

Extends the standard OpenEnv MCPToolClient with convenience async
helpers tailored for the trading domain.

Usage:
    async with TradeExecClient(base_url="http://localhost:7860") as client:
        await client.reset(task_id="task1_twap_beater", seed=42)
        state = await client.get_market_state()
        result = await client.execute_trade(participation_rate=0.05)
        reward = await client.get_reward()
"""

import re
from typing import Optional
from openenv.core.mcp_client import MCPToolClient


class TradeExecClient(MCPToolClient):
    """
    Client for TradeExecGym Smart Order Router Environment.

    Provides domain-specific async helpers over the raw call_tool API.
    All methods are async since MCPToolClient is inherently async.
    """

    async def get_market_state(self) -> str:
        """Read the current market state narrative."""
        return await self.call_tool("get_market_state")

    async def get_baseline_comparison(self) -> str:
        """Compare your execution vs TWAP/VWAP/AC Optimal baselines."""
        return await self.call_tool("get_baseline_comparison")

    async def execute_trade(
        self,
        participation_rate: float,
        use_dark_pool: bool = False,
        dark_pool_fraction: float = 0.0,
        order_type: str = "MARKET",
        limit_offset_bps: float = 0.0,
    ) -> str:
        """Execute block shares at the specified participation rate.

        Args:
            participation_rate: Fraction of ADV to trade [0.0, 0.25].
            use_dark_pool: Route portion of order via dark pool (Task 3+).
            dark_pool_fraction: Fraction to route dark [0.0, 1.0] (0.3 recommended).
            order_type: Order type string (default "MARKET").
            limit_offset_bps: Limit price offset in basis points.

        Returns:
            str: Execution result narrative with fills, IS, and grader score.
        """
        return await self.call_tool(
            "execute_trade",
            participation_rate=participation_rate,
            use_dark_pool=use_dark_pool,
            dark_pool_fraction=dark_pool_fraction,
            order_type=order_type,
            limit_offset_bps=limit_offset_bps,
        )

    async def get_reward(self) -> float:
        """Fetch the latest per-step reward via MCP tool.

        Returns:
            float: Reward in approximately [-2.0, 2.0].
        """
        result = await self.call_tool("get_reward")
        # call_tool may return str or float depending on server version
        if isinstance(result, (int, float)):
            return float(result)
        try:
            return float(str(result).strip())
        except (ValueError, TypeError):
            return 0.0

    async def get_grader_score(self) -> Optional[float]:
        """Parse the current estimated grader score from market state.

        Returns:
            float or None: Grader score in [0.0001, 0.9999], or None if unavailable.
        """
        state_text = await self.get_market_state()
        for pattern in ["Est. Score:", "Grader Score:", "Score:"]:
            if pattern in state_text:
                try:
                    raw = state_text.split(pattern)[1].split("/")[0].strip().split()[0]
                    val = float(raw)
                    if 0.0 <= val <= 1.0:
                        return val
                except (ValueError, IndexError):
                    continue
        return None

    async def get_numeric_state(self) -> dict:
        """Get a structured numeric state dict from the market state text.

        Parses the text narrative to extract the 5-dimensional numeric observation.
        Returns a dict matching TradeObservation fields.

        Returns:
            dict: Numeric state with keys: remaining_shares, steps_left, current_is,
                  vol_ratio, progress_pct, etc.
        """
        state_text = await self.get_market_state()
        result = {}
        try:
            if "Remaining:" in state_text:
                result["remaining_shares"] = int(
                    state_text.split("Remaining:")[1].split("shares")[0]
                    .replace(",", "").strip()
                )
            if "Time left:" in state_text:
                result["steps_left"] = int(
                    state_text.split("Time left:")[1].split("steps")[0].strip()
                )
            if "Your IS:" in state_text:
                raw = state_text.split("Your IS:")[1].strip()
                result["current_is_bps"] = float(
                    re.sub(r"[^\d.]", "", raw.split()[0])
                )
            if "Volume Ratio:" in state_text:
                raw = state_text.split("Volume Ratio:")[1].strip().split("x")[0].strip()
                result["vol_ratio"] = float(raw)
            if "Mid Price:" in state_text:
                raw = state_text.split("Mid Price:")[1].strip().replace("$", "").split()[0]
                result["mid_price"] = float(raw)
            if "Executed:" in state_text and "/" in state_text.split("Executed:")[1].split("\n")[0]:
                exec_line = state_text.split("Executed:")[1].split("\n")[0]
                parts = exec_line.split("/")
                executed = int(parts[0].replace(",", "").strip())
                total = int(parts[1].strip().replace(",", "").split()[0])
                result["shares_executed"] = executed
                result["total_shares"] = total
                result["progress_pct"] = executed / max(1, total)
                result["remaining_pct"] = 1.0 - result["progress_pct"]
        except Exception:
            pass
        return result

    async def run_twap_episode(
        self,
        task_id: str = "task1_twap_beater",
        n_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """Run a full TWAP (uniform slicing) baseline episode via client.

        Args:
            task_id: Task to run (default task1_twap_beater).
            n_steps: Max steps override (auto-detected from reset metadata).
            verbose: Print step-by-step results.

        Returns:
            dict: Episode results with task_id, steps_taken, final_is_bps, grader_score.
        """
        obs = await self.reset(task_id=task_id)

        # Auto-detect max_steps from reset metadata
        max_steps = n_steps or 30
        try:
            if hasattr(obs, "metadata") and obs.metadata:
                meta = obs.metadata
                if isinstance(meta, dict):
                    max_steps = int(meta.get("max_steps", max_steps))
        except Exception:
            pass

        if verbose:
            print(f"[TWAP] Starting {task_id} | max_steps={max_steps}")

        results = {
            "task_id": task_id,
            "steps_taken": 0,
            "final_is_bps": None,
            "grader_score": None,
        }

        for step in range(max_steps):
            # TWAP: uniform equal-slice rate
            twap_rate = 0.05
            result = await self.execute_trade(participation_rate=twap_rate)
            results["steps_taken"] += 1

            if verbose:
                print(f"[TWAP] Step {step+1}: {result[:100]}...")

            if "EPISODE COMPLETE" in result:
                for line in result.split("\n"):
                    line = line.strip()
                    if "Final IS:" in line:
                        try:
                            raw = line.split("Final IS:")[1].strip()
                            results["final_is_bps"] = float(
                                raw.lower().replace("bps", "").strip().split()[0]
                            )
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

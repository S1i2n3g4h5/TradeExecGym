#!/usr/bin/env python3
"""
eval_baselines.py — Evaluate baseline agents against TradeExecGym.
Executes TWAP, VWAP, and AC Optimal via TradeExecClient.
"""

import argparse
import asyncio
import sys

from baselines.twap import get_twap_action
from baselines.vwap import get_vwap_action
from baselines.ac_optimal import get_ac_optimal_action

try:
    from client import TradeExecClient
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from client import TradeExecClient

async def run_baseline_episode(base_url: str, task_id: str, agent_type: str, verbose: bool = False) -> dict:
    results = {
        "agent": agent_type,
        "task_id": task_id,
        "success": False,
        "final_is_bps": None,
        "grader_score": None,
        "error": None
    }

    try:
        async with TradeExecClient(base_url=base_url) as client:
            obs = await client.reset(task_id=task_id)
            # Robustly extract metadata from potentially nested StepResult/Observation
            meta = {}
            # Try obs.metadata first (if Observation directly)
            if hasattr(obs, 'metadata') and isinstance(getattr(obs, 'metadata', None), dict):
                meta = obs.metadata
            # Try obs.observation.metadata (if StepResult wrapping Observation)
            if not meta and hasattr(obs, 'observation'):
                inner = obs.observation
                if hasattr(inner, 'metadata') and isinstance(getattr(inner, 'metadata', None), dict):
                    meta = inner.metadata
                elif isinstance(inner, dict):
                    meta = inner.get('metadata', {})
            
            total_shares = meta.get("total_shares", 100_000)
            max_steps = meta.get("max_steps", 120)

            shares_remaining = total_shares
            
            for step in range(max_steps):
                if agent_type == "twap":
                    rate = get_twap_action(step, max_steps, shares_remaining, total_shares)
                elif agent_type == "vwap":
                    rate = get_vwap_action(step, max_steps, shares_remaining, total_shares)
                elif agent_type == "ac_optimal":
                    rate = get_ac_optimal_action(step, max_steps, shares_remaining, total_shares)
                else:
                    raise ValueError(f"Unknown agent: {agent_type}")
                
                # We can simulate dark pool access for AC Optimal as an advanced strategy
                use_dark = (agent_type == "ac_optimal")
                df = 0.5 if use_dark else 0.0

                result = await client.execute_trade(
                    participation_rate=rate,
                    use_dark_pool=use_dark,
                    dark_pool_fraction=df
                )
                
                # Approximate shares filed from the text result for shares_remaining updates
                # "Remaining: X shares"
                for line in result.split("\n"):
                    if "Remaining:" in line and "shares" in line:
                        try:
                            rem = line.split(":")[1].replace("shares", "").replace(",", "").strip()
                            shares_remaining = int(rem)
                        except Exception:
                            pass

                if "EPISODE COMPLETE" in result:
                    for line in result.split("\n"):
                        if "Final IS:" in line:
                            try:
                                results["final_is_bps"] = float(line.split(":")[1].replace(" bps", "").strip())
                            except Exception:
                                pass
                        if "Grader Score:" in line:
                            try:
                                results["grader_score"] = float(line.split(":")[1].split("/")[0].strip())
                            except Exception:
                                pass
                    break

            results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print(f"  ❌ Error in {agent_type}: {e}")

    return results

async def main():
    parser = argparse.ArgumentParser(description="Evaluate Baselines")
    parser.add_argument("--url", default="http://localhost:7860", help="Server API URL")
    parser.add_argument("--task", default="task1_twap_beater", help="Task ID to evaluate on (e.g. task1_twap_beater)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    task_id = args.task
    
    print(f"Evaluating Baselines on {task_id} @ {base_url}")
    print("-" * 60)
    
    agents = ["twap", "vwap", "ac_optimal"]
    for agent in agents:
        res = await run_baseline_episode(base_url, task_id, agent)
        if res["success"]:
            is_str = f"{res['final_is_bps']:>6.2f}" if res['final_is_bps'] is not None else "  N/A "
            sc_str = f"{res['grader_score']:>6.4f}" if res['grader_score'] is not None else "  N/A "
            print(f"{agent.upper():<12} | IS: {is_str} bps | Score: {sc_str}")
        else:
            print(f"{agent.upper():<12} | FAILED: {res['error']}")

if __name__ == "__main__":
    asyncio.run(main())

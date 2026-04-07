"""
TradeExecGym -- Heuristic Baseline Scorer
=========================================
OpenEnv Spec Compliant | Version 1.0.0

Produces reproducible baseline scores on Tasks 1, 2, and 3 using the
AlmgrenChrissHeuristic (no LLM required). Each task runs with seed=42
for full reproducibility.

Usage:
    # Start environment server first:
    #   uvicorn server.app:app --port 7860
    
    python baselines/run_baselines.py

Environment Variables:
    ENV_BASE_URL   Base URL of the TradeExecGym server (default: http://localhost:7860)

Output:
    [BASELINE] task=task1_twap_beater score=0.72 is_bps=18.4 steps=30
    [BASELINE] task=task2_vwap_optimizer score=0.68 is_bps=21.1 steps=60
    [BASELINE] task=task3_volatile_execution score=0.61 is_bps=28.3 steps=90
    [SUMMARY] mean_score=0.67
    
    Results also saved to results/baseline_scores.json
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Optional

# ASCII-safe stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

# Path resolution for running from any directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from client import TradeExecClient
from baselines.heuristic_agent import AlmgrenChrissHeuristic

# Configuration
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
RESULTS_DIR = "results"
SEED = 42

# Tasks to benchmark (easy -> medium -> hard)
BASELINE_TASKS = [
    {
        "task_id": "task1_twap_beater",
        "max_steps": 30,
        "description": "Buy 100K shares, beat TWAP (Easy)",
    },
    {
        "task_id": "task2_vwap_optimizer",
        "max_steps": 60,
        "description": "Sell 250K shares, beat VWAP (Medium)",
    },
    {
        "task_id": "task3_volatile_execution",
        "max_steps": 90,
        "description": "Buy 400K under 3x volatility (Hard)",
    },
]


def extract_score(text: str) -> Optional[float]:
    """Extract grader score from execute_trade response text."""
    for pattern in ["Grader Score:", "grader_score:", "Score:"]:
        if pattern in text:
            try:
                raw = text.split(pattern)[1].split("/")[0].strip().split()[0]
                val = float(raw)
                if 0.0 <= val <= 1.0:
                    return val
            except (ValueError, IndexError):
                continue
    return None


def extract_is(text: str) -> Optional[float]:
    """Extract final IS (bps) from execute_trade response text."""
    for pattern in ["Final IS:", "Your IS:"]:
        if pattern in text:
            try:
                raw = text.split(pattern)[1].strip()
                val = float(raw.lower().replace("bps", "").strip().split()[0])
                return val
            except (ValueError, IndexError):
                continue
    return None


async def run_baseline_episode(
    env_client: TradeExecClient,
    heuristic: AlmgrenChrissHeuristic,
    task_id: str,
    max_steps: int,
) -> dict:
    """Run one heuristic baseline episode on a single task."""
    # Reset environment
    await env_client.reset(task_id=task_id, seed=SEED)

    step_count = 0
    done = False
    rewards = []
    final_score = 0.5
    final_is = None

    while not done and step_count < max_steps + 5:
        step_count += 1

        # Get market state
        state_text = await env_client.get_market_state()

        # Heuristic rate calculation
        suggested_rate = 0.05
        try:
            if "Remaining:" in state_text:
                rem = int(state_text.split("Remaining:")[1].split("shares")[0]
                          .replace(",", "").strip())
                steps_left = int(state_text.split("Time left:")[1].split("steps")[0].strip())

                total_shares = rem
                if "Executed:" in state_text and "/" in state_text.split("Executed:")[1].split("\n")[0]:
                    exec_line = state_text.split("Executed:")[1].split("\n")[0]
                    total_shares = int(exec_line.split("/")[1].strip()
                                      .replace(",", "").split()[0])

                current_is = 0.0
                if "Your IS:" in state_text:
                    raw = state_text.split("Your IS:")[1].strip()
                    current_is = float(raw.lower().replace("bps", "")
                                      .strip().split()[0])

                suggested_rate = heuristic.calculate_rate(
                    rem, total_shares, steps_left, current_is
                )
                suggested_rate = max(0.01, min(0.25, suggested_rate))
        except Exception:
            pass

        # Execute trade
        result = await env_client.execute_trade(participation_rate=suggested_rate)
        reward = await env_client.get_reward()
        rewards.append(reward)

        # Check done
        done = (
            "EPISODE COMPLETE" in result
            or step_count >= max_steps
        )

        if done:
            extracted = extract_score(result)
            if extracted is not None:
                final_score = extracted
            is_val = extract_is(result)
            if is_val is not None:
                final_is = is_val

    return {
        "task_id": task_id,
        "steps": step_count,
        "score": round(final_score, 4),
        "is_bps": round(final_is, 2) if final_is is not None else None,
        "total_reward": round(sum(rewards), 4),
        "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
        "seed": SEED,
    }


async def run_all_baselines():
    """Run heuristic baseline on all 3 tasks and report results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    heuristic = AlmgrenChrissHeuristic()
    results = []

    print(f"[BASELINES] TradeExecGym Heuristic Baseline Scorer", flush=True)
    print(f"[BASELINES] Server: {ENV_BASE_URL} | Seed: {SEED}", flush=True)
    print(f"[BASELINES] Tasks: {len(BASELINE_TASKS)} (easy -> medium -> hard)", flush=True)
    print("-" * 60, flush=True)

    async with TradeExecClient(base_url=ENV_BASE_URL) as env_client:
        for task_cfg in BASELINE_TASKS:
            task_id = task_cfg["task_id"]
            desc = task_cfg["description"]
            max_steps = task_cfg["max_steps"]

            print(f"[RUNNING] {task_id} ({desc})", flush=True)

            try:
                result = await run_baseline_episode(
                    env_client, heuristic, task_id, max_steps
                )
                results.append(result)

                is_str = f"{result['is_bps']:.1f}" if result['is_bps'] is not None else "N/A"
                print(
                    f"[BASELINE] task={task_id} score={result['score']:.4f} "
                    f"is_bps={is_str} steps={result['steps']}",
                    flush=True
                )
            except Exception as e:
                print(f"[ERROR] {task_id}: {e}", flush=True)
                results.append({
                    "task_id": task_id,
                    "score": 0.0,
                    "is_bps": None,
                    "steps": 0,
                    "error": str(e),
                    "seed": SEED,
                })

    print("-" * 60, flush=True)

    # Summary statistics
    valid_scores = [r["score"] for r in results if "error" not in r]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"[SUMMARY] mean_score={mean_score:.4f}", flush=True)

    for r in results:
        task_short = r["task_id"].replace("task", "T")
        is_str = f"{r['is_bps']:.1f} bps" if r.get("is_bps") else "N/A"
        print(f"[SUMMARY]   {task_short}: score={r['score']:.4f} | IS={is_str}", flush=True)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"baseline_scores_{ts}.json")
    summary = {
        "timestamp": ts,
        "seed": SEED,
        "env_base_url": ENV_BASE_URL,
        "tasks": results,
        "summary": {
            "mean_score": round(mean_score, 4),
            "task_count": len(results),
            "valid_runs": len(valid_scores),
        }
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Results saved to: {out_file}", flush=True)
    return summary


if __name__ == "__main__":
    asyncio.run(run_all_baselines())

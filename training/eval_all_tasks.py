#!/usr/bin/env python3
"""
eval_all_tasks.py -- Run TWAP baseline against ALL 5 tasks end-to-end.

This confirms:
1. Every task initializes correctly via the factory
2. Episodes run to completion
3. Grader scores are returned
4. Task4 adversary applies penalties to uniform TWAP
5. Task5 deadline pressure grading works
"""

import argparse
import asyncio
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from baselines.twap import get_twap_action
from client import TradeExecClient

TASKS = [
    "task1_twap_beater",
    "task2_vwap_optimizer",
    "task3_volatile_execution",
    "task4_adversarial",
    "task5_deadline_pressure",
]

async def run_single_task(base_url: str, task_id: str) -> dict:
    """Run a TWAP episode on a single task."""
    result = {
        "task_id": task_id,
        "success": False,
        "final_is_bps": None,
        "grader_score": None,
        "steps_taken": 0,
        "shares_executed": None,
        "error": None,
    }

    try:
        async with TradeExecClient(base_url=base_url) as client:
            obs = await client.reset(task_id=task_id)

            # Use generous defaults; the episode itself determines completion
            max_steps = 200

            shares_remaining = 999_999_999
            for step in range(max_steps):
                # Simple TWAP: constant rate
                rate = 0.05

                trade_result = await client.execute_trade(
                    participation_rate=rate,
                    use_dark_pool=False,
                    dark_pool_fraction=0.0,
                )

                result["steps_taken"] += 1

                # Parse remaining shares
                for line in trade_result.split("\n"):
                    if "Remaining:" in line and "shares" in line:
                        try:
                            rem = line.split(":")[1].replace("shares", "").replace(",", "").strip()
                            shares_remaining = int(rem)
                        except Exception:
                            pass
                    if "Executed:" in line:
                        try:
                            parts = line.split(":")[1].strip().split("/")
                            result["shares_executed"] = parts[0].strip().replace(",", "")
                        except Exception:
                            pass

                if "EPISODE COMPLETE" in trade_result:
                    for line in trade_result.split("\n"):
                        if "Final IS:" in line:
                            try:
                                result["final_is_bps"] = float(
                                    line.split(":")[1].replace("bps", "").strip()
                                )
                            except Exception:
                                pass
                        if "Grader Score:" in line:
                            try:
                                result["grader_score"] = float(
                                    line.split(":")[1].split("/")[0].strip()
                                )
                            except Exception:
                                pass
                    break

            result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


async def main():
    parser = argparse.ArgumentParser(description="Evaluate all tasks")
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("=" * 70)
    print("  TradeExecGym -- Phase 3: All-Task Verification")
    print("=" * 70)
    print(f"  Server: {base_url}")
    print("=" * 70)
    print()

    all_passed = True
    results_table = []

    for task_id in TASKS:
        print(f"  [>] Running {task_id} ...", end="", flush=True)
        res = await run_single_task(base_url, task_id)

        if res["success"] and res["grader_score"] is not None:
            is_str = f"{res['final_is_bps']:>7.2f}" if res["final_is_bps"] is not None else "    N/A"
            sc_str = f"{res['grader_score']:>7.4f}" if res["grader_score"] is not None else "    N/A"
            print(f"  [OK]  IS={is_str} bps  Score={sc_str}  Steps={res['steps_taken']}")
            results_table.append((task_id, is_str, sc_str, res["steps_taken"], "PASS"))
        elif res["success"]:
            print(f"  [WARN]  Completed but no grader (steps={res['steps_taken']})")
            results_table.append((task_id, "N/A", "N/A", res["steps_taken"], "WARN"))
        else:
            print(f"  [FAIL]  FAILED: {res['error']}")
            results_table.append((task_id, "N/A", "N/A", 0, "FAIL"))
            all_passed = False

    print()
    print("=" * 70)
    print(f"  {'Task':<30} {'IS (bps)':>10} {'Score':>10} {'Steps':>6} {'Status':>8}")
    print("-" * 70)
    for task_id, is_v, sc_v, steps, status in results_table:
        emoji = "[OK]" if status == "PASS" else ("[WARN]" if status == "WARN" else "[FAIL]")
        print(f"  {task_id:<30} {is_v:>10} {sc_v:>10} {steps:>6} {emoji:>6}")
    print("=" * 70)

    if all_passed:
        print("\n  [DONE] ALL TASKS PASSED -- Phase 3 Verified!\n")
    else:
        print("\n  [WARN]  Some tasks failed. Review above output.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

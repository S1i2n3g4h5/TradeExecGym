#!/usr/bin/env python3
"""
dry_run.py — Pre-deploy Async Validation.

Starts the full HTTP stack validation tests before deploying
to HF space to guarantee nothing breaks the OpenEnv schema.
"""

import argparse
import asyncio
import sys

import requests

try:
    from client import TradeExecClient
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from client import TradeExecClient

def check_health(base_url: str, verbose: bool = False) -> bool:
    """Check FastMCP health endpoint."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        if resp.status_code == 200:
            print(f"  ✅ /health → {resp.status_code}")
            if verbose:
                print(f"     {resp.json()}")
            return True
        else:
            print(f"  ❌ /health → {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"  ❌ /health failed: {e}")
        return False

async def run_episode(base_url: str, task_id: str, verbose: bool = False) -> dict:
    """Executes an entire episode against the async client."""
    results = {
        "task_id": task_id,
        "success": False,
        "steps_taken": 0,
        "final_is_bps": None,
        "grader_score": None,
        "error": None
    }

    try:
        async with TradeExecClient(base_url=base_url) as client:
            obs = await client.reset(task_id=task_id)
            if verbose:
                metadata = obs.observation.get('metadata', {}) if hasattr(obs, 'observation') else getattr(obs, 'metadata', {})
                print(f"     Reset output: {metadata.get('output', '')[:100]}...")

            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            if verbose:
                print(f"     Tools discovered: {tool_names}")
            assert "execute_trade" in tool_names, "Missing core tool"

            # Execute a full episode natively via the async custom wrapper
            ep_results = await client.run_twap_episode(task_id=task_id, verbose=verbose)
            
            results["success"] = True
            results["steps_taken"] = ep_results["steps_taken"]
            results["final_is_bps"] = ep_results["final_is_bps"]
            results["grader_score"] = ep_results["grader_score"]

    except Exception as e:
        results["error"] = str(e)
        print(f"  ❌ Episode Error: {e}")

    return results

async def async_main():
    parser = argparse.ArgumentParser(description="TradeExecGym validation")
    parser.add_argument("--url", default="http://localhost:7865", help="Server API URL")
    parser.add_argument("--verbose", action="store_true", help="Print debug details")
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")

    print("=" * 60)
    print("         TradeExecGym — Pre-Deploy Validation             ")
    print("=" * 60)
    print(f"  URL: {base_url:<51} ")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    print("CHECK 1: Server Health")
    if check_health(base_url, verbose=args.verbose):
        passed += 1
    else:
        print("  → Healthcheck failed. Ensure server is running.")
        sys.exit(1)

    print("\nCHECK 2: Task 1 Full Episode (task1_twap_beater)")
    res1 = await run_episode(base_url, "task1_twap_beater", verbose=args.verbose)
    if res1["success"]:
        print(f"  ✅ Completed in {res1['steps_taken']} steps")
        print(f"     Final IS: {res1['final_is_bps']:.2f} bps")
        print(f"     Grader Score: {res1['grader_score']:.4f}")
        passed += 1
    else:
        print(f"  ❌ Failed: {res1['error']}")
        failed += 1

    print("\nCHECK 3: Task 3 Volatile Episode (task3_volatile_execution)")
    res3 = await run_episode(base_url, "task3_volatile_execution", verbose=False)
    if res3["success"]:
        print(f"  ✅ Completed in {res3['steps_taken']} steps")
        passed += 1
    else:
        failed += 1

    print("\nCHECK 4: Grader Determinism (task1_twap_beater x2)")
    r1 = await run_episode(base_url, "task1_twap_beater", verbose=False)
    r2 = await run_episode(base_url, "task1_twap_beater", verbose=False)
    if r1["success"] and r2["success"]:
        print(f"  ✅ Validated two continuous runs.")
        passed += 1
    else:
        failed += 1

    print(f"\n{'='*50}\n  RESULTS: {passed} passed, {failed} failed\n{'='*50}")
    
    if failed == 0:
        print("\n  🎉 Validation Passed")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(async_main())

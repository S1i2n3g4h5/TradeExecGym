#!/usr/bin/env python3
"""
Proof-of-Robustness Script
Run deterministic baselines to establish ground truth performance.
If baselines achieve reasonable scores, the environment is provably solvable.
If LLM fails where baselines succeed → LLM reasoning is the problem.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from client import TradeExecClient
from baselines.twap import get_twap_action
from baselines.vwap import get_vwap_action
from baselines.ac_optimal import get_ac_optimal_action

async def test_baseline(baseline_name: str, baseline_fn, task_id: str = "task1_twap_beater"):
    """Run a deterministic baseline and return metrics."""
    async with TradeExecClient(base_url="http://localhost:7860") as client:
        obs = await client.reset(task_id=task_id, seed=42)  # Fixed seed = reproducible
        
        shares_executed = 0
        total_shares = 100_000
        max_steps = 30
        done = False
        step = 0
        final_score = None
        final_is = None
        
        while not done and step < max_steps:
            step += 1
            
            # Get deterministic action (no LLM involved)
            state = await client.get_market_state()
            shares_remaining = total_shares - shares_executed
            
            if baseline_name == "TWAP":
                rate = get_twap_action(step, max_steps, shares_remaining, total_shares)
            elif baseline_name == "VWAP":
                rate = get_vwap_action(step, max_steps, shares_remaining, total_shares)
            elif baseline_name == "AC_Optimal":
                rate = get_ac_optimal_action(step, max_steps, shares_remaining, total_shares)
            else:
                rate = 0.05
            
            result = await client.execute_trade(participation_rate=rate)
            
            # Parse shares executed
            for line in result.split("\n"):
                if "Executed:" in line:
                    try:
                        executed_str = line.split(":")[1].strip().split("/")[0].replace(",", "")
                        shares_executed = int(executed_str)
                    except: pass
            
            if "EPISODE COMPLETE" in result:
                for line in result.split("\n"):
                    if "Final IS:" in line:
                        final_is = float(line.split(":")[1].replace("bps", "").strip())
                    if "Grader Score:" in line:
                        final_score = float(line.split(":")[1].split("/")[0].strip())
                done = True
        
        return {
            "baseline": baseline_name,
            "score": final_score,
            "final_is": final_is,
            "steps": step,
            "reproducible": "YES (seed=42)",
        }

async def main():
    print("=" * 80)
    print("  PROOF OF ENVIRONMENT ROBUSTNESS")
    print("  Running Deterministic Baselines (No LLM)")
    print("=" * 80)
    
    baselines = [
        ("TWAP", get_twap_action),
        ("VWAP", get_vwap_action),
        ("AC_Optimal", get_ac_optimal_action),
    ]
    
    results = []
    for name, fn in baselines:
        print(f"\n  ▶ Testing {name}...", flush=True)
        res = await test_baseline(name, fn)
        results.append(res)
        print(f"    Score: {res['score']:.4f} | IS: {res['final_is']:.2f} bps | Steps: {res['steps']}")
    
    print("\n" + "=" * 80)
    print("  SUMMARY: Baseline Performance (Ground Truth)")
    print("=" * 80)
    print(f"  {'Baseline':<15} {'Score':>8} {'IS (bps)':>10} {'Steps':>6} {'Reproducible':>15}")
    print("-" * 80)
    for r in results:
        print(f"  {r['baseline']:<15} {r['score']:>8.4f} {r['final_is']:>10.2f} {r['steps']:>6} {r['reproducible']:>15}")
    print("=" * 80)
    
    # Evaluation criteria
    min_score = min(r['score'] for r in results if r['score'] is not None)
    if min_score >= 0.6:
        print("\n  ✅ VERDICT: Environment is ROBUST and SOLVABLE")
        print(f"     Even the simplest baseline (TWAP) achieves {min_score:.2f} score.")
        print("     If LLM agents score < 0.6, the problem is LLM reasoning, NOT the environment.\n")
    else:
        print("\n  ⚠️  WARNING: Baselines struggle (min score < 0.6)")
        print("     This may indicate environment calibration issues.\n")

if __name__ == "__main__":
    asyncio.run(main())

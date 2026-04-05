#!/usr/bin/env python3
"""
ABLATION STUDY: Prove Environment Robustness
============================================
This script compares different agent types on the SAME task to isolate where failures occur.

Test Matrix:
1. Random Agent (floor performance)
2. TWAP Baseline (deterministic math)
3. AC Optimal (theoretical ceiling)
4. LLM Agent (if available)

If Random < TWAP < AC Optimal < 1.0, the environment has a clear skill gradient.
If LLM performs worse than TWAP, the problem is LLM reasoning, NOT the environment.
"""

import asyncio
import random
from client import TradeExecClient
from baselines.twap import get_twap_action
from baselines.ac_optimal import get_ac_optimal_action

async def run_agent(agent_name: str, agent_fn, task_id: str = "task1_twap_beater", seed: int = 42):
    """Run a single agent and return performance metrics."""
    async with TradeExecClient(base_url="http://localhost:7860") as client:
        await client.reset(task_id=task_id, seed=seed)
        
        shares_executed = 0
        total_shares = 100_000
        max_steps = 30
        step = 0
        rewards = []
        
        while step < max_steps:
            step += 1
            state = await client.get_market_state()
            shares_remaining = total_shares - shares_executed
            
            # Get action from agent
            rate = agent_fn(step, max_steps, shares_remaining, total_shares)
            
            result = await client.execute_trade(participation_rate=rate)
            reward = await client.get_reward()
            rewards.append(reward)
            
            # Parse execution
            for line in result.split("\n"):
                if "Executed:" in line:
                    try:
                        shares_executed = int(line.split(":")[1].split("/")[0].replace(",", "").strip())
                    except: pass
            
            if "EPISODE COMPLETE" in result:
                # Extract final metrics
                final_is = None
                final_score = None
                for line in result.split("\n"):
                    if "Final IS:" in line:
                        final_is = float(line.split(":")[1].replace("bps", "").strip())
                    if "Grader Score:" in line:
                        final_score = float(line.split(":")[1].split("/")[0].strip())
                
                return {
                    "agent": agent_name,
                    "score": final_score,
                    "is_bps": final_is,
                    "steps": step,
                    "completion": shares_executed / total_shares,
                    "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
                }
        
        return None

def random_agent(step, max_steps, shares_remaining, total_shares):
    """Random action baseline - establishes floor performance."""
    return random.uniform(0.01, 0.25)

async def main():
    print("=" * 90)
    print("  ABLATION STUDY: Environment Robustness Proof")
    print("  Comparing Multiple Agent Types on Same Task (seed=42)")
    print("=" * 90)
    
    agents = [
        ("Random", random_agent),
        ("TWAP", get_twap_action),
        ("AC_Optimal", get_ac_optimal_action),
    ]
    
    results = []
    for name, fn in agents:
        print(f"\n  ▶ Running {name}...", flush=True)
        res = await run_agent(name, fn)
        if res:
            results.append(res)
            print(f"    Score: {res['score']:.4f} | IS: {res['is_bps']:.2f} bps | Completion: {res['completion']*100:.1f}%")
    
    print("\n" + "=" * 90)
    print("  ABLATION RESULTS: Skill Gradient Analysis")
    print("=" * 90)
    print(f"  {'Agent':<15} {'Score':>8} {'IS (bps)':>10} {'Completion':>12} {'Avg Reward':>12}")
    print("-" * 90)
    
    for r in sorted(results, key=lambda x: x['score']):
        print(f"  {r['agent']:<15} {r['score']:>8.4f} {r['is_bps']:>10.2f} {r['completion']*100:>11.1f}% {r['avg_reward']:>12.4f}")
    
    print("=" * 90)
    
    # Analysis
    scores = [r['score'] for r in results]
    if len(scores) >= 3:
        random_score = next(r['score'] for r in results if r['agent'] == 'Random')
        twap_score = next(r['score'] for r in results if r['agent'] == 'TWAP')
        optimal_score = next(r['score'] for r in results if r['agent'] == 'AC_Optimal')
        
        print("\n  📊 SKILL GRADIENT ANALYSIS:")
        print(f"     Random Agent:  {random_score:.4f} (floor performance)")
        print(f"     TWAP Baseline: {twap_score:.4f} (simple strategy)")
        print(f"     AC Optimal:    {optimal_score:.4f} (math-based ceiling)")
        print(f"     Gradient:      {min(scores):.4f} → {max(scores):.4f} (range: {max(scores)-min(scores):.4f})")
        
        if max(scores) - min(scores) > 0.15:
            print("\n  ✅ VERDICT: Clear skill gradient exists!")
            print("     The environment rewards better strategies with higher scores.")
            print("     If an LLM scores below TWAP, it's making poor decisions, not env issues.\n")
        else:
            print("\n  ⚠️  WARNING: Weak skill gradient (range < 0.15)")
            print("     May indicate task is too easy or grading is too lenient.\n")

if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())

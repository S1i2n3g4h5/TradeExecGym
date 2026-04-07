import asyncio
import numpy as np
import random
from client import TradeExecClient

async def run_policy(name, policy_type="uniform"):
    async with TradeExecClient(base_url="http://localhost:7865") as client:
        obs = await client.reset(task_id="task4_adversarial", seed=42)
        total_penalized_steps = 0
        
        for i in range(120):
            if policy_type == "uniform":
                rate = 0.10
            elif policy_type == "alternating":
                rate = 0.05 if i % 2 == 0 else 0.15
            elif policy_type == "jitter":
                # Ensure std_dev > 0.005 and randomized pattern
                rate = 0.10 + random.uniform(-0.04, 0.04)
            else: # AC Proxy
                # Hyperbolic-like decay (simplified for script)
                p = (i+1)/120
                rate = max(0.02, 0.20 * (1 - p**0.5))
            
            result = await client.execute_trade(participation_rate=rate)
            
            if "⚠️ ADVERSARY" in result or "detector fired" in result:
                total_penalized_steps += 1
                
            if "EPISODE COMPLETE" in result:
                # Parse final score and IS
                final_is = 0.0
                score = 0.0
                for line in result.split("\n"):
                    if "Final IS:" in line:
                        final_is = float(line.split(":")[1].replace("bps", "").strip())
                    if "Grader Score:" in line:
                        score = float(line.split(":")[1].split("/")[0].strip())
                return name, final_is, score, total_penalized_steps
        return name, 0, 0, 0

async def main():
    print("="*60)
    print(" TASK 4 SKILL GRADIENT ANALYSIS — PATTERN BREAKING PROOF")
    print("="*60)
    print(f"{'Strategy':<20} {'Final IS':>10} {'Score':>10} {'Penalties':>10}")
    print("-" * 60)
    
    tasks = [
        run_policy("Uniform (TWAP)", "uniform"),
        run_policy("Periodic (Pulse)", "alternating"),
        run_policy("Random Jitter", "jitter"),
    ]
    
    results = await asyncio.gather(*tasks)
    for name, is_val, score, penalties in results:
        print(f"{name:<20} {is_val:>8.2f} bps {score:>9.4f} {penalties:>10}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

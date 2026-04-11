"""
generate_grpo_dataset.py — Generate GRPO training dataset from TradeExecGym.

Runs the Almgren-Chriss heuristic agent through N episodes and collects
(prompt, completion) pairs suitable for GRPOTrainer.from_dict() or
saved as a HuggingFace Dataset to disk.

Each record = one decision point in a trading episode:
  prompt:     [system_msg, user_msg(market_state)]
  completion: JSON action string the model should output

The heuristic provides "reasonable but imperfect" completions to bootstrap
the rollout distribution. GRPO then refines these via group-relative rewards.

Usage:
  python training/generate_grpo_dataset.py --episodes 300 --output training/grpo_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.trade_environment import TradeExecEnvironment
from baselines.heuristic_agent import AlmgrenChrissHeuristic
from inference import SYSTEM_PROMPT

# Tasks to sample from (weight tasks by difficulty variety)
TASK_POOL = [
    ("task_1", 1),  # easy
    ("task_2", 1),  # medium
    ("task_3", 1),  # hard (volatile — dark pool)
    ("task_4", 1),  # adversarial HFT
    ("task_5", 1),  # deadline cliff
]

# Strategy classification thresholds
_AGGRESSIVE_THRESHOLD = 0.14
_DARK_THRESHOLD = 0.10


def _classify_strategy(
    rate: float,
    market_text: str,
    shares_remaining: int,
    total_shares: int,
    steps_left: int,
    max_steps: int,
) -> str:
    """Map a rate + context to the closest named strategy."""
    pct_done = 1.0 - shares_remaining / max(1, total_shares)
    pct_time_used = 1.0 - steps_left / max(1, max_steps)
    is_behind = pct_done < pct_time_used - 0.10

    has_adversary = "ADVERSARY" in market_text and "DETECTED" in market_text
    has_volatility = "sigma=0.06" in market_text or "VOLATILE" in market_text or "3x" in market_text

    if has_adversary:
        return "RANDOMIZE"
    if is_behind and rate >= _AGGRESSIVE_THRESHOLD:
        return "AGGRESSIVE"
    if has_volatility:
        return "DARK"
    if rate >= _AGGRESSIVE_THRESHOLD:
        return "AGGRESSIVE"
    return "PASSIVE"


def _make_reasoning(
    strategy: str,
    shares_remaining: int,
    total_shares: int,
    steps_left: int,
    rate: float,
) -> str:
    """Generate a concise reasoning string for the heuristic decision."""
    pct_rem = shares_remaining / max(1, total_shares) * 100
    if strategy == "RANDOMIZE":
        return f"HFT adversary detected — adding jitter (rate={rate:.3f}) to break autocorrelation."
    if strategy == "AGGRESSIVE":
        return f"{pct_rem:.0f}% shares remain, {steps_left} steps left — accelerating to catch deadline."
    if strategy == "DARK":
        return f"High volatility session — routing {rate:.3f} via dark pool to hide impact."
    return f"{pct_rem:.0f}% shares remain, {steps_left} steps — passive pace to minimize market impact."


def generate_dataset(
    n_episodes: int = 200,
    tasks: list[str] | None = None,
    use_jitter: bool = True,
    seed_start: int = 0,
) -> list[dict]:
    """
    Run heuristic agent on TradeExecGym, collect (prompt, completion) pairs.

    Args:
        n_episodes: Number of episodes to simulate.
        tasks: List of task_ids to sample. Defaults to all 5 tasks.
        use_jitter: Add Gaussian noise to rates for diversity (recommended).
        seed_start: Starting seed for reproducibility.

    Returns:
        List of dicts with 'prompt' and 'completion' keys (HF Dataset format).
    """
    if tasks is None:
        tasks = [t for t, _ in TASK_POOL]

    env = TradeExecEnvironment()
    heuristic = AlmgrenChrissHeuristic()
    dataset = []

    print(f"[>>] Generating {n_episodes} episodes across tasks: {tasks}")

    for ep_idx in range(n_episodes):
        task_id = tasks[ep_idx % len(tasks)]
        seed = seed_start + ep_idx * 7 + 42

        try:
            obs = env.reset(task_id=task_id, seed=seed)
        except Exception as e:
            print(f"[WARN] Reset failed for {task_id} seed={seed}: {e}")
            continue

        max_steps = env._max_steps
        total_shares = env._total_shares

        # Episode loop
        for step in range(max_steps):
            shares_remaining = env._shares_remaining
            steps_left = max(1, max_steps - step)

            if shares_remaining <= 0:
                break

            # Get market state text (what the LLM sees)
            try:
                market_text = env.get_market_state()
            except Exception:
                market_text = f"Step {step}: shares_remaining={shares_remaining}"

            # Heuristic decision
            if use_jitter:
                rate = heuristic.calculate_rate_with_jitter(
                    shares_remaining=shares_remaining,
                    total_shares=total_shares,
                    steps_left=steps_left,
                    current_is=0.0,
                    jitter_std=0.015,
                )
            else:
                rate = heuristic.calculate_rate(
                    shares_remaining=shares_remaining,
                    total_shares=total_shares,
                    steps_left=steps_left,
                    current_is=0.0,
                )

            strategy = _classify_strategy(
                rate=rate,
                market_text=market_text,
                shares_remaining=shares_remaining,
                total_shares=total_shares,
                steps_left=steps_left,
                max_steps=max_steps,
            )
            reasoning = _make_reasoning(strategy, shares_remaining, total_shares, steps_left, rate)

            # Dark pool fraction: use it in volatile sessions
            dark_frac = 0.0
            if strategy == "DARK":
                dark_frac = round(0.30 + (hash(seed + step) % 10) / 100, 2)

            completion = json.dumps({
                "strategy": strategy,
                "participation_rate": round(rate, 4),
                "dark_pool_fraction": dark_frac,
                "reasoning": reasoning,
            })

            # Prompt in chat format (matches GRPOTrainer expected format)
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": market_text},
            ]

            dataset.append({"prompt": prompt, "completion": completion})

            # Step the environment
            try:
                env.execute_trade(
                    participation_rate=rate,
                    use_dark_pool=(dark_frac > 0),
                    dark_pool_fraction=dark_frac,
                )
            except Exception as e:
                print(f"[WARN] Step failed ep={ep_idx} step={step}: {e}")
                break

            if env._episode_done:
                break

        if (ep_idx + 1) % 50 == 0:
            print(f"    Episode {ep_idx + 1}/{n_episodes} — dataset size: {len(dataset)}")

    print(f"[OK] Dataset generation complete: {len(dataset)} samples from {n_episodes} episodes")
    return dataset


def save_dataset(dataset: list[dict], output_path: str) -> None:
    """Save dataset to disk in HuggingFace datasets format."""
    try:
        import datasets as hfds
        ds = hfds.Dataset.from_list(dataset)
        ds.save_to_disk(output_path)
        print(f"[OK] Saved HF dataset → {output_path}/")
        print(f"     Columns: {ds.column_names}")
        print(f"     Rows:    {len(ds)}")
    except ImportError:
        # Fallback: save as JSONL
        jsonl_path = output_path + ".jsonl"
        os.makedirs(os.path.dirname(jsonl_path) if os.path.dirname(jsonl_path) else ".", exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row) + "\n")
        print(f"[OK] Saved JSONL dataset → {jsonl_path} (install 'datasets' for HF format)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GRPO training dataset")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of episodes to simulate (default: 200)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task IDs to sample (default: all 5)")
    parser.add_argument("--output", default="training/grpo_dataset",
                        help="Output path for the dataset")
    parser.add_argument("--no-jitter", action="store_true",
                        help="Disable rate jitter (less diverse dataset)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed for reproducibility")
    args = parser.parse_args()

    data = generate_dataset(
        n_episodes=args.episodes,
        tasks=args.tasks,
        use_jitter=not args.no_jitter,
        seed_start=args.seed,
    )
    os.makedirs(args.output, exist_ok=True) if not args.output.endswith(".jsonl") else None
    save_dataset(data, args.output)

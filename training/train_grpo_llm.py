"""
train_grpo_llm.py — Train an LLM on TradeExecGym using TRL GRPOTrainer.

This is the primary training script. Uses:
  - Qwen2.5-0.5B-Instruct (fits 4 GB VRAM; or runs CPU-only ~4h)
  - TRL GRPOTrainer with 3 verifiable reward functions
  - TradeExecGym heuristic dataset as training corpus

The reward signal is fully verifiable (no human labels):
  format_reward   → Is it valid JSON with correct keys?
  strategy_reward → Does strategy match market context?
  quality_reward  → Is the rate in the AC-optimal range?

Hardware targets:
  Minimum:     CPU-only (use --bf16=false --fp16=false)
  Recommended: GPU 8 GB+ (T4 / A10 on Colab)
  Fast:        A100 on Colab Pro

Quickstart (CPU dry run, 5 episodes):
  python training/train_grpo_llm.py --dry-run

Full training (GPU, 200 episodes):
  python training/train_grpo_llm.py --episodes 200 --model Qwen/Qwen2.5-0.5B-Instruct

After training, use the model in inference.py:
  MODEL_NAME = "models/grpo_llm"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency check (training extras are optional)
# ─────────────────────────────────────────────────────────────────────────────

def _check_deps() -> bool:
    """Return True if all training deps are available."""
    missing = []
    for pkg in ["trl", "transformers", "datasets", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing training dependencies: {missing}")
        print("        Install with:")
        print("          pip install trl transformers datasets torch accelerate peft")
        print("        Or: pip install 'trade_exec_gym[training-llm]'")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = "models/grpo_llm"
DEFAULT_DATASET = "training/grpo_dataset"


def train_grpo(
    model_name: str = DEFAULT_MODEL,
    dataset_path: str = DEFAULT_DATASET,
    output_dir: str = DEFAULT_OUTPUT,
    n_episodes: int = 200,
    num_generations: int = 4,    # GRPO group size G
    max_new_tokens: int = 256,
    learning_rate: float = 5e-7,
    batch_size: int = 1,
    grad_accum: int = 4,
    use_bf16: bool = False,
    use_fp16: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Full GRPO training pipeline.

    Steps:
      1. Generate dataset (if not already on disk)
      2. Load tokenizer + model
      3. Configure GRPOTrainer with 3 reward functions
      4. Train
      5. Save model + training summary

    Returns:
        dict with training_loss, global_step, model_path.
    """
    if not _check_deps():
        sys.exit(1)

    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoTokenizer
    import datasets as hfds

    from training.reward_functions import format_reward, strategy_reward, quality_reward

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    if dry_run:
        print("[DRY RUN] Generating mini dataset (10 episodes)...")
        from training.generate_grpo_dataset import generate_dataset
        data = generate_dataset(n_episodes=10, tasks=["task_1"])
        dataset = hfds.Dataset.from_list(data)
    elif os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        print(f"[>>] Loading existing dataset from {dataset_path}...")
        dataset = hfds.load_from_disk(dataset_path)
    else:
        print(f"[>>] Dataset not found at '{dataset_path}'. Generating {n_episodes} episodes...")
        from training.generate_grpo_dataset import generate_dataset, save_dataset
        data = generate_dataset(n_episodes=n_episodes)
        save_dataset(data, dataset_path)
        dataset = hfds.Dataset.from_list(data)

    print(f"[OK] Dataset: {len(dataset)} samples ready")

    if dry_run and len(dataset) < 4:
        print("[DRY RUN] Dataset too small for GRPO (need ≥ 4 samples). Padding...")
        # Duplicate rows to meet minimum
        rows = [dataset[i % len(dataset)] for i in range(8)]
        dataset = hfds.Dataset.from_list(rows)

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    print(f"[>>] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. GRPO Config ────────────────────────────────────────────────────────
    # Dry run: minimal steps to verify the pipeline works without GPU
    training_steps = 5 if dry_run else None  # None = full epoch

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=max_new_tokens,
        temperature=0.9,
        logging_steps=1 if dry_run else 10,
        save_steps=50,
        report_to="none" if dry_run else "tensorboard",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        # Limit steps in dry-run mode
        max_steps=training_steps if dry_run else -1,
    )

    # ── 4. Trainer ────────────────────────────────────────────────────────────
    print(f"[>>] Building GRPOTrainer...")
    print(f"     Model:          {model_name}")
    print(f"     Dataset size:   {len(dataset)} samples")
    print(f"     Group size G:   {num_generations}")
    print(f"     Max tokens:     {max_new_tokens}")
    print(f"     LR:             {learning_rate}")
    print(f"     BF16/FP16:      {use_bf16}/{use_fp16}")
    print(f"     Dry run:        {dry_run}")

    trainer = GRPOTrainer(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[
            format_reward,    # Weight: JSON structure compliance (fast)
            strategy_reward,  # Weight: Strategy-context alignment (fast)
            quality_reward,   # Weight: Rate quality + reasoning (fast)
        ],
        processing_class=tokenizer,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("[>>] Starting GRPO training...")
    train_result = trainer.train()

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    if not dry_run:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[OK] Model saved → {output_dir}/")

    summary = {
        "model": model_name,
        "dataset_size": len(dataset),
        "training_loss": round(train_result.training_loss, 6),
        "global_step": train_result.global_step,
        "output_dir": output_dir,
        "dry_run": dry_run,
    }

    summary_path = os.path.join(output_dir if not dry_run else ".", "training_summary.json")
    os.makedirs(os.path.dirname(summary_path) if os.path.dirname(summary_path) else ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("GRPO Training Complete")
    print("=" * 60)
    print(f"  Training loss:  {summary['training_loss']:.6f}")
    print(f"  Steps:          {summary['global_step']}")
    print(f"  Model saved:    {output_dir if not dry_run else '(skipped — dry run)'}")
    print()
    print("  To use this model in inference.py, set:")
    print(f'    MODEL_NAME = "{output_dir}"')
    print("=" * 60)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LLM on TradeExecGym using TRL GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Episodes to generate if dataset doesn't exist",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="Path to saved HF dataset (generated if missing)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--generations", type=int, default=4,
        help="GRPO group size G (completions per prompt)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-7,
        help="Learning rate",
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use bfloat16 (requires Ampere+ GPU)",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use float16 (T4 / older GPUs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 5 training steps to validate pipeline (no GPU needed)",
    )
    args = parser.parse_args()

    train_grpo(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        n_episodes=args.episodes,
        num_generations=args.generations,
        learning_rate=args.lr,
        use_bf16=args.bf16,
        use_fp16=args.fp16,
        dry_run=args.dry_run,
    )

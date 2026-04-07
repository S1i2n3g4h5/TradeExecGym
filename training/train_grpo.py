"""
train_grpo.py -- PPO-based RL training with Group-Relative reward shaping.

This script trains a Proximal Policy Optimization (PPO) agent on the
TradeExecGym environment. The reward function implements Group-Relative
reward shaping: each step's reward is computed relative to the TWAP shadow
baseline, so the agent learns by comparing its performance to a reference
group (the TWAP strategy) rather than using absolute reward signals.

This is conceptually aligned with GRPO (Group Relative Policy Optimization,
Shao et al. 2024): the advantage signal is the IS delta vs TWAP, not an
absolute value. The agent that consistently beats TWAP within a rollout
batch receives positive relative advantage.
"""

from stable_baselines3 import PPO
from env.gym_wrapper import TradeExecGymEnv
import os


def train_ppo_with_relative_reward(
    task_id: str = "task1_twap_beater",
    total_timesteps: int = 5_000,
    base_url: str = "http://localhost:7860",
) -> None:
    """Train a PPO agent with group-relative (TWAP-benchmarked) reward shaping.

    The environment's reward function (env/reward.py) computes IS delta vs
    the TWAP shadow baseline at each step, which implements the group-relative
    advantage signal. A positive reward means the agent is beating TWAP on
    this rollout; negative means it is worse than the TWAP reference group.

    Args:
        task_id: Which task to train on (default: task1_twap_beater).
        total_timesteps: Total environment steps for training.
        base_url: URL of the running TradeExecGym server.
    """
    env = TradeExecGymEnv(base_url=base_url, task_id=task_id)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    print(f"[>>] Training PPO agent (group-relative reward vs TWAP) on {task_id}...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    os.makedirs("models", exist_ok=True)
    model.save("models/grpo_agent")
    print("[OK] PPO agent (GRPO-style relative reward) trained and saved to models/grpo_agent.")


if __name__ == "__main__":
    train_ppo_with_relative_reward()

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env.gym_wrapper import TradeExecGymEnv
import os

def train():
    # Setup environment
    env = TradeExecGymEnv(task_id="task1_twap_beater")
    
    # Model configuration
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=2000, 
        save_path='./models/',
        name_prefix='sor_ppo_agent'
    )
    
    print("🚀 Starting Grade-level RL Training...")
    model.learn(
        total_timesteps=10_000, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/sor_ppo_final")
    print("✅ Training complete. Model saved to models/sor_ppo_final.zip")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    train()

# 🥇 Judges' Cheat Sheet: Why TradeExecGym Wins

**TradeExecGym** is not just another RL environment. It is a production-grade, agent-centric trading infrastructure built for the Meta x HuggingFace OpenEnv Hackathon.

## 1. Scientific Depth (Almgren-Chriss Physics)
Most trading environments use simple random walks. We use the **Almgren-Chriss (2000)** market impact model—the industry standard for institutional block trades. Every action has permanent and temporary price impact, governed by real-world friction.

## 2. Institutional Realism (Microstructure)
We simulate the **L2 Order Book**. Agents don't just see "the price"; they see the spread, the bid/ask queue depth, and the volume imbalance. This forces LLMs to reason like high-frequency traders, not just simple calculators.

## 3. Recursive Intelligence (GRPO Training)
We've moved beyond standard PPO. Our environment supports **Group Relative Policy Optimization (GRPO)** using the `trl` framework.
- **Verifiable Rewards**: Our 3-component reward setup automatically grades LLM completions for format compliance, strategic alignment, and execution quality.
- **Bootrapped Learning**: We include a heuristic-to-dataset generator for accelerating the cold-start problematic in RL.

## 4. The Adversarial "X-Factor"
Task 4 features a **Pattern-Matching HFT Adversary**. If an agent's trading participation rate is too uniform or autocorrelated, the adversary detects the pattern and front-runs the order, penalizing the score. This forces the LLM to learn **strategic randomization** (stealth execution).

## 5. Procedural Market Regimes
The environment is alive. It procedurally transitions through:
- **Flash Crashes**: Sudden 5x volatility spikes and wide spreads.
- **Liquidity Crises**: Dark pools go offline, forcing agents to lit venues.
- **Momentum Shifts**: Structural price trends that reward patience or aggression.

## 6. Rigorous Robustness Validation
We include a **5-Layer Robustness Gauntlet** that mathematically proves the reward signal is sane. It verifies that a random agent *cannot* beat a TWAP agent, and a TWAP agent *cannot* beat an AC-Optimal agent. The skill gradient is monotonic and scientifically verified.

## 7. Unified SOR Architecture
We have solved the common "port conflict" problem in Hugging Face Spaces by mounting the Gradio UI directly onto the FastAPI server. This provides a unified entry point at port 7860 where the API is instant (<10ms) and the UI is beautiful.

---
**TradeExecGym: Real Physics. Real Finance. Real Intelligence.**

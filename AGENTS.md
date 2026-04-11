# TradeExecGym: Agentic Training Strategy

TradeExecGym moves beyond standard Reinforcement Learning by implementing **Group Relative Policy Optimization (GRPO)** specifically for financial decision-making.

## 🧠 The Agentic Core
Our agents utilize a structured **Chain-of-Thought (CoT)** reasoning loop to handle institutional constraints:

1.  **State Observation**: The model receives a rich L2 Order Book snapshot and Market Regime identifier.
2.  **Rationalization**: The model reasons about alpha decay, spread costs, and potential HFT adversaries.
3.  **Action**: The model outputs a structured JSON block (Strategy, Rate, Dark Pool Usage, Reasoning).

## 🚀 GRPO: Group Relative Policy Optimization
Instead of training against a flat reward signal, our GRPO pipeline (implemented in `training/train_grpo_llm.py`) uses a group of completions to calculate relative advantage.

### Verifiable Reward Functions
We use three deterministic, stateless reward functions that provide "Rule-based Alignment" for the model without human labels:

*   **Format Reward**: Rewards the model for maintaining strict JSON schemas.
*   **Context Reward**: Rewards the model for selecting strategies that match the market regime (e.g., using `DARK` during `LIQUIDITY_CRISIS`).
*   **Efficiency Reward**: Rewards participation rates that align with the mathematically optimal **Almgren-Chriss** execution schedule.

## 🛠️ The Curriculum
We train across an adaptive 5-task curriculum:
1.  **TWAP Beater**: Basic linear execution.
2.  **VWAP Optimizer**: Tracking U-shaped volume distributions.
3.  **Volatile Execution**: Managing impact in 3x Sigma environments.
4.  **Adversarial HFT**: Learning to randomize rates to evade pattern detectors.
5.  **Deadline Pressure**: High-stress terminal inventory clearance.

## 📈 Performance Benchmarking
Every agent is gauged against three baseline tiers:
- **Naive TWAP**: The minimum standard.
- **VWAP Baseline**: Volume-weighted tracking.
- **AC Optimal**: The theoretical mathematical limit.

TradeExecGym enables models to bridge the gap between "standard heuristics" and "agentic awareness," allowing them to outperform math-only solutions by detecting structural market regimes.

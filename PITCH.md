# 💎 TradeExecGym: The Institutional-Grade Trading Agent Playground

**Mission**: Bridging the gap between toy RL environments and the chaotic reality of high-frequency institutional trading.

---

## 🚀 The "Wow" Factor: Phase 2 Recursive Update

TradeExecGym has evolved from a simple baseline into a multi-layered simulation engine that challenges even the most advanced LLMs.

### 1. 🧬 Group Relative Policy Optimization (GRPO) Ready
We have implemented the full **GRPO training infrastructure**. Using the `trl` library, our environment allows agents to self-improve via a triple-reward alignment system:
- **Logical Coherence**: Verifies the agent's Chain-of-Thought reasoning.
- **Strategic Alignment**: Rewards matching market regimes (e.g., being patient in low liquidity).
- **Execution Quality**: Penalizes deviations from the mathematical Almgren-Chriss optimum.

### 2. 📈 Institutional Microstructure (L2 Book)
We moved beyond "price point" simulation. Agents now observe a **10-level deep L2 Order Book**, providing real-time spread and depth data. They must detect and respond to:
- **Iceberg Orders**: Hidden liquidity that masks true market impact.
- **Volume Imbalance**: Predictive signals for short-term price movements.
- **Adversarial HFTs**: Predatory bots that attempt to front-run predictable patterns.

### 🛡️ Scientifically Robust
Our **5-Layer Robustness Gauntlet** proves the environment's integrity. By verifying a monotonic skill gradient from random agents to mathematical optima, we guarantee that the environment provides a genuine, non-noisy learning signal for LLMs.

---

## 🏛️ Architecture: The Unified SOR
We follow a **Unified Smart Order Router (SOR)** architecture. The FastAPI backend serves low-latency OpenEnv API calls, while the Gradio UI provides high-fidelity "LLM Observability."
- **Institutional Physics**: Almgren-Chriss engine (Citadel/GS standard).
- **Unified Port 7860**: Everything—API and UI—runs on a single port for seamless deployment.
- **Agent Sandbox**: Native support for Llama 3, GPT-4, and custom local models.

---

## 🏆 Why We Win
TradeExecGym isn't just an environment; it's a **recursive improvement loop**. It allows an agent to trade, analyze its transaction costs (TCA), and then train itself to do better on the next episode.

**Physics + Finance + Intelligence. TradeExecGym.**

# 🏦 TradeExecGym — The Gold Standard SOR Training Environment
## Physics-Grounded Order Execution for Meta OpenEnv

Institutional execution is where "general" RL usually breaks. You can't just train a model to trade on raw price arrays and expect it to survive a $100M block trade. TradeExecGym is built to bridge that gap. It is a high-fidelity Smart Order Router (SOR) simulator grounded in the **Almgren-Chriss (2001)** market impact model — the same quantitative physics engine used by global hedge funds to manage billions in liquidity.

> [!IMPORTANT]
> **The Problem:** Institutional traders lose $50B–$100B annually to "Implementation Shortfall" (IS). If you buy too fast, you crash the order book (Market Impact). If you buy too slow, the price drifts away (Volatility Risk). TradeExecGym provides the definitive testbed for RL agents to solve this $100B optimization problem.

---

## 🔬 The Physics Engine: Almgren-Chriss (2001)

We abandoned the "random walk" toy models common in FinRL. TradeExecGym uses a path-dependent liquidity model based on the *Journal of Risk* gold standard:

1.  **Permanent Price Impact ($\gamma$):** Every trade you make permanently shifts the stock's fundamental price level by removing available supply. This impact is irreversible.
2.  **Temporary Price Impact ($\eta$):** The "liquidity premium" you pay to cross the spread and eat the immediate order book. This impact fades after the trade.
3.  **Brownian Motion ($\sigma$):** The underlying "fair value" wanders randomly. Staying in the market too long exposes you to "Variance Risk."

**The Core Dynamic:**
- **Trade FAST:** High impact, low drift risk (High Slippage).
- **Trade SLOW:** Low impact, high drift risk (Inventory Risk).
- **Optimal Policy:** The agent must find the mathematical "Sweet Spot" described by the AC-Optimal hyperbolic sine solution.

---

## 🤖 LLM-Native Observability (CoT Enabled)

TradeExecGym is built natively for Meta's **OpenEnv**. While standard environments return unlabelled float arrays like `[150.22, 0.45, 0.12]`, TradeExecGym generates **Market Narratives**.

Every environmental step translates raw physics into plain-English situational reports:
> *"MARKET STATE: Mid Price is $150.40. Volume is normal. ⚠️ ADVERSARY ALERT: HFT pattern detection is Active. Uniform trading behavior is being penalized!"*

This allows modern LLMs and GRPO agents to use **Chain-of-Thought (CoT)** reasoning. The model doesn't just see a number; it understands *why* it’s being punished by a predatory adversary.

---

## 🏁 The Curriculum: 5 Tiers of Difficulty

We designed a progressive training path to move agents from baseline behavior to institutional-grade execution:

| Task | Market Regime | Key Challenge |
| :--- | :--- | :--- |
| **Tier 1: TWAP Beater** | Stable | Beat the naive Time-Weighted Average Price. |
| **Tier 2: VWAP Optimizer** | U-Shaped | Track intraday volume surges at Open/Close. |
| **Tier 3: Fat-Tail Volatility** | Chaotic | Manage risk during 5.0% annualized variance spikes. |
| **Tier 4: Adversary Hunter** | Predatory | Dodge HFT bots that front-run predictable patterns. |
| **Tier 5: The Deadline Cliff** | Liquidation | Solve the "Panic Sell" problem at the 4:00 PM close. |

---

## 📊 The Professional Dashboard

TradeExecGym includes a multi-tab Gradio application optimized for judge evaluation:

*   **Auto Simulation:** Benchmark trained RL agents (PPO/GRPO) against TWAP and VWAP baselines in real-time.
*   **Human Challenge:** Manually execute steps to experience "Order Book Pressure" first-hand.
*   **Performance Scorecard:** Live tracking of Slippage (bps) and Grader Scores vs the AC-Optimal frontier.

### Quick Start
```bash
# 1. Start the Market Engine
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860

# 2. Launch the Control Dashboard
uv run python ui/app.py --port 7861
```

---

## 🛡️ Hackathon Compliance
- ✅ **OpenEnv Native:** Full implementation of `reset()`, `step()`, and `state()` via FastAPI.
- ✅ **Standardized Schemas:** Uses Pydantic for all `ExecutionAction` and `Observation` models.
- ✅ **3+ Baselines:** Comparison against TWAP, VWAP, and mathematical optimum.
- ✅ **Physics-Grounded:** Almgren, R., & Chriss, N. (2001). *Optimal execution of portfolio transactions.* Journal of Risk.

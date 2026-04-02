# TradeExecGym

Reinforcement learning for finance usually fails because agents get handed raw arrays of unlabelled numbers and are expected to magically learn market micro-structure. We built TradeExecGym to fix that data format problem. It runs natively on Meta's OpenEnv framework.

We spent half the hackathon wrestling the Almgren-Chriss market impact equations into a fast Python backend. The result is a system where trading actually hurts. If you buy 1 million shares of a stock at once, you will crash the order book and ruin your fill price. You have to slice the order delicately over time.

## The Physics Engine

We abandoned fake random walks. The market state is calculated using standard quantitative models that penalize poor execution.

1. **Permanent Price Impact.** When you buy, you remove supply. The fundamental price of the stock shifts upward permanently.
2. **Temporary Price Impact.** Eating through the order book's immediate liquidity costs extra cash right now.
3. **Brownian Motion Drift.** The underlying asset price wanders randomly while you try to execute your order. Stay in the market too long and volatility might kill your trade.
4. **Information Leakage.** High-frequency traders watch the tape. If your execution pattern is too predictable, they front-run you.

Success is measured strictly by Implementation Shortfall (IS). That means the difference between the price when you decided to buy and the average price you actually paid. We map this to a flat 0.0 to 1.0 grader score so the RL optimizers don't break during high volatility.

## The Curriculum

We designed five specific scenarios to train agents. 

* **The TWAP Beater.** A baseline intro. Execute an order smoothly.
* **VWAP Optimizer.** Market volume surges at the open and the close. The agent has to find the U-shape.
* **Volatile Execution.** The market is chaotic. 
* **Adversarial HFT.** A predatory algorithm tracks your trades. If your participation rate standard deviation drops below 0.005, it hits you with a massive slippage penalty.
* **Deadline Cliff.** Any shares left over at the end of the session get force-liquidated at awful prices.

## LLM-First Observability

This is why we built on OpenEnv. 

Standard RL tools return `[150.2, 0.45, 120]`. You can't feed that to Llama 3 and expect it to reason about High-Frequency Trading predators.

We injected an LLM Narrative hook directly into the step loop. Every time the task advances, the math is translated into plain-English strings. The environment actively warns the agent with alerts like "⚠️ ADVERSARY ALERT: HFT pattern detection isActive." 

Language models can actually read this status and deploy Chain-of-Thought logic to change their trading cadence. 

## The Dashboard

We included a multi-tab web application built in Gradio on port 7861. 

You can load a trained model checkpoint to watch the GRPO agent trade entirely on its own. Or you can select the human challenge mode and manually execute step-by-step to see how hard it is to beat a simple time-weighted average price baseline without triggering the adversary detector. 

To run it locally:
```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
uv run python ui/app.py --port 7861
```

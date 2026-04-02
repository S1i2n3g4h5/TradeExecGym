---
title: TradeExecGym
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
tags:
- openenv
- reinforcement-learning
- finance
---

# TradeExecGym: institutional Smart Order Router (SOR)

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-emerald)](https://github.com/meta-pytorch/OpenEnv)
[![Built for Hackathon](https://img.shields.io/badge/Hackathon-Meta%20x%20HuggingFace-blue)](https://huggingface.co/spaces)

TradeExecGym is a high-fidelity Reinforcement Learning environment for **Institutional Trade Execution**. It simulates the "Implementation Shortfall" (IS) problem: How to buy or sell millions of shares without moving the market price against yourself.

## 📈 Real-World Utility
Institutional traders do not buy 1,000,000 shares in one click. Doing so would exhaust the order book and cause massive "Slippage." Instead, they use Smart Order Routers (SORs) to slice the order into thousands of small trades. 

TradeExecGym uses the **Almgren-Chriss (2000)** market impact model to provide a physics-grounded simulation of:
- **Temporary Impact:** The instant cost of liquidity consumption.
- **Permanent Impact:** The supply/demand shift caused by your trade.
- **Adversarial HFT:** Predatory bots that detect and exploit predictable trading patterns.

---

## 🏗️ Environment Specification (OpenEnv)
- **Observation Space:** Structured market data (Mid-price, Volatility, Inventory, Time-remaining) and high-level LLM Narratives.
- **Action Space:** `participation_rate` (Continuous: 0.0 to 0.25). Represents the percentage of market volume the agent targets.
- **Rewards:** Continuous reward based on the delta between the agent's execution price and the Arrival Price (IS minimization), with bonuses for beating TWAP/VWAP baselines.

---

## 🎯 Curriculum Tasks
1. **task1_twap_beater (Easy):** 100K shares, 30 steps. Low volatility.
2. **task2_vwap_optimizer (Medium):** 250K shares. Must follow the U-shaped intraday volume curve.
3. **task3_volatile_execution (Hard):** 400K shares. High volatility (3x). Requires Dark Pool usage.
4. **task4_adversarial (Very Hard):** HFT bots are active. Predictable strategies will be "sniped."
5. **task5_deadline_pressure (Extreme):** 1M shares. Extreme penalties for failing to complete the order.

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
uv pip install -e .
```

### 2. Launch the Server
```bash
uv run uvicorn server.app:app --port 7860
```

### 3. Run Inference (Compliance Mode)
```bash
$env:HF_TOKEN = "your_token"
uv run python inference.py
```

---

## 🐳 Docker Deployment
```bash
docker build -t trade-exec-gym .
docker run -p 7860:7860 trade-exec-gym
```

## 📊 Baseline Scores (GP-4o Hybrid)
| Task | Avg IS (bps) | Grader Score |
|------|--------------|--------------|
| Task 1 | 24.2 | 0.88 |
| Task 2 | 19.5 | 0.84 |
| Task 3 | 42.1 | 0.79 |
| Task 4 | 55.0 | 0.72 |
| Task 5 | 88.0 | 0.65 |

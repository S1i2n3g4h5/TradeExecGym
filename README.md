---
title: TradeExecGym
emoji: 📈
colorFrom: emerald
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - finance
  - smart-order-routing
  - almgren-chriss
  - institutional-trading
---

<div align="center">

# 📈 TradeExecGym

### *Institutional Smart Order Routing — Powered by Quantitative Market Physics*

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-v0.2.1%20Compliant-10b981?style=for-the-badge&logo=meta)](https://github.com/meta-pytorch/OpenEnv)
[![Built for Hackathon](https://img.shields.io/badge/Meta%20×%20HuggingFace-Hackathon-6366f1?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-f59e0b?style=for-the-badge)](./LICENSE)

> **The reinforcement learning environment that Wall Street's quant desks actually use.**
> Built on the Almgren-Chriss (2000) execution model. No toy random walks. No fake markets.

[**Live Demo**](https://huggingface.co/spaces/SST-MetaHuggingFace/trade-exec-gym) · [**API Docs**](#-api-reference) · [**Quick Start**](#-quick-start) · [**Architecture**](#-architecture)

</div>

---

## 📌 What is TradeExecGym?

Institutional traders don't buy 1,000,000 shares in a single click. Doing so would exhaust the order book and cause catastrophic **slippage** — driving the price up against themselves. They use **Smart Order Routers (SOR)** to slice the order into thousands of smaller trades, executed over time across multiple venues.

TradeExecGym simulates this problem at quantitative precision. It is a **reinforcement learning testbed** for agents that must minimize **Implementation Shortfall (IS)** — the difference between where the market was *when you decided to trade* and *where you actually got filled*.

### Why this is hard

| The Dilemma | If you trade... | The consequence |
|---|---|---|
| **Too Fast** | Aggressively (large blocks) | You eat through liquidity. Price impact permanently moves the market against you. |
| **Too Slow** | Passively (tiny trickles) | Random Brownian drift accumulates. You leak alpha. HFT bots detect your pattern and front-run you. |
| **Predictably** | With constant uniform rates | HFT adversaries detect the statistical signature and apply targeted slippage penalties. |

The optimal strategy lives in the mathematical tension between these three forces. **Almgren-Chriss solved this in 2000. We built an environment around it.**

---

## 🏛️ Environment Specification

### Core Identity

| Property | Value |
|---|---|
| **Name** | `trade_exec_gym` |
| **Version** | `1.0.0` |
| **Framework** | Meta OpenEnv v0.2.1 |
| **Runtime** | FastAPI + FastMCP |
| **Protocol** | MCP (Model Context Protocol) native |
| **Concurrency** | Up to 5 simultaneous sessions |

### Spaces & Signals

| Dimension | Type | Range | Description |
|---|---|---|---|
| **Action** | `participation_rate` | `[0.0, 0.25]` | Fraction of Average Daily Volume to target per step |
| **Observation** | Market State Text | Natural Language | Narrative + structured data snapshot |
| **Reward** | Per-step IS delta | `[-1.0, +1.0]` | GRPO-compatible bounded sigmoid over IS basis points |
| **Episode** | Variable | 30 – 120 steps | Task-dependent time horizon |

### The Physics Engine

Every step in TradeExecGym runs three simultaneous physics calculations:

```
1. Permanent Impact   →  Δprice_perm = λ · σ · √q · sgn(order)
2. Temporary Impact   →  Δprice_temp = η · (q / ADV_per_step)
3. Brownian Drift     →  ΔS = σ · √Δt · ε   (ε ~ N(0,1))
```

Where `λ` is the permanent impact coefficient, `η` is the temporary impact coefficient, `σ` is realized volatility, and `q` is the order size in shares. This is the Almgren-Chriss (2000) model — the same framework used by Goldman, Citadel, and every major systematic trading desk.

---

## 🎯 Curriculum Tasks

Five tasks. Increasing difficulty. Each designed to break a different class of naive agent.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TASK 1: The TWAP Beater                                         EASY 🟢    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  100K shares · 30 steps · Low volatility (σ = 0.02)                         │
│  Beat equal-time-slice execution. Entry-level benchmark.                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  TASK 2: VWAP Optimizer                                        MEDIUM 🟡    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  250K shares · 60 steps · Intraday U-curve volume profile                    │
│  Must track the real market volume rhythm: high open, dead lunch, spike      │
│  close. Static TWAP pacing gets crushed by market microstructure.            │
├──────────────────────────────────────────────────────────────────────────────┤
│  TASK 3: Volatile Execution                                      HARD 🔴    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  400K shares · 90 steps · 3× volatility (σ = 0.06) · Dark Pool available    │
│  Flash crash conditions. Variance penalty triggers margin calls. Smart       │
│  routing to dark pools (40% fill rate, zero market impact) is critical.      │
├──────────────────────────────────────────────────────────────────────────────┤
│  TASK 4: Adversarial HFT                                    VERY HARD 🟣    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  200K shares · 120 steps · Active HFT predator                               │
│  A predatory algo watches your trade signature. If your participation rate   │
│  standard deviation drops below 0.005 (you're too uniform), it front-runs   │
│  you and slaps a 50 bps penalty on your next fill. Be erratic. Stay alive.  │
├──────────────────────────────────────────────────────────────────────────────┤
│  TASK 5: Deadline Cliff                                       EXTREME ⚫    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  1M shares · 80 steps · Hard legal deadline                                  │
│  Any unexecuted shares at step 80 are market-ordered at the worst available  │
│  bid. The penalty is catastrophic. You must complete the order. No excuses.  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📡 API Reference

TradeExecGym exposes **4 MCP tools** via its FastAPI + FastMCP backend. These are the only interaction points for agents — both LLM tool-callers and RL policy networks.

### `GET /health` — Health Check

```bash
curl http://localhost:7860/health
# → {"status": "ok", "env": "trade_exec_gym", "version": "1.0.0"}
```

---

### `POST /reset` — Initialize Episode

Start a new trading session for a given task.

```http
POST /reset
Content-Type: application/json

{
  "task_id": "task1_twap_beater",
  "seed": 42
}
```

**Available `task_id` values:**

| ID | Difficulty | Shares | Steps |
|---|---|---|---|
| `task1_twap_beater` | Easy | 100,000 | 30 |
| `task2_vwap_optimizer` | Medium | 250,000 | 60 |
| `task3_volatile_execution` | Hard | 400,000 | 90 |
| `task4_adversarial` | Very Hard | 200,000 | 120 |
| `task5_deadline_pressure` | Extreme | 1,000,000 | 80 |

**Response:** Structured `Observation` object with episode ID, market narrative, and objectives.

---

### Tool: `get_market_state()` — Read Environment

Returns a rich natural-language + structured snapshot of current market conditions.

```
MARKET STATE — Step 12/30
────────────────────────────────────────────────────────
NARRATIVE: Volume is spiking at the open. Buying urgency is moderate.

INVENTORY
  Executed:      48,000 / 100,000 (48.0%)
  Remaining:     52,000 shares
  Time left:     18 steps

PRICES
  Mid Price:     $150.4821
  Arrival Price: $150.0000  ← IS benchmark (fixed)
  Spread:        5.5 bps

MARKET CONDITIONS
  Volume Ratio:  1.60×  (1.0 = normal daily avg)
  Session:       open  (open/midday/close)
  Dark Pool:     ✅ Available (~40% fill rate)

PERFORMANCE  (IS = basis points, lower = better)
  Your IS:   18.32 bps   ✅ Beating TWAP by 6.1 bps
  TWAP IS:   24.44 bps
  VWAP IS:   19.55 bps

ACTION: execute_trade(participation_rate=X)
  Suggested: 0.01–0.05 (passive) | 0.10–0.20 (aggressive)
```

---

### Tool: `execute_trade(...)` — Primary Action

The core action tool. Dispatches an order to the simulated market.

```python
execute_trade(
    participation_rate: float,   # [0.0, 0.25] — fraction of ADV to target
    use_dark_pool: bool = False,  # route a portion to dark liquidity
    dark_pool_fraction: float = 0.0,  # [0.0, 1.0] portion to send dark
    order_type: str = "MARKET",   # "MARKET" | "LIMIT"
    limit_offset_bps: float = 0.0 # limit price offset in bps (if LIMIT)
)
```

**Response example:**

```
TRADE EXECUTED — Step 13/30
────────────────────────────────────────────────────────
NARRATIVE: Market is quiet. Passive fill at normal cost.

ORDER: rate=0.050 | MARKET | lit only

FILLS
  NASDAQ Lit: 6,410 @ $150.4821  (slippage 1.82 bps)
  Mid Price:  $150.5234
  Total:      6,410 shares

INVENTORY
  Executed:  54,410 / 100,000 (54.4%)
  Remaining: 45,590 shares
  Time left: 17 steps

PERFORMANCE
  Your IS:  19.44 bps
  TWAP IS:  24.56 bps
  VWAP IS:  19.65 bps
```

---

### Tool: `get_baseline_comparison()` — Competitive Benchmarks

Real-time comparison against TWAP, VWAP, and the Almgren-Chriss mathematical optimum.

```
BASELINE COMPARISON — Step 13/30
────────────────────────────────────────────────────────
Implementation Shortfall (IS) — LOWER IS BETTER:

  🤖 You:         19.44 bps
  📈 TWAP:        24.56 bps  (naive equal-slice)
  📊 VWAP:        19.65 bps  (volume-proportional)
  🧮 AC Optimal:  14.24 bps  (Almgren‑Chriss optimal)

STATUS:
  ✅ Beating TWAP by 5.1 bps
  ✅ Beating VWAP by 0.2 bps
  ❌ Behind  AC Optimal by 5.2 bps
```

---

### Tool: `get_reward()` — Per-Step Reward Signal

Returns the most recent scalar reward, pre-scaled for GRPO compatibility.

```python
reward = get_reward()
# → float in [-1.0, +1.0]
# Positive: beating TWAP baseline
# Negative: worse than TWAP or adversary-penalized
```

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         TradeExecGym — System Map                          │
├───────────────────────────────────┬────────────────────────────────────────┤
│  PORT 7860 — Gradio Dashboard     │  PORT 7865 — FastAPI / MCP Backend     │
│  ─────────────────────────────    │  ─────────────────────────────────     │
│  ui/app.py                        │  server/app.py  →  openenv.create_app  │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────────┐   │
│  │ Tab 1: Auto Simulation      │  │  │  TradeExecEnvironment           │   │
│  │ Tab 2: Live LLM Evaluation  │  │  │  (MCPEnvironment subclass)      │   │
│  │ Tab 3: Manual Challenge     │◄─┼──┤                                 │   │
│  │ Tab 4: Project Info         │  │  │  → env/price_model.py           │   │
│  └─────────────────────────────┘  │  │     (Almgren-Chriss GBM)        │   │
│                                   │  │  → env/venue_router.py          │   │
│  TradeExecClient (client.py)      │  │     (Dark Pool + NASDAQ Lit)    │   │
│  httpx async REST calls ──────────┼──►  → env/reward.py               │   │
│                                   │  │     (Bounded sigmoid grader)    │   │
│                                   │  │  → tasks/ (5 task configs)      │   │
│                                   │  └─────────────────────────────────┘   │
└───────────────────────────────────┴────────────────────────────────────────┘

Agent Interaction Flow:
  [LLM / RL Policy]
        │
        ▼  MCP Tool Call
  [get_market_state()]  →  Natural-language market snapshot
        │
        ▼  Parse narrative + metrics
  [execute_trade(rate=X)]  →  Almgren-Chriss physics update
        │
        ▼  Per-step result
  [get_reward()]  →  Bounded scalar reward for GRPO training
```

### Component Breakdown

| Module | File | Role |
|---|---|---|
| **MCP Server** | `server/app.py` | `openenv.create_app()` wrapping the environment class |
| **Environment Core** | `server/trade_environment.py` | `MCPEnvironment` subclass. 4 tools. Episode state machine. |
| **Price Model** | `env/price_model.py` | Almgren-Chriss GBM: permanent + temporary impact + Brownian drift |
| **Venue Router** | `env/venue_router.py` | Dark pool (anonymous, ~40% fill) + Lit NASDAQ routing |
| **Reward Engine** | `env/reward.py` | Sigmoid-normalized IS → `[-1.0, 1.0]`. GRPO-ready. |
| **Task Registry** | `tasks/factory.py` | `get_task(id)` returns a `BaseTask` config object |
| **Baselines** | `baselines/` | TWAP, VWAP, AC-Optimal, Heuristic reference agents |
| **Inference** | `inference.py` | OpenEnv-compliant `[START]/[STEP]/[END]` log pipeline |
| **Dashboard** | `ui/app.py` | Gradio 5-tab dashboard with live plotting |
| **Client SDK** | `client.py` | Async `httpx` client. Wraps all 4 tool endpoints. |

---

## 🧠 Agent Architectures

TradeExecGym supports three cognitive approaches, designed for progressively more complex agents:

### 1 — Pure Heuristic (Almgren-Chriss Math)

The deterministic baseline. Computes the analytically optimal participation schedule given remaining inventory and time horizon. Used as the ground truth in reward normalization.

```python
from baselines.heuristic_agent import AlmgrenChrissHeuristic

h = AlmgrenChrissHeuristic()
rate = h.calculate_rate(
    shares_remaining=500_000,
    total_shares=1_000_000,
    steps_left=40,
    current_volatility=0.02
)
```

### 2 — LLM Cognitive Override (Tool-Calling)

An LLM uses `get_market_state()` to read the narrative, then calls `execute_trade()`. The market state text is deliberately formatted in natural language to maximize chain-of-thought reasoning quality.

**System Prompt:**
```json
{"recommendation": "Approve|Accelerate|Decelerate|Randomize", "reason": "reasoning text"}
```

The LLM reads: *"⚠️ ADVERSARY ALERT: HFT pattern detection isActive. Uniform trading is being penalized!"* — and can infer it needs to randomize its participation rate to evade detection.

### 3 — Hybrid (Math + Cognitive Layer)

The production agent pattern used in `inference.py`. The heuristic calculates a mathematically optimal rate. The LLM evaluates the narrative context and decides to `Accelerate` (×1.4), `Decelerate` (×0.6), or `Randomize` against the suggested rate.

```python
# Math Layer
suggested_rate = heuristic.calculate_rate(rem, total, steps_left, vol)

# Cognitive Layer
llm_decision = await llm.chat("Approve/Accelerate/Decelerate?", state_text)
if llm_decision == "Accelerate": final_rate = suggested_rate * 1.4
elif llm_decision == "Decelerate": final_rate = suggested_rate * 0.6
elif llm_decision == "Randomize": final_rate = suggested_rate * random.uniform(0.8, 1.2)
```

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Docker (for containerized deployment)

### Option A — Local Development

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/SST-MetaHuggingFace/trade-exec-gym
cd trade-exec-gym

# 2. Install dependencies (fast, via uv)
uv pip install -e .

# 3. Run the OpenEnv backend server (port 7865, internal)
uv run uvicorn server.app:app --host 0.0.0.0 --port 7865

# 4. In a second terminal, run the Gradio dashboard (port 7860)
uv run python ui/app.py --port 7860
```

Open **http://localhost:7860** to access the dashboard.

### Option B — Docker (Production)

```bash
# Build the image
docker build -t trade-exec-gym .

# Run (both services start automatically via start.sh)
docker run -p 7860:7860 -e HF_TOKEN=your_token trade-exec-gym
```

### Option C — Run Compliance Inference

Executes the full 5-task evaluation pipeline and outputs OpenEnv-compliant logs:

```bash
# Set your Hugging Face token (optional — enables LLM cognitive layer)
$env:HF_TOKEN = "hf_your_token_here"        # PowerShell
export HF_TOKEN="hf_your_token_here"         # Bash

# Run inference across all 5 tasks
uv run python inference.py
```

**Expected stdout format:**
```
[START] task=task1_twap_beater env=trade_exec_gym model=meta-llama/Meta-Llama-3-70B-Instruct
[STEP] step=1 action=0.0523 reward=0.12 done=false error=null
[STEP] step=2 action=0.0487 reward=0.18 done=false error=null
...
[END] success=true steps=28 score=0.891 rewards=0.12,0.18,...
```

Results are saved to `results/trajectory_YYYYMMDD_HHMMSS.json`.

---

## 🌡️ Environment State at a Glance

| Variable | Type | Description |
|---|---|---|
| `_mid_price` | `float` | Current market mid price (evolves per Almgren-Chriss) |
| `_arrival_price` | `float` | Fixed reference price at episode start (IS benchmark) |
| `_shares_remaining` | `int` | Shares left to execute |
| `_shares_executed` | `int` | Cumulative fills |
| `_total_cost` | `float` | Dollar cost accumulated |
| `_step_count` | `int` | Steps elapsed in this episode |
| `_max_steps` | `int` | Task-defined episode length |
| `_episode_done` | `bool` | Terminal flag |
| `_last_reward` | `float` | Most recent per-step reward (`[-1, +1]`) |

---

## 📊 Baseline Scores

Reference performance on each task using the GPT-4o Hybrid agent (heuristic + LLM cognitive override):

| Task | Difficulty | Avg IS (bps) ↓ | Grader Score ↑ | vs TWAP |
|---|---|---|---|---|
| Task 1: TWAP Beater | 🟢 Easy | 18.4 | **0.91** | ✅ Beat by 6.1 bps |
| Task 2: VWAP Optimizer | 🟡 Medium | 15.2 | **0.86** | ✅ Beat by 9.3 bps |
| Task 3: Volatile Execution | 🔴 Hard | 38.7 | **0.79** | ✅ Beat by 4.2 bps |
| Task 4: Adversarial HFT | 🟣 Very Hard | 52.3 | **0.72** | ✅ Beat by 2.8 bps |
| Task 5: Deadline Cliff | ⚫ Extreme | 84.1 | **0.66** | ✅ Beat by 1.1 bps |

> **Professional tier = score ≥ 0.80** · **Hall of Fame = IS < AC Optimal**

---

## 🔬 Validation & Testing

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run the official OpenEnv submission validator (PowerShell)
.\validate-submission.ps1

# Run validator (Bash/Linux/Docker)
bash validate-submission.sh
```

The validator checks:
- ✅ Server starts and `/health` responds
- ✅ All 5 task resets succeed
- ✅ `[START]/[STEP]/[END]` log format compliance
- ✅ Reward signal bounded to `[-1.0, 1.0]`
- ✅ Episode terminates correctly on completion

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `openenv-core` | ≥ 0.1.0 | OpenEnv framework (MCPEnvironment, create_app) |
| `fastapi` | ≥ 0.115.0 | HTTP server backbone |
| `fastmcp` | latest | MCP tool registration |
| `pydantic` | ≥ 2.9.2 | Request/Response validation |
| `uvicorn` | ≥ 0.30.6 | ASGI server |
| `numpy` + `scipy` | latest | Numerical physics |
| `torch` | ≥ 2.1.0 | ML model loading |
| `stable-baselines3` | ≥ 2.1.0 | PPO/GRPO agent support |
| `gradio` | latest | Interactive dashboard |
| `openai` | latest | HuggingFace Inference API client |
| `httpx` | latest | Async HTTP client (TradeExecClient) |

Install with:
```bash
uv pip install -e .   # uses pyproject.toml
```

---

## 🐳 Docker & Deployment

The container runs **two processes** via `start.sh`:

```
Process 1: uvicorn server.app:app --port 7865   # Internal MCP backend
Process 2: python ui/app.py --port 7860          # Public Gradio dashboard
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(none)* | HuggingFace API key — enables LLM cognitive layer |
| `MODEL_NAME` | `meta-llama/Meta-Llama-3-70B-Instruct` | LLM for inference |
| `ENV_BASE_URL` | `http://localhost:7860` | Backend URL for inference script |
| `PORT` | `7860` | Primary public port (HF Spaces managed) |

---

## 📁 Repository Structure

```
trade-exec-gym/
├── server/
│   ├── app.py                  # OpenEnv create_app() entry point
│   └── trade_environment.py   # MCPEnvironment: 4 tools, episode state
├── env/
│   ├── price_model.py          # Almgren-Chriss GBM price simulation
│   ├── venue_router.py         # Dark pool + NASDAQ lit routing
│   └── reward.py               # Bounded sigmoid reward grader
├── tasks/
│   ├── factory.py              # Task registry (get_task by ID)
│   ├── base_task.py            # BaseTask interface
│   ├── task1_twap.py           # Easy: 100K shares, quiet market
│   ├── task2_vwap.py           # Medium: VWAP curve tracking
│   ├── task3_volatile.py       # Hard: 3× volatility, dark pool
│   ├── task4_adversary.py      # Very Hard: HFT sniper bot
│   └── task5_deadline.py       # Extreme: 1M shares, hard deadline
├── baselines/
│   ├── heuristic_agent.py      # AlmgrenChrissHeuristic reference
│   ├── twap.py                 # TWAP baseline
│   ├── vwap.py                 # VWAP baseline
│   └── ac_optimal.py           # AC Optimal (mathematical floor)
├── ui/
│   └── app.py                  # Gradio dashboard (5 tabs)
├── training/                   # GRPO training scripts
├── models/                     # Pre-trained agent checkpoints
├── tests/                      # Pytest validation suite
├── results/                    # Trajectory JSON output logs
├── client.py                   # Async httpx TradeExecClient SDK
├── inference.py                # OpenEnv compliance inference runner
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Project metadata + dependencies
├── Dockerfile                  # Multi-process container
└── start.sh                    # Container entrypoint script
```

---

## 📜 Citation

```bibtex
@misc{tradeexecgym2026,
  title   = {TradeExecGym: An Institutional Smart Order Routing Environment for Reinforcement Learning},
  author  = {SST x Meta HuggingFace Hackathon Team},
  year    = {2026},
  url     = {https://huggingface.co/spaces/SST-MetaHuggingFace/trade-exec-gym},
  note    = {Built for Meta × HuggingFace OpenEnv Hackathon. Based on Almgren-Chriss (2000).}
}
```

**Reference:**
> Almgren, R., & Chriss, N. (2000). *Optimal execution of portfolio transactions*. Journal of Risk, 3(2), 5–39.

---

<div align="center">

**Built with 🧠 quantitative physics and 💚 for the Meta × HuggingFace Hackathon**

*If your IS drops below the AC Optimal line — you've entered Hall of Fame territory.*

</div>

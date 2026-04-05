# 🧠 PRD: LLM Context Budget Manager
## Complete Build Specification | OpenEnv Hackathon 2026
### ContextGym — The First RL Environment for Training LLM Memory Policies

---

## 1. Executive Summary

**The Problem:** Every production LLM deployment degrades over long conversations. After 50+ turns, models hallucinate, lose facts, and contradict themselves — because their context window fills with low-value tokens crowding out critical information.

**The Current State:** MemGPT/Letta (static OS-inspired algorithm), xMemory (static hierarchy), BudgetThinker (SFT+RL for a single model, not a training environment). None of these are a **standardized RL benchmark** for training memory management policies.

**What We Build:** `ContextGym` — an OpenEnv environment where an agent *learns* to be a better memory manager than any hardcoded algorithm. The agent decides, per message: **keep verbatim → compress to summary → archive to KV store → evict**. Graded by downstream QA accuracy and token efficiency.

**Why it wins:** HF's own TRL team is working on this. Memory-R1 (arxiv 2025) proves RL produces better memory managers than static algorithms. We are building the **standardized training ground** for the next Memory-R1.

---

## 2. Competitive Moat

| What Exists | Why We're Different |
|-------------|---------------------|
| MemGPT/Letta | Static algorithm. Cannot learn. We are the *training environment* it should have had. |
| Memory-R1 (arxiv 2025) | Trains a specific model. Not a reusable environment. No OpenEnv spec. |
| BudgetThinker (arxiv 2025) | Single-model SFT+RL experiment. Not packaged as a benchmark. |
| xMemory | Static hierarchy. Closed source. |
| **ContextGym (ours)** | **First OpenEnv-spec, Docker-deployed, multi-task RL benchmark for memory management** |

> **The key claim judges cannot dispute:** No repository on the OpenEnv Hub provides a standardized step()/reset() environment for training LLM memory policies. We are first.

---

## 3. Environment Architecture

### 3.1 The 4-Tier Memory Model (AlphaStar-Inspired Hierarchy)

Unlike simple keep/evict, we model 4 memory tiers with different cost/benefit tradeoffs:

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1: Active Context (Hot)                               │
│  - Messages in the live LLM context window                  │
│  - Token cost: FULL (1 token = 1 token)                     │
│  - Retrieval latency: 0ms                                   │
│  - Capacity: 8,192 tokens (hard constraint)                 │
├─────────────────────────────────────────────────────────────┤
│  TIER 2: Summarized Cache (Warm)                            │
│  - Compressed summaries of past message blocks              │
│  - Token cost: ~15% of original (8:1 compression ratio)     │
│  - Quality loss: ~8% on QA benchmark                        │
│  - Capacity: 32 summary slots                               │
├─────────────────────────────────────────────────────────────┤
│  TIER 3: Semantic Archive (Cold)                            │
│  - Key facts extracted and stored as structured JSON        │
│  - Token cost: ~5% of original                              │
│  - Retrieval: keyword-based lookup (mock vector search)     │
│  - Capacity: 500 fact entries                               │
├─────────────────────────────────────────────────────────────┤
│  TIER 4: Evicted (Gone)                                     │
│  - Permanently dropped. Irrecoverable.                      │
│  - Used sparingly — agent penalized for evicting key facts  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Topic Continuity Engine (The Complexity Layer)

Each message is tagged with a **topic vector** (6 fixed topics per scenario). The agent must learn:
- Messages from the *active topic* are high-value → keep
- Messages from *completed topics* can be summarized → tier 2
- *Historical facts* referenced later are critical → archive, never evict

This prevents the trivially greedy solution of "just evict oldest." A naive LRU cache fails the Hard tasks because important facts appear early and are referenced much later.

```python
# Topic tracking state (invisible to agent, drives grader)
topics = {
    "user_preferences": {"messages": [3, 17, 42], "referenced_later": True},
    "technical_specs":  {"messages": [5, 6, 7, 8], "referenced_later": True},
    "small_talk":       {"messages": [1, 2, 9, 11], "referenced_later": False},
    "error_debugging":  {"messages": [15, 20, 21], "referenced_later": True},
}
```

### 3.3 Importance Scoring System

Every message has a hidden **importance_score** (0.0–1.0) computed at environment creation. The agent doesn't see this directly — it must *infer* importance from:
- Message recency
- Explicit references in later messages ("as I mentioned earlier...")
- Question type (declarative facts → high importance, pleasantries → low)
- Entity density (named entities + numbers → higher importance)

The grader uses the hidden importance scores to penalize evictions of high-importance content.

---

## 4. OpenEnv Specification

### 4.1 `openenv.yaml`
```yaml
name: context-budget-manager
version: "1.0.0"
description: >
  RL environment for training LLM memory management policies.
  Agent manages a 4-tier memory hierarchy under token budget constraints
  while maximizing downstream QA accuracy.
author: "SST x Meta/HF Hackathon Team"
tasks:
  - name: "warm_up"
    difficulty: easy
    max_steps: 20
  - name: "topic_juggler"
    difficulty: medium
    max_steps: 50
  - name: "adversarial_recall"
    difficulty: medium_hard
    max_steps: 80
  - name: "long_horizon_cfo"
    difficulty: hard
    max_steps: 150
  - name: "stress_test"
    difficulty: expert
    max_steps: 300
reward_range: [-10.0, 10.0]
observation_space: dict
action_space: discrete+continuous
```

### 4.2 State Schema (Pydantic)

```python
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class MemoryTier(str, Enum):
    HOT = "hot"           # Active context
    WARM = "warm"         # Summarized
    COLD = "cold"         # Archived facts
    EVICTED = "evicted"   # Gone

class MessageState(BaseModel):
    id: str
    role: str                    # user | assistant | system
    content_preview: str         # First 100 chars only (agent can't see full content)
    token_count: int
    tier: MemoryTier
    recency_rank: int            # 1 = most recent
    topic_hint: str              # "preferences" | "technical" | "small_talk" | ...
    entity_density: float        # 0.0–1.0
    has_explicit_reference: bool # Was this message cited later?
    # Note: importance_score is HIDDEN from agent (used by grader only)

class ContextState(BaseModel):
    messages: List[MessageState]
    hot_tokens_used: int
    hot_token_budget: int         # Hard limit: 8192
    warm_slots_used: int
    warm_slot_budget: int         # Hard limit: 32
    cold_entries_used: int
    cold_entry_budget: int        # Hard limit: 500
    qa_score_last_3_turns: float  # Rolling window QA accuracy
    qa_score_baseline: float      # What FIFO would achieve (benchmark)
    steps_remaining: int
    scenario_topic_count: int
    active_topic: str
```

### 4.3 Action Schema

```python
class ActionType(str, Enum):
    KEEP = "keep"             # Leave in hot context, no change
    COMPRESS = "compress"     # Move to warm tier with auto-summary
    ARCHIVE = "archive"       # Extract key facts, move to cold tier
    EVICT = "evict"           # Delete permanently
    PROMOTE = "promote"       # Pull from warm/cold back to hot

class ContextAction(BaseModel):
    message_id: str
    action: ActionType
    custom_summary: Optional[str] = None   # Agent can write its own summary
    archive_key: Optional[str] = None      # Semantic key for cold storage
    reasoning: Optional[str] = None        # Chain-of-thought (for LLM agents)
```

### 4.4 API Endpoints

```
POST /reset?task={task_name}    → Returns initial ContextState
POST /step                      → Accepts ContextAction, returns (ContextState, reward, done, info)
GET  /state                     → Returns current ContextState
GET  /tasks                     → Returns all task schemas
GET  /grader?task={task_name}   → Returns grading rubric
POST /baseline?task={task_name} → Runs FIFO baseline, returns score
GET  /health                    → Docker health check
```

---

## 5. Task Tiers (AlphaStar Curriculum)

> AlphaStar's key insight: train on a **league** of progressively harder opponents. Our curriculum progressively introduces complexity layers that can't be shortcuts around.

### Task 1: Warm Up (Easy)
**Scenario:** 20-turn customer support chat. Single topic. User asks about product features, gets answers. Near end: asks a clarifying question about turn 3.

- **Budget:** 2048 tokens (tight but manageable)
- **Topics:** 1 (product features)
- **Challenge:** Learn that early declarative facts are needed later
- **Baseline (FIFO) score:** 0.72
- **Target agent score:** > 0.90
- **Win condition:** QA accuracy > 0.88 in < 20 actions

### Task 2: Topic Juggler (Medium)
**Scenario:** 50-turn conversation across 3 topics: (1) debugging a Python script, (2) planning a team meeting, (3) discussing product requirements. Topics interleave. Late questions span all 3.

- **Budget:** 4096 tokens
- **Topics:** 3 interleaved
- **Challenge:** Must track which topic each message belongs to. Cannot evict debugging details even when meeting planning is active.
- **Baseline (FIFO) score:** 0.58 (FIFO fails because it evicts important early debug context)
- **Target agent score:** > 0.82
- **Win condition:** QA accuracy > 0.80, zero critical fact evictions

### Task 3: Adversarial Recall (Medium-Hard)
**Scenario:** 80-turn conversation. Planted "sleeper facts" — information mentioned casually in turn 5 that becomes *critical* in turn 75. Also includes 30 turns of low-value small talk that looks important (jargon-heavy but semantically empty).

- **Budget:** 6144 tokens
- **Special mechanic:** 5 "sleeper facts" are embedded. Agent that learns to archive them wins; agent that evicts any loses catastrophically.
- **Decoy layer:** Small talk messages have high entity density to fool naive importance estimators
- **Baseline (FIFO) score:** 0.41
- **Target agent score:** > 0.78

### Task 4: Long Horizon CFO (Hard)
**Scenario:** 150-turn financial planning conversation. User is a CFO discussing quarterly budget. Early turns: company-wide budget allocations. Mid turns: department deep-dives (some irrelevant). Late turns: final decisions that must reference early allocation numbers exactly.

- **Budget:** 6144 tokens (extremely tight for 150 turns)
- **Special mechanic:** Numbers must be *exact* when recalled. "Approximately" is penalized. This forces the agent to archive numeric facts verbatim.
- **Multi-layer reward:** Dense (per-action quality), delayed (weekly QA check), sparse (final CFO report accuracy)
- **Baseline (FIFO) score:** 0.29
- **Target agent score:** > 0.71

### Task 5: Stress Test (Expert)
**Scenario:** 300-turn multi-user enterprise support thread. 3 different users join the conversation at different points. Each user has different authorization levels. Information shared with User A must not be surfaced when User B asks.

- **Budget:** 8192 tokens (maximum)
- **Special mechanic:** Privacy boundary enforcement. Cross-user information leakage = -5.0 reward per incident.
- **Baseline (FIFO) score:** 0.18
- **Target agent score:** > 0.65
- **This task is intentionally hard** — it demonstrates the research depth of our environment

---

## 6. Reward Function Design

### 6.1 Three-Component Reward (StaffingGym Pattern)

```python
def compute_reward(state_before: ContextState,
                   state_after: ContextState,
                   action: ContextAction,
                   hidden_importance: float,
                   qa_result: float) -> float:

    # === COMPONENT 1: Dense Reward (per action) ===
    # Immediate signal for quality decisions
    r_dense = 0.0

    if action.action == "evict" and hidden_importance > 0.7:
        r_dense -= 2.0   # Penalize evicting important content

    if action.action == "archive" and hidden_importance > 0.7:
        r_dense += 0.5   # Reward correct archiving of important content

    if action.action == "evict" and hidden_importance < 0.2:
        r_dense += 0.3   # Reward evicting genuinely useless content

    if action.action == "compress" and hidden_importance in [0.4, 0.7]:
        r_dense += 0.2   # Reward appropriate compression of medium content

    # === COMPONENT 2: Delayed Reward (per QA evaluation, every 10 steps) ===
    r_delayed = 0.0
    qa_improvement = qa_result - state_before.qa_score_baseline
    r_delayed += qa_improvement * 5.0   # Beat baseline → reward

    # Token efficiency bonus
    token_utilization = state_after.hot_tokens_used / state_after.hot_token_budget
    if token_utilization < 0.85:        # Efficient use of budget
        r_delayed += 0.5
    if token_utilization > 0.99:        # Critically over budget
        r_delayed -= 1.5

    # === COMPONENT 3: Sparse Reward (episode end) ===
    r_sparse = 0.0
    final_qa = evaluate_full_qa_suite(state_after)   # Full QA battery at end

    if final_qa > 0.90:
        r_sparse += 10.0    # Excellent memory management
    elif final_qa > 0.80:
        r_sparse += 5.0     # Good
    elif final_qa > 0.70:
        r_sparse += 2.0     # Acceptable
    else:
        r_sparse -= 5.0     # Worse than FIFO baseline = failure

    # Privacy violation check (Task 5 only)
    if state_after.privacy_violation_count > 0:
        r_sparse -= state_after.privacy_violation_count * 5.0

    return r_dense + r_delayed + r_sparse
```

### 6.2 Why Removing Any Layer Breaks Training

| Remove Dense | Agent has no signal for 300 steps — training collapses |
|---|---|
| Remove Delayed | Agent over-compresses everything — QA degrades before end penalty hits |
| Remove Sparse | Agent learns locally greedy policy — passes easy tasks, fails hard ones |

---

## 7. QA Grader Design

The grader is the **most critical** component. It must be deterministic, fast, and meaningful.

### 7.1 Three Grader Tiers

**Tier 1 — Exact Match Grader (Tasks 1-2)**
```python
def exact_match_grader(question: str, expected_answer: str,
                        context: List[MessageState]) -> float:
    # Build context string from hot + warm + cold tiers
    context_str = build_retrieval_context(context)
    # Use GPT-4o-mini to answer the question given context
    answer = llm_answer(question, context_str)
    return 1.0 if normalize(answer) == normalize(expected_answer) else 0.0
```

**Tier 2 — Semantic Similarity Grader (Tasks 3-4)**
```python
def semantic_grader(question: str, expected_answer: str,
                     context: List[MessageState]) -> float:
    context_str = build_retrieval_context(context)
    answer = llm_answer(question, context_str)
    # Use sentence-transformers cosine sim
    return cosine_similarity(embed(answer), embed(expected_answer))
```

**Tier 3 — Factual Precision Grader (Task 4-5)**
```python
def factual_grader(question: str, key_facts: List[str],
                    context: List[MessageState]) -> float:
    context_str = build_retrieval_context(context)
    answer = llm_answer(question, context_str)
    # Check what % of key_facts are present in the answer
    return sum(fact in answer for fact in key_facts) / len(key_facts)
```

### 7.2 QA Dataset Design (Pre-generated, Not LLM-dependent)

Each task scenario comes with a **gold QA set** generated at environment creation time. The questions are fixed — the grader doesn't generate new questions at runtime.

```python
# Example gold QA set for Task 4 (CFO scenario)
gold_qa = [
    {"q": "What was the Q1 engineering budget allocation?", "a": "$2.4M", "turn_defined": 5},
    {"q": "Which department requested a budget increase?", "a": "Product design", "turn_defined": 8},
    {"q": "What was the approved headcount for Q2?", "a": "47 FTEs", "turn_defined": 12},
    # ... 20 more questions spanning turns 3–140
]
```

---

## 8. Baseline Agents

### 8.1 FIFO Baseline (Required by OpenEnv Spec)
```python
class FIFOBaseline:
    """Evicts oldest message when budget exceeded. Simple and dumb."""
    def act(self, state: ContextState) -> ContextAction:
        if state.hot_tokens_used > state.hot_token_budget * 0.95:
            oldest = min(state.messages, key=lambda m: m.recency_rank,
                         filter=lambda m: m.tier == MemoryTier.HOT)
            return ContextAction(message_id=oldest.id, action=ActionType.EVICT)
        return ContextAction(message_id=state.messages[0].id, action=ActionType.KEEP)
```

### 8.2 Importance-Aware Greedy Baseline
```python
class ImportanceGreedyBaseline:
    """Uses entity_density + has_explicit_reference as proxy for importance."""
    def score(self, msg: MessageState) -> float:
        return msg.entity_density * 0.6 + int(msg.has_explicit_reference) * 0.4

    def act(self, state: ContextState) -> ContextAction:
        if state.hot_tokens_used > state.hot_token_budget * 0.92:
            hot_msgs = [m for m in state.messages if m.tier == MemoryTier.HOT]
            worst = min(hot_msgs, key=self.score)
            if self.score(worst) < 0.3:
                return ContextAction(message_id=worst.id, action=ActionType.EVICT)
            return ContextAction(message_id=worst.id, action=ActionType.COMPRESS)
        return ContextAction(message_id=state.messages[0].id, action=ActionType.KEEP)
```

### 8.3 RL Agent (Training Target — GRPO)
```python
# Training with GRPO (Group Relative Policy Optimization)
# Same algorithm used by StaffingGym winner (Qwen3-0.6B → Qwen3-8B)
# Compatible with HF TRL's GRPOTrainer

from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    model_name="Qwen/Qwen3-0.6B",
    env_name="context-budget-manager",
    max_steps=10000,
    reward_fn="composite",    # our 3-component reward
    curriculum=True,          # AlphaStar-style: easy → hard
    num_generations=8,        # Group size for GRPO
    temperature=0.9,
)
```

---

## 9. AlphaStar Curriculum Training Protocol

Directly inspired by AlphaStar's "league training" — the agent starts against the simplest tasks and unlocks harder ones on passing conditions:

```
Phase 1 (Steps 0–2000):     Train exclusively on Task 1 (Warm Up)
                             Unlock Task 2 when avg_score > 0.85

Phase 2 (Steps 2000–5000):  Train on Task 1 + Task 2 (20%/80% mix)
                             Unlock Task 3 when Task 2 avg_score > 0.78

Phase 3 (Steps 5000–8000):  Train on Task 2 + Task 3 (30%/70% mix)
                             Unlock Task 4 when Task 3 avg_score > 0.72

Phase 4 (Steps 8000–12000): Train on Task 3 + Task 4 (40%/60% mix)
                             Expert Task 5 unlocked for best 10% of rollouts

Phase 5 (Steps 12000+):     Full league training across all 5 tasks
                             Tasks weighted by current failure rate
```

**Why this matters:** Without curriculum, the agent quits on Task 4 immediately (sparse rewards, no dense signal for 150 steps). With curriculum, it arrives at Task 4 already knowing the fundamental memory management policy.

---

## 10. Infrastructure Stack

### 10.1 Project Structure
```
context-budget-manager/
├── openenv.yaml
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── env/
│   │   ├── __init__.py
│   │   ├── context_env.py      # Core environment logic
│   │   ├── memory_tiers.py     # 4-tier memory model
│   │   ├── topic_engine.py     # Topic tracking + importance scoring
│   │   └── reward.py           # 3-component reward function
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── warm_up.py
│   │   ├── topic_juggler.py
│   │   ├── adversarial_recall.py
│   │   ├── long_horizon_cfo.py
│   │   └── stress_test.py
│   ├── grader/
│   │   ├── __init__.py
│   │   ├── exact_match.py
│   │   ├── semantic_grader.py
│   │   └── factual_grader.py
│   ├── baselines/
│   │   ├── fifo_baseline.py
│   │   └── importance_greedy.py
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── schemas.py          # Pydantic models
│       └── routes.py           # All endpoints
├── training/
│   ├── train_grpo.py           # GRPO training script
│   ├── curriculum.py           # AlphaStar curriculum scheduler
│   └── eval.py                 # Evaluation harness
├── dashboard/
│   └── app.py                  # Gradio live dashboard (HF Spaces)
└── tests/
    ├── test_env.py
    ├── test_grader.py
    └── test_baselines.py
```

### 10.2 Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 10.3 Key Dependencies
```
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.7.0
tiktoken==0.7.0           # Token counting
openai==1.40.0            # QA grader (GPT-4o-mini)
sentence-transformers     # Semantic similarity grader
networkx                  # Topic dependency graphs
trl>=0.9.0               # GRPO training
transformers>=4.40.0
torch>=2.3.0
gradio>=4.0               # Live dashboard
pytest                    # Test suite
```

---

## 11. Live Dashboard (HF Spaces)

A Gradio dashboard showing:
- **Real-time memory visualization:** Color-coded tier map of all messages (hot=red, warm=yellow, cold=blue, evicted=grey)
- **Token budget gauge:** Live bar showing token consumption vs budget
- **QA score vs FIFO benchmark:** Live comparison chart
- **Reward trace:** All 3 reward components plotted per step
- **Human-vs-AI challenge:** Human can manually manage memory and compete against the trained agent

The dashboard is the **demo weapon** — judges see the memory tier map fill up and watch the agent make compression decisions in real time.

---

## 12. Hackathon Submission Checklist

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Real-world task simulation | LLM memory management (real enterprise pain) | ✅ |
| OpenEnv spec: step/reset/state | FastAPI with Pydantic schemas | ✅ |
| openenv.yaml | 5 tasks, easy→expert | ✅ |
| Minimum 3 tasks | 5 tasks (3 required + 2 bonus) | ✅ |
| Agent graders (0.0–1.0) | 3-tier grader: exact/semantic/factual | ✅ |
| Meaningful reward | 3-component: dense+delayed+sparse | ✅ |
| Baseline inference script | FIFO + Importance-Greedy + GRPO agent | ✅ |
| Reproducible scores | All scenarios seeded, gold QA sets fixed | ✅ |
| Deploy to HF Spaces | Dockerfile + Gradio dashboard | ✅ |
| README | Full environment description | ✅ |

---

## 13. 10-Day Build Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1** | Core env scaffold — MemoryTier, MessageState, 4-tier model | `src/env/` working locally |
| **Day 2** | Task 1 + Task 2 implementation + FIFO baseline | 2 tasks passing `openenv validate` |
| **Day 3** | QA grader (exact match + semantic) | Grader returning 0.0–1.0 scores |
| **Day 4** | 3-component reward function + reward unit tests | Reward stable on Tasks 1-2 |
| **Day 5** | FastAPI endpoints: reset/step/state/tasks/grader/baseline | Full OpenEnv API working |
| **Day 6** | Tasks 3-5 implementation (adversarial + CFO + stress) | All 5 tasks working |
| **Day 7** | Dockerfile + HF Spaces deployment | Live URL working |
| **Day 8** | GRPO training script + curriculum scheduler | Agent training running |
| **Day 9** | Gradio dashboard + baseline comparison scores | Dashboard live on HF Spaces |
| **Day 10** | README, openenv.yaml polish, submission | **Submit** |

---

## 14. The "They Pulled It Off" Factors

The elements that make judges say *"how did they build this in 10 days?"*:

1. **4-Tier Memory Model** — Not keep/evict. Four tiers with real cost/benefit tradeoffs.
2. **Topic Continuity Engine** — LRU is trivially defeated. Topic-aware retention is not.
3. **Adversarial Recall Task** — Planted "sleeper facts" that trap greedy strategies.
4. **AlphaStar Curriculum** — Not just 3 tasks. A full league training protocol with unlock conditions.
5. **Privacy Boundary Enforcement (Task 5)** — Multi-user information isolation in a conversation. This is a genuinely novel RL problem.
6. **Live Memory Tier Visualization** — Dashboard shows memory management decisions in real time. Nobody will forget this demo.
7. **3-Component Reward with Published Justification** — cite StaffingGym paper + Memory-R1 paper to show the reward design is principled, not made up.

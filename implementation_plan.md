# TradeExecGym — Master Implementation Plan
**Last Updated:** 2026-04-07 | **Status:** Approved for Implementation

## Executive Summary

**Project:** Smart Order Router RL Environment (OpenEnv hackathon — Meta x HuggingFace)

**Evaluator Feedback (Verbatim):**
- "UI is optional and not part of scoring"
- "inference.py should run one complete, reproducible interaction (episode)"
- "It should demonstrate: reset → step → state flow"
- "Avoid running multiple tasks or building a task loop inside it"

**Scoring Criteria:**
| Criterion | Weight |
|-----------|--------|
| Real-world utility | 30% |
| Task & grader quality | 25% |
| Environment design | 20% |
| Code quality & spec compliance | 15% |
| Creativity & novelty | 10% |

---

## Current State Diagnosis

### What's Already Solid (Do NOT Touch)
- Core Almgren-Chriss physics engine (`env/price_model.py`, `env/venue_router.py`)
- Shadow baseline caching (TWAP/VWAP/AC Optimal — same seed = fair comparison)
- 5-task curriculum with task-specific graders (0.0–1.0)
- 3-component reward (dense IS improvement + milestone sparse + terminal bonus)
- `inference.py` already restructured to single-episode (evaluator requirement met)
- Server/client async architecture (`server/app.py`, `client.py`)
- Task adversary mechanics (dual-gate pattern detection)

### Critical Bug Found (P0 — Fix First)
**`_milestones_reached` NOT reset between episodes** in `server/trade_environment.py` `reset()`.
Milestones from episode 1 carry into episode 2 → milestone rewards never fire again after the first episode. Directly breaks "meaningful reward over full trajectory" criterion.
```python
# MISSING from reset() method:
self._milestones_reached = set()
```

### Gaps vs OpenEnv Spec
1. `openenv.yaml` missing `observation_space`, `action_space`, `reward_range`, `entry_point`
2. No typed Pydantic `Observation/Action/Reward` models exposed to `openenv validate`
3. `inference.py` has emoji in stdout → `UnicodeEncodeError` risk on evaluator machines
4. `OPENAI_API_KEY` not read as primary credential (spec requirement)
5. Done signal detected via text parsing (`"EPISODE COMPLETE" in result`) — fragile
6. Score extraction via string split — fragile
7. `_build_market_state_text()` has emojis — potential crash on non-UTF8 terminals

---

## PHASE 0: Critical Bug Fix (Immediate — Before Anything Else)

**File:** `server/trade_environment.py`  
**Change:** Add `self._milestones_reached = set()` inside the `reset()` method, alongside the other state resets.

**Why critical:** Without this, every test run after the first episode will show 0 sparse reward, which means our reward function appears to only produce dense signal — failing the "meaningful reward over full trajectory" spec requirement.

---

## PHASE 1: OpenEnv Spec Compliance (Code Quality 15% weight)

### 1a. Rewrite `openenv.yaml` (Full Rewrite)
Current YAML is minimal (10 lines). Full spec-compliant rewrite:

**Add these sections:**
- `entry_point: "server.app:app"` — required by openenv validate
- `observation_space` block with all 5 numeric fields typed with ranges:
  - `price_norm: float [0.5, 2.0]` — normalized current price vs arrival
  - `progress_pct: float [0.0, 1.0]` — fraction of episode elapsed
  - `remaining_pct: float [0.0, 1.0]` — fraction of shares remaining
  - `vol_ratio: float [0.0, 5.0]` — intraday volume ratio
  - `current_is_bps: float [0.0, 100.0]` — implementation shortfall
- `action_space` block with full `participation_rate` spec
- `reward_range: [-2.0, 2.0]`
- Per-task `difficulty`, `max_episode_steps`, `scoring_criteria` fields

**Task ID alignment check:**
Ensure all task IDs in YAML exactly match factory keys:
- `task1_twap_beater` ✓
- `task2_vwap_optimizer` ✓
- `task3_volatile_execution` ✓
- `task4_adversarial` ✓
- `task5_deadline_pressure` ✓

### 1b. Typed Pydantic Models in `server/trade_environment.py`
Add three spec-compliant Pydantic models at the top of the file (after imports):

```python
class TradeObservation(BaseModel):
    price_norm: float        # current_price / arrival_price
    progress_pct: float      # step_count / max_steps
    remaining_pct: float     # shares_remaining / total_shares
    vol_ratio: float         # intraday volume multiplier
    current_is_bps: float    # implementation shortfall in bps
    done: bool
    step: int
    info: dict = {}

class TradeAction(BaseModel):
    participation_rate: float = Field(ge=0.0, le=0.25, default=0.05)
    use_dark_pool: bool = False
    dark_pool_fraction: float = Field(ge=0.0, le=1.0, default=0.0)

class TradeReward(BaseModel):
    value: float             # total step reward
    dense: float             # IS-improvement component
    sparse: float            # milestone bonus component
    terminal: float          # end-of-episode bonus component
```

### 1c. Add `_build_numeric_observation()` Helper
New private method in `TradeExecEnvironment` that returns a `dict` matching `TradeObservation`:
```python
def _build_numeric_observation(self) -> dict:
    return {
        "price_norm": self._mid_price / self._arrival_price,
        "progress_pct": self._step_count / max(1, self._max_steps),
        "remaining_pct": self._shares_remaining / max(1, self._total_shares),
        "vol_ratio": self._volume_ratio(),
        "current_is_bps": self._compute_current_is(),
        "done": self._episode_done,
        "step": self._step_count,
        "info": {}
    }
```

### 1d. Fix `reset()` Return Value
`reset()` must return a structured `Observation` with numeric fields (not just text) for `openenv validate`. Add the numeric dict to the observation metadata.

### 1e. Add `tests/test_openenv_validate.py`
New test file verifying spec compliance:
- Import environment, call `reset()`, verify return type
- Call `execute_trade()`, verify string return
- Call `get_reward()`, verify float return in `[-2.0, 2.0]`
- Call `state` property, verify fields present
- Run `_build_numeric_observation()`, verify all 5 fields in correct ranges

---

## PHASE 2: `inference.py` Hardening

### 2a. ASCII-Safe STDOUT (Zero-Crash Guarantee)
**Change:** Add at top of file (after imports):
```python
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
```

**Why:** On Windows or restricted terminals, UTF-8 emojis in print() cause `UnicodeEncodeError`. The evaluator's parser sees a crash instead of `[END]` log line.

### 2b. Replace Emoji in System Prompt & Log Functions
**In `HYBRID_SYSTEM_PROMPT`:**
- `⚠️ DETECTED` → `[DETECTED]`
- `📉 improving` → `[improving]`
- `📈 worsening` → `[worsening]`

**In log functions:** Already ASCII-safe (`[START]`, `[STEP]`, `[END]`) — verify no emoji slips in.

### 2c. OPENAI_API_KEY as Primary Credential (Spec Requirement)
**Current:** Uses `HF_TOKEN` as primary  
**Fix:** Add `OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")` and use as primary:
```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
_api_key = OPENAI_API_KEY or HF_TOKEN or "dummy"
llm_client = OpenAI(api_key=_api_key, base_url=API_BASE_URL)
```

### 2d. Done Signal Fix
**Current:** `done_bool = "EPISODE COMPLETE" in execute_result` — text parsing  
**Fix:** Add step counter guard AND keep text parse as primary:
```python
done_bool = ("EPISODE COMPLETE" in execute_result) or (step_count >= 150)
```
Also add: if `execute_result` starts with "❌ Episode already complete" → set `done = True`.

### 2e. Score Extraction Robustness
**Current:** Single `split("Grader Score:")` — crashes if text format differs  
**Fix:** Multi-pattern fallback chain:
```python
for pattern in ["Grader Score:", "score:", "Score:"]:
    if pattern in execute_result:
        raw = execute_result.split(pattern)[1].split("/")[0].strip().split()[0]
        final_score = float(raw)
        break
```
Wrap entire block in `try/except` with `final_score = 0.5` fallback (not 0.0001).

### 2f. Hero Task: Task 4 (Adversarial) — Confirmed
- `DEFAULT_TASK = "task4_adversarial"` (already set)
- `seed=42` locked (already set)
- 120 steps → rich trajectory for evaluator inspection
- Adversary metrics show reactive LLM intelligence — most impressive task

### 2g. Add Clear Script Header
Add docstring at top explaining:
- Single-episode design (per evaluator spec)
- Environment: `reset() → get_market_state() → execute_trade() → get_reward()`
- Credential: `OPENAI_API_KEY` environment variable
- Task: Task 4 Adversarial HFT (default)

---

## PHASE 3: Create `baselines/run_baselines.py` (NEW FILE)

**Purpose:** The spec says "Produces a reproducible baseline score on all 3 tasks." But `inference.py` must be single-episode. Resolution: separate baseline utility script.

**Design:**
- Pure heuristic (no LLM needed) using `AlmgrenChrissHeuristic`
- Runs Tasks 1, 2, 3 sequentially (each with `seed=42`)
- Uses `TradeExecClient` async interface
- Reads `ENV_BASE_URL` from environment variable
- Output format:
  ```
  [BASELINE] task=task1_twap_beater score=0.72 is_bps=18.4 steps=30
  [BASELINE] task=task2_vwap_optimizer score=0.68 is_bps=21.1 steps=60
  [BASELINE] task=task3_volatile_execution score=0.61 is_bps=28.3 steps=90
  [SUMMARY] mean_score=0.67
  ```
- Saves `results/baseline_scores.json` with full trajectory data
- All ASCII output — no emoji

**Why 3 tasks (not 4 or 5):**
- Tasks 1-3 are the "standard" difficulty range (easy/medium/hard)
- Tasks 4-5 are adversarial/extreme — need LLM to score well
- Heuristic baseline on Tasks 1-3 gives meaningful comparison point

---

## PHASE 4: Environment Design Polish (20% weight)

### 4a. ASCII-Safe State Narratives
`_build_market_state_text()` has emoji that could crash on non-UTF8 terminals. Add ASCII fallback mapping:
```python
EMOJI_TO_ASCII = {
    "✅": "[OK]", "❌": "[NO]", "⚠️": "[!]",
    "📉": "[DOWN]", "📈": "[UP]", "🏁": "[DONE]",
    "⚡": "[!]", "🔴": "[CRIT]", "💡": "[TIP]"
}
```
Apply when `ascii_safe=True` flag is passed (default False for Gradio UI compatibility).

### 4b. Verify Task 4 `participation_history` Reset Isolation
- Confirm `TaskAdversary.reset()` clears `self.participation_history = []`
- Already implemented — verify it's called in `TradeExecEnvironment.reset()`

### 4c. Zero-Rate Penalty in `env/reward.py`
Prevent degenerate always-zero-rate strategy:
```python
# In compute_reward():
if participation_rate == 0.0 (or slippage == 0 and no shares filled):
    return base_reward - 0.05  # small penalty for non-trading step
```
Add `participation_rate` parameter to `compute_reward()` signature.

### 4d. `_step_impl()` Enhancement
Currently returns generic "use MCP tools" message. Update to include current numeric state in metadata for non-MCP callers.

### 4e. Episode Boundary Clarity
Verify `"EPISODE COMPLETE"` string appears in BOTH completion conditions:
1. `self._shares_remaining <= 0` (all filled)
2. `steps_after <= 0` (time limit reached)
Both already return `completion_block` — but confirm the exact string is present in both paths.

---

## PHASE 5: Task & Grader Quality (25% weight)

### 5a. Add Determinism Tests to `tests/test_tasks.py`
New test class `TestGraderDeterminism`:
- **Same seed → same score**: Run episode with seed=42 twice, assert `score1 == score2`
- **Range test**: For each task, verify grader returns value in `[0.0001, 0.9999]`
- **Difficulty progression**: Task1 heuristic score > Task3 heuristic score
- **Task5 cliff test**: Completion < 0.999 → score ≤ 0.15

### 5b. Richer Task Narratives (Actionable LLM Guidance)
Each task's `get_market_narrative()` should give unambiguous, actionable hints:

**Task 1 (TWAP Beater):**
- Add: `"TWAP equal-slice rate = {1/max_steps:.4f}. To beat TWAP: increase rate at open/close, reduce midday."`

**Task 2 (VWAP):**
- Add: `"Volume profile: Open=1.6x, Midday=0.5x, Close=1.8x. Match these weights."`

**Task 3 (Volatile):**
- Add: `"Tip: use dark_pool_fraction=0.3 to reduce market impact during high-vol periods."`

**Task 4 (Adversarial):**
- Already has rich narrative — verify jitter instruction is prominent
- Add explicit: `"Jitter target: rate in [0.05, 0.15] changing by ±0.03 each step"`

**Task 5 (Deadline):**
- Add exact pace calculation: `"REQUIRED PACE: {shares_remaining}/{steps_left} = {pace:.0f} shares/step"`

### 5c. Grader Score in Every Step
Confirm `grader_hint` is included in `_build_market_state_text()` at EVERY step (not just terminal). Already implemented — verify it shows.

---

## PHASE 6: Code Quality & Structure (15% weight)

### 6a. Import Cleanup in `server/trade_environment.py`
Move from inside functions to top of file:
- `import statistics` (currently inside `_build_market_state_text()`)
- `import traceback` (currently inside except block)
- `from uuid import uuid4` (verify it's at top)

### 6b. Add Return Type Annotations
```python
def reset(self, ...) -> Observation: ...
def get_market_state(self) -> str: ...
def execute_trade(self, ...) -> str: ...
def get_reward(self) -> float: ...
def _build_numeric_observation(self) -> dict: ...
def _compute_grader_score(self) -> float: ...
```

### 6c. Module Docstrings for Task Files
Each task file should have a module-level docstring:
```python
"""
Task 1: TWAP Beater
Objective: Execute 100,000 shares in 30 steps, beating TWAP (uniform slicing).
Difficulty: Easy
Winning Strategy: Front-load at open (1.6x vol), reduce midday (0.5x), surge at close (1.8x).
Grader: 50% IS quality vs AC Optimal, 30% completion, 20% baseline beating.
"""
```

### 6d. `client.py` Enhancement
Add convenience helpers:
```python
async def get_grader_score(self) -> float:
    """Parse grader score from baseline comparison text."""
    ...

async def get_numeric_state(self) -> dict:
    """Get structured numeric observation dict."""
    ...
```
Fix `run_twap_episode()` to dynamically set `max_steps` from reset response instead of hardcoded 30.

### 6e. Test Suite Additions
- `tests/test_reward.py`: Add zero-rate penalty test
- `tests/phase1_validation.py`: Add milestone reset test (run 2 episodes, verify milestones fire both times)
- `tests/test_tasks.py`: Add determinism + cliff tests (from 5a)
- Create `tests/test_openenv_validate.py`: Spec compliance test (from 1e)

---

## PHASE 7: Docker & Deployment (Code Quality 15%)

### 7a. Fix Dockerfile Healthcheck
**Current:** Pings port 7860 (Gradio UI)  
**Fix:** Ping port 7865 (FastAPI backend — the actual environment server)
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7865/health || exit 1
```

### 7b. Add Inference-Only Mode to `start.sh`
When `INFERENCE_MODE=true`, skip Gradio and only start FastAPI:
```bash
if [ "${INFERENCE_MODE:-false}" = "true" ]; then
    echo "Inference mode: starting FastAPI only on port 7865"
    uvicorn server.app:app --host 0.0.0.0 --port 7865
else
    # existing logic with Gradio
fi
```
This lets evaluators run: `docker run -e INFERENCE_MODE=true -p 7865:7865 trade-exec-gym`

### 7c. Review `.dockerignore`
Ensure excluded:
- `tensorboard/` (large event files)
- `results/` (generated output)
- `models/` (trained model weights)

Ensure included:
- `tests/` (evaluator may run tests)
- All source code

---

## PHASE 8: README & Documentation Polish

### 8a. Quick Start at Top (3-Command Pattern)
```bash
# 1. Start environment server
uvicorn server.app:app --port 7865

# 2. Run single-episode inference demo (hero task: Task 4 Adversarial)
OPENAI_API_KEY=your_key python inference.py

# 3. Run heuristic baselines on Tasks 1-3
python baselines/run_baselines.py
```

### 8b. OpenEnv Validate Instructions
```bash
# Validate environment spec compliance
openenv validate --config openenv.yaml
```

### 8c. Evaluation Rubric Mapping Table
Show explicitly how each feature maps to scoring criteria:

| Feature | Scoring Criterion | Weight |
|---------|------------------|--------|
| Almgren-Chriss physics | Real-world utility | 30% |
| 5-task curriculum | Task & grader quality | 25% |
| Shadow baselines (fair comparison) | Environment design | 20% |
| Typed Pydantic models | Code quality | 15% |
| Adversary HFT detection | Creativity & novelty | 10% |

### 8d. Architecture Flow Diagram
```
inference.py
    ↓ (OPENAI_API_KEY + OpenAI client)
client.py (TradeExecClient)
    ↓ (async HTTP)
server/app.py (FastAPI + OpenEnv create_app)
    ↓
server/trade_environment.py (TradeExecEnvironment)
    ├── tasks/ (task1-5 graders + narratives)
    ├── env/price_model.py (Almgren-Chriss GBM)
    ├── env/venue_router.py (dark pool routing)
    └── env/reward.py (3-component reward)
```

---

## Files Summary

### Files to CREATE (New)
| File | Purpose |
|------|---------|
| `baselines/run_baselines.py` | 3-task heuristic baseline scorer |
| `tests/test_openenv_validate.py` | OpenEnv spec compliance test |

### Files to MODIFY (Existing)
| File | Changes | Priority |
|------|---------|---------|
| `server/trade_environment.py` | Milestone reset bug, Pydantic models, imports, ASCII narratives | P0+P1 |
| `openenv.yaml` | Full rewrite with obs/action/reward schema | P1 |
| `inference.py` | ASCII-safe, OPENAI_API_KEY, done signal, score extraction | P1 |
| `env/reward.py` | Zero-rate penalty | P2 |
| `tasks/base_task.py` | Richer narratives, module docstring | P2 |
| `tasks/task1_twap.py` | Richer narrative hints, module docstring | P2 |
| `tasks/task2_vwap.py` | Richer narrative hints, module docstring | P2 |
| `tasks/task3_volatile.py` | Richer narrative hints, module docstring | P2 |
| `tasks/task4_adversary.py` | Emoji cleanup, module docstring | P2 |
| `tasks/task5_deadline.py` | Module docstring | P2 |
| `client.py` | Helper methods, dynamic max_steps | P3 |
| `tests/test_tasks.py` | Determinism + cliff + difficulty tests | P2 |
| `tests/test_reward.py` | Zero-rate penalty test | P2 |
| `tests/phase1_validation.py` | Milestone reset test | P2 |
| `Dockerfile` | Healthcheck fix, INFERENCE_MODE | P3 |
| `README.md` | Quickstart, architecture, rubric mapping | P3 |

### Files NOT to Touch
| File | Reason |
|------|--------|
| `env/price_model.py` | Physics is correct and well-tested |
| `env/venue_router.py` | Routing logic is correct |
| `env/gym_wrapper.py` | Gymnasium wrapper, not scored |
| `training/` | Not scored |
| `ui/` | Explicitly not scored (evaluator confirmed) |
| `tasks/factory.py` | Works correctly |
| `server/app.py` | Minimal, correct |

---

## Expected Score Impact

| Criterion | Weight | Current Est. | After Plan | Delta |
|-----------|--------|-------------|------------|-------|
| Real-world utility | 30% | 22/30 | 26/30 | +4 |
| Task & grader quality | 25% | 18/25 | 22/25 | +4 |
| Environment design | 20% | 13/20 | 17/20 | +4 |
| Code quality & spec | 15% | 8/15 | 13/15 | +5 |
| Creativity & novelty | 10% | 8/10 | 9/10 | +1 |
| **Total** | **100%** | **69/100** | **87/100** | **+18** |

---

## Implementation Order (Sequential)

1. **P0 Bug Fix** → `server/trade_environment.py` milestone reset
2. **P1a** → Rewrite `openenv.yaml`
3. **P1b-d** → Typed models + numeric observation in `trade_environment.py`
4. **P2** → Harden `inference.py` (ASCII, OPENAI_API_KEY, done, score)
5. **P3** → Create `baselines/run_baselines.py`
6. **P4a** → ASCII-safe state text in `trade_environment.py`
7. **P4c** → Zero-rate penalty in `env/reward.py`
8. **P5b** → Richer narratives in all task files
9. **P6a-b** → Import cleanup + type annotations
10. **P5a + P6e** → Add tests
11. **P6d** → Enhance `client.py`
12. **P7** → Docker fixes
13. **P8** → README polish

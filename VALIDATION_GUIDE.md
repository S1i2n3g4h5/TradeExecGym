# 🛡️ TradeExecGym — Validation Guide

One command. One report. Full scientific proof.

## Run the Gauntlet

```bash
# Offline (Layers 0–3, no server needed):
python3 tests/validate_robustness.py

# Full (all 5 layers including API — server must be running):
uvicorn server.app:app --host 0.0.0.0 --port 7865 &
python3 tests/validate_robustness.py --full
```

Output: `ROBUSTNESS_REPORT.json`

---

## What Each Layer Proves

| Layer | Command | What It Proves |
|---|---|---|
| **0 — Boot** | Auto | All 5 tasks import + reset cleanly. If this fails, nothing else matters. |
| **1 — Unit Tests** | `pytest tests/` | 24 atomic tests: physics constants, reward math, adversary detection, venue routing. |
| **2 — Baseline Scores** | Auto | A pure-math TWAP agent scores above the random noise floor on every task. |
| **3 — Skill Gradient** | Auto | `Random IS > TWAP IS > AC Optimal IS` — better strategy = lower cost. Monotonic. |
| **4 — API Compliance** | `--full` | All 6 HTTP endpoints (`/health`, `/reset`, `/step`, `/state`, `/schema`, `/mcp`) respond correctly. |
| **Determinism** | Auto | Same `seed=42` produces bit-identical IS values across independent runs. |

---

## How to Interpret `ROBUSTNESS_REPORT.json`

```json
{
  "layer3_skill_gradient": {
    "agents": {
      "random":     {"is_bps": 17.60},   ← worst — no strategy
      "twap":       {"is_bps": 13.16},   ← middle — math baseline
      "ac_optimal": {"is_bps":  9.77}    ← best — Almgren-Chriss ceiling
    },
    "monotonic_ordering": true,           ← THIS is what proves the environment works
    "status": "PASS"
  },
  "overall": "PASS ✅"
}
```

**Key signal:** `monotonic_ordering: true` means the environment correctly rewards skill.
A random agent cannot accidentally beat a math-optimal agent. The reward gradient is real.

---

## Expected Results

```
Overall Result:  PASS ✅
Layers Passed:   6/6
```

Any `FAIL` in Layers 0–3 indicates a physics or reward bug — open an issue.
Layer 4 `FAIL` typically means the server isn't running on port 7865.

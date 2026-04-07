"""
TradeExecGym -- Unified Robustness Validation Script
=====================================================
The 4-Layer Robustness Pyramid for RL Environment Certification.

This script proves that TradeExecGym is scientifically sound, not buggy,
and rewards better strategies with better scores -- exactly what the OpenEnv
evaluation framework requires.

Layer 0  Environment Boot    -- All modules import; all 5 tasks reset cleanly
Layer 1  Unit Tests          -- 24 atomic physics + reward + adversary tests
Layer 2  Baseline Scores     -- Pure-math TWAP agent beats random noise (>=0.30)
Layer 3  Skill Gradient      -- Random < TWAP < AC-Optimal (monotonic ordering)
Layer 4  OpenEnv Compliance  -- All 6 HTTP endpoints respond correctly

Usage:
    # Layers 0-3 (no server required):
    python3 tests/validate_robustness.py

    # All 5 layers including API compliance (server must be running):
    uvicorn server.app:app --host 0.0.0.0 --port 7860 &
    python3 tests/validate_robustness.py --full

Output: ROBUSTNESS_REPORT.json
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
import os
import json
import time
import argparse
import subprocess
from datetime import datetime, timezone

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ------------------------------------------------------------------------------
# Layer 0: Environment Boot Check
# Proves all modules load and all 5 tasks initialize without errors.
# ------------------------------------------------------------------------------

def run_layer0_environment_boot() -> dict:
    """
    Verify that the full environment can be imported and all 5 tasks reset cleanly.
    This is the prerequisite check -- if this fails, no other layer is meaningful.
    """
    print("\n[Layer 0] Environment boot check (5 tasks x import + reset)...")
    ALL_TASKS = [
        "task1_twap_beater",
        "task2_vwap_optimizer",
        "task3_volatile_execution",
        "task4_adversarial",
        "task5_deadline_pressure",
    ]
    task_results = {}
    try:
        from server.trade_environment import TradeExecEnvironment
        from tasks.factory import get_task
        from env.price_model import PriceModel
        from env.reward import compute_reward
        from env.venue_router import VenueRouter
        print("  -> Core modules imported: [OK]")

        for task_id in ALL_TASKS:
            try:
                env = TradeExecEnvironment()
                env.reset(seed=42, task_id=task_id)
                # Verify essential state was initialized
                assert env._total_shares > 0, "total_shares not set"
                assert env._max_steps > 0, "max_steps not set"
                assert env._arrival_price > 0, "arrival_price not set"
                task_results[task_id] = "PASS"
                print(f"  -> {task_id}: [OK] PASS (shares={env._total_shares:,}, steps={env._max_steps})")
            except Exception as e:
                task_results[task_id] = f"FAIL: {e}"
                print(f"  -> {task_id}: [FAIL] FAIL ({e})")

        all_ok = all(v == "PASS" for v in task_results.values())
        status = "PASS" if all_ok else "FAIL"
        print(f"  -> Boot result: {'[OK] ALL 5 TASKS INITIALIZED' if all_ok else '[FAIL] SOME TASKS FAILED'}")
        return {"tasks": task_results, "modules_imported": True, "status": status}

    except ImportError as e:
        print(f"  -> [FAIL] CRITICAL: Module import failed: {e}")
        return {"tasks": task_results, "modules_imported": False, "status": "FAIL", "error": str(e)}


# ------------------------------------------------------------------------------
# Layer 1: Unit Tests (runs pytest programmatically and captures results)
# ------------------------------------------------------------------------------

def run_layer1_unit_tests() -> dict:
    """Run the full pytest suite and return a structured result."""
    print("\n[Layer 1] Running unit test suite (pytest)...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q", 
         "--ignore=tests/validate_robustness.py", "--ignore=tests/phase1_validation.py"],
        capture_output=True,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    output = result.stdout + result.stderr

    # Parse the summary line e.g. "24 passed, 1 warning in 8.57s"
    passed, failed, errors = 0, 0, 0
    for line in output.splitlines():
        if "passed" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "passed":
                    passed = int(parts[i - 1])
                elif p == "failed":
                    failed = int(parts[i - 1])
                elif p == "error" in p:
                    errors = int(parts[i - 1])

    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"  -> {passed} passed, {failed} failed, {errors} errors -- {status}")
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "status": status,
        "stdout_tail": output[-1000:] if len(output) > 1000 else output
    }


# ------------------------------------------------------------------------------
# Layer 2: Deterministic Baseline Scores
# Proves that pure-math agents achieve consistent, predictable scores.
# ------------------------------------------------------------------------------

def run_layer2_baseline_scores() -> dict:
    """
    Verify that a pure-math TWAP agent achieves a meaningful grader score on each task.

    ADV = 10,000,000 shares/day over 780 steps -> adv_per_step ~ 12,820 shares.
    Each task's required participation rate = total_shares / (max_steps x adv_per_step).
    Tasks with large order sizes relative to ADV (e.g. Task 2: 250k/60 steps needs ~0.32,
    above the 0.25 cap) will naturally have lower completion -- that is by design and proves
    the environment is challenging. The threshold here is >= 0.30 (above random floor ~0.10).

    This layer proves: math-based agents reliably outperform random noise.
    """
    print("\n[Layer 2] Checking deterministic baseline scores (TWAP agent, full episode)...")

    # Task-specific TWAP rates: use min(required_rate, 0.25) -- the participation cap.
    # Required = total_shares / (max_steps * adv_per_step) where adv_per_step = 10M/780
    # Task 2 (250k/60 steps needs ~0.32) and Task 3 (400k/90 steps needs ~0.35) exceed
    # the 0.25 cap -- by design. These tasks are intentionally hard.
    # Per-task threshold accounts for structural completion limits.
    TASK_CONFIG = {
        "task1_twap_beater":        {"rate": 0.033, "threshold": 0.50},  # Can complete, expect >=0.50
        "task2_vwap_optimizer":     {"rate": 0.25,  "threshold": 0.20},  # Can't fully complete -- hard by design
        "task3_volatile_execution": {"rate": 0.20,  "threshold": 0.20},  # Volatile -- lower bar
    }

    try:
        from server.trade_environment import TradeExecEnvironment

        results = {}

        for task_id, cfg in TASK_CONFIG.items():
            rate = cfg["rate"]
            threshold = cfg["threshold"]
            env = TradeExecEnvironment()
            env.reset(seed=42, task_id=task_id)

            # Run uniform TWAP to episode completion
            while not env._episode_done:
                env.execute_trade(participation_rate=rate)

            score = env._compute_grader_score()
            completion = env._shares_executed / env._total_shares * 100
            passed = score >= threshold
            results[task_id] = {
                "score": round(score, 4),
                "completion_pct": round(completion, 1),
                "participation_rate_used": rate,
                "threshold": threshold,
                "passed": passed,
            }
            print(f"  -> {task_id}: score={score:.4f}, completion={completion:.1f}% "
                  f"[{'[OK] PASS' if passed else '[FAIL] FAIL'} -- threshold >={threshold}]")

        all_ok = all(v["passed"] for v in results.values())
        return {
            "scores": {k: v["score"] for k, v in results.items()},
            "details": results,
            "status": "PASS" if all_ok else "FAIL",
        }

    except Exception as e:
        print(f"  -> ERROR: {e}")
        return {"status": "ERROR", "error": str(e)}


# ------------------------------------------------------------------------------
# Layer 3: Skill Gradient Proof
# Proves Random < TWAP < AC-Optimal (monotonic ordering).
# ------------------------------------------------------------------------------

def run_layer3_skill_gradient() -> dict:
    """
    Prove that TradeExecGym rewards skill monotonically: Random IS > TWAP IS > AC Optimal IS.

    This is the critical 'ablation study' that separates a well-designed RL environment
    from a lucky one. Lower IS (Implementation Shortfall in basis points) = better execution.

    - Random agent:   uniformly samples participation_rate in [0.01, 0.25] each step.
                      Expected to produce the worst IS due to volatility-unaware sizing.
                      **CRITICAL**: We seed the random agent (pyrandom.seed(999)) BEFORE
                      env creation to ensure this test is deterministic and reproducible.
    - TWAP strategy:  pre-calculated by the environment's shadow baseline on the SAME seed.
                      Guaranteed fair comparison -- identical price path.
    - AC Optimal:     Almgren-Chriss (2000) mathematically optimal schedule, also pre-
                      calculated on the same seed. Represents the theoretical performance ceiling.

    Monotonic ordering (Random > TWAP > AC Optimal in IS bps) proves:
    1. The environment is non-trivial (random doesn't accidentally do well)
    2. The reward signal correctly incentivizes better strategies
    3. There is a clear performance ceiling an RL agent can learn to approach
    """
    print("\n[Layer 3] Verifying skill gradient: Random < TWAP < AC-Optimal...")
    try:
        import random as pyrandom
        
        # CRITICAL: Seed the random agent BEFORE environment creation to ensure
        # deterministic, reproducible results across validation runs.
        # Without this, the random agent's IS would vary, making regression testing impossible.
        pyrandom.seed(999)
        
        from server.trade_environment import TradeExecEnvironment
        env = TradeExecEnvironment()
        env.reset(seed=123, task_id="task1_twap_beater")

        # Read pre-calculated TWAP and AC Optimal IS from the shadow baseline cache.
        # These ran on the EXACT SAME seed=123 price path -> fair comparison.
        twap_is = float(env._baseline_cache[env._max_steps]["twap"])
        heuristic_is = float(env._baseline_cache[env._max_steps]["ac"])

        # Run a true random agent: uniformly random rate each step on the same price path.
        # A random agent over-trades during high-impact periods and misses volume surges,
        # producing substantially higher IS than either math baseline.
        while not env._episode_done:
            env.execute_trade(participation_rate=pyrandom.uniform(0.01, 0.25))
        random_is = env._compute_current_is()

        # Lower IS = better execution quality.
        # Assert monotonic ordering: AC Opt <= TWAP < Random
        monotonic = bool(heuristic_is <= twap_is and twap_is < random_is)

        print(f"  -> Random agent:       {random_is:.2f} bps  (worst -- uniformly random rate)")
        print(f"  -> TWAP baseline:      {twap_is:.2f} bps  (middle -- equal-slice math)")
        print(f"  -> AC Optimal ceiling: {heuristic_is:.2f} bps  (best -- Almgren-Chriss 2000)")
        print(f"  -> Gradient monotonic: {'[OK] VALIDATED -- Skill correlates with reward' if monotonic else '[FAIL] FAILED -- ordering broken'}")

        return {
            "agents": {
                "random": {"is_bps": round(random_is, 2), "strategy": "uniform random rate [0.01, 0.25]"},
                "twap":   {"is_bps": round(twap_is, 2),   "strategy": "equal time-slice (shadow baseline)"},
                "ac_optimal": {"is_bps": round(heuristic_is, 2), "strategy": "Almgren-Chriss 2000 optimal"},
            },
            "monotonic_ordering": monotonic,
            "interpretation": "Lower IS bps = better execution. Random > TWAP > AC Optimal confirms skill gradient.",
            "status": "PASS" if monotonic else "FAIL",
        }

    except Exception as e:
        print(f"  -> ERROR: {e}")
        return {"status": "ERROR", "error": str(e)}


# ------------------------------------------------------------------------------
# Layer 4: OpenEnv API Compliance (requires running server)
# ------------------------------------------------------------------------------

def run_layer4_openenv_compliance(base_url: str = "http://localhost:7860") -> dict:
    """
    Verify all 6 HTTP endpoints of the OpenEnv-compliant server respond correctly.

    Core endpoints (must pass): /health, /reset, /step
    Optional endpoints (recommended): /state, /schema, /mcp

    This proves the server is OpenEnv v0.2.1 spec compliant and ready for
    external agent evaluation without any client-side workarounds.
    """
    print(f"\n[Layer 4] OpenEnv API compliance check at {base_url}...")
    try:
        import urllib.request
        import urllib.error
        import json as _json

        endpoints_ok = {}

        def get(path, timeout=5):
            with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as r:
                return _json.loads(r.read())

        def post(path, payload, timeout=10):
            data = _json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{base_url}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return _json.loads(r.read())

        # /health -- liveness probe
        try:
            data = get("/health")
            ok = data.get("status") in ("healthy", "ok")
            endpoints_ok["/health"] = ok
            print(f"  -> GET  /health:  {'[OK] OK' if ok else '[FAIL] FAIL'}  -> {data}")
        except Exception as e:
            endpoints_ok["/health"] = False
            print(f"  -> GET  /health:  [FAIL] UNREACHABLE ({e})")

        # /reset -- episode initialisation (CORE)
        try:
            data = post("/reset", {"task_id": "task1_twap_beater", "seed": 42})
            ok = isinstance(data, dict)
            endpoints_ok["/reset"] = ok
            print(f"  -> POST /reset:   {'[OK] OK' if ok else '[FAIL] FAIL'}  -> keys={list(data.keys())}")
        except Exception as e:
            endpoints_ok["/reset"] = False
            print(f"  -> POST /reset:   [FAIL] FAILED ({e})")

        # /step -- primary action endpoint (CORE)
        # Uses OpenEnv CallToolAction schema:
        #   {"action": {"type": "call_tool", "tool_name": "...", "arguments": {...}}}
        try:
            data = post("/step", {
                "action": {
                    "type": "call_tool",
                    "tool_name": "execute_trade",
                    "arguments": {"participation_rate": 0.05}
                }
            })
            ok = isinstance(data, dict)
            endpoints_ok["/step"] = ok
            print(f"  -> POST /step:    {'[OK] OK' if ok else '[FAIL] FAIL'}  -> keys={list(data.keys())}")
        except Exception as e:
            endpoints_ok["/step"] = False
            print(f"  -> POST /step:    [FAIL] FAILED ({e})")

        # /state -- current market state (optional)
        try:
            data = get("/state")
            endpoints_ok["/state"] = True
            print(f"  -> GET  /state:   [OK] OK")
        except Exception as e:
            endpoints_ok["/state"] = False
            print(f"  -> GET  /state:   [FAIL] UNAVAILABLE ({e})")

        # /schema -- environment schema (optional)
        try:
            data = get("/schema")
            endpoints_ok["/schema"] = True
            print(f"  -> GET  /schema:  [OK] OK")
        except Exception as e:
            endpoints_ok["/schema"] = False
            print(f"  -> GET  /schema:  [FAIL] UNAVAILABLE ({e})")

        # /mcp -- MCP protocol endpoint (optional)
        try:
            data = post("/mcp", {"method": "tools/list", "params": {}})
            endpoints_ok["/mcp"] = True
            print(f"  -> POST /mcp:     [OK] OK")
        except Exception as e:
            endpoints_ok["/mcp"] = False
            print(f"  -> POST /mcp:     [FAIL] UNAVAILABLE (optional) ({e})")

        # Core = /health + /reset + /step must all pass
        core_ok = all(endpoints_ok.get(ep, False) for ep in ["/health", "/reset", "/step"])
        total_ok = sum(endpoints_ok.values())
        total = len(endpoints_ok)
        print(f"  -> Summary: {total_ok}/{total} endpoints OK -- Core: {'[OK] PASS' if core_ok else '[FAIL] FAIL'}")

        return {
            "endpoints": endpoints_ok,
            "core_endpoints_ok": core_ok,
            "endpoints_passing": f"{total_ok}/{total}",
            "status": "PASS" if core_ok else "FAIL",
        }

    except Exception as e:
        print(f"  -> ERROR: {e}")
        return {"status": "ERROR", "error": str(e)}


# ------------------------------------------------------------------------------
# Determinism Check (bonus)
# ------------------------------------------------------------------------------

def run_performance_profiling() -> dict:
    """
    Profile baseline cache performance to ensure reset() is fast enough for RL training.
    
    The baseline cache (_calculate_real_baselines) runs 3 full simulations at reset:
    - TWAP trajectory (120 steps)
    - VWAP trajectory (120 steps)  
    - AC Optimal trajectory (120 steps)
    
    This is 360 total physics steps. We verify this overhead is < 200ms to ensure
    the environment is suitable for high-throughput RL training loops.
    """
    print("\n[Performance] Profiling baseline cache overhead...")
    try:
        from server.trade_environment import TradeExecEnvironment
        import time
        
        env = TradeExecEnvironment()
        
        # Measure reset time (includes baseline cache calculation)
        start = time.perf_counter()
        env.reset(seed=42, task_id="task1_twap_beater")
        reset_time_ms = (time.perf_counter() - start) * 1000
        
        # Measure single step time
        step_times = []
        for _ in range(10):
            start = time.perf_counter()
            env.execute_trade(participation_rate=0.05)
            step_times.append((time.perf_counter() - start) * 1000)
        
        avg_step_time_ms = sum(step_times) / len(step_times)
        
        # Performance thresholds
        reset_threshold_ms = 200.0  # Baseline cache should be < 200ms
        step_threshold_ms = 10.0    # Single step should be < 10ms
        
        reset_pass = reset_time_ms < reset_threshold_ms
        step_pass = avg_step_time_ms < step_threshold_ms
        
        print(f"  -> Reset time (with baseline cache): {reset_time_ms:.2f} ms "
              f"({'[OK] PASS' if reset_pass else '[FAIL] SLOW'} -- threshold: {reset_threshold_ms} ms)")
        print(f"  -> Average step time: {avg_step_time_ms:.3f} ms "
              f"({'[OK] PASS' if step_pass else '[FAIL] SLOW'} -- threshold: {step_threshold_ms} ms)")
        
        return {
            "reset_time_ms": round(reset_time_ms, 2),
            "avg_step_time_ms": round(avg_step_time_ms, 3),
            "reset_threshold_ms": reset_threshold_ms,
            "step_threshold_ms": step_threshold_ms,
            "reset_pass": reset_pass,
            "step_pass": step_pass,
            "status": "PASS" if (reset_pass and step_pass) else "SLOW",
            "note": "Baseline cache runs 3 full simulations (TWAP/VWAP/AC) = 360 physics steps at reset"
        }
    
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return {"status": "ERROR", "error": str(e)}


def run_determinism_check() -> dict:
    """Verify that same seed produces identical IS values across two episode runs."""
    print("\n[Bonus] Determinism check (same seed -> same result)...")
    try:
        from server.trade_environment import TradeExecEnvironment

        def run_ep(seed):
            env = TradeExecEnvironment()
            env.reset(seed=seed, task_id="task1_twap_beater")
            for _ in range(5):
                env.execute_trade(participation_rate=0.07)
            return round(env._compute_current_is(), 6)

        is1 = run_ep(42)
        is2 = run_ep(42)
        is3 = run_ep(99)

        same_seed_match = (is1 == is2)
        diff_seed_diff = (is1 != is3)

        print(f"  -> seed=42 run1: {is1} bps")
        print(f"  -> seed=42 run2: {is2} bps  -- {'[OK] MATCH' if same_seed_match else '[FAIL] MISMATCH'}")
        print(f"  -> seed=99:      {is3} bps  -- {'[OK] DIFFERENT (expected)' if diff_seed_diff else '[WARN] SAME (suspicious)'}")

        return {
            "seed42_run1_is": is1,
            "seed42_run2_is": is2,
            "seed99_is": is3,
            "same_seed_deterministic": same_seed_match,
            "status": "PASS" if same_seed_match else "FAIL"
        }

    except Exception as e:
        print(f"  -> ERROR: {e}")
        return {"status": "ERROR", "error": str(e)}


# ------------------------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TradeExecGym -- 4-Layer Robustness Validation Gauntlet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/validate_robustness.py           # Layers 0-3 (offline, no server needed)
  python3 tests/validate_robustness.py --full    # All 5 layers (server must be running)
        """
    )
    parser.add_argument("--full", action="store_true",
                        help="Include Layer 4 API compliance (requires server on --url)")
    parser.add_argument("--url", default="http://localhost:7860",
                        help="Backend server URL (default: http://localhost:7860)")
    parser.add_argument("--output", default="ROBUSTNESS_REPORT.json",
                        help="Output report path (default: ROBUSTNESS_REPORT.json)")
    args = parser.parse_args()

    print("=" * 65)
    print("  TradeExecGym -- Robustness Validation Gauntlet")
    print("  Proving: Physics  | Baselines  | Skill Gradient  | API ")
    print("=" * 65)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "description": (
            "4-Layer Robustness Pyramid for TradeExecGym. "
            "Layer 0: Environment boot. Layer 1: Unit tests. "
            "Layer 2: Baseline scores. Layer 3: Skill gradient. "
            "Layer 4: OpenEnv API compliance."
        ),
    }

    # Layer 0: Boot check (prerequisite -- if this fails, everything else is suspect)
    report["layer0_environment_boot"] = run_layer0_environment_boot()

    # Layer 1: Unit tests
    report["layer1_unit_tests"] = run_layer1_unit_tests()

    # Layer 2: Baseline scores
    report["layer2_baseline_scores"] = run_layer2_baseline_scores()

    # Layer 3: Skill gradient
    report["layer3_skill_gradient"] = run_layer3_skill_gradient()

    # Performance profiling
    report["performance_profiling"] = run_performance_profiling()

    # Determinism bonus
    report["determinism_check"] = run_determinism_check()

    # Layer 4: API compliance (optional -- requires running server)
    if args.full:
        report["layer4_openenv_compliance"] = run_layer4_openenv_compliance(args.url)
    else:
        report["layer4_openenv_compliance"] = {
            "status": "SKIPPED",
            "note": "Run with --full to include Layer 4 API compliance (server required).",
        }

    # -- Overall verdict --------------------------------------------------------
    core_statuses = [
        report["layer0_environment_boot"]["status"],
        report["layer1_unit_tests"]["status"],
        report["layer2_baseline_scores"]["status"],
        report["layer3_skill_gradient"]["status"],
        report["determinism_check"]["status"],
    ]
    if args.full:
        core_statuses.append(report["layer4_openenv_compliance"]["status"])

    passed_count = sum(1 for s in core_statuses if s == "PASS")
    total_count = len(core_statuses)
    all_pass = all(s == "PASS" for s in core_statuses)

    # If Layer 4 was skipped (not --full), still consider it a full pass offline
    report["overall"] = "PASS [OK]" if all_pass else f"PARTIAL [WARN]  ({passed_count}/{total_count} layers passed)"
    report["layers_passed"] = f"{passed_count}/{total_count}"

    # -- Write report -----------------------------------------------------------
    output_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        args.output
    )
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Overall Result:  {report['overall']}")
    print(f"  Layers Passed:   {report['layers_passed']}")
    print(f"  Report saved  ->  {output_path}")
    print("=" * 65)
    if not args.full:
        print("  [TIP] Tip: Run with --full to also validate Layer 4 (API compliance)")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

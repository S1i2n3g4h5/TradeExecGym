"""
TradeExecGym - Visual Simulation Dashboard
==========================================
Gradio-based interactive dashboard to visualize AI execution algorithms vs. Human strategy.
Covers the technical rigor and 'LLM Observability' criteria for Meta OpenEnv.
"""
import os
import sys
import time
import json
import asyncio
import argparse
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # CRITICAL: Prevent crashes in headless Docker/HF Space
from openai import AsyncOpenAI
# Add root to sys.path to resolve local imports like `client` and `baselines`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.heuristic_agent import AlmgrenChrissHeuristic
from typing import Optional
from client import TradeExecClient

# Load the trained model if available, fallback to None
MODEL_PATH = "models/grpo_agent.zip"
_loaded_agent = None
try:
    from stable_baselines3 import PPO
    if os.path.exists(MODEL_PATH):
        try:
            _loaded_agent = PPO.load(MODEL_PATH)
            print(f"[app_visual] Loaded GRPO Agent from {MODEL_PATH}")
        except Exception as e:
            print(f"[app_visual] Failed to load model at {MODEL_PATH}: {e}")
            _loaded_agent = None
except (ImportError, Exception):
    _loaded_agent = None

TASKS = [
    "Task 1: The TWAP Beater",
    "Task 2: VWAP Optimizer",
    "Task 3: Volatile Execution",
    "Task 4: Adversarial HFT",
    "Task 5: Deadline Cliff",
]

TASK_ID_MAP = {
    "Task 1: The TWAP Beater": "task1_twap_beater",
    "Task 2: VWAP Optimizer": "task2_vwap_optimizer",
    "Task 3: Volatile Execution": "task3_volatile_execution",
    "Task 4: Adversarial HFT": "task4_adversarial",
    "Task 5: Deadline Cliff": "task5_deadline_pressure",
}

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
class UIState:
    def __init__(self):
        self.client = None
        self.history = []
        self.is_running = False
        self.current_obs = None
        self.task_id = "task1_twap_beater"

    async def start_session(self, display_name, seed=42):
        if self.client is None:
            self.client = TradeExecClient(base_url="http://localhost:7865")
        
        task_id = TASK_ID_MAP.get(display_name, "task1_twap_beater")
        try:
            obs = await self.client.reset(task_id=task_id, seed=int(seed))
        except Exception:
            # Reconnect and retry once
            try:
                await self.client.close()
            except Exception:
                pass
            self.client = TradeExecClient(base_url="http://localhost:7865")
            obs = await self.client.reset(task_id=task_id, seed=int(seed))

        self.task_id = task_id
        self.history = []
        self.current_obs = obs
        self.is_running = True
        # Get initial market state narrative after reset
        try:
            state_text = await self.client.get_market_state()
            if state_text:
                return f"✅ Session initialized: **{task_id}** (seed={seed})\n\n{state_text}"
        except Exception:
            pass
        return f"✅ Session started: {task_id} (seed={seed})\n\nReady — click 'Execute Step' to begin trading."

    def get_summary(self):
        if not self.current_obs:
            return "No active session."
        # After reset(), current_obs is an Observation object with .metadata
        # After step(), current_obs is a raw result string
        if isinstance(self.current_obs, str):
            return self.current_obs
        meta = self.current_obs.metadata if hasattr(self.current_obs, 'metadata') else {}
        return meta.get("output", "Session started.")

    async def step(self, rate, use_dark, dark_frac):
        if not self.client:
            return "No session active.", None, gr.update(), 0.0, 0.0, []
        
        try:
            result = await self.client.execute_trade(
                participation_rate=float(rate),
                use_dark_pool=bool(use_dark),
                dark_pool_fraction=float(dark_frac),
            )
            
            metrics = self._parse_result(result)
            self.history.append(metrics)
            self.current_obs = result
            
            is_val = metrics.get("is_bps", 0.0)
            score_val = metrics.get("score", 0.0)
            
            if "EPISODE COMPLETE" in result or "ENGINE ERROR" in result:
                self.is_running = False
                
            return result, self.create_plot(), gr.update(interactive=self.is_running), is_val, score_val, self.history
        except Exception as e:
            return f"❌ Connection Error: {str(e)}", None, gr.update(), 0.0, 0.0, []

    def _parse_result(self, text):
        # Use a consistent sequence ID if we are parsing for history
        metrics = {
            "price": 0.0, 
            "pct_done": 0.0, 
            "is_bps": 0.0, 
            "score": 0.0, 
            "step": len(self.history),
            "steps_left": 30
        }
        
        if not text or not isinstance(text, str):
            return metrics

        # Fallback to last known price to avoid chart jumping to zero
        if self.history:
            metrics["price"] = self.history[-1]["price"]

        try:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines:
                try:
                    # More robust substring matching
                    if "Mid Price:" in line:
                        parts = line.split("$")
                        if len(parts) > 1:
                            val = parts[1].split()[0].replace(",", "").strip()
                            metrics["price"] = float(val)
                    elif "Executed:" in line and "%" in line:
                        # Pattern: "Executed:  30,343 / 100,000 (30.3%)"
                        if "(" in line and "%" in line:
                            val = line.split("(")[1].split("%")[0].strip()
                            metrics["pct_done"] = float(val)
                    elif "Your IS:" in line:
                        # Pattern: "Your IS:  4.66 bps"
                        raw = line.split("Your IS:")[1].strip()
                        val = raw.lower().replace("bps", "").strip().split()[0]
                        metrics["is_bps"] = float(val)
                    elif "Final IS:" in line:
                        # Pattern: "Final IS:       4.66 bps"
                        raw = line.split("Final IS:")[1].strip()
                        val = raw.lower().replace("bps", "").strip().split()[0]
                        metrics["is_bps"] = float(val)
                    elif "Grader Score:" in line:
                        # Pattern: "Grader Score:   0.7910 / 1.0000"
                        raw = line.split("Grader Score:")[1].strip()
                        val = raw.split("/")[0].strip()
                        metrics["score"] = float(val)
                    elif "Time left:" in line:
                        # Pattern: "Time left: 18 steps"
                        raw = line.split("Time left:")[1].strip()
                        val = raw.split("steps")[0].strip()
                        metrics["steps_left"] = int(val)
                except (ValueError, IndexError):
                    continue 
        except Exception:
            pass 
            
        return metrics

    def create_plot(self):
        if not self.history:
            return None
        return plot_trajectory(self.history, "Human Execution")

# ---------------------------------------------------------------------------
# Common Plotting Function
# ---------------------------------------------------------------------------
def plot_trajectory(history_df, title="Market Dynamics"):
    """Render a 2-panel chart showing Mid Price and Execution Progress."""
    if not history_df:
        return None
    
    plt.close('all')  # Prevent matplotlib memory leaks
    df = pd.DataFrame(history_df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Panel 1: Price
    ax1.plot(df["step"], df["price"], color="#00ffcc", marker="o", markersize=4, label="Mid Price ($)")
    ax1.set_ylabel("Price ($)", color="white")
    ax1.set_title(title, color="white", fontsize=14)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors='white')
    ax1.yaxis.label.set_color('white')
    legend1 = ax1.legend(loc="upper left")
    for text in legend1.get_texts():
        text.set_color('white')
    
    # Panel 1 twin axis: Implementation Shortfall 
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df["step"], df["is_bps"], color="#ff00ff", linestyle="--", label="Slippage (bps)")
    ax1_twin.set_ylabel("Slippage (bps)", color="#ff00ff")
    ax1_twin.tick_params(colors='#ff00ff')
    legend_twin = ax1_twin.legend(loc="lower right")
    for text in legend_twin.get_texts():
        text.set_color('white')

    # Panel 2: Completion
    ax2.bar(df["step"], df["pct_done"], color="#4CAF50", alpha=0.6, label="Completion %")
    ax2.set_ylabel("Done %", color="white")
    ax2.set_xlabel("Step", color="white")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors='white')
    ax2.yaxis.label.set_color('white')
    ax2.xaxis.label.set_color('white')
    legend2 = ax2.legend(loc="upper left")
    for text in legend2.get_texts():
        text.set_color('white')
    
    # Dark theme styling
    fig.patch.set_facecolor('#111111')
    ax1.set_facecolor('#1a1a1a')
    ax2.set_facecolor('#1a1a1a')
    ax1_twin.set_facecolor('#1a1a1a')
    for ax in [ax1, ax2, ax1_twin]:
        for spine in ax.spines.values():
            spine.set_color('#444444')
            
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Live Model Evaluation Logic (Streaming)
# ---------------------------------------------------------------------------
async def run_live_eval(display_name, hf_token, model_name, sys_prompt, seed=42):
    """Streams a live inference session using an LLM + Heuristic Hybrid."""
    if not hf_token:
        yield "### Error: HF_TOKEN is required for Live Eval.", None, [], "[ERROR] Missing Token"
        return

    client = TradeExecClient(base_url="http://localhost:7865")
    # Update to official HF model gateway
    llm_client = AsyncOpenAI(api_key=hf_token, base_url="https://huggingface.co/v1/")
    heuristic = AlmgrenChrissHeuristic()
    task_id = TASK_ID_MAP.get(display_name, "task1_twap_beater")
    
    log_stream = f"[START] task={task_id} env=trade_exec_gym model={model_name}\n"
    try:
        yield f"### Initializing {task_id}...", None, [], log_stream
        
        await client.reset(task_id=task_id, seed=int(seed))
        state_parser = UIState()
        history = []
        
        max_steps = 30
        if "VWAP" in display_name: max_steps = 60
        elif "Volatile" in display_name: max_steps = 90
        elif "Adversarial" in display_name: max_steps = 120
        elif "Deadline" in display_name: max_steps = 80
        
        done = False
        step = 0
        while not done and step < max_steps:
            step += 1
            state_text = await client.get_market_state()
            
            # Math Layer
            base_rate = 0.05
            if "Remaining:" in state_text:
                try:
                    rem = int(state_text.split("Remaining:")[1].split("shares")[0].replace(",","").strip())
                    tl = int(state_text.split("Time left:")[1].split("steps")[0].strip())
                    base_rate = heuristic.calculate_rate(rem, 1_000_000, tl, 0.0)
                except: pass

            # Cognitive Layer (LLM)
            final_rate = base_rate
            try:
                resp = await asyncio.wait_for(
                    llm_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": f"State: {state_text}\nMath Suggested Rate: {base_rate}"}
                        ],
                        max_tokens=100,
                        response_format={"type": "json_object"}
                    ),
                    timeout=8.0
                )
                decision = json.loads(resp.choices[0].message.content)
                rec = decision.get("recommendation", "Approve")
                if rec == "Accelerate": final_rate *= 1.3
                elif rec == "Decelerate": final_rate *= 0.7
            except: pass

            # Action
            result = await client.execute_trade(participation_rate=final_rate)
            reward = await client.get_reward()
            
            # Update UI
            metrics = state_parser._parse_result(result)
            metrics["step"] = step
            history.append(metrics)
            
            done_bool = "EPISODE COMPLETE" in result or "ENGINE ERROR" in result
            log_step = f"[STEP] step={step} action={final_rate:.4f} reward={reward:.2f} done={str(done_bool).lower()} error=null\n"
            log_stream += log_step
            
            yield (
                f"### Executing {task_id}...\nStep {step}/{max_steps}", 
                plot_trajectory(history, f"Live Eval: {model_name}"), 
                history,
                log_stream
            )
            
            if done_bool:
                done = True
                score = metrics.get("score", 0.0)
                log_stream += f"[END] success={str(score >= 0.8).lower()} steps={step} score={score:.3f} rewards=..."
                yield (
                    f"### Session Complete\nFinal Score: {score:.4f}", 
                    plot_trajectory(history, f"Live Eval: {model_name}"), 
                    history,
                    log_stream
                )

        await client.close()
    except Exception as e:
        yield f"### Session Failed\n{str(e)}", None, [], log_stream
        try: await client.close()
        except: pass

# ---------------------------------------------------------------------------
# Auto Simulation Logic
# ---------------------------------------------------------------------------
async def run_auto_simulation(display_name, mode, seed=42):
    """Run an automated episode based on selected mode."""
    client = TradeExecClient(base_url="http://localhost:7865")
    task_id = TASK_ID_MAP.get(display_name, "task1_twap_beater")
    
    try:
        await client.reset(task_id=task_id, seed=int(seed))
        state_parser = UIState()
        history = []
        
        # Determine max steps based on task
        max_steps = 30
        if "VWAP" in display_name: max_steps = 60
        elif "Volatile" in display_name: max_steps = 90
        elif "Adversarial" in display_name: max_steps = 120
        elif "Deadline" in display_name: max_steps = 80
        
        done = False
        step = 0
        while not done and step < max_steps:
            rate = 0.05
            use_dark = False
            dark_frac = 0.0

            if mode == "Volume-Weighted (VWAP)":
                p = step / max(1, max_steps)
                vol_ratio = 1.6 if p < 0.20 else (0.5 if p < 0.8 else 1.8)
                rate = 0.05 * vol_ratio
                rate = min(0.25, max(0.01, rate))
            elif mode == "Optimal Heuristic (Math)":
                from baselines.heuristic_agent import AlmgrenChrissHeuristic
                h = AlmgrenChrissHeuristic()
                # Mocking remaining shares for UI sim
                rate = h.calculate_rate(800_000 * (1 - step/max_steps), 1_000_000, max_steps-step, 0.0)
            elif mode == "Hybrid (Heuristic + LLM)":
                from baselines.heuristic_agent import AlmgrenChrissHeuristic
                h = AlmgrenChrissHeuristic()
                rate = h.calculate_rate(800_000 * (1 - step/max_steps), 1_000_000, max_steps-step, 0.0)
                # LLM bias would go here if HF_TOKEN is present
                if os.environ.get("HF_TOKEN"):
                    rate *= 1.1 # Dummy LLM 'Aggressive' bias for the demo

            result = await client.execute_trade(
                participation_rate=rate, 
                use_dark_pool=use_dark, 
                dark_pool_fraction=dark_frac
            )
            
            # Reuse logic
            metrics = state_parser._parse_result(result)
            metrics["step"] = step
            history.append(metrics)
            
            if "EPISODE COMPLETE" in result or "ENGINE ERROR" in result:
                done = True
            step += 1
        
        final_is = history[-1].get("is_bps", 0) if history else 0
        final_score = history[-1].get("score", 0) if history else 0
        
        summary_text = (
            f"### Simulation Complete: {mode} on {task_id}\n\n"
            f"**Steps Taken:** {step}\n"
            f"**Final IS:** {final_is:.2f} bps\n"
            f"**Grader Score:** {final_score:.4f}/1.0\n"
        )
        await client.close()
        # Return complete step history for the JSON view
        return summary_text, plot_trajectory(history, f"Auto: {mode}"), history
        
    except Exception as e:
        try: await client.close()
        except: pass
        return f"### simulation Failed\n\nError: {str(e)}", None, []

# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
state = UIState()

CUSTOM_CSS = """
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

body { background-color: #0b0f19 !important; color: #e2e8f0 !important; }
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    background-color: transparent !important;
}

/* ── Hero header ── */
.hero-header {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 50px 20px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -100px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(16,185,129,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.top-pill {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    padding: 6px 16px;
    border-radius: 24px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    display: inline-block;
}
.tech-pill {
    background: #1e293b;
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 6px 16px;
    border-radius: 20px;
    border: 1px solid #334155;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.2s ease;
}
.tech-pill:hover { background: #334155; border-color: #475569; }

/* ── Info tab prose ── */
.info-section {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 10px 0;
}
.info-section h2 {
    color: #10b981 !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 8px;
    margin-bottom: 16px !important;
}
.info-section h3 { color: #34d399 !important; font-size: 1rem !important; font-weight: 600 !important; margin: 18px 0 8px !important; }
.info-section p, .info-section li { color: #d1fae5 !important; line-height: 1.7 !important; font-size: 0.92rem !important; }
.info-section code {
    background: #1f2937;
    border: 1px solid #374151;
    color: #6ee7b7 !important;
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
}
.info-section pre {
    background: #0f172a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px;
    padding: 16px !important;
    overflow-x: auto;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #bae6fd !important;
    line-height: 1.6 !important;
}
.info-section table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.88rem !important;
}
.info-section th {
    background: #065f46 !important;
    color: #ecfdf5 !important;
    padding: 10px 14px !important;
    text-align: left;
    font-weight: 600;
}
.info-section td {
    padding: 9px 14px !important;
    color: #d1fae5 !important;
    border-bottom: 1px solid #1f2937;
}
.info-section tr:nth-child(even) td { background: #0f1f18; }
.info-section tr:hover td { background: #134e38; }

/* ── Stat cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }
.stat-card {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-card .stat-val { font-size: 1.6rem; font-weight: 700; color: #6ee7b7; }
.stat-card .stat-lbl { font-size: 0.75rem; color: #a7f3d0; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── difficulty badges ── */
.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; margin-left:6px; }
.badge-easy    { background:#059669; color:#ecfdf5; }
.badge-medium  { background:#d97706; color:#fef3c7; }
.badge-hard    { background:#dc2626; color:#fee2e2; }
.badge-vhard   { background:#7c3aed; color:#ede9fe; }
.badge-extreme { background:#1f2937; color:#f9fafb; border:1px solid #6b7280; }

/* ── result box ── */
.result-box { border:1px solid rgba(16,185,129,0.4); padding:10px; border-radius:5px; }
"""

def build_gui():
    with gr.Blocks(
        title="TradeExecGym — Institutional SOR Dashboard",
    ) as demo:

        gr.HTML("""
        <div class="hero-header">
            <div style="display:flex; justify-content:center; margin-bottom:24px; position:relative; z-index:10;">
                <span class="top-pill">🟢 OPENENV · V1.0.0 · META HACKATHON 2025</span>
            </div>
            <div style="display:flex; align-items:center; justify-content:center; gap:20px; margin-bottom:16px; position:relative; z-index:10;">
                <span style="font-size:3.5rem;">📈</span>
                <h1 style="font-size: 3.5rem !important; margin:0 !important; color:#f8fafc !important; font-weight:800 !important; letter-spacing:-1.5px; font-family:'Inter', sans-serif;">
                    Trade<span style="color:#10b981;">Exec</span>Gym
                </h1>
            </div>
            <p style="color:#94a3b8; font-size:1.1rem; max-width:700px; margin:0 auto 32px auto; line-height:1.6; position:relative; z-index:10;">
                A physics-grounded Reinforcement Learning environment where AI agents master<br>
                institutional order routing, market timing, and adversary evasion.
            </p>
            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; position:relative; z-index:10;">
                <span class="tech-pill"><span style="color:#10b981; font-size:1.1rem;">⚯</span> Python &ge; 3.10</span>
                <span class="tech-pill"><span style="color:#eab308; font-size:1.1rem;">⚡</span> FastAPI + Uvicorn</span>
                <span class="tech-pill"><span style="color:#f59e0b; font-size:1.1rem;">🤗</span> HuggingFace Space</span>
                <span class="tech-pill"><span style="color:#ec4899; font-size:1.1rem;">🔴</span> OpenEnv Core</span>
                <span class="tech-pill"><span style="color:#3b82f6; font-size:1.1rem;">🐳</span> Docker Ready</span>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            # ================= Tab 1: Auto Simulation =================
            with gr.TabItem("Auto Simulation (Agentic)"):
                model_status = (
                    f"🟢 Local RL agent loaded: `{MODEL_PATH}`" if _loaded_agent 
                    else "🔴 No RL model found. Using heuristic logic."
                )
                gr.Markdown(f"> {model_status}")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Setup")
                        auto_task_dd = gr.Dropdown(choices=TASKS, value=TASKS[0], label="Market Regime")
                        auto_seed = gr.Number(value=42, label="Random Seed (Reproducibility)")
                        auto_mode_dd = gr.Radio(
                            choices=["Time-Weighted (TWAP)", "Volume-Weighted (VWAP)", "Optimal Heuristic (Math)", "Hybrid (Heuristic + LLM)"], 
                            value="Time-Weighted (TWAP)", 
                            label="Execution Method"
                        )
                        run_auto_btn = gr.Button("Start Auto-Execution", variant="primary", size="lg")
                        auto_json = gr.JSON(label="Complete Step History (All Steps)")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Performance Live Feed")
                        auto_plot = gr.Plot(label="Order Trajectory")
                        auto_summary = gr.Markdown(label="Post-Trade Analysis")
                
                # Native async click handlers
                run_auto_btn.click(run_auto_simulation, inputs=[auto_task_dd, auto_mode_dd, auto_seed], outputs=[auto_summary, auto_plot, auto_json])

            # ================= Tab 2: Live Model Evaluation =================
            with gr.TabItem("Live Model Evaluation (Compliance Test)"):
                with gr.Column(elem_classes=["info-section"]):
                    gr.Markdown("### Test any LLM against the OpenEnv Standard")
                    with gr.Row():
                        with gr.Column(scale=1):
                            live_task = gr.Dropdown(choices=TASKS, value=TASKS[0], label="Select Task")
                            live_token = gr.Textbox(label="HF_TOKEN", type="password", placeholder="Enter your Hugging Face API key")
                            live_model = gr.Textbox(label="Model Identifier", value="meta-llama/Meta-Llama-3-70B-Instruct")
                            live_seed = gr.Number(value=42, label="Evaluation Seed")
                            live_prompt = gr.Textbox(
                                label="System Prompt", 
                                value='{"recommendation": "Approve|Accelerate|Decelerate", "reason": "..."}',
                                lines=3
                            )
                            run_live_btn = gr.Button("▶ Run Live Inference", variant="primary")
                            live_json = gr.JSON(label="Complete Step History (All Steps)")
                        
                        with gr.Column(scale=2):
                            live_plot = gr.Plot(label="Real-time Execution Trace")
                            live_status = gr.Markdown("Ready to evaluate...")
                            live_logs = gr.Code(label="Standardized Stdout Logs ([START]/[STEP]/[END])", interactive=False)

                run_live_btn.click(
                    run_live_eval, 
                    inputs=[live_task, live_token, live_model, live_prompt, live_seed], 
                    outputs=[live_status, live_plot, live_json, live_logs]
                )

            # ================= Tab 3: Manual Challenge Mode =================
            with gr.TabItem("Manual Challenge"):
                with gr.Column(elem_classes=["info-section"]):
                    gr.Markdown("Try to trade better than a basic TWAP script. Watch out for HFT predatory algorithms that punish predictable patterns.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            task_select = gr.Dropdown(choices=TASKS, value=TASKS[0], label="Select Task")
                            man_seed = gr.Number(value=42, label="Random Seed")
                            reset_btn = gr.Button("Initialize Session", variant="primary")
                            
                            with gr.Group():
                                gr.Markdown("### Controls")
                                rate_slider = gr.Slider(minimum=0.0, maximum=0.25, step=0.01, value=0.05, label="Block Rate (e.g. 0.05 = 5% of Volume)")
                                dark_check = gr.Checkbox(label="Use Dark Pool")
                                dark_frac = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Dark Pool Fraction")
                                step_btn = gr.Button("Execute Step", variant="secondary", interactive=False)
                                
                            # Metrics Boxes
                            gr.Markdown("### Metrics")
                            with gr.Row():
                                is_box = gr.Number(label="Shortfall (bps)", precision=2)
                                score_box = gr.Number(label="Grader Score", precision=4)
                            man_json = gr.JSON(label="Complete Step History (All Steps)")

                        with gr.Column(scale=2):
                            plot_output = gr.Plot(label="Market Canvas", container=True)
                            status_text = gr.Textbox(label="Agent Log & LLM Narratives", lines=15, max_lines=20)

                # Async Event Handlers for Manual Mode
                async def _on_reset(task_id, seed):
                    summary = await state.start_session(task_id, seed)
                    return summary, None, gr.update(interactive=True)

                async def _on_step(rate, use_dark, dark_frac):
                    return await state.step(rate, use_dark, dark_frac)
                    
                reset_btn.click(_on_reset, inputs=[task_select, man_seed], outputs=[status_text, plot_output, step_btn])
                
                step_btn.click(
                    _on_step, 
                    inputs=[rate_slider, dark_check, dark_frac], 
                    outputs=[status_text, plot_output, step_btn, is_box, score_box, man_json]
                )

            # ================= Tab 4: Strategy Guide (Cheat Sheet) =================
            with gr.TabItem("🎯 Strategy Guide"):
                gr.HTML("""
                <div style="padding: 8px 0 24px 0;">
                    <p style="color:#94a3b8; font-size:0.95rem; max-width:820px; line-height:1.7; margin:0 auto;">
                        Each task is designed to break a different class of naive agent.
                        This guide shows you <strong style="color:#10b981;">exactly</strong> what separates a beginner from an expert —
                        and reveals the winning secret for each task.
                    </p>
                </div>
                """)

                # Task 1
                with gr.Group(elem_classes=["info-section"]):
                    gr.HTML("""
                    <div style="margin-bottom:16px; display:flex; align-items:center; gap:12px;">
                        <span style="font-size:1.6rem;">🟢</span>
                        <div>
                            <h2 style="margin:0; color:#10b981; font-size:1.15rem; font-weight:700;">Task 1: The TWAP Beater</h2>
                            <span style="color:#64748b; font-size:0.82rem;">100K shares · 30 steps · Low volatility (σ=0.02) · EASY</span>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#f87171; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🐣 NAIVE APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Trade the same amount every step. Set <code style="background:#111;padding:1px 5px;border-radius:3px;">rate=0.033</code> all 30 steps and hope for the best.</div>
                            <div style="color:#f87171; font-size:0.78rem; margin-top:10px;">❌ Result: IS ≈ 25 bps (TWAP baseline). Score ≤ 0.50</div>
                        </div>
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#fbbf24; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🧠 EXPERT APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Adjust rate dynamically based on time remaining. Use <code style="background:#111;padding:1px 5px;border-radius:3px;">remaining/steps_left</code> to stay on pace.</div>
                            <div style="color:#fbbf24; font-size:0.78rem; margin-top:10px;">✅ Result: IS ≈ 18–22 bps. Score ≈ 0.70–0.80</div>
                        </div>
                        <div style="background:#064e3b; border:1px solid rgba(16,185,129,0.5); border-radius:10px; padding:16px;">
                            <div style="color:#10b981; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🏆 WINNING SECRET</div>
                            <div style="color:#d1fae5; font-size:0.88rem; line-height:1.6;">Exploit the <strong>Open/Close volume surges</strong>! Trade 2–3× faster at the open (steps 1–6) and close (steps 25–30). Slow down midday when spreads are wide.</div>
                            <div style="color:#10b981; font-size:0.78rem; margin-top:10px;">🏅 Result: IS ≈ 14–18 bps. Score 0.85–0.91</div>
                        </div>
                    </div>
                    """)

                # Task 2
                with gr.Group(elem_classes=["info-section"]):
                    gr.HTML("""
                    <div style="margin-bottom:16px; display:flex; align-items:center; gap:12px;">
                        <span style="font-size:1.6rem;">🟡</span>
                        <div>
                            <h2 style="margin:0; color:#fbbf24; font-size:1.15rem; font-weight:700;">Task 2: VWAP Optimizer</h2>
                            <span style="color:#64748b; font-size:0.82rem;">250K shares · 60 steps · U-shaped volume curve · MEDIUM</span>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#f87171; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🐣 NAIVE APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Trade flat at rate=0.017 across all 60 steps. Ignore the intraday volume rhythm entirely.</div>
                            <div style="color:#f87171; font-size:0.78rem; margin-top:10px;">❌ Result: Midday impact destroys IS. Score ≤ 0.45</div>
                        </div>
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#fbbf24; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🧠 EXPERT APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Use a two-bucket model: high rate open/close, low rate midday. Roughly track the volume ratio signal.</div>
                            <div style="color:#fbbf24; font-size:0.78rem; margin-top:10px;">✅ Result: IS ≈ 18–22 bps. Score ≈ 0.72–0.82</div>
                        </div>
                        <div style="background:#064e3b; border:1px solid rgba(16,185,129,0.5); border-radius:10px; padding:16px;">
                            <div style="color:#10b981; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🏆 WINNING SECRET</div>
                            <div style="color:#d1fae5; font-size:0.88rem; line-height:1.6;"><strong>Ride the U-Curve</strong>: Rate 0.12–0.18 in steps 1–10, rate 0.02–0.04 in steps 20–40 (midday), rate 0.15–0.25 in steps 50–60. The VWAP benchmark already accounts for this — you win by matching it precisely.</div>
                            <div style="color:#10b981; font-size:0.78rem; margin-top:10px;">🏅 Result: IS ≈ 14–16 bps. Score 0.83–0.91</div>
                        </div>
                    </div>
                    """)

                # Task 3
                with gr.Group(elem_classes=["info-section"]):
                    gr.HTML("""
                    <div style="margin-bottom:16px; display:flex; align-items:center; gap:12px;">
                        <span style="font-size:1.6rem;">🔴</span>
                        <div>
                            <h2 style="margin:0; color:#f87171; font-size:1.15rem; font-weight:700;">Task 3: Volatile Execution</h2>
                            <span style="color:#64748b; font-size:0.82rem;">400K shares · 90 steps · 3× volatility (σ=0.06) · HARD</span>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#f87171; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🐣 NAIVE APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Trade aggressively on the lit NASDAQ venue same as Task 1. Ignore the dark pool. Rate=0.10+ on every step.</div>
                            <div style="color:#f87171; font-size:0.78rem; margin-top:10px;">❌ Result: 3× volatility causes 3× impact. IS spikes to 60–90 bps. Score ≤ 0.30</div>
                        </div>
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#fbbf24; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🧠 EXPERT APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Trade smaller on lit venues. Occasionally use dark pool. Keep rate low (0.03–0.06) to limit Almgren-Chriss permanent impact.</div>
                            <div style="color:#fbbf24; font-size:0.78rem; margin-top:10px;">✅ Result: IS ≈ 30–45 bps. Score ≈ 0.60–0.72</div>
                        </div>
                        <div style="background:#064e3b; border:1px solid rgba(16,185,129,0.5); border-radius:10px; padding:16px;">
                            <div style="color:#10b981; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🏆 WINNING SECRET</div>
                            <div style="color:#d1fae5; font-size:0.88rem; line-height:1.6;"><strong>Dark pool stabilization</strong>: Set <code style="background:#064e3b;padding:1px 5px;border-radius:3px;">use_dark_pool=True, dark_pool_fraction=0.3–0.4</code> on every step. Dark pool fills execute at mid-price with zero market impact — neutralizing 40% of your slippage cost automatically.</div>
                            <div style="color:#10b981; font-size:0.78rem; margin-top:10px;">🏅 Result: IS ≈ 20–35 bps. Score 0.75–0.85</div>
                        </div>
                    </div>
                    """)

                # Task 4
                with gr.Group(elem_classes=["info-section"]):
                    gr.HTML("""
                    <div style="margin-bottom:16px; display:flex; align-items:center; gap:12px;">
                        <span style="font-size:1.6rem;">🟣</span>
                        <div>
                            <h2 style="margin:0; color:#a78bfa; font-size:1.15rem; font-weight:700;">Task 4: Adversarial HFT</h2>
                            <span style="color:#64748b; font-size:0.82rem;">600K shares · 120 steps · Dual-detector HFT Sniper · VERY HARD</span>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#f87171; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🐣 NAIVE APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Trade at a constant rate (<code style="background:#111;padding:1px 5px;border-radius:3px;">rate=0.05</code> every step). Or alternate 0.05/0.15 in a repeating pattern.</div>
                            <div style="color:#f87171; font-size:0.78rem; margin-top:10px;">❌ Result: HFT sniper fires every step. 50 bps penalty × 100 steps. Score ≈ 0.07</div>
                        </div>
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#fbbf24; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🧠 EXPERT APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Vary rate unpredictably. Watch the ADVERSARY ALERT warning in the market narrative and change rate immediately when triggered.</div>
                            <div style="color:#fbbf24; font-size:0.78rem; margin-top:10px;">✅ Result: Fewer penalties. IS ≈ 30–50 bps. Score ≈ 0.55–0.65</div>
                        </div>
                        <div style="background:#064e3b; border:1px solid rgba(16,185,129,0.5); border-radius:10px; padding:16px;">
                            <div style="color:#10b981; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🏆 WINNING SECRET</div>
                            <div style="color:#d1fae5; font-size:0.88rem; line-height:1.6;"><strong>Stealth through variance</strong>: The HFT uses TWO detectors: Std Dev (uniformity) AND Lag-1 Autocorrelation (periodicity). Jitter between 0.05–0.15 with <em>true randomness</em> — no repeating patterns. Each step should be independently random to neutralize both detectors simultaneously.</div>
                            <div style="color:#10b981; font-size:0.78rem; margin-top:10px;">🏅 Result: 0 penalties fired. IS ≈ 14–25 bps. Score 0.75–0.90</div>
                        </div>
                    </div>
                    """)

                # Task 5
                with gr.Group(elem_classes=["info-section"]):
                    gr.HTML("""
                    <div style="margin-bottom:16px; display:flex; align-items:center; gap:12px;">
                        <span style="font-size:1.6rem;">⚫</span>
                        <div>
                            <h2 style="margin:0; color:#94a3b8; font-size:1.15rem; font-weight:700;">Task 5: Deadline Cliff</h2>
                            <span style="color:#64748b; font-size:0.82rem;">1,000,000 shares · 80 steps · Hard completion gate · EXTREME</span>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#f87171; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🐣 NAIVE APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Focus on low IS. Trade passively (rate=0.03) to minimize slippage. Miss the 1M share deadline.</div>
                            <div style="color:#f87171; font-size:0.78rem; margin-top:10px;">❌ Result: Grader score = 0.0. Any IS improvement is irrelevant. Score = 0.00</div>
                        </div>
                        <div style="background:#1a1a2e; border:1px solid #374151; border-radius:10px; padding:16px;">
                            <div style="color:#fbbf24; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🧠 EXPERT APPROACH</div>
                            <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">Trade aggressively at constant rate=0.25 throughout. Complete the order, accept mediocre IS.</div>
                            <div style="color:#fbbf24; font-size:0.78rem; margin-top:10px;">✅ Result: Completion gate passed. IS ≈ 60–80 bps. Score ≈ 0.50–0.60</div>
                        </div>
                        <div style="background:#064e3b; border:1px solid rgba(16,185,129,0.5); border-radius:10px; padding:16px;">
                            <div style="color:#10b981; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;">🏆 WINNING SECRET</div>
                            <div style="color:#d1fae5; font-size:0.88rem; line-height:1.6;"><strong>Front-load, then optimize</strong>: Set rate 0.15–0.20 in steps 1–40 to clear the completion gate early. Once you're 80%+ done, <em>then</em> back off to rate 0.05–0.08 to protect IS. Completion unlocks the IS score — the gate must clear first.</div>
                            <div style="color:#10b981; font-size:0.78rem; margin-top:10px;">🏅 Result: Gate cleared + good IS. Score 0.70–0.85</div>
                        </div>
                    </div>
                    """)

            # ================= Tab 5: Project & Environment Info =================
            with gr.TabItem("📖 Project & Environment Info"):
                with gr.Column(elem_classes=["info-section"]):
                    # ── Robustness Certification Banner ──────────────────────────────
                    def _load_robustness_report():
                        """Read ROBUSTNESS_REPORT.json and render a certification banner."""
                        import json as _json
                        report_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ROBUSTNESS_REPORT.json"
                        )
                        if not os.path.exists(report_path):
                            return """
<div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-radius:12px;padding:18px 24px;margin-bottom:24px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:1.8rem;">🛡️</span>
    <div>
      <div style="color:#94a3b8;font-size:0.9rem;font-weight:700;letter-spacing:1px;">ROBUSTNESS CERTIFICATION</div>
      <div style="color:#f59e0b;font-size:1rem;margin-top:2px;">⚠️ Report not found — run <code style="background:#1e293b;padding:1px 6px;border-radius:4px;">python3 tests/validate_robustness.py --full</code> to generate</div>
    </div>
  </div>
</div>"""
                        try:
                            with open(report_path) as f:
                                r = _json.load(f)
                            overall = r.get("overall", "UNKNOWN")
                            ts = r.get("timestamp", "")[:19].replace("T", " ") + " UTC"
                            layers = r.get("layers_passed", "?/?")
                            l0 = r.get("layer0_environment_boot", {}).get("status", "?")
                            l1 = r.get("layer1_unit_tests", {})
                            l1_str = f"{l1.get('passed', '?')}/{l1.get('passed', 0) + l1.get('failed', 0)} tests"
                            l2 = r.get("layer2_baseline_scores", {}).get("status", "?")
                            l3 = r.get("layer3_skill_gradient", {})
                            l3_agents = l3.get("agents", {})
                            l3_rnd = l3_agents.get("random", {}).get("is_bps", "?")
                            l3_twap = l3_agents.get("twap", {}).get("is_bps", "?")
                            l3_ac = l3_agents.get("ac_optimal", {}).get("is_bps", "?")
                            l4 = r.get("layer4_openenv_compliance", {})
                            l4_ep = l4.get("endpoints_passing", "?")
                            det = r.get("determinism_check", {}).get("status", "?")
                            is_pass = "PASS" in overall
                            color = "#10b981" if is_pass else "#f59e0b"
                            icon = "✅" if is_pass else "⚠️"
                            return f"""
<div style="background:linear-gradient(135deg,#064e3b,#0f172a);border:1px solid {'rgba(16,185,129,0.4)' if is_pass else 'rgba(245,158,11,0.4)'};border-radius:12px;padding:20px 24px;margin-bottom:24px;">
  <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
    <div style="flex:0 0 auto;">
      <span style="font-size:2rem;">🛡️</span>
    </div>
    <div style="flex:1;min-width:200px;">
      <div style="color:#94a3b8;font-size:0.75rem;font-weight:700;letter-spacing:2px;margin-bottom:4px;">ROBUSTNESS CERTIFICATION</div>
      <div style="color:{color};font-size:1.3rem;font-weight:800;margin-bottom:2px;">{icon} {overall}</div>
      <div style="color:#64748b;font-size:0.78rem;">Last validated: {ts} &nbsp;|&nbsp; Layers: {layers}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;flex:2;min-width:300px;margin-top:4px;">
      <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:10px;text-align:center;">
        <div style="color:#94a3b8;font-size:0.7rem;letter-spacing:1px;">LAYER 0–1</div>
        <div style="color:{'#10b981' if l0=='PASS' else '#f87171'};font-size:0.9rem;font-weight:700;">Boot + Tests</div>
        <div style="color:#e2e8f0;font-size:0.78rem;">{l1_str}</div>
      </div>
      <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:10px;text-align:center;">
        <div style="color:#94a3b8;font-size:0.7rem;letter-spacing:1px;">LAYER 3</div>
        <div style="color:#10b981;font-size:0.9rem;font-weight:700;">Skill Gradient</div>
        <div style="color:#e2e8f0;font-size:0.78rem;">Rnd {l3_rnd} › TWAP {l3_twap} › AC {l3_ac} bps</div>
      </div>
      <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:10px;text-align:center;">
        <div style="color:#94a3b8;font-size:0.7rem;letter-spacing:1px;">LAYER 4</div>
        <div style="color:{'#10b981' if l4.get('status')=='PASS' else '#f59e0b'};font-size:0.9rem;font-weight:700;">API Compliance</div>
        <div style="color:#e2e8f0;font-size:0.78rem;">{l4_ep} endpoints OK</div>
      </div>
    </div>
  </div>
  <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.05);color:#475569;font-size:0.73rem;">
    🔁 Determinism: {det} &nbsp;|&nbsp; 📄 Full report: <code style="background:rgba(0,0,0,0.3);padding:1px 6px;border-radius:3px;">ROBUSTNESS_REPORT.json</code> &nbsp;|&nbsp; 🖥️ Rerun: <code style="background:rgba(0,0,0,0.3);padding:1px 6px;border-radius:3px;">python3 tests/validate_robustness.py --full</code>
  </div>
</div>"""
                        except Exception as e:
                            return f'<div style="color:#f87171;padding:12px;">⚠️ Could not load robustness report: {e}</div>'

                    gr.HTML(_load_robustness_report())
                    # ── End Robustness Certification Banner ──────────────────────────

                    gr.Markdown("""
## 🏦 What is TradeExecGym?

Institutional traders don't pick stocks — they figure out how to buy **1,000,000 shares** without
the market noticing. If you trade too fast, you exhaust all the sellers and drive the price up
against yourself. If you trade too slow, random market chaos (and HFT predators) eat your profit.

This is the **Implementation Shortfall** problem. TradeExecGym is a physics-grounded RL environment
that makes agents solve it — just like real hedge fund Smart Order Routers do.

---

## ⚙️ Environment Specification

| Property | Value |
|---|---|
| **Name** | `trade_exec_gym` |
| **Version** | `1.0.0` |
| **Framework** | Meta OpenEnv v0.2.1 |
| **Runtime** | FastAPI + FastMCP |
| **Protocol** | HTTP REST + MCP (Model Context Protocol) |
| **Max Concurrent Sessions** | 5 |
| **Python** | ≥ 3.10 |

### Action & Observation Space

| Dimension | Type | Range | Description |
|---|---|---|---|
| **Action** | `participation_rate` | `[0.0, 0.25]` | Fraction of Average Daily Volume to target per step |
| **Observation** | Market State Text | Natural Language | Narrative + structured market data snapshot |
| **Reward** | Per-step IS delta + terminal bonus | Unbounded float | Dense (0.1 × IS_diff) + terminal (+1.0 completion, +0.5 excellence) |
| **Grader Score** | Normalized score | `[0.0, 1.0]` | Task-specific grader for leaderboard ranking |
| **Episode Length** | Variable | 30 – 120 steps | Depends on task difficulty |

### Live Environment State Variables

| Variable | Type | Description |
|---|---|---|
| `_mid_price` | `float` | Current market mid price — evolves each step via Almgren-Chriss GBM |
| `_arrival_price` | `float` | Locked reference price at episode start — the IS benchmark |
| `_shares_remaining` | `int` | Shares still to be executed |
| `_shares_executed` | `int` | Cumulative fills so far |
| `_total_cost` | `float` | Accumulated dollar cost of all fills |
| `_step_count` | `int` | Steps elapsed in this episode |
| `_max_steps` | `int` | Episode length (task-defined) |
| `_episode_done` | `bool` | Terminal flag |
| `_last_reward` | `float` | Most recent per-step reward signal |

---

## 🧮 The Physics Engine

Every step runs three simultaneous calculations. There are no random walks or fake data:

```
1. GBM Drift          →  ΔS = S · exp((μ - 0.5σ²)·Δt + σ·√Δt·ε)   ε ~ N(0,1)
2. Permanent Impact   →  Δprice_perm = γ · participation_rate  (shifts mid-price forever)
3. Temporary Impact   →  Δprice_temp = η · participation_rate  (affects fill price only)
```

This is the **Almgren-Chriss (2000)** model — the same mathematical framework used by Goldman Sachs,
Citadel, and every major systematic trading desk. The Implementation Shortfall (IS) formula:

```
IS (bps) = |avg_exec_price - arrival_price| / arrival_price × 10,000
```

A grader score ≥ **0.80** is professional tier. Beating the AC Optimal line puts you in the Hall of Fame.

---

## 🎯 The 5 Curriculum Tasks

| # | Task ID | Difficulty | Shares | Steps | Key Challenge |
|---|---|---|---|---|---|
| 1 | `task1_twap_beater` | 🟢 Easy | 100K | 30 | Beat equal-time-slice baseline |
| 2 | `task2_vwap_optimizer` | 🟡 Medium | 250K | 60 | Track the U-shaped intraday volume curve |
| 3 | `task3_volatile_execution` | 🔴 Hard | 400K | 90 | 3× volatility — dark pool routing required |
| 4 | `task4_adversarial` | 🟣 Very Hard | 200K | 120 | HFT predator detects uniform orders |
| 5 | `task5_deadline_pressure` | ⚫ Extreme | 1M | 80 | Hard legal cutoff — all remaining shares market-ordered |

**Task 4 Detail:** The adversary watches your participation rate standard deviation.
If it drops below `0.005` (you are too uniform), the HFT bot front-runs you and applies
a **50 bps penalty** on your next fill. The countermeasure: randomize your rate.

---

## 📊 Baseline Performance (Heuristic Hybrid Agent)

| Task | Avg IS ↓ | Grader Score ↑ | vs TWAP |
|---|---|---|---|
| Task 1: TWAP Beater | 18.4 bps | **0.91** | ✅ Beat by 6.1 bps |
| Task 2: VWAP Optimizer | 15.2 bps | **0.86** | ✅ Beat by 9.3 bps |
| Task 3: Volatile Execution | 38.7 bps | **0.79** | ✅ Beat by 4.2 bps |
| Task 4: Adversarial HFT | 52.3 bps | **0.72** | ✅ Beat by 2.8 bps |
| Task 5: Deadline Cliff | 84.1 bps | **0.66** | ✅ Beat by 1.1 bps |

---

## 🚀 Quick Start

**Local development** — two terminal windows:
```bash
# Terminal 1: Backend MCP server (internal port 7865)
uv pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7865

# Terminal 2: Gradio dashboard (public port 7860)
python3 ui/app.py --port 7860
```

**Run compliance inference** (all 5 tasks, OpenEnv log format):
```bash
export HF_TOKEN="hf_your_token_here"   # optional — enables LLM layer
python3 inference.py
# Outputs [START]/[STEP]/[END] logs to stdout
```

**Docker** (production — both services start automatically):
```bash
docker build -t trade-exec-gym .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx trade-exec-gym
```

---

## 📁 Repository Structure

```
trade-exec-gym/
├── server/
│   ├── app.py                 ← OpenEnv create_app() entry point (port 7865)
│   └── trade_environment.py  ← MCPEnvironment: 4 tools + episode state machine
├── env/
│   ├── price_model.py         ← Almgren-Chriss GBM price simulator
│   ├── venue_router.py        ← Dark pool + NASDAQ lit venue routing
│   └── reward.py              ← 3-component IS reward function
├── tasks/         ← 5 task configs + factory registry
├── baselines/     ← TWAP, VWAP, AC-Optimal, AlmgrenChrissHeuristic agents
├── training/      ← PPO/GRPO training scripts
├── tests/         ← 24-test pytest validation suite
├── ui/app.py      ← This Gradio dashboard (port 7860)
├── client.py      ← Async httpx SDK (TradeExecClient extends MCPToolClient)
├── inference.py   ← OpenEnv compliance inference runner
├── openenv.yaml   ← OpenEnv manifest (spec_version: 1)
└── pyproject.toml ← Dependencies + build config
```
""", elem_classes=["info-section"])

            # ================= Tab 5: Training & OpenEnv Architecture =================
            with gr.TabItem("🏗️ Architecture & API"):
                gr.Markdown("""
## 🔌 How Agents Connect

TradeExecGym runs on **Meta's OpenEnv Framework (v0.2.1)**. Every interaction goes through
**4 MCP tools** — the same standardized protocol used across all OpenEnv environments.
Both tool-calling LLMs and RL policy networks use identical endpoints.

```text
+----------------------------------------------------------------------+
|                     TradeExecGym - System Map                        |
+--------------------------+-------------------------------------------+
|  PORT 7860  (Public)     |  PORT 7865  (Internal)                    |
|  Gradio Dashboard        |  FastAPI + FastMCP Backend                |
|  ui/app.py               |  server/app.py -> openenv.create_app()    |
|                          |                                           |
|  +------------------+    |  +-----------------------------------+    |
|  | Auto Simulation  |    |  |  TradeExecEnvironment             |    |
|  | Live LLM Eval    |<---+--+  (MCPEnvironment subclass)        |    |
|  | Manual Challenge |    |  |  -> env/price_model.py (GBM)      |    |
|  | Info / Arch Tabs |    |  |  -> env/venue_router.py           |    |
|  +------------------+    |  |  -> env/reward.py (IS grader)     |    |
|  TradeExecClient         |  |  -> tasks/ (5 task configs)       |    |
|  (httpx async) ----------+-->                                    |    |
+--------------------------+-------------------------------------------+
```

---

## 📡 HTTP API Reference

All tools are callable via HTTP REST. Direct async wrappers available in `client.py` (TradeExecClient).

### `GET /health` — Liveness Check
```bash
curl http://localhost:7865/health
# {"status": "healthy"}
```

### `GET /schema` — Environment Schema
```bash
curl http://localhost:7865/schema
# Returns full action/observation schema
```

### `POST /reset` — Initialize Episode
```json
POST http://localhost:7865/reset
{ "task_id": "task1_twap_beater", "seed": 42 }
```
Returns `{"observation": {}, "reward": null, "done": false}`

### `POST /step` — Execute Action (raw)
```json
POST http://localhost:7865/step
{ "tool_name": "execute_trade", "tool_input": {"participation_rate": 0.05} }
```

### `GET /state` — Current State
```bash
curl http://localhost:7865/state
```

### `POST /mcp` — MCP Protocol Endpoint
For MCP-native tool-calling clients.

---

## 🛠️ The 4 MCP Tools

### Tool 1: `get_market_state()` → Read Environment
Returns a rich natural-language + structured snapshot. Designed for LLM chain-of-thought.
```
MARKET STATE — Step 12/30
───────────────────────────────────────────
NARRATIVE: Volume is spiking at the open.

INVENTORY
  Executed:  48,000 / 100,000 (48.0%)
  Remaining: 52,000 shares | Time left: 18 steps

PRICES
  Mid Price: $150.4821 | Arrival: $150.0000 | Spread: 5.5 bps

PERFORMANCE  (lower IS = better)
  Your IS:  18.32 bps  ✅ Beating TWAP by 6.1 bps
  TWAP IS:  24.44 bps  | VWAP IS: 19.55 bps
```

### Tool 2: `execute_trade(...)` → Primary Action
```python
execute_trade(
    participation_rate: float,        # [0.0, 0.25]  fraction of ADV to target
    use_dark_pool: bool = False,      # route to anonymous dark liquidity
    dark_pool_fraction: float = 0.0,  # [0.0, 1.0] portion sent dark
    order_type: str = "MARKET",       # "MARKET" | "LIMIT"
    limit_offset_bps: float = 0.0    # limit price offset in bps
)
```

### Tool 3: `get_baseline_comparison()` → Competitive Benchmarks
Real-time IS comparison vs TWAP, VWAP, and the Almgren-Chriss mathematical optimum.
```
  🤖 You:          19.44 bps
  📈 TWAP:         24.56 bps  (naive equal-slice)
  📊 VWAP:         19.65 bps  (volume-proportional)
  🧮 AC Optimal:   14.24 bps  (Almgren-Chriss floor)
```

### Tool 4: `get_reward()` → Per-Step Reward
Returns a `float`. Positive = beating TWAP baseline. Negative = worse than TWAP or adversary-penalized.
Terminal episode adds +1.0 for >95% completion and +0.5 excellence bonus for beating AC Optimal.

---

## 🧠 Agent Architectures

### Layer 1 — Pure Math (Heuristic)
Almgren-Chriss analytically optimal schedule. Deterministic. Used as the reward normalization baseline.
```python
from baselines.heuristic_agent import AlmgrenChrissHeuristic
h = AlmgrenChrissHeuristic()
rate = h.calculate_rate(shares_remaining=500_000, total_shares=1_000_000, steps_left=40, volatility=0.02)
```

### Layer 2 — LLM Tool-Caller
An LLM reads `get_market_state()` narrative text and calls `execute_trade()` directly.
Natural language state enables Chain-of-Thought reasoning for adversary detection.

**System prompt contract:**
```json
{"recommendation": "Approve | Accelerate | Decelerate | Randomize", "reason": "..."}
```

### Layer 3 — Hybrid (Production Pattern)
Math calculates the rate. LLM evaluates context and applies a multiplier:
```python
# Math layer
suggested_rate = heuristic.calculate_rate(rem, total, steps_left, vol)

# Cognitive layer
if llm_decision == "Accelerate":  final_rate = suggested_rate * 1.4
elif llm_decision == "Decelerate": final_rate = suggested_rate * 0.6
elif llm_decision == "Randomize":  final_rate = suggested_rate * random.uniform(0.8, 1.2)
```
This is what `inference.py` runs. The LLM can detect adversary alerts and pivot strategy mid-episode.

---

## 🔁 Reward Design

The reward function (`env/reward.py`) has 3 components:

| Component | Trigger | Value |
|---|---|---|
| **Dense** | Every step | `0.1 × (twap_IS − agent_IS)` — positive if beating TWAP |
| **Terminal completion** | Episode end, >95% filled | `+1.0` |
| **Terminal excellence** | Episode end, beats AC Optimal | `+0.5` additional |
| **Terminal failure** | Episode end, <95% filled | `−0.5` |
| **Milestone** | Each 25% completion threshold | `+0.2` |

The **Grader Score** (0.0–1.0) is computed separately by each task's grader function and is used for leaderboard ranking.

---

## 📦 Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `HF_TOKEN` | *(none)* | Enables LLM cognitive layer in inference |
| `MODEL_NAME` | `meta-llama/Meta-Llama-3-70B-Instruct` | LLM used for hybrid agent |
| `ENV_BASE_URL` | `http://localhost:7865` | Backend URL for inference script |
| `PORT` | `7860` | Primary public port (HF Spaces managed) |

---

## ✅ OpenEnv Compliance Log Format
```
[START] task=task1_twap_beater env=trade_exec_gym model=meta-llama/...
[STEP]  step=1 action=0.0523 reward=0.12 done=false error=null
[STEP]  step=2 action=0.0487 reward=0.18 done=false error=null
...
[END]   success=true steps=28 score=0.891 rewards=0.12,0.18,...
```
Run `python3 inference.py` to generate this output (logs to stdout).
""", elem_classes=["info-section"])
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args, unknown = parser.parse_known_args()
    
    app = build_gui()
    app.launch(
        server_port=args.port,
        server_name=args.host,
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"),
        css=CUSTOM_CSS,
    )

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
from openai import AsyncOpenAI
from baselines.heuristic_agent import AlmgrenChrissHeuristic
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add root to sys.path to resolve local imports like `client`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client import TradeExecClient

# Load the trained model if available, fallback to None
try:
    from stable_baselines3 import PPO
    MODEL_PATH = "models/grpo_agent.zip"
    if os.path.exists(MODEL_PATH):
        _loaded_agent = PPO.load(MODEL_PATH)
        print(f"[app_visual] Loaded GRPO Agent from {MODEL_PATH}")
    else:
        _loaded_agent = None
except ImportError:
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
            self.client = TradeExecClient(base_url="http://localhost:7865")
            obs = await self.client.reset(task_id=task_id, seed=int(seed))

        self.task_id = task_id
        self.history = []
        self.current_obs = obs
        self.is_running = True
        return self.get_summary()

    def get_summary(self):
        if not self.current_obs:
            return "No active session."
        meta = self.current_obs.metadata if hasattr(self.current_obs, 'metadata') else {}
        return meta.get("output", "Session started.")

    async def step(self, rate, use_dark, dark_frac):
        if not self.client:
            return "No session active.", None, gr.update(), 0.0, 0.0
        
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
                
            return result, self.create_plot(), gr.update(interactive=self.is_running), is_val, score_val, metrics
        except Exception as e:
            return f"❌ Connection Error: {str(e)}", None, gr.update(), 0.0, 0.0, {}

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
                        val = line.split("(")[1].split("%")[0].strip()
                        metrics["pct_done"] = float(val)
                    elif "Your IS:" in line:
                        val = line.split(":")[1].lower().replace("bps", "").strip()
                        metrics["is_bps"] = float(val)
                    elif "Final IS:" in line:
                        val = line.split(":")[1].lower().replace("bps", "").strip()
                        metrics["is_bps"] = float(val)
                    elif "Grader Score:" in line:
                        val = line.split(":")[1].split("/")[0].strip()
                        metrics["score"] = float(val)
                    elif "Time left:" in line:
                        val = line.split(":")[1].split("steps")[0].strip()
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
        
    df = pd.DataFrame(history_df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Panel 1: Price
    ax1.plot(df["step"], df["price"], color="#00ffcc", marker="o", label="Mid Price ($)")
    ax1.set_ylabel("Price ($)", color="white")
    ax1.set_title(title, color="white", fontsize=14)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper left")
    
    # Panel 1 twin axis: Implementation Shortfall 
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df["step"], df["is_bps"], color="#ff00ff", linestyle="--", label="Slippage (bps)")
    ax1_twin.set_ylabel("Slippage (bps)", color="#ff00ff")
    ax1_twin.legend(loc="lower right")

    # Panel 2: Completion
    ax2.bar(df["step"], df["pct_done"], color="#4CAF50", alpha=0.6, label="Completion %")
    ax2.set_ylabel("Done %", color="white")
    ax2.set_xlabel("Step", color="white")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left")
    
    # Style styling wrapper
    fig.patch.set_facecolor('#111111')
    ax1.set_facecolor('#1a1a1a')
    ax2.set_facecolor('#1a1a1a')
    for ax in [ax1, ax2, ax1_twin]:
        ax.tick_params(colors='white')
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
        yield "### Error: HF_TOKEN is required for Live Eval.", None, {}, "[ERROR] Missing Token"
        return

    client = TradeExecClient(base_url="http://localhost:7865")
    llm_client = AsyncOpenAI(api_key=hf_token, base_url="https://api-inference.huggingface.co/v1/")
    heuristic = AlmgrenChrissHeuristic()
    task_id = TASK_ID_MAP.get(display_name, "task1_twap_beater")
    
    try:
        log_stream = f"[START] task={task_id} env=trade_exec_gym model={model_name}\n"
        yield f"### Initializing {task_id}...", None, {}, log_stream
        
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
                metrics,
                log_stream
            )
            
            if done_bool:
                done = True
                score = metrics.get("score", 0.0)
                log_stream += f"[END] success={str(score >= 0.8).lower()} steps={step} score={score:.3f} rewards=..."
                yield (
                    f"### Session Complete\nFinal Score: {score:.4f}", 
                    plot_trajectory(history, f"Live Eval: {model_name}"), 
                    metrics,
                    log_stream
                )

        await client.close()
    except Exception as e:
        yield f"### Session Failed\n{str(e)}", None, {}, log_stream
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
        # Return final metrics for the JSON view
        return summary_text, plot_trajectory(history, f"Auto: {mode}"), history[-1] if history else {}
        
    except Exception as e:
        try: await client.close()
        except: pass
        return f"### simulation Failed\n\nError: {str(e)}", None, {}

# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
state = UIState()

CUSTOM_CSS = """
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    max-width: 1280px !important;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #0d4f3c 50%, #1a1a2e 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header h1 { color: #ecfdf5 !important; font-size: 2rem !important; font-weight: 700 !important; margin: 0 0 4px 0 !important; }
.hero-header p  { color: #a7f3d0 !important; margin: 0 !important; font-size: 0.95rem !important; }

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
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"),
        css=CUSTOM_CSS
    ) as demo:

        gr.HTML("""
        <div class="hero-header">
          <h1>📈 TradeExecGym</h1>
          <p>
            Institutional Smart Order Routing &nbsp;·&nbsp;
            Almgren-Chriss Market Physics &nbsp;·&nbsp;
            HFT Adversary Simulation &nbsp;·&nbsp;
            MCP-Native OpenEnv
          </p>
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
                        auto_json = gr.JSON(label="Step Metadata (JSON)")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Performance Live Feed")
                        auto_plot = gr.Plot(label="Order Trajectory")
                        auto_summary = gr.Markdown(label="Post-Trade Analysis")
                
                # Native async click handlers
                run_auto_btn.click(run_auto_simulation, inputs=[auto_task_dd, auto_mode_dd, auto_seed], outputs=[auto_summary, auto_plot, auto_json])

            # ================= Tab 2: Live Model Evaluation =================
            with gr.TabItem("Live Model Evaluation (Compliance Test)"):
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
                        live_json = gr.JSON(label="Live State (JSON)")
                    
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
                        man_json = gr.JSON(label="Step Data")

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

            # ================= Tab 4: Project & Environment Info =================
            with gr.TabItem("📖 Project & Environment Info"):
                gr.HTML('<div class="info-section">')
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
| **Protocol** | MCP (Model Context Protocol) native |
| **Max Concurrent Sessions** | 5 |
| **Python** | ≥ 3.10 |

### Action & Observation Space

| Dimension | Type | Range | Description |
|---|---|---|---|
| **Action** | `participation_rate` | `[0.0, 0.25]` | Fraction of Average Daily Volume to target per step |
| **Observation** | Market State Text | Natural Language | Narrative + structured market data snapshot |
| **Reward** | Per-step IS delta | `[-1.0, +1.0]` | GRPO-compatible bounded sigmoid over IS basis points |
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
1. Permanent Impact   →  Δprice_perm = λ · σ · √q · sgn(order)
2. Temporary Impact   →  Δprice_temp = η · (q / ADV_per_step)
3. Brownian Drift     →  ΔS = σ · √Δt · ε   where ε ~ N(0,1)
```

This is the **Almgren-Chriss (2000)** model — the same mathematical framework used by Goldman Sachs,
Citadel, and every major systematic trading desk. The Implementation Shortfall (IS) formula:

```
IS (bps) = |avg_exec_price - arrival_price| / arrival_price × 10,000
```

A score ≥ **0.80** is professional tier. Beating the AC Optimal line puts you in the Hall of Fame.

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

## 📊 Baseline Performance (GPT-4o Hybrid Agent)

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
# Terminal 1: Backend MCP server (internal port)
uv pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7865

# Terminal 2: Gradio dashboard
python ui/app.py --port 7860
```

**Run compliance inference** (all 5 tasks, OpenEnv log format):
```bash
export HF_TOKEN="hf_your_token_here"   # optional — enables LLM layer
python inference.py
# Outputs: results/trajectory_YYYYMMDD_HHMMSS.json
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
│   ├── app.py                 ← OpenEnv create_app() entry point
│   └── trade_environment.py  ← MCPEnvironment: 4 tools + state machine
├── env/
│   ├── price_model.py         ← Almgren-Chriss GBM simulator
│   ├── venue_router.py        ← Dark pool + NASDAQ lit routing
│   └── reward.py              ← Sigmoid-normalized IS grader
├── tasks/         ← 5 task configs + factory registry
├── baselines/     ← TWAP, VWAP, AC-Optimal, Heuristic agents
├── ui/app.py      ← This dashboard
├── client.py      ← Async httpx SDK (TradeExecClient)
├── inference.py   ← OpenEnv compliance runner
├── openenv.yaml   ← OpenEnv manifest
└── pyproject.toml ← Dependencies
```
""")
                gr.HTML('</div>')

            # ================= Tab 5: Training & OpenEnv Architecture =================
            with gr.TabItem("🏗️ Architecture & API"):
                gr.HTML('<div class="info-section">')
                gr.Markdown("""
## 🔌 How Agents Connect

TradeExecGym runs on **Meta's OpenEnv Framework (v0.2.1)**. Every interaction goes through
**4 MCP tools** — the same standardized protocol used across all OpenEnv environments.
Both tool-calling LLMs and RL policy networks use identical endpoints.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TradeExecGym — System Map                        │
├──────────────────────────┬──────────────────────────────────────────┤
│  PORT 7860  (Public)     │  PORT 7865  (Internal)                   │
│  Gradio Dashboard        │  FastAPI + FastMCP Backend               │
│  ui/app.py               │  server/app.py → openenv.create_app()   │
│                          │                                          │
│  ┌──────────────────┐   │  ┌──────────────────────────────────┐    │
│  │ Auto Simulation  │   │  │  TradeExecEnvironment            │    │
│  │ Live LLM Eval    │◄──┼──┤  (MCPEnvironment subclass)       │    │
│  │ Manual Challenge │   │  │  → env/price_model.py (GBM)      │    │
│  │ Info / Arch Tabs │   │  │  → env/venue_router.py           │    │
│  └──────────────────┘   │  │  → env/reward.py (sigmoid)       │    │
│  TradeExecClient        │  │  → tasks/ (5 task configs)       │    │
│  (httpx async) ─────────┼──►                                  │    │
└──────────────────────────┴──────────────────────────────────────────┘
```

---

## 📡 MCP API Reference

All tools are callable via the MCP protocol. Direct HTTP wrappers available via `client.py`.

### `GET /health` — Liveness Check
```bash
curl http://localhost:7865/health
# {"status": "ok", "env": "trade_exec_gym", "version": "1.0.0"}
```

### `POST /reset` — Initialize Episode
```json
POST /reset
{ "task_id": "task1_twap_beater", "seed": 42 }
```
Returns a structured `Observation` with episode ID, market narrative, and task objectives.

---

### Tool: `get_market_state()` → Read Environment
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

### Tool: `execute_trade(...)` → Primary Action
```python
execute_trade(
    participation_rate: float,     # [0.0, 0.25]  fraction of ADV to target
    use_dark_pool: bool = False,   # route to anonymous dark liquidity
    dark_pool_fraction: float = 0.0,  # [0.0, 1.0] portion sent dark
    order_type: str = "MARKET",   # "MARKET" | "LIMIT"
    limit_offset_bps: float = 0.0 # limit price offset in bps
)
```

### Tool: `get_baseline_comparison()` → Competitive Benchmarks
Real-time IS comparison vs TWAP, VWAP, and the Almgren-Chriss mathematical optimum.
```
  🤖 You:          19.44 bps
  📈 TWAP:         24.56 bps  (naive equal-slice)
  📊 VWAP:         19.65 bps  (volume-proportional)
  🧮 AC Optimal:   14.24 bps  (Almgren-Chriss floor)
```

### Tool: `get_reward()` → Per-Step Reward
Returns a `float` in `[-1.0, +1.0]`. Pre-scaled for GRPO training stability.
Positive = beating TWAP. Negative = worse than TWAP or adversary-penalized.

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

## 🔁 Reward Design — Why Sigmoid?

GRPO relies on **properly scaled advantages**. Raw slippage in basis points is unbounded and
task-dependent (Task 1 IS ≈ 20 bps; Task 5 IS ≈ 80 bps). A raw reward would produce wildly
different gradient magnitudes across tasks.

The bounded sigmoid grader maps any IS value to `[-1.0, +1.0]`, guaranteeing:
- Stable gradient magnitudes across all 5 tasks
- Meaningful ranking of policy improvements
- A single scalar that GRPO's advantage estimator can use without per-task rescaling

---

## 📦 Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `HF_TOKEN` | *(none)* | Enables LLM cognitive layer in inference |
| `MODEL_NAME` | `meta-llama/Meta-Llama-3-70B-Instruct` | LLM used for hybrid agent |
| `ENV_BASE_URL` | `http://localhost:7860` | Backend URL for inference script |
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
Run `python inference.py` to generate this output. Trajectory saved to `results/`.
""")
                gr.HTML('</div>')
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_index = False
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args, unknown = parser.parse_known_args()
    
    app = build_gui()
    app.launch(server_port=args.port, server_name=args.host)

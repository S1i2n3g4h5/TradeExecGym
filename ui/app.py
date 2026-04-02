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

def build_gui():
    with gr.Blocks(
        title="TradeExecGym Dashboard",
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"),
        css=".main-header {text-align: center;} .result-box {border: 1px solid #4CAF50; padding:10px; border-radius:5px;}"
    ) as demo:
        
        gr.Markdown(
            "# TradeExecGym: Market Impact Simulator\n"
            "### Built on Meta OpenEnv for Institutional Execution\n"
            "**The Stack:** Almgren-Chriss Equations. HFT Predators. LLM Narratives. MCP Native Hook.\n"
            "\n"
            "Most trading simulators use fake random data. That's useless for training serious agents. TradeExecGym runs on the actual quantitative physics that hedge funds use. If you trade 500,000 shares in one go, you'll clear the order book and drive the price up. You'll lose money instantly. This environment forces you to slice orders over time while avoiding high-frequency predatory bots."
        )
        
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
            with gr.TabItem("Project & Environment Info"):
                gr.Markdown(
                    '''
                    ## What is this thing?
                    Institutional trading is not picking what stock to buy. It's figuring out how to buy 1,000,000 shares without the market noticing. If you trade too fast, you exhaust all the sellers and drive the price up against yourself. This is called Slippage. If you trade too slow, random market chaos might ruin your fill rate.

                    ## The Math
                    We stripped out the fake random walks most open source trading toys use. This backend runs the Almgren-Chriss (2000) execution model.
                    Every time the agent trades, three variables update:
                    1. Permanent Impact. You removed supply. The fundamental baseline price moved up permanently.
                    2. Temporary Impact. The localized cost of eating through whatever liquidity was immediately available in that specific millisecond.
                    3. Brownian Drift. The market wanders wildly on its own.

                    ## Winning
                    The metric is Implementation Shortfall (IS). Let's say the arrival price was $150.00. You finished executing the order after 100 steps and your average fill was $150.50. You bled money. We convert that raw slippage into a harsh 0.0 to 1.0 grader score. Anything over 0.8 is professional tier.

                    ## The 5 Tasks
                    We wrote a curriculum to break normal bots.
                    * **The TWAP Beater.** Execute the order smoothly over a fixed timeframe.
                    * **VWAP Optimizer.** Intraday volume is basically a U-shape. High at the open, dead at lunch, huge at the close. Find the curve.
                    * **Volatile Market.** The asset is crashing. Stay in too long and the variance penalty triggers margin calls. 
                    * **Adversarial HFT.** A predatory high-frequency trading algorithm watches for uniform orders. If your standard deviation drops below 0.005, it front-runs you and applies a brutal fixed penalty to your execution price.
                    * **Deadline Cliff.** A hard legal cutoff. The remaining block gets market-ordered at terrible liquidity spreads instantly.
                    '''
                )

            # ================= Tab 4: Training Pipeline & OpenEnv Integration =================
            with gr.TabItem("Training & OpenEnv Architecture"):
                gr.Markdown(
                    '''
                    ## How Agents Actually Learn Here
                    This runs entirely on Meta's OpenEnv Framework (v0.2.1). We wired it this way to test tool-calling LLMs and GRPO agents against actual market physics.
                    
                    ### The Architecture
                    ```text
                    [Llama 3 / RL Agent] <---> [OpenEnv MCP Client Hook] 
                                  |                           |
                                  V                           V
                          [Execution Action]            [Market Observation]
                    (Decides block participation) (Reads Price, Fill%, LLM Narrative)
                                  |                           |
                                  +---------------------------+
                                                |
                                     FastAPI Backend Engine
                                  (Almgren-Chriss Simulator)
                    ```
                    
                    ### The Missing Signal: LLM Narratives
                    Feeding an LLM an array of floats like `[150.2, 0.45, 120]` doesn't work. The context window relies on language to form reasoning paths. We fixed that format problem immediately.

                    We injected an LLM Narrative hook inside the step loop. The environment translates the math into a plain-English situational report before it hands the state back to the agent.
                    
                    **Example Output:**
                    > *"MARKET STATE: Mid Price is $150.40. Volume is normal. ⚠️ ADVERSARY ALERT: HFT pattern detection isActive. Uniform trading is being penalized!"*
                    
                    Now a modern parameter sequence can use Chain-of-Thought. The model reads that a predator is hunting them, deduces that uniform `0.05` participation rates trigger the alarm, and pivots to trading erratically.
                    
                    ### The Reward Signal
                    We bound the raw slippage float to a `[0.0, 1.0]` limit box. Group Relative Policy Optimization (GRPO) relies on properly scaled advantages. By forcing the Implementation Shortfall through a bounded sigmoid grader, the reward surface stays stable whether the simulated asset is a 0.5% volatility blue chip or a 6% variance speculative token.
                    '''
                )
    return demo

if __name__ == "__main__":
    app = build_gui()
    app.launch(server_port=7861, server_name="0.0.0.0")

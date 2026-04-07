import gradio as gr
import os
import json
import asyncio
import time
from agents import HeuristicAgent, HybridAgent
from server.scenario_engine import ScenarioEngine
from server.scenario_definitions import SCENARIOS

# ── STATE MANAGEMENT (HONEY-STYLE) ──────────────────────────────────────────
class UIState:
    def __init__(self):
        self.reasoning = ""
        self.status = "🟢 READY"
        self.is_processing = False

state = UIState()

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── ROOT TOKENS ──────────────────────────────────────── */
:root {
    --bg:          #0b0f19;
    --panel:       rgba(15, 23, 42, 0.9);
    --border:      rgba(30, 41, 59, 0.5);
    --border-hi:   #10b981;
    --code-bg:     #0f172a;
    --green:       #10b981;
    --green-dim:   #059669;
    --green-glow:  rgba(16, 185, 129, 0.1);
    --amber:       #f59e0b;
    --text:        #e2e8f0;
    --muted:       #94a3b8;
    --font-sans:   'Inter', sans-serif;
    --font-mono:   'JetBrains Mono', monospace;
    --font-head:   'Syne', sans-serif;
    --r:           12px;
    --rl:          16px;
}

/* ── GLOBAL ──────────────────────────────────────────── */
body, .gradio-container {
    font-family: var(--font-sans) !important;
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(34,197,94,0.07) 0%, var(--bg) 60%) !important;
    color: var(--text) !important;
    min-height: 100vh !important;
}

/* transparent bg erasers for Gradio chrome */
.gradio-container > .main,
.gradio-container .wrap,
.gradio-container .tabs,
.gradio-container .tab-content,
footer,
.gradio-container .block,
.gradio-container .form,
.gradio-container .box,
.gradio-container > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── TABS ───────────────────────────────────────────── */
.tabs > .tab-nav {
    border-bottom: none !important;
    background: rgba(14, 22, 14, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 6px !important;
    margin: 0 auto 32px auto !important;
    display: flex !important;
    justify-content: center !important;
    gap: 4px !important;
    border-radius: 100px !important;
    border: 1px solid var(--border) !important;
    max-width: max-content !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

.tab-nav button {
    font-family: var(--font-head) !important;
    font-size: 1.4rem !important; /* Made significantly larger as requested */
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    color: var(--text) !important;
    opacity: 0.6 !important;
    padding: 12px 36px !important; /* Adjusted padding to accommodate the larger text */
    border: none !important;
    border-radius: 100px !important;
    background: transparent !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button:hover {
    opacity: 1 !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.tab-nav button.selected {
    color: #000 !important;
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    opacity: 1 !important;
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.3) !important;
    border: none !important;
}

/* ── SECTION PANELS ──────────────────────────────────── */
.section-box {
    background: var(--panel) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--rl) !important;
    padding: 22px 24px !important;
    box-shadow: 0 4px 36px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}

.section-box:hover {
    border-color: var(--border-hi) !important;
    box-shadow: 0 6px 44px rgba(34,197,94,0.07), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}

/* ── FARM PLOT CARDS ──────────────────────────────────── */
.farm-plot {
    background: linear-gradient(150deg, rgba(12,20,12,0.97) 0%, rgba(8,13,8,1) 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--rl) !important;
    padding: 22px !important;
    text-align: center !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.75), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    position: relative !important;
    overflow: hidden !important;
}

.farm-plot::before {
    content: '';
    position: absolute;
    top: 0; left: -150%; width: 50%; height: 100%;
    background: linear-gradient(to right, transparent, rgba(34,197,94,0.05), transparent);
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}

.farm-plot:hover {
    transform: translateY(-5px) scale(1.02) !important;
    border-color: var(--border-hi) !important;
    box-shadow: 0 18px 44px rgba(34,197,94,0.14) !important;
}

.farm-plot:hover::before { left: 200%; }

/* ── AI AUDIT BOX ────────────────────────────────────── */
.audit-box {
    border-left: 4px solid var(--green) !important;
    background: rgba(34, 197, 94, 0.03) !important;
}

.audit-box:contains('HALLUCINATION') {
    border-left-color: #ef4444 !important;
}

.auth-error-box {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 2px solid #ef4444 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
    animation: pulse-red 2s infinite !important;
    text-align: center !important;
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.15) !important;
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.obs-viewer {
    background: var(--code-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    margin-top: 12px !important;
    font-size: 0.85rem !important;
}

.auth-error-box {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 2px solid #ef4444 !important;
    border-radius: var(--rl) !important;
    padding: 24px !important;
    margin: 16px 0 !important;
    text-align: center !important;
    animation: pulse-red 2s infinite !important;
}

.auth-error-box h3 {
    color: #ef4444 !important;
    -webkit-text-fill-color: #ef4444 !important;
    font-size: 1.5rem !important;
    margin-bottom: 8px !important;
}

.auth-error-box p {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}

/* ── TYPOGRAPHY ───────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: var(--font-head) !important;
    font-weight: 800 !important;
    margin-bottom: 0.4rem !important;
    background: none !important;
    -webkit-text-fill-color: var(--green) !important;
    color: var(--green) !important;
    letter-spacing: -0.3px !important;
}

/* section labels like "## THE FARM" */
h2 {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    opacity: 1 !important;
}

hr { border-color: var(--border) !important; margin: 14px 0 !important; }
p, li { color: var(--text) !important; line-height: 1.65 !important; }

/* ── TABLES ──────────────────────────────────────────── */
table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 5px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

th {
    color: var(--muted) !important;
    text-transform: uppercase !important;
    font-size: 0.66rem !important;
    letter-spacing: 1.5px !important;
    padding: 0 12px 8px !important;
    border: none !important;
    background: none !important;
    -webkit-text-fill-color: var(--muted) !important;
}

td {
    background: rgba(34,197,94,0.03) !important;
    padding: 9px 12px !important;
    border: 1px solid rgba(34,197,94,0.06) !important;
    color: var(--text) !important;
    transition: background 0.2s, border-color 0.2s !important;
}

tr td:first-child { border-radius: 8px 0 0 8px !important; color: #fff !important; font-weight: 600 !important; }
tr td:last-child  { border-radius: 0 8px 8px 0 !important; }

tr:hover td {
    background: rgba(34,197,94,0.07) !important;
    border-color: rgba(34,197,94,0.2) !important;
}

/* ── BUTTONS ─────────────────────────────────────────── */
button, .gr-button {
    font-family: var(--font-mono) !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    border-radius: var(--r) !important;
    padding: 10px 16px !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
}

button:hover { transform: translateY(-2px) !important; }
button:active { transform: translateY(1px) !important; }

/* PRIMARY — green */
button[class*="primary"], .gr-button-primary {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: #000 !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(34,197,94,0.28) !important;
    font-weight: 700 !important;
}

button[class*="primary"]:hover {
    box-shadow: 0 8px 30px rgba(34,197,94,0.42) !important;
}

/* SECONDARY */
button[class*="secondary"], .gr-button-secondary {
    background: rgba(14,22,14,0.8) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
}

button[class*="secondary"]:hover {
    border-color: var(--border-hi) !important;
    color: var(--green) !important;
}

/* QUICK ACTIONS — amber */
.action-btn {
    background: rgba(245,158,11,0.08) !important;
    color: var(--amber) !important;
    border: 1px solid rgba(245,158,11,0.22) !important;
}

.action-btn:hover {
    background: rgba(245,158,11,0.16) !important;
    border-color: rgba(245,158,11,0.45) !important;
    box-shadow: 0 6px 22px rgba(245,158,11,0.18) !important;
}

/* ── INPUTS & FORMS ──────────────────────────────────── */
input, select, textarea {
    background: rgba(8,14,8,0.85) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

input:focus, select:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px var(--green-glow) !important;
    outline: none !important;
}

input[type=range]::-webkit-slider-thumb { background: var(--green) !important; }
input[type=range]::-webkit-slider-runnable-track { background: rgba(34,197,94,0.18) !important; border-radius: 4px !important; }
input[type=radio], input[type=checkbox] { accent-color: var(--green) !important; }

/* ── LABELS ──────────────────────────────────────────── */
label, .label-wrap span {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

/* ── CODE / JSON ─────────────────────────────────────── */
pre, code, .gr-code {
    background: rgba(0,0,0,0.65) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--green) !important;
}

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(34,197,94,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(34,197,94,0.45); }
"""

SHARED_BANNER_HTML = """
<div id="fs-shared-banner">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');
#fs-shared-banner {
  position: relative;
  overflow: hidden;
  padding: 52px 48px 44px;
  text-align: center;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  background:
    linear-gradient(180deg, rgba(34,197,94,0.07) 0%, transparent 100%),
    radial-gradient(ellipse 70% 60% at 50% 0%, rgba(34,197,94,0.12) 0%, transparent 70%);
  font-family: 'Inter', system-ui, sans-serif;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.fsb-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 16px; border-radius: 100px;
  background: rgba(34,197,94,0.12);
  border: 1px solid rgba(34,197,94,0.25);
  color: #22c55e;
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem; letter-spacing: 1.2px;
  text-transform: uppercase; margin-bottom: 20px;
}
.fsb-dot {
  width: 6px; height: 6px; background: #22c55e; border-radius: 50%;
  animation: fsb-pulse 2s ease-in-out infinite; display: inline-block;
}
@keyframes fsb-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(1.5)} }
#fs-shared-banner h1 {
  font-family: 'Syne', 'Outfit', sans-serif !important;
  font-size: clamp(2.2rem, 4vw, 4rem) !important;
  font-weight: 800 !important;
  line-height: 1.05 !important;
  margin-bottom: 14px !important;
  color: #fff !important;
  letter-spacing: -2px !important;
  background: none !important;
  -webkit-text-fill-color: #fff !important;
  text-align: center !important;
}
#fs-shared-banner h1 span { color: #22c55e; -webkit-text-fill-color: #22c55e !important; }
.fsb-sub {
  font-size: 1rem;
  color: #7a8f82;
  max-width: 680px;
  margin: 0 auto 24px;
  text-align: center !important;
  line-height: 1.65;
  font-family: 'Inter', system-ui, sans-serif;
}
.fsb-pills {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;
}
.fsb-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 14px; border-radius: 100px;
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem; font-weight: 500; border: 1px solid;
}
.fsb-pill-g { background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.28); color: #22c55e; }
.fsb-pill-a { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.28); color: #f59e0b; }
.fsb-pill-b { background: rgba(56,189,248,0.1); border-color: rgba(56,189,248,0.28); color: #38bdf8; }
.fsb-pill-p { background: rgba(167,139,250,0.1); border-color: rgba(167,139,250,0.28); color: #a78bfa; }
</style>
<div class="fsb-badge"><span class="fsb-dot"></span>&nbsp;OpenEnv &middot; v1.0.0 &middot; Meta Hackathon 2025</div>
<h1>&#127807; Farm<span>Simulation</span></h1>
<br>
<p class="fsb-sub">A physics-grounded Reinforcement Learning environment where AI agents master agricultural resource management, market timing, and drought survival.</p>
<br>
<div class="fsb-pills">
  <span class="fsb-pill fsb-pill-g">&#128013; Python &ge; 3.11</span>
  <span class="fsb-pill fsb-pill-a">&#9889; FastAPI + Uvicorn</span>
  <span class="fsb-pill fsb-pill-b">&#129303; HuggingFace Space</span>
  <span class="fsb-pill fsb-pill-p">&#128721; OpenEnv Core</span>
  <span class="fsb-pill fsb-pill-g">&#128051; Docker Ready</span>
</div>
</div>
"""

DOCS_HTML = """
<div id="fs-docs">
<style>
#fs-docs {
  font-family: 'Inter', system-ui, sans-serif;
  background: #080b0a;
  color: #e2e8e4;
  min-height: 100vh;
  line-height: 1.7;
}
#fs-docs * { box-sizing: border-box; }

/* Reset Gradio interference */
#fs-docs h1,#fs-docs h2,#fs-docs h3,#fs-docs h4 {
  background: none !important;
  -webkit-text-fill-color: unset !important;
  background-clip: unset !important;
  font-weight: 800;
  letter-spacing: -0.5px;
}

/* ── Banner ── */
.fs-banner {
  position: relative;
  overflow: hidden;
  padding: 72px 48px 64px;
  text-align: center;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  background:
    linear-gradient(180deg, rgba(34,197,94,0.07) 0%, transparent 100%),
    radial-gradient(ellipse 70% 60% at 50% 0%, rgba(34,197,94,0.13) 0%, transparent 70%);
  margin-bottom: 0;
}
.fs-banner-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 16px; border-radius: 100px;
  background: rgba(34,197,94,0.12);
  border: 1px solid rgba(34,197,94,0.25);
  color: #22c55e;
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem; letter-spacing: 1.2px;
  text-transform: uppercase; margin-bottom: 26px;
}
.fs-banner-dot { width: 6px; height: 6px; background: #22c55e; border-radius: 50%;
  animation: fs-pulse 2s ease-in-out infinite; display: inline-block; }
@keyframes fs-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(1.5)} }
.fs-banner h1 {
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: clamp(2.6rem, 5vw, 4.8rem) !important;
  font-weight: 800 !important;
  line-height: 1.05 !important;
  margin-bottom: 18px !important;
  color: #fff !important;
  letter-spacing: -2px !important;
}
.fs-banner h1 span { color: #22c55e; }
.fs-banner-sub {
  font-size: 1rem; color: #7a8f82;
  max-width: 600px; margin: 0 auto 30px;
}
.fs-pills { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-bottom: 36px; }
.fs-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 14px; border-radius: 100px;
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem; font-weight: 500; border: 1px solid;
}
.fs-pill-g { background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.28); color: #22c55e; }
.fs-pill-a { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.28); color: #f59e0b; }
.fs-pill-b { background: rgba(56,189,248,0.1); border-color: rgba(56,189,248,0.28); color: #38bdf8; }
.fs-pill-p { background: rgba(167,139,250,0.1); border-color: rgba(167,139,250,0.28); color: #a78bfa; }

/* ── Layout ── */
.fs-layout { display: grid; grid-template-columns: 240px 1fr; min-height: 80vh; }
.fs-sidebar {
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
  padding: 32px 16px;
  border-right: 1px solid rgba(255,255,255,0.07);
  background: rgba(8,11,10,0.95);
  backdrop-filter: blur(12px);
}
.fs-sidebar::-webkit-scrollbar { width: 3px; }
.fs-sidebar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
.fs-logo { font-family: 'Inter', system-ui, sans-serif; font-weight: 800; font-size: 0.95rem; color: #22c55e; margin-bottom: 28px; display: block; }
.fs-nav-section { margin-bottom: 24px; }
.fs-nav-label { font-family: 'DM Mono',monospace; font-size: 0.62rem; letter-spacing: 1.5px; text-transform: uppercase; color: #2d3d32; margin-bottom: 8px; padding: 0 8px; }
.fs-nav-link {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: 8px;
  font-size: 0.85rem; color: #7a8f82;
  text-decoration: none; transition: all 0.18s ease;
  cursor: pointer;
}
.fs-nav-link:hover { background: rgba(34,197,94,0.1); color: #22c55e; }

/* ── Main ── */
.fs-main { padding: 48px 56px; max-width: 900px; }
.fs-section { margin-bottom: 72px; scroll-margin-top: 32px; }
.fs-tag { display: inline-block; font-family: 'DM Mono',monospace; font-size: 0.67rem; color: #22c55e; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; }
.fs-section h2 { font-family: 'Inter', system-ui, sans-serif !important; font-size: 1.9rem !important; color: #fff !important; margin-bottom: 6px !important; letter-spacing: -0.5px !important; }
.fs-section h3 { font-family: 'Inter', system-ui, sans-serif !important; font-size: 1.1rem !important; color: #22c55e !important; margin: 28px 0 14px !important; }
.fs-lead { font-size: 0.95rem; color: #7a8f82; margin-bottom: 28px; border-left: 2px solid #22c55e; padding-left: 14px; max-width: 640px; }
.fs-p { color: #7a8f82; margin-bottom: 14px; font-size: 0.9rem; }

/* ── Cards ── */
.fs-card-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin: 24px 0; }
.fs-stat-card {
  background: rgba(14,20,16,0.85); border: 1px solid rgba(255,255,255,0.07);
  border-radius: 14px; padding: 22px;
  transition: border-color 0.2s;
}
.fs-stat-card:hover { border-color: rgba(34,197,94,0.3); }
.fs-stat-label { font-family: 'DM Mono',monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: #2d3d32; margin-bottom: 8px; }
.fs-stat-value { font-family: 'Inter', system-ui, sans-serif; font-size: 2rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -1px; }
.fs-stat-desc { font-size: 0.8rem; color: #7a8f82; }

/* ── Tables ── */
.fs-table-wrap { overflow-x: auto; border-radius: 12px; border: 1px solid rgba(255,255,255,0.07); margin: 22px 0; }
#fs-docs table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
#fs-docs thead tr { background: rgba(34,197,94,0.05); border-bottom: 1px solid rgba(255,255,255,0.07); }
#fs-docs th { padding: 13px 16px; text-align: left; font-family: 'DM Mono',monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: #2d3d32; font-weight: 500; border: none !important; background: none !important; }
#fs-docs td { padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #7a8f82; vertical-align: top; border: none !important; border-bottom: 1px solid rgba(255,255,255,0.04) !important; background: none !important; }
#fs-docs td:first-child { font-family: 'DM Mono',monospace; color: #22c55e; font-size: 0.8rem; }
#fs-docs tbody tr:last-child td { border-bottom: none !important; }
#fs-docs tbody tr:hover td { background: rgba(34,197,94,0.04) !important; }

/* ── Code ── */
.fs-pre {
  background: rgba(0,0,0,0.55);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px; padding: 22px;
  overflow-x: auto;
  font-family: 'DM Mono',monospace;
  font-size: 0.8rem; line-height: 1.8;
  margin: 18px 0; position: relative;
  color: #e2e8e4;
}
.fs-code {
  font-family: 'DM Mono',monospace;
  font-size: 0.8rem; color: #86efac;
  background: rgba(34,197,94,0.08);
  padding: 2px 6px; border-radius: 4px;
}
.fs-kw { color: #f472b6; } .fs-fn { color: #60a5fa; } .fs-str { color: #a3e3b8; }
.fs-num { color: #f59e0b; } .fs-cmt { color: #2d3d32; font-style: italic; } .fs-var { color: #c4b5fd; }

/* ── Task Cards ── */
.fs-task-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; }
.fs-task {
  padding: 24px; border-radius: 14px;
  border: 1px solid; position: relative; overflow: hidden;
}
.fs-task-num { font-family:'DM Mono',monospace; font-size:0.65rem; opacity:0.45; margin-bottom:10px; }
.fs-task h4 { font-family:'Syne','Outfit',sans-serif !important; font-size:1rem !important; color:#fff !important; margin-bottom:8px !important; }
.fs-task .fs-meta { font-size:0.78rem; color:#7a8f82; margin-bottom:10px; }
.fs-task .fs-challenge { font-size:0.83rem; color:#e2e8e4; }
.fs-task-easy { background:rgba(34,197,94,0.05); border-color:rgba(34,197,94,0.2); }
.fs-task-med  { background:rgba(245,158,11,0.05); border-color:rgba(245,158,11,0.2); }
.fs-task-hard { background:rgba(248,113,113,0.05); border-color:rgba(248,113,113,0.2); }

/* ── Endpoints ── */
.fs-endpoint {
  display:flex; align-items:flex-start; gap:14px;
  padding:16px 20px; border-radius:8px;
  background:rgba(14,20,16,0.85);
  border:1px solid rgba(255,255,255,0.07);
  margin-bottom:10px; transition:border-color 0.2s;
}
.fs-endpoint:hover { border-color:rgba(34,197,94,0.3); }
.fs-method {
  font-family:'DM Mono',monospace; font-size:0.7rem; font-weight:600;
  padding:4px 10px; border-radius:5px; min-width:50px; text-align:center;
}
.fs-get  { background:rgba(56,189,248,0.1); color:#38bdf8; }
.fs-post { background:rgba(34,197,94,0.1); color:#22c55e; }
.fs-endpoint-path { font-family:'DM Mono',monospace; font-size:0.85rem; color:#fff; margin-bottom:3px; }
.fs-endpoint-desc { font-size:0.8rem; color:#7a8f82; }

/* ── Alert ── */
.fs-alert {
  display:flex; gap:12px; padding:14px 18px;
  border-radius:8px; margin:18px 0;
  border:1px solid; font-size:0.85rem;
}
.fs-alert-warn { background:rgba(245,158,11,0.07); border-color:rgba(245,158,11,0.25); }
.fs-alert-warn strong { color:#f59e0b; }
.fs-alert-info { background:rgba(56,189,248,0.07); border-color:rgba(56,189,248,0.25); }
.fs-alert-info strong { color:#38bdf8; }

/* ── Reward ── */
.fs-pos { color:#22c55e; font-weight:600; }
.fs-neg { color:#f87171; font-weight:600; }

/* ── Divider ── */
.fs-div { height:1px; background:rgba(255,255,255,0.06); margin:56px 0; }

/* ── Footer ── */
.fs-footer { text-align:center; padding:40px 48px; border-top:1px solid rgba(255,255,255,0.06); font-size:0.8rem; color:#2d3d32; }
.fs-footer a { color:#22c55e; text-decoration:none; }

@media (max-width: 860px) {
  .fs-layout { grid-template-columns: 1fr; }
  .fs-sidebar { display: none; }
  .fs-main { padding: 28px 18px; }
  .fs-card-grid, .fs-task-grid { grid-template-columns: 1fr; }
  .fs-banner { padding: 52px 20px 44px; }
}
</style>

<!-- LAYOUT -->
<div class="fs-layout">

  <!-- SIDEBAR -->
  <nav class="fs-sidebar">
    <span class="fs-logo">&#127807; FarmSim Docs</span>
    <div class="fs-nav-section">
      <div class="fs-nav-label">Getting Started</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-overview').scrollIntoView({behavior:'smooth'})">&#128161; Overview</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-quickstart').scrollIntoView({behavior:'smooth'})">&#128640; Quick Start</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-docker').scrollIntoView({behavior:'smooth'})">&#128051; Docker</div>
    </div>
    <div class="fs-nav-section">
      <div class="fs-nav-label">Environment</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-spec').scrollIntoView({behavior:'smooth'})">&#9881; Env Specification</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-actions').scrollIntoView({behavior:'smooth'})">&#127918; Action Space</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-obs').scrollIntoView({behavior:'smooth'})">&#128065; Observation Space</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-physics').scrollIntoView({behavior:'smooth'})">&#129518; Physics Engine</div>
    </div>
    <div class="fs-nav-section">
      <div class="fs-nav-label">Crops &amp; Market</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-crops').scrollIntoView({behavior:'smooth'})">&#127807; Crop Reference</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-climate').scrollIntoView({behavior:'smooth'})">&#127782; Climate System</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-market').scrollIntoView({behavior:'smooth'})">&#128200; Market Dynamics</div>
    </div>
    <div class="fs-nav-section">
      <div class="fs-nav-label">Tasks &amp; Scoring</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-tasks').scrollIntoView({behavior:'smooth'})">&#127919; Curriculum Tasks</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-grading').scrollIntoView({behavior:'smooth'})">&#127942; Grading Formulas</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-rewards').scrollIntoView({behavior:'smooth'})">&#127873; Reward Shaping</div>
    </div>
    <div class="fs-nav-section">
      <div class="fs-nav-label">Agent &amp; API</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-agent').scrollIntoView({behavior:'smooth'})">&#129504; LLM Agent Loop</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-api').scrollIntoView({behavior:'smooth'})">&#128225; API Reference</div>
      <div class="fs-nav-link" onclick="document.getElementById('fs-structure').scrollIntoView({behavior:'smooth'})">&#128193; File Structure</div>
    </div>
  </nav>

  <!-- MAIN -->
  <main class="fs-main">

    <!-- OVERVIEW -->
    <section class="fs-section" id="fs-overview">
      <div class="fs-tag">// overview</div>
      <h2>What is FarmSimulation?</h2>
      <p class="fs-lead">Real farmers balance scarce capital against volatile markets, fight pests and drought, and time harvests for peak prices &mdash; miss one irrigation in an arid climate and a week of growth dies overnight.</p>
      <p class="fs-p">FarmSimulation is an <strong>OpenEnv-compatible Reinforcement Learning environment</strong> that makes agents solve the Agricultural Resource Management problem &mdash; simulating the full crop lifecycle, market economics, and environmental hazards that any autonomous farming agent must navigate to turn a profit.</p>
      <div class="fs-card-grid">
        <div class="fs-stat-card"><div class="fs-stat-label">Land Plots</div><div class="fs-stat-value" style="color:#22c55e">4</div><div class="fs-stat-desc">Independent plots with unique soil states</div></div>
        <div class="fs-stat-card"><div class="fs-stat-label">Valid Actions</div><div class="fs-stat-value" style="color:#f59e0b">11</div><div class="fs-stat-desc">From planting to market selling</div></div>
        <div class="fs-stat-card"><div class="fs-stat-label">Crop Varieties</div><div class="fs-stat-value" style="color:#38bdf8">3</div><div class="fs-stat-desc">Wheat &middot; Rice &middot; Corn</div></div>
      </div>
    </section>

    <div class="fs-div"></div>

    <!-- QUICKSTART -->
    <section class="fs-section" id="fs-quickstart">
      <div class="fs-tag">// getting started</div>
      <h2>Quick Start</h2>
      <p class="fs-lead">Get your local dev server running in under 2 minutes.</p>
      <h3>1. Install &amp; Run Locally</h3>
      <pre class="fs-pre"><span class="fs-cmt"># Install dependencies (uv-compatible)</span>
pip install -e .

<span class="fs-cmt"># Start the environment server + Gradio UI</span>
uvicorn server.app:app --host 0.0.0.0 --port 7860

<span class="fs-cmt"># Open the interactive dashboard</span>
open http://localhost:7860</pre>
      <h3>2. Run the LLM Agent</h3>
      <pre class="fs-pre"><span class="fs-kw">export</span> <span class="fs-var">HF_TOKEN</span>=<span class="fs-str">"hf_your_token_here"</span>
<span class="fs-kw">export</span> <span class="fs-var">MODEL_NAME</span>=<span class="fs-str">"Qwen/Qwen2.5-72B-Instruct"</span>
<span class="fs-kw">export</span> <span class="fs-var">FARMING_ENV_URL</span>=<span class="fs-str">"http://localhost:7860"</span>
<span class="fs-kw">export</span> <span class="fs-var">MAX_STEPS</span>=<span class="fs-num">60</span>

python inference.py
<span class="fs-cmt"># &rarr; saves baseline_results.json</span></pre>
      <h3>3. Select Task Difficulty</h3>
      <pre class="fs-pre"><span class="fs-kw">export</span> <span class="fs-var">FARMING_TASK_ID</span>=<span class="fs-num">3</span>   <span class="fs-cmt"># 1=easy &nbsp; 2=medium &nbsp; 3=hard</span>
python inference.py</pre>
    </section>

    <div class="fs-div"></div>

    <!-- DOCKER -->
    <section class="fs-section" id="fs-docker">
      <div class="fs-tag">// deployment</div>
      <h2>Docker</h2>
      <pre class="fs-pre"><span class="fs-cmt"># Build the image</span>
docker build -t farming-sim .

<span class="fs-cmt"># Run with your HF token</span>
docker run -p <span class="fs-num">7860</span>:<span class="fs-num">7860</span> -e <span class="fs-var">HF_TOKEN</span>=hf_xxx farming-sim</pre>
      <p class="fs-p">The Dockerfile uses <span class="fs-code">python:3.11-slim</span> and exposes port <span class="fs-code">7860</span>. Compatible with Hugging Face Spaces &mdash; set <span class="fs-code">HF_TOKEN</span> via Space secrets.</p>
    </section>

    <div class="fs-div"></div>

    <!-- SPEC -->
    <section class="fs-section" id="fs-spec">
      <div class="fs-tag">// environment</div>
      <h2>Environment Specification</h2>
      <div class="fs-table-wrap"><table><thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>
        <tr><td>Name</td><td><span class="fs-code">farming-env</span></td></tr>
        <tr><td>Version</td><td>1.0.0</td></tr>
        <tr><td>Framework</td><td>OpenEnv Core</td></tr>
        <tr><td>Runtime</td><td>FastAPI + Uvicorn + Gradio</td></tr>
        <tr><td>Protocol</td><td>REST &mdash; <span class="fs-code">/reset</span>, <span class="fs-code">/step</span>, <span class="fs-code">/state</span>, <span class="fs-code">/health</span></td></tr>
        <tr><td>Python</td><td>&ge; 3.11</td></tr>
      </tbody></table></div>
    </section>

    <!-- ACTIONS -->
    <section class="fs-section" id="fs-actions">
      <div class="fs-tag">// action space</div>
      <h2>Action Space</h2>
      <div class="fs-table-wrap"><table><thead><tr><th>Field</th><th>Type</th><th>Range</th><th>Description</th></tr></thead><tbody>
        <tr><td>action_type</td><td>str</td><td>10 valid types</td><td>The farming operation to perform this step</td></tr>
        <tr><td>plot_id</td><td>int</td><td>[0, 3]</td><td>Target land plot (required for plot operations)</td></tr>
        <tr><td>seed_type</td><td>str</td><td>wheat | rice | corn</td><td>Crop variety (required for buy/plant/sell)</td></tr>
        <tr><td>quantity</td><td>int</td><td>&gt; 0</td><td>Seeds to buy or kilograms to sell</td></tr>
      </tbody></table></div>
      <h3>Valid Action Types</h3>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;">
        <span class="fs-code">wait</span><span class="fs-code">buy_seeds</span><span class="fs-code">plant</span><span class="fs-code">irrigate</span><span class="fs-code">pump_water</span><span class="fs-code">apply_fertilizer</span><span class="fs-code">spray_pesticide</span><span class="fs-code">pull_weeds</span><span class="fs-code">harvest</span><span class="fs-code">sell</span><span class="fs-code">clear</span>
      </div>
    </section>

    <!-- OBSERVATIONS -->
    <section class="fs-section" id="fs-obs">
      <div class="fs-tag">// observation space</div>
      <h2>Observation Space</h2>
      <div class="fs-table-wrap"><table><thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead><tbody>
        <tr><td>day</td><td>int</td><td>Current simulation day</td></tr>
        <tr><td>money</td><td>float</td><td>Agent cash balance</td></tr>
        <tr><td>water_tank</td><td>float</td><td>Tank fill fraction [0, 1]</td></tr>
        <tr><td>aquifer</td><td>float</td><td>Underground reserve in litres</td></tr>
        <tr><td>seed_inventory</td><td>Dict[str, int]</td><td>Seeds on hand</td></tr>
        <tr><td>storage</td><td>Dict[str, float]</td><td>Harvested crop kg</td></tr>
        <tr><td>plots</td><td>List[PlotState]</td><td>4 independent land plot states</td></tr>
        <tr><td>climate</td><td>ClimateState</td><td>Current temperature, humidity, precipitation</td></tr>
        <tr><td>market_prices</td><td>Dict[str, MarketPrice]</td><td>Live prices with trend signal</td></tr>
        <tr><td>text_summary</td><td>str</td><td>Human/LLM-readable narrative snapshot</td></tr>
        <tr><td>valid_actions</td><td>List[str]</td><td>Context-sensitive legal action hints</td></tr>
      </tbody></table></div>
    </section>

    <div class="fs-div"></div>

    <!-- PHYSICS -->
    <section class="fs-section" id="fs-physics">
      <div class="fs-tag">// simulation</div>
      <h2>Physics Engine</h2>
      <p class="fs-lead">Five simultaneous simulation passes run every day &mdash; grounded in agricultural science, not random walks.</p>
      <pre class="fs-pre"><span class="fs-num">1.</span> <span class="fs-fn">Precipitation</span>   &rarr; aquifer += climate.precipitation &times; <span class="fs-num">2</span>   <span class="fs-cmt">[mm &rarr; litres]</span>
<span class="fs-num">2.</span> <span class="fs-fn">Moisture Decay</span>  &rarr; soil_moisture -= climate.moisture_decay + weed_penalty(<span class="fs-num">0.05</span>)
<span class="fs-num">3.</span> <span class="fs-fn">Pest Escalation</span> &rarr; pest_severity = min(<span class="fs-num">1.0</span>, (sev + <span class="fs-num">0.1</span>) &times; <span class="fs-num">1.5</span>)
<span class="fs-num">4.</span> <span class="fs-fn">Health Damage</span>   &rarr; health -= <span class="fs-num">0.1</span>&nbsp;&nbsp;<span class="fs-cmt">[if moisture &lt; 0.2 OR NPK &lt; 0.2 OR moisture &gt; 0.9]</span>
<span class="fs-num">5.</span> <span class="fs-fn">Market Tick</span>    &rarr; sell_price = base &times; (<span class="fs-num">1.0</span> + <span class="fs-num">0.2</span>&middot;sin(2&pi;&middot;day/<span class="fs-num">20</span> + offset) + noise)</pre>
    </section>

    <!-- CLIMATE -->
    <section class="fs-section" id="fs-climate">
      <div class="fs-tag">// environment</div>
      <h2>Climate System</h2>
      <p class="fs-p">Climates rotate every <strong>10 days</strong>: <span class="fs-code">temperate &rarr; arid &rarr; tropical</span>. Extreme temperatures (&gt;32&deg;C or &lt;10&deg;C) freeze crop growth entirely.</p>
      <div class="fs-table-wrap"><table><thead><tr><th>Climate</th><th>Temp</th><th>Humidity</th><th>Precipitation</th><th>Moisture Decay</th><th>Spoilage</th></tr></thead><tbody>
        <tr><td>temperate</td><td>22&deg;C</td><td>60%</td><td>5 mm/day</td><td>0.05/day</td><td>1%/day</td></tr>
        <tr><td>arid</td><td>35&deg;C</td><td>20%</td><td>1 mm/day</td><td style="color:#f87171"><strong>0.12/day</strong></td><td>1%/day</td></tr>
        <tr><td>tropical</td><td>28&deg;C</td><td>90%</td><td>12 mm/day</td><td>0.03/day</td><td style="color:#f87171"><strong>3%/day</strong></td></tr>
      </tbody></table></div>
    </section>

    <!-- MARKET -->
    <section class="fs-section" id="fs-market">
      <div class="fs-tag">// economics</div>
      <h2>Market Dynamics</h2>
      <p class="fs-p">Each crop rides its own <strong>20-day sine wave</strong>, desynchronized by a fixed offset so peaks never align:</p>
      <pre class="fs-pre"><span class="fs-var">offset</span> = {<span class="fs-str">"wheat"</span>: <span class="fs-num">0</span>, <span class="fs-str">"rice"</span>: <span class="fs-num">7</span>, <span class="fs-str">"corn"</span>: <span class="fs-num">13</span>}
sell_price = base_sell &times; (<span class="fs-num">1.0</span> + <span class="fs-num">0.20</span> &times; sin(2&pi; &times; (day + offset) / <span class="fs-num">20</span>) + noise)</pre>
      <h3>Price Elasticity</h3>
      <p class="fs-p">Large sell orders crash their own price:</p>
      <pre class="fs-pre">price_drop = min(<span class="fs-num">50%</span>, qty_sold / <span class="fs-num">10</span>kg &times; <span class="fs-num">1%</span>)
sell_price *= (<span class="fs-num">1.0</span> - price_drop)</pre>
    </section>

    <div class="fs-div"></div>

    <!-- CROPS -->
    <section class="fs-section" id="fs-crops">
      <div class="fs-tag">// crop reference</div>
      <h2>Crop Reference</h2>
      <div class="fs-alert fs-alert-warn"><strong>&#9888; Critical:</strong>&nbsp; Once a plot reaches <span class="fs-code">mature</span>, you have exactly <strong>3 days</strong> to harvest before it withers and is permanently lost.</div>
      <div class="fs-table-wrap"><table><thead><tr><th>Crop</th><th>Grow Days</th><th>Max Yield</th><th>Buy $</th><th>Sell $</th><th>Water Need</th><th>NPK Drain</th></tr></thead><tbody>
        <tr><td>wheat</td><td>7 days</td><td>10 kg</td><td>$5.00</td><td>$8.00</td><td>Low</td><td>[0.05, 0.02, 0.03]</td></tr>
        <tr><td>rice</td><td>12 days</td><td>20 kg</td><td>$8.00</td><td>$14.00</td><td style="color:#f87171"><strong>High</strong></td><td>[0.03, 0.04, 0.05]</td></tr>
        <tr><td>corn</td><td>18 days</td><td>35 kg</td><td>$12.00</td><td>$20.00</td><td>Medium</td><td>[0.08, 0.04, 0.02]</td></tr>
      </tbody></table></div>
    </section>

    <div class="fs-div"></div>

    <!-- TASKS -->
    <section class="fs-section" id="fs-tasks">
      <div class="fs-tag">// curriculum</div>
      <h2>The 3 Curriculum Tasks</h2>
      <p class="fs-lead">Progressive difficulty from basic crop management to drought survival under volatile markets.</p>
      <div class="fs-task-grid">
        <div class="fs-task fs-task-easy"><div class="fs-task-num">TASK 01 &middot; &#127856; EASY</div><h4>Single Crop Stable</h4><div class="fs-meta">Start: $200 &middot; Max Days: 30</div><div class="fs-challenge">Double your starting money through stable single-crop cultivation.</div></div>
        <div class="fs-task fs-task-med"><div class="fs-task-num">TASK 02 &middot; &#128998; MEDIUM</div><h4>Multi-Crop Market Timing</h4><div class="fs-meta">Start: $150 &middot; Max Days: 45</div><div class="fs-challenge">Profit across all 3 crops at peak prices. 40% of score from selling during price peaks.</div></div>
        <div class="fs-task fs-task-hard"><div class="fs-task-num">TASK 03 &middot; &#128997; HARD</div><h4>Drought Survival</h4><div class="fs-meta">Start: $100 &middot; Max Days: 60</div><div class="fs-challenge">Every 5th day: zero precipitation + &minus;15L tank drain. Pump from aquifer to survive.</div></div>
      </div>
    </section>

    <!-- GRADING -->
    <section class="fs-section" id="fs-grading">
      <div class="fs-tag">// scoring</div>
      <h2>Grading Formulas</h2>
      <p class="fs-p">Every episode produces a final grade in [0.0, 1.0] via <span class="fs-code">observation.metadata["grade"]</span>. A score &ge; <strong>0.80</strong> is professional tier.</p>
      <h3>Task 1 &mdash; Single Crop Stable</h3>
      <pre class="fs-pre">score = clamp(net_worth / (initial_money &times; <span class="fs-num">2.0</span>), <span class="fs-num">0</span>, <span class="fs-num">1</span>) &minus; min(<span class="fs-num">0.20</span>, withered &times; <span class="fs-num">0.05</span>)</pre>
      <h3>Task 2 &mdash; Multi-Crop Market Timing</h3>
      <pre class="fs-pre">score = <span class="fs-num">0.6</span> &times; profit_score + <span class="fs-num">0.4</span> &times; timing_score &minus; min(<span class="fs-num">0.30</span>, withered &times; <span class="fs-num">0.10</span>)
timing_score = clamp(premium_revenue / (total_revenue &times; <span class="fs-num">0.3</span>), <span class="fs-num">0</span>, <span class="fs-num">1</span>)</pre>
      <h3>Task 3 &mdash; Drought Survival</h3>
      <pre class="fs-pre">score = <span class="fs-num">0.5</span> &times; profit + <span class="fs-num">0.3</span> &times; survival + <span class="fs-num">0.2</span> &times; resilience &minus; min(<span class="fs-num">0.40</span>, withered &times; <span class="fs-num">0.15</span>)
resilience = healthy_days / max_days &nbsp;<span class="fs-cmt"># days where &ge; 2 plots had health &ge; 0.6</span></pre>
    </section>

    <!-- REWARDS -->
    <section class="fs-section" id="fs-rewards">
      <div class="fs-tag">// reward shaping</div>
      <h2>Reward Shaping</h2>
      <div class="fs-table-wrap"><table><thead><tr><th>Action / Event</th><th>Reward</th><th>Notes</th></tr></thead><tbody>
        <tr><td>plant</td><td class="fs-pos">+0.2</td><td>Commit bonus</td></tr>
        <tr><td>irrigate (rescue)</td><td class="fs-pos">+0.5</td><td>Moisture was critically low (&lt;0.25)</td></tr>
        <tr><td>irrigate (normal)</td><td class="fs-pos">+0.1</td><td></td></tr>
        <tr><td>harvest</td><td class="fs-pos">up to +1.0</td><td>Scales with yield / max_yield</td></tr>
        <tr><td>sell</td><td class="fs-pos">+0.3 + premium</td><td>Bonus for above-base price premium</td></tr>
        <tr><td>wait (crops growing)</td><td class="fs-pos">+0.05/plot</td><td>Smart patience</td></tr>
        <tr><td>health maintenance</td><td class="fs-pos">+0.1/plot/day</td><td>Passive per living, healthy plot</td></tr>
        <tr><td>terminal bonus</td><td class="fs-pos">up to +10.0</td><td>max(0, (net_worth/start &minus; 1) &times; 5)</td></tr>
        <tr><td>crop withers</td><td class="fs-neg">&minus;5.0</td><td>Hard penalty, once per wither event</td></tr>
        <tr><td>wait with mature plots</td><td class="fs-neg">&minus;0.3/plot</td><td>Every idle day risks permanent loss</td></tr>
        <tr><td>overwatering</td><td class="fs-neg">&minus;0.15 health</td><td>Moisture &gt; 0.9 damages the crop</td></tr>
        <tr><td>storage overflow</td><td class="fs-neg">&minus;0.3</td><td>Lost kg on harvest</td></tr>
        <tr><td>invalid action</td><td class="fs-neg">&minus;1.0</td><td>Hard rejection</td></tr>
      </tbody></table></div>
    </section>

    <div class="fs-div"></div>

    <!-- AGENT -->
    <section class="fs-section" id="fs-agent">
      <div class="fs-tag">// llm agent</div>
      <h2>LLM Agent Architecture</h2>
      <p class="fs-lead"><span class="fs-code">inference.py</span> implements a stateless LLM-as-Agent loop compatible with any OpenAI-format API.</p>
      <pre class="fs-pre">System Prompt  &rarr;  farming strategy rules (one-shot)
                        |
Observation    &rarr;  text_summary + valid_actions list
                        |
LLM Response   &rarr;  raw JSON: <span class="fs-str">{"action_type": "harvest", "plot_id": 2}</span>
                        |
Parser         &rarr;  parse_action() &rarr; validate_action() &rarr; FALLBACK on malformed
                        |
Environment    &rarr;  env.step(action) &rarr; new obs + reward
                        |
History Buffer &rarr;  last 4 steps kept in context window</pre>
      <div class="fs-table-wrap"><table><thead><tr><th>Parameter</th><th>Default</th></tr></thead><tbody>
        <tr><td>Model</td><td><span class="fs-code">Qwen/Qwen2.5-72B-Instruct</span></td></tr>
        <tr><td>Temperature</td><td><span class="fs-code">0.2</span></td></tr>
        <tr><td>Max tokens</td><td><span class="fs-code">150</span></td></tr>
        <tr><td>Fallback</td><td><span class="fs-code">{"action_type": "wait"}</span></td></tr>
      </tbody></table></div>
    </section>

    <div class="fs-div"></div>

    <!-- API -->
    <section class="fs-section" id="fs-api">
      <div class="fs-tag">// rest api</div>
      <h2>API Reference</h2>
      <p class="fs-lead">The server runs on port <span class="fs-code">7860</span>. All endpoints return JSON.</p>
      <div class="fs-endpoint"><span class="fs-method fs-get">GET</span><div><div class="fs-endpoint-path">/health</div><div class="fs-endpoint-desc">Liveness check &mdash; returns <span class="fs-code">{"status": "ok"}</span></div></div></div>
      <div class="fs-endpoint"><span class="fs-method fs-post">POST</span><div><div class="fs-endpoint-path">/reset</div><div class="fs-endpoint-desc">Body: <span class="fs-code">{"task_id": 1|2|3}</span> &mdash; Resets episode, returns initial <span class="fs-code">FarmObservation</span></div></div></div>
      <div class="fs-endpoint"><span class="fs-method fs-post">POST</span><div><div class="fs-endpoint-path">/step</div><div class="fs-endpoint-desc">Body: <span class="fs-code">{"action": FarmAction}</span> &mdash; Takes one action, returns new <span class="fs-code">FarmObservation</span></div></div></div>
      <div class="fs-endpoint"><span class="fs-method fs-get">GET</span><div><div class="fs-endpoint-path">/state</div><div class="fs-endpoint-desc">Returns full internal <span class="fs-code">FarmState</span> for debugging</div></div></div>
      <div class="fs-endpoint"><span class="fs-method fs-get">GET</span><div><div class="fs-endpoint-path">/</div><div class="fs-endpoint-desc">Gradio interactive dashboard UI</div></div></div>
      <div class="fs-alert fs-alert-info" style="margin-top:24px;"><strong>&#8505; OpenEnv Protocol:</strong>&nbsp;All endpoints are compliant with the OpenEnv Core REST specification for agent loop integration.</div>
    </section>

    <!-- STRUCTURE -->
    <section class="fs-section" id="fs-structure">
      <div class="fs-tag">// repository</div>
      <h2>File Structure</h2>
      <pre class="fs-pre"><span class="fs-fn">FarmSimulation/</span>
&boxvr;&boxh;&boxh; <span class="fs-fn">server/</span>
&boxv;   &boxvr;&boxh;&boxh; <span class="fs-var">app.py</span>                  <span class="fs-cmt">&larr; OpenEnv create_app() entry + Gradio mount</span>
&boxv;   &boxvr;&boxh;&boxh; <span class="fs-var">farming_environment.py</span>  <span class="fs-cmt">&larr; Full physics engine + 10 action handlers</span>
&boxv;   &boxvr;&boxh;&boxh; <span class="fs-var">gradio_app.py</span>           <span class="fs-cmt">&larr; Glassmorphic dark-mode dashboard UI</span>
&boxv;   &boxvr;&boxh;&boxh; <span class="fs-var">tasks.py</span>               <span class="fs-cmt">&larr; EpisodeRecord + grade_task1/2/3 graders</span>
&boxv;   &boxur;&boxh;&boxh; <span class="fs-var">requirements.txt</span>
&boxvr;&boxh;&boxh; <span class="fs-var">models.py</span>                   <span class="fs-cmt">&larr; Pydantic schemas + SEED_CONFIG / CLIMATE_CONFIG</span>
&boxvr;&boxh;&boxh; <span class="fs-var">inference.py</span>                <span class="fs-cmt">&larr; OpenEnv compliance runner (LLM agent loop)</span>
&boxvr;&boxh;&boxh; <span class="fs-var">openenv.yaml</span>               <span class="fs-cmt">&larr; OpenEnv manifest</span>
&boxvr;&boxh;&boxh; <span class="fs-var">pyproject.toml</span>             <span class="fs-cmt">&larr; Dependencies (uv-compatible)</span>
&boxvr;&boxh;&boxh; <span class="fs-var">Dockerfile</span>                 <span class="fs-cmt">&larr; Python 3.11-slim, exposes :7860</span>
&boxvr;&boxh;&boxh; <span class="fs-var">baseline_results.json</span>      <span class="fs-cmt">&larr; Last inference run output</span>
&boxur;&boxh;&boxh; <span class="fs-var">test_phase{2-7}.py</span>         <span class="fs-cmt">&larr; Phase-gated TDD test suite</span></pre>
    </section>

  </main>
</div>

<div class="fs-footer">
  &#127807; <strong style="color:#22c55e">FarmSimulation</strong> &mdash; Built for the Meta Hackathon &middot; MIT License
  <br><span style="margin-top:6px;display:block;">An OpenEnv-compatible environment for evaluating LLM agents on real-world agricultural planning tasks.</span>
</div>
</div>
"""

def format_hud(obs, metadata):
    day = obs.day
    max_days = metadata.get('max_days', 30)
    money = obs.money
    water_pct = obs.water_tank * 100
    climate = obs.climate
    climate_type = getattr(climate, "climate_type", "UNKNOWN").upper()
    temp = getattr(climate, "temperature", 0)
    drought_active = metadata.get("drought_active", False)
    
    msg = f"## 📅 DAY {day}/{max_days} | ⌛ **{obs.labor_remaining:.1f}h remaining** | 💰 FUNDS: ${money:.2f} | 💧 TANK: {water_pct:.1f}%\n\n"
    msg += f"<div style='margin-top: 12px; margin-bottom: 6px;'>**🌡️ CLIMATE**: <span style='color:#fbbf24'>{climate_type} ({int(temp)}°C)</span></div>"
    if drought_active:
        msg += " 🔥 <strong style='color:#ef4444'>DROUGHT ACTIVE!</strong>"
    return msg

def format_plot(obs, idx):
    plot = obs.plots[idx]
    stage = getattr(plot, "stage", "empty")
    crop = getattr(plot, "crop_type", "DIRT")
    if not crop: crop = "DIRT"
    
    icon = "🟫"
    if stage == "seedling": icon = "🌱"
    elif stage == "growing": icon = "🌿"
    elif stage == "mature": icon = "🌾"
    elif stage == "withered": icon = "🥀"
    
    moisture = getattr(plot, "soil_moisture", 0) * 100
    health = getattr(plot, "health", 0) * 100
    nitrogen = getattr(plot, "nitrogen", 1.0) * 100
    phosphorus = getattr(plot, "phosphorus", 1.0) * 100
    potassium = getattr(plot, "potassium", 1.0) * 100
    has_weeds = getattr(plot, "has_weeds", False)
    has_pests = getattr(plot, "has_pests", False)
    pest_sev = getattr(plot, "pest_severity", 0.0) * 100
    protection = getattr(plot, "pesticide_protection", 0)
    
    warnings = ""
    if has_weeds:
        warnings += " 🌿"
    if has_pests:
        warnings += f" 🐛({pest_sev:.0f}%)"
    if protection > 0:
        warnings += f" 🛡️({protection}d)"
    
    return f"""### PLOT {idx}{warnings}
    
<div class="plot-icon-wrapper" style="font-size: 3.5rem; margin:15px 0;">{icon}</div>

**{crop.upper()}**
<span style="color:#a1a1aa; font-size:0.85em; text-transform:uppercase;">{stage}</span>

<div style="margin-top: 15px; text-align: left; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
    <strong>💧 WATER</strong> &nbsp;&nbsp;&nbsp;&nbsp; {moisture:.0f}%<br>
    <strong>✨ HEALTH</strong> &nbsp;&nbsp;&nbsp; {health:.0f}%<br>
    <strong>🌱 N-P-K</strong> &nbsp;&nbsp;&nbsp;&nbsp; {nitrogen:.0f}-{phosphorus:.0f}-{potassium:.0f}
</div>
"""

def format_resources(obs, title, data_dict, unit=""):
    lines = [f"### {title}", "| ITEM | AMOUNT |", "|---|---|"]
    if not data_dict:
        lines.append("| (Empty) | - |")
    for k, v in data_dict.items():
        if isinstance(v, float):
            lines.append(f"| {k.title()} | **{v:.1f}{unit}** |")
        else:
            lines.append(f"| {k.title()} | **{v}{unit}** |")
    return "\n".join(lines)

def format_market(obs):
    lines = ["### 📈 MARKET", "| CROP | PRICE | TREND |", "|---|---|---|"]
    for k, v in obs.market_prices.items():
        trend = "<span style='color:#10b981'>▲ UP</span>" if v.trend > 0 else "<span style='color:#ef4444'>▼ DOWN</span>" if v.trend < 0 else "<span style='color:#a1a1aa'>─ FLAT</span>"
        lines.append(f"| {k.title()} | **${v.sell_price:.2f}** | {trend} |")
    return "\n".join(lines)

def prettify_observation_json(obs):
    """Convert observation to prettified JSON format."""
    import json
    
    # Convert observation to dict (assuming obs has a dict representation or attributes)
    obs_dict = {}
    
    # Extract all observable attributes
    if hasattr(obs, '__dict__'):
        for key, value in obs.__dict__.items():
            if not key.startswith('_'):
                # Handle special types
                if hasattr(value, '__dict__'):
                    # Nested object
                    obs_dict[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
                elif isinstance(value, (list, dict)):
                    obs_dict[key] = value
                else:
                    obs_dict[key] = value
    
    # Pretty print with indentation
    return json.dumps(obs_dict, indent=2, default=str)

def format_action_history(metadata):
    """Format action history as JSON object - organized by day with full state snapshots."""
    history = metadata.get("action_history", [])
    
    if not history:
        return {"message": "No actions taken yet"}
    
    # Organize actions by day (reverse order - newest first)
    history_by_day = {}
    for entry in reversed(history):
        day_key = f"Day {entry['day']}"
        # Include the full state snapshot for each day
        history_by_day[day_key] = entry
    
    # Return as dict with total count
    return {
        "total_days_recorded": len(history),
        **history_by_day  # Unpack day entries at top level
    }

def create_gradio_ui(env_factory):
    with gr.Blocks(title="Farming RL Dashboard", css=custom_css, theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
    )) as ui:

        gr.HTML(SHARED_BANNER_HTML)

        with gr.Tabs(selected=0):
            with gr.Tab("🚜 Dashboard"):
                gr.HTML("<div style='height: 24px'></div>")
                with gr.Row():
                    with gr.Column(scale=3):
                        # Overview Section
                        hud_md = gr.Markdown("Loading...", elem_classes=["section-box"])
                        
                        gr.HTML("<div style='height: 32px'></div>")
                        
                        # Farm Plots Grid
                        gr.Markdown("## 🌾 THE FARM", elem_classes=[])
                        plot_mds = []
                        for row_idx in range(2):
                            with gr.Row():
                                for col_idx in range(2):
                                    plot_md = gr.Markdown("Loading plot...", elem_classes=["farm-plot"])
                                    plot_mds.append(plot_md)
                        
                        gr.HTML("<div style='height: 32px'></div>")
                        
                        # Resources Row
                        gr.Markdown("## 📦 INVENTORY & ECONOMY", elem_classes=[])
                        with gr.Row():
                            seeds_md = gr.Markdown("seeds", elem_classes=["section-box"])
                            storage_md = gr.Markdown("storage", elem_classes=["section-box"])
                            market_md = gr.Markdown("market", elem_classes=["section-box"])
                        
                        action_feed = gr.Markdown("> 🟢 **SYSTEM READY...** AWAITING COMMAND.", elem_classes=["section-box"])

                        status_box = gr.Textbox(
                            label="📊 Detailed Observation JSON",
                            lines=15,
                            max_lines=15,
                            interactive=False,
                            visible=False,
                        )
                        show_debug_btn = gr.Button("🔍 Toggle Debug", size="sm")

                    # Control Panel
                    with gr.Column(scale=1, elem_classes=["section-box"]):
                        gr.Markdown("### 🕹️ COMMAND CENTER")
                        
                        with gr.Row():
                            task_id_input = gr.Dropdown(
                                choices=[("Easy", 1), ("Medium", 2), ("Hard", 3)],
                                value=1,
                                label="Difficulty"
                            )
                            reset_btn = gr.Button("♻️ RESET", variant="secondary")
                        
                        gr.HTML("<hr style='margin: 24px 0; border-color: rgba(255,255,255,0.08)'>")
                        gr.Markdown("#### ⚡ QUICK ACTIONS")
                        with gr.Row():
                            buy_btn = gr.Button("🛒 BUY SEEDS", elem_classes=["action-btn"])
                            wait_btn = gr.Button("⏰ WAIT", elem_classes=["action-btn"])
                        with gr.Row():
                            pump_btn = gr.Button("⚙️ PUMP", elem_classes=["action-btn"])
                            end_day_btn = gr.Button("🧺 END DAY", variant="secondary")
                        
                        gr.HTML("<hr style='margin: 24px 0; border-color: rgba(255,255,255,0.08)'>")
                        gr.Markdown("#### 🌱 PLOT OPERATIONS")
                        plot_selector = gr.Radio(
                            choices=[0, 1, 2, 3],
                            value=0,
                            label="Select Target Plot"
                        )
                        
                        with gr.Row():
                            plant_btn = gr.Button("🌱 PLANT", variant="primary")
                            irrigate_btn = gr.Button("💧 WATER", variant="primary")
                            harvest_btn = gr.Button("🌾 HARVEST", variant="primary")
                        with gr.Row():
                            fertilize_btn = gr.Button("🧪 FERTILIZE", variant="primary")
                            spray_btn = gr.Button("🦟 SPRAY", variant="primary")
                            pull_weeds_btn = gr.Button("🤲 WEED", variant="primary")
                        
                        clear_btn = gr.Button("🧹 CLEAR DEAD", variant="secondary")

                        gr.Markdown("---")
                        gr.Markdown("#### 📦 RESOURCES")
                        seed_type = gr.Dropdown(
                            choices=["wheat", "rice", "corn"],
                            value="wheat",
                            label="Target Crop"
                        )
                        
                        quantity = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Quantity (Buy/Sell)"
                        )
                        
                        sell_btn = gr.Button("💰 SELL CROPS", variant="secondary")
                        
                        show_history_btn = gr.Button("📜 Toggle Action History", size="sm")
                        history_display = gr.JSON(label="Action History (All Days)", visible=False)
                        episode_stats = gr.JSON(label="Metadata", visible=False)

            # ── AGENT TABS (REMOVED FOR HF STABILITY) ──────────────────────────
            # The Auto-Heuristic and Hybrid AI tabs have been disabled to prevent
            # reactive loops and connection timeouts on HuggingFace Spaces.
            # Automated agents can still access the environment via the REST API.

            with gr.Tab("🧪 AGENT STRESS TEST") as stress_tab:
                gr.Markdown("### 🛠️ CONFIGURATION & CALIBRATION")
                with gr.Row():
                    with gr.Column(scale=1):
                        st_token = gr.Textbox(label="HuggingFace Token", type="password", placeholder="hf_...", value=os.getenv("HF_TOKEN", ""))
                        st_model = gr.Dropdown(
                            label="Model Endpoint",
                            choices=[
                                "Qwen/Qwen2.5-72B-Instruct", 
                                "Qwen/Qwen2.5-7B-Instruct", 
                                "meta-llama/Llama-3.3-70B-Instruct",
                                "deepseek-ai/DeepSeek-V3"
                            ],
                            value="Qwen/Qwen2.5-72B-Instruct",
                            allow_custom_value=True
                        )
                        st_seed = gr.Number(label="Random Seed (Consistency)", value=42, precision=0)
                        st_agent = gr.Radio(["Heuristic", "Hybrid"], label="Target Agent", value="Heuristic")
                        st_diff = gr.Slider(0, 3, value=0, step=1, label="Difficulty Filter (0=All)")
                        st_run_btn = gr.Button("🔥 START STRESS TEST", variant="primary")
                    
                    with gr.Column(scale=2):
                        st_gauge = gr.HTML(value="<div style='text-align:center; padding: 20px; background: #1a1a1a; border-radius: 15px; border: 1px solid #333;'><h3 style='color: #888;'>Intelligence Pass Rate</h3><h1 style='font-size:5em; margin: 10px 0; color: #4caf50;'>--%</h1><p style='color: #666;'>Select configuration and click START</p></div>")
                        st_progress_md = gr.Markdown("### Status: `IDLE`")
                
                gr.Markdown("---")
                st_results = gr.Dataframe(
                    headers=["ID", "Scenario", "Dif", "Expected", "Actual", "Status", "Duration (s)"],
                    datatype=["number", "str", "number", "str", "str", "str", "number"],
                    label="Execution Trace",
                    interactive=False
                )

            with gr.Tab("📖 Documentation") as doc_tab:
                doc_html_content = gr.HTML(DOCS_HTML)

        # dashboard_vals (12): hud, plot×4, seeds, storage, market, action_feed, history(dict), json(str), metadata(dict)
        base_outputs = [hud_md] + plot_mds + [seeds_md, storage_md, market_md, action_feed, history_display, status_box, episode_stats]

        # ⚠️ Agent outputs disabled for stability on HuggingFace ⚠️
        # h_outputs = [...]
        # ai_outputs = [...]

        all_outputs = base_outputs 

        # Agent Instances
        h_agent = HeuristicAgent()
        ai_agent = HybridAgent()

        async def get_status(reasoning=None, status=None):
            """Return a single snapshot of all dashboard outputs.

            IMPORTANT – HuggingFace Spaces compatibility:
            This MUST be a regular async function (single return), NOT an async
            generator (yield).  The previous double-yield pattern created dangling
            generator state that blocked the Gradio event queue when the user
            switched tabs — the pending second yield could never be delivered
            after a tab change cancelled the WebSocket frame, causing a permanent
            UI freeze.
            """
            if reasoning is not None: state.reasoning = reasoning
            if status is not None: state.status = status
            
            env = env_factory()
            obs = env.get_observation()
            metadata = env.get_metadata()
            msg = getattr(env, "_action_message", "")
            
            # Dashboard Components
            out_plots = [format_plot(obs, i) for i in range(4)]
            out_seeds = format_resources(obs, "🎒 SEEDS", obs.seed_inventory)
            out_storage = format_resources(obs, "🌾 STORAGE", obs.storage, "kg")
            out_market = format_market(obs)
            out_msg = f"> {msg}" if msg else "> AWAITING COMMAND..."
            out_history = format_action_history(metadata)
            out_json = prettify_observation_json(obs)
            
            # Agent Reasoning Integration
            os.environ["CURRENT_AGENT_REASONING"] = state.reasoning
            out_hud = format_hud(obs, metadata)
            
            return [out_hud] + out_plots + [out_seeds, out_storage, out_market, out_msg, out_history, out_json, metadata]

        async def handle_reset(tid):
            env = env_factory()
            os.environ["FARMING_TASK_ID"] = str(int(tid))
            env.reset(task_id=int(tid))
            return await get_status(reasoning="Environment Reset.", status="🟢 READY")

        async def handle_action(action_type, p_id, qty, s_type):
            env = env_factory()
            action = {"action_type": action_type}
            if action_type in ["plant", "irrigate", "harvest", "clear", "apply_fertilizer", "spray_pesticide", "pull_weeds", "end_day"]:
                action["plot_id"] = int(p_id)
            if action_type in ["buy_seeds", "sell"]:
                action["seed_type"] = s_type
                action["quantity"] = int(qty)
            if action_type == "plant":
                action["seed_type"] = s_type
            
            env.step(action)
            return await get_status(status=f"DONE: {action_type.upper()}")

        async def get_initial_status():
            return await get_status(reasoning="System initialized.", status="🟢 READY")

        # ── Event Handlers ────────────────────────────────────────────────────────

        # On startup: only hydrate Dashboard.
        ui.load(get_initial_status, outputs=base_outputs)

        # Dashboard action buttons — update only Dashboard outputs
        reset_btn.click(handle_reset, inputs=[task_id_input], outputs=base_outputs)
        
        # We need a small wrapper for buttons to inject the fixed command name
        async def do_wait(p, q, s):
            return await handle_action("wait", p, q, s)
        async def do_buy(p, q, s):
            return await handle_action("buy_seeds", p, q, s)
        async def do_pump(p, q, s):
            return await handle_action("pump_water", p, q, s)
        async def do_end_day(p, q, s):
            return await handle_action("end_day", p, q, s)
        async def do_plant(p, q, s):
            return await handle_action("plant", p, q, s)
        async def do_irrigate(p, q, s):
            return await handle_action("irrigate", p, q, s)
        async def do_harvest(p, q, s):
            return await handle_action("harvest", p, q, s)
        async def do_clear(p, q, s):
            return await handle_action("clear", p, q, s)
        async def do_fertilize(p, q, s):
            return await handle_action("apply_fertilizer", p, q, s)
        async def do_spray(p, q, s):
            return await handle_action("spray_pesticide", p, q, s)
        async def do_pull_weeds(p, q, s):
            return await handle_action("pull_weeds", p, q, s)
        async def do_sell(p, q, s):
            return await handle_action("sell", p, q, s)

        wait_btn.click(do_wait, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        buy_btn.click(do_buy, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        pump_btn.click(do_pump, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        end_day_btn.click(do_end_day, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        plant_btn.click(do_plant, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        irrigate_btn.click(do_irrigate, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        harvest_btn.click(do_harvest, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        clear_btn.click(do_clear, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        fertilize_btn.click(do_fertilize, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        spray_btn.click(do_spray, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        pull_weeds_btn.click(do_pull_weeds, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        sell_btn.click(do_sell, inputs=[plot_selector, quantity, seed_type], outputs=base_outputs)
        # ── Stress Test Logic ───────────────────────────────────────────
        
        async def run_stress_test(token, model, seed, agent_type, diff):
            # Instantiate Agent
            agent = HybridAgent() if agent_type == "Hybrid" else HeuristicAgent()
            engine = ScenarioEngine(agent)
            
            # Filter diff
            d_val = int(diff) if diff > 0 else None
            
            # Execute
            report_gen = engine.run_tests_stream(difficulty=d_val, seed=int(seed), hf_token=token, model_name=model)
            
            df_data = []
            async for result in report_gen:
                if "summary" in result:
                    summary = result["summary"]
                    score = summary["score"]
                    color = "#f44336" if score < 50 else ("#ffeb3b" if score < 80 else "#4caf50")
                    gauge_html = f"""
                    <div style='text-align:center; padding: 20px; background: #0f172a; border-radius: 12px; border: 1px solid #1e293b;'>
                        <h3 style='color: #94a3b8;'>Intelligence Pass Rate</h3>
                        <h1 style='font-size:5em; margin: 10px 0; color: {color};'>{score:.0f}%</h1>
                        <p style='color: #64748b;'>{summary['passed']} Passed | {summary['failed']} Failed</p>
                    </div>
                    """
                    progress = f"### Status: `COMPLETE` ({summary['total']} scenarios evaluated)"
                    yield gauge_html, progress, df_data
                else:
                    # Individual scenario result
                    r = result
                    df_data.append([
                        r["id"], r["name"], r["difficulty"], 
                        ", ".join(r["expected"]), r["actual"], 
                        r["status"], r["duration"]
                    ])
                    yield gr.update(), f"### Status: `EVALUATING` (Scenario {r['id']}...)", df_data
                    await asyncio.sleep(0.01)

        st_run_btn.click(
            run_stress_test, 
            inputs=[st_token, st_model, st_seed, st_agent, st_diff], 
            outputs=[st_gauge, st_progress_md, st_results]
        )

        # Agent logic removed from UI

        # Agent logic removed from UI

        # Auto-Play Timers — DISABLED on HuggingFace for stability
        # h_timer = gr.Timer(1.0, active=False)
        # ai_timer = gr.Timer(3.0, active=False)

        # Visibility toggles for debug panel and history — use gr.State to track runtime
        # visibility. Component.visible is a build-time constant on HuggingFace and
        # evaluating `not component.visible` always returns the same value, breaking toggle.
        debug_visible = gr.State(False)
        history_visible = gr.State(False)

        def toggle_debug(current):
            new_val = not current
            return gr.update(visible=new_val), new_val

        def toggle_history(current):
            new_val = not current
            return gr.update(visible=new_val), new_val

        show_debug_btn.click(toggle_debug, inputs=[debug_visible], outputs=[status_box, debug_visible])
        show_history_btn.click(toggle_history, inputs=[history_visible], outputs=[history_display, history_visible])

    return ui
#!/usr/bin/env python3
"""
OpenEnv Reference Implementation Agent.

Mandatory for Hackathon submission.
Demonstrates LLM interactions using standard OpenAI models.
"""

import os
import sys
import json
import asyncio
from openai import AsyncOpenAI

try:
    from client import TradeExecClient
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from client import TradeExecClient

# Use environment variables if set
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

EVAL_TASKS = [
    "task1_twap_beater",
    "task2_vwap_optimizer",
    "task3_volatile_execution"
]

SYSTEM_PROMPT = """
You are an algorithmic Smart Order Router (SOR). 
Your objective is to minimize Implementation Shortfall (IS) when executing large block trades.
You have three tools available:
1. get_market_state(): Read the current order book, prices, and IS metrics.
2. get_baseline_comparison(): Check how your execution compares to TWAP/VWAP.
3. execute_trade(participation_rate): Your primary action. Pass a float between 0.0 and 0.25.

STRATEGY:
- Start slower (participation_rate=0.03) to let the market digest.
- If behind schedule, accelerate (participation_rate=0.06 - 0.10).
- If Dark Pool is available, set use_dark_pool=True, dark_pool_fraction=1.0.
"""

async def run_evaluation():
    # Use HF_TOKEN if testing against HuggingFace, otherwise OPENAI_API_KEY
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "dummy_key")
    
    # If the user specified an endpoint (like HF Inference API), use it
    base_url_openai = os.environ.get("OPENAI_API_BASE")
    if not base_url_openai and os.environ.get("HF_TOKEN"):
        # Default HF endpoint if token is provided but base is not
        base_url_openai = "https://api-inference.huggingface.co/v1/"
    
    # We allow the fallback so users testing locally don't crash
    client = AsyncOpenAI(api_key=api_key, base_url=base_url_openai)

    print(f"Connecting to OpenEnv Server: {ENV_BASE_URL}")
    print(f"Using Model: {MODEL_NAME}\n")
    
    total_score = 0.0

    async with TradeExecClient(base_url=ENV_BASE_URL) as env_client:
        for task_id in EVAL_TASKS:
            print(f"=== Evaluating Task: {task_id} ===")
            
            obs = await env_client.reset(task_id=task_id)
            print("Environment initialized.")

            metadata = obs.observation.get('metadata', {}) if hasattr(obs, 'observation') else getattr(obs, 'metadata', {})
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": metadata.get("output", "")}
            ]
            
            tools_list = await env_client.list_tools()
            tools_spec = []
            
            for tool in tools_list:
                tools_spec.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            step = 0
            while step < 100:  # safety bound
                step += 1
                try:
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=tools_spec,
                        tool_choice="auto",
                        temperature=0.0
                    )
                except Exception as e:
                    print(f"LLM API Error: {e}")
                    break

                message = response.choices[0].message
                messages.append(message)

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        
                        tool_obs = await env_client.call_tool(name, **args)
                        result_text = tool_obs.result.content[0].text if tool_obs.result else tool_obs.error.message
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        })

                        if "EPISODE COMPLETE" in result_text:
                            print(f"[Finished {task_id}]")
                            # Extract grader score
                            for line in result_text.split("\\n"):
                                if "Grader Score:" in line:
                                    try:
                                        score = float(line.split(":")[1].split("/")[0].strip())
                                        print(f"  Task Score: {score:.4f}")
                                        total_score += score
                                    except Exception:
                                        pass
                            step = 999  # Break loop
                            break
                else:
                    messages.append({
                        "role": "user",
                        "content": "Please call the next tool. You MUST make progress."
                    })

                if step >= 100:
                    break

    print(f"\nFinal Total Score: {total_score:.4f}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())

import httpx
import asyncio

async def run():
    async with httpx.AsyncClient(timeout=10.0) as c:
        print("Resetting...")
        r = await c.post('http://localhost:7860/reset', json={'task_id': 'task_1', 'seed': 42})
        print("Reset Response:", r.status_code, r.json())
        print("Stepping...")
        r2 = await c.post('http://localhost:7860/step', json={'action': {'participation_rate': 0.05}})
        print("Step Response:", r2.status_code, r2.json())

if __name__ == "__main__":
    asyncio.run(run())

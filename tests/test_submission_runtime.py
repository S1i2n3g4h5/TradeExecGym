from __future__ import annotations

from collections import namedtuple

from fastapi.testclient import TestClient

import inference
import server.app as server_app
from models import YourRlObservation


def test_grade_endpoint_uses_live_env_state():
    server_app.GLOBAL_ENV = None
    client = TestClient(server_app.app)

    before = client.get("/grade/task_1").json()
    client.post("/reset", json={"task_id": "task_1", "seed": 42})
    client.post("/step", json={"action": {"participation_rate": 0.05}})
    after = client.get("/grade/task_1").json()

    assert before == {"score": 0.01, "reward": 0.01}
    assert after["score"] != before["score"]
    assert after["score"] == after["reward"]


def test_run_task_uses_task_specific_step_budget(monkeypatch):
    Result = namedtuple("Result", ["observation", "done", "reward"])

    class FakeSyncEnv:
        def __init__(self) -> None:
            self.max_steps = 60
            self.step_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def reset(self, task_id=None, seed=None):
            obs = YourRlObservation.model_validate(
                {
                    "text_summary": "",
                    "task": {
                        "task_id": task_id or "task_2",
                        "description": "Synthetic task for step-budget validation",
                    },
                    "info": {"max_steps": self.max_steps},
                    "reward": 0.0,
                    "step": 0,
                    "task_achieved": False,
                }
            )
            return Result(observation=obs, done=False, reward=0.0)

        def step(self, action):
            self.step_calls += 1
            done = self.step_calls >= self.max_steps
            obs = YourRlObservation.model_validate(
                {
                    "text_summary": "",
                    "task": {
                        "task_id": "task_2",
                        "description": "Synthetic task for step-budget validation",
                    },
                    "info": {"max_steps": self.max_steps},
                    "reward": 0.5,
                    "step": self.step_calls,
                    "task_achieved": False,
                }
            )
            return Result(observation=obs, done=done, reward=0.5)

    class FakeClient:
        last_env: FakeSyncEnv | None = None

        def __init__(self, base_url: str):
            self.base_url = base_url

        def sync(self):
            self.__class__.last_env = FakeSyncEnv()
            return self.__class__.last_env

    monkeypatch.setattr(inference, "YourRlEnv", FakeClient)
    monkeypatch.setattr(inference, "_build_llm_client", lambda: None)
    monkeypatch.setattr(inference, "REQUIRE_LLM_PROXY", False)
    monkeypatch.delenv("MAX_STEPS", raising=False)

    inference.run_task("http://example.test", task_id="task_2")

    assert FakeClient.last_env is not None
    assert FakeClient.last_env.step_calls == 60


def test_resolve_task_step_limit_supports_http_observation_shape(monkeypatch):
    monkeypatch.delenv("MAX_STEPS", raising=False)
    obs = YourRlObservation.model_validate(
        {
            "text_summary": "",
            "task": {"task_id": "task_2", "description": "desc"},
            "info": {"task_id": "task_2"},
            "max_steps": 60,
        }
    )

    assert inference._resolve_task_step_limit(obs) == 60

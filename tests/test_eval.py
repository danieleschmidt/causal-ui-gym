"""Tests for EvalHarness and EvalResult."""

import pytest
from causal_ui_gym import (
    CheckoutFlow,
    EvalHarness,
    EvalResult,
    FormValidationFlow,
    GreedyUIAgent,
    LoginFlow,
    RandomUIAgent,
    UIAction,
)
from causal_ui_gym.eval import EpisodeResult
from causal_ui_gym.core import UIState


class TestEpisodeResult:
    def _ep(self, success=True, steps=5, optimal=None):
        info = {}
        if optimal is not None:
            info["optimal_steps"] = optimal
        return EpisodeResult(
            task_name="test",
            success=success,
            steps=steps,
            total_reward=1.0 if success else -0.5,
            final_state=UIState(),
            info=info,
        )

    def test_path_efficiency_with_optimal(self):
        ep = self._ep(steps=5, optimal=3)
        assert ep.path_efficiency == pytest.approx(3 / 5)

    def test_path_efficiency_without_optimal(self):
        ep = self._ep(steps=5, optimal=None)
        assert ep.path_efficiency is None

    def test_path_efficiency_perfect(self):
        ep = self._ep(steps=3, optimal=3)
        assert ep.path_efficiency == pytest.approx(1.0)


class TestEvalResult:
    def _make_result(self, successes, steps_list=None):
        if steps_list is None:
            steps_list = [5] * len(successes)
        episodes = [
            EpisodeResult(
                task_name="task",
                success=s,
                steps=st,
                total_reward=1.0 if s else 0.0,
                final_state=UIState(),
            )
            for s, st in zip(successes, steps_list)
        ]
        return EvalResult(agent_name="TestAgent", episodes=episodes)

    def test_success_rate_all_success(self):
        r = self._make_result([True, True, True])
        assert r.success_rate == pytest.approx(1.0)

    def test_success_rate_none(self):
        r = self._make_result([False, False])
        assert r.success_rate == pytest.approx(0.0)

    def test_success_rate_mixed(self):
        r = self._make_result([True, False, True, False])
        assert r.success_rate == pytest.approx(0.5)

    def test_mean_steps(self):
        r = self._make_result([True, False], steps_list=[4, 6])
        assert r.mean_steps == pytest.approx(5.0)

    def test_empty_result(self):
        r = EvalResult(agent_name="empty")
        assert r.success_rate == 0.0
        assert r.mean_steps == 0.0

    def test_by_task(self):
        ep1 = EpisodeResult("taskA", True, 3, 1.0, UIState())
        ep2 = EpisodeResult("taskB", False, 7, -0.5, UIState())
        ep3 = EpisodeResult("taskA", False, 5, 0.0, UIState())
        r = EvalResult(agent_name="X", episodes=[ep1, ep2, ep3])
        by_task = r.by_task()
        assert set(by_task.keys()) == {"taskA", "taskB"}
        assert len(by_task["taskA"].episodes) == 2

    def test_summary_contains_agent_name(self):
        r = self._make_result([True, False])
        s = r.summary()
        assert "TestAgent" in s

    def test_mean_reward(self):
        r = self._make_result([True, False])
        assert r.mean_reward == pytest.approx(0.5)


class TestEvalHarness:
    def test_run_returns_eval_result(self):
        harness = EvalHarness(n_episodes=3)
        agent = RandomUIAgent(seed=0)
        task = LoginFlow()
        result = harness.run(agent, task)
        assert isinstance(result, EvalResult)
        assert len(result.episodes) == 3

    def test_episode_count(self):
        for n in [1, 5, 10]:
            harness = EvalHarness(n_episodes=n)
            result = harness.run(RandomUIAgent(seed=0), LoginFlow())
            assert len(result.episodes) == n

    def test_run_many(self):
        harness = EvalHarness(n_episodes=2)
        agent = RandomUIAgent(seed=0)
        tasks = [LoginFlow(), CheckoutFlow(), FormValidationFlow()]
        result = harness.run_many(agent, tasks)
        assert len(result.episodes) == 6

    def test_optimal_agent_on_login(self):
        """
        A hand-crafted oracle agent should always succeed on LoginFlow.
        """
        from causal_ui_gym.agents import UIAgent

        class OracleLoginAgent(UIAgent):
            def __init__(self):
                super().__init__()
                self._step = 0

            def reset(self):
                self._step = 0

            def act(self, state, available_actions):
                self._step += 1
                plan = [
                    UIAction("fill_username", {"value": "alice"}),
                    UIAction("fill_password", {"value": "hunter2"}),
                    UIAction("click_submit"),
                ]
                if self._step <= len(plan):
                    return plan[self._step - 1]
                return available_actions[0]

        harness = EvalHarness(n_episodes=5)
        result = harness.run(OracleLoginAgent(), LoginFlow())
        assert result.success_rate == pytest.approx(1.0)

    def test_agent_name_in_result(self):
        harness = EvalHarness(n_episodes=1)
        result = harness.run(RandomUIAgent(seed=0), LoginFlow())
        assert result.agent_name == "RandomUIAgent"

    def test_verbose_runs_without_error(self, capsys):
        harness = EvalHarness(n_episodes=2, verbose=True)
        harness.run(RandomUIAgent(seed=0), LoginFlow())
        captured = capsys.readouterr()
        assert "RandomUIAgent" in captured.out

    def test_bare_env_wrapped(self):
        """EvalHarness should accept a plain CausalUIEnvironment."""
        from causal_ui_gym.environments.login import _LoginEnv
        harness = EvalHarness(n_episodes=2)
        result = harness.run(RandomUIAgent(seed=0), _LoginEnv())
        assert len(result.episodes) == 2

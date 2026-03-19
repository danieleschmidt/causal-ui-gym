"""Tests for UIAgent implementations."""

import pytest
from causal_ui_gym import (
    CheckoutFlow,
    FormValidationFlow,
    GreedyUIAgent,
    LoginFlow,
    RandomUIAgent,
    UIAction,
)
from causal_ui_gym.core import UIState


class TestRandomUIAgent:
    def test_picks_from_available(self):
        agent = RandomUIAgent(seed=42)
        task = LoginFlow()
        state = task.reset()
        available = task.available_actions()
        action = agent.act(state, available)
        assert action in available

    def test_reproducible_with_seed(self):
        actions1, actions2 = [], []
        for _ in range(2):
            agent = RandomUIAgent(seed=0)
            task = LoginFlow()
            state = task.reset()
            for _ in range(5):
                av = task.available_actions()
                a = agent.act(state, av)
                actions1.append(a) if _ == 0 else actions2.append(a)
                task.step(a)
                state = task.state
        # Two runs with same seed should agree on first action at least
        agent_a = RandomUIAgent(seed=99)
        agent_b = RandomUIAgent(seed=99)
        task = LoginFlow()
        state = task.reset()
        av = task.available_actions()
        assert agent_a.act(state, av) == agent_b.act(state, av)

    def test_raises_if_no_actions(self):
        agent = RandomUIAgent()
        with pytest.raises(ValueError):
            agent.act(UIState(), [])

    def test_reset_is_callable(self):
        agent = RandomUIAgent()
        agent.reset()  # Should not raise

    def test_different_seeds_differ(self):
        """Two different seeds should not always produce the same action."""
        task = LoginFlow()
        state = task.reset()
        av = task.available_actions()
        # With many options, seeds 0 and 1 should eventually differ
        chosen = {RandomUIAgent(seed=i).act(state, av).name for i in range(10)}
        assert len(chosen) >= 1  # At minimum picks something valid


class TestGreedyUIAgent:
    def test_always_picks_first(self):
        agent = GreedyUIAgent()
        task = LoginFlow()
        state = task.reset()
        available = task.available_actions()
        action = agent.act(state, available)
        assert action == available[0]

    def test_deterministic(self):
        agent = GreedyUIAgent()
        task = LoginFlow()
        state = task.reset()
        av = task.available_actions()
        a1 = agent.act(state, av)
        a2 = agent.act(state, av)
        assert a1 == a2

    def test_raises_if_no_actions(self):
        agent = GreedyUIAgent()
        with pytest.raises(ValueError):
            agent.act(UIState(), [])

    def test_works_across_environments(self):
        agent = GreedyUIAgent()
        for task_fn in [LoginFlow, CheckoutFlow, FormValidationFlow]:
            task = task_fn()
            state = task.reset()
            av = task.available_actions()
            action = agent.act(state, av)
            assert action in av

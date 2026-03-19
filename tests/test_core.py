"""Tests for core UIState, UIAction, ActionResult, CausalUIEnvironment, CausalUITask."""

import pytest
from causal_ui_gym.core import (
    ActionResult,
    CausalUIEnvironment,
    CausalUITask,
    UIAction,
    UIState,
)


# ---------------------------------------------------------------------------
# UIState
# ---------------------------------------------------------------------------

class TestUIState:
    def test_defaults(self):
        s = UIState()
        assert s.screen == "start"
        assert s.fields == {}
        assert s.flags == {}
        assert s.errors == []
        assert s.history == []

    def test_with_screen_immutable(self):
        s = UIState(screen="login")
        s2 = s.with_screen("dashboard")
        assert s.screen == "login"
        assert s2.screen == "dashboard"

    def test_with_field(self):
        s = UIState()
        s2 = s.with_field("username", "alice")
        assert s.fields == {}
        assert s2.fields == {"username": "alice"}

    def test_with_flag(self):
        s = UIState()
        s2 = s.with_flag("logged_in", True)
        assert not s.has_flag("logged_in")
        assert s2.has_flag("logged_in")

    def test_with_error(self):
        s = UIState()
        s2 = s.with_error("Oops")
        assert s.errors == []
        assert s2.errors == ["Oops"]

    def test_clear_errors(self):
        s = UIState(errors=["A", "B"])
        s2 = s.clear_errors()
        assert s2.errors == []
        assert s.errors == ["A", "B"]  # original unchanged

    def test_record_action(self):
        s = UIState(screen="login")
        s2 = s.record_action("click_submit")
        assert s.history == []
        assert s2.history == [("click_submit", "login")]

    def test_get_field_default(self):
        s = UIState()
        assert s.get_field("username") == ""
        assert s.get_field("username", "anon") == "anon"

    def test_chaining(self):
        s = UIState()
        s2 = (
            s.with_screen("dashboard")
            .with_flag("logged_in", True)
            .with_field("user", "alice")
        )
        assert s2.screen == "dashboard"
        assert s2.has_flag("logged_in")
        assert s2.get_field("user") == "alice"


# ---------------------------------------------------------------------------
# UIAction
# ---------------------------------------------------------------------------

class TestUIAction:
    def test_str_no_payload(self):
        a = UIAction("click_submit")
        assert str(a) == "click_submit"

    def test_str_with_payload(self):
        a = UIAction("fill_username", {"value": "alice"})
        assert "fill_username" in str(a)

    def test_equality(self):
        a1 = UIAction("click", {"x": 1})
        a2 = UIAction("click", {"x": 1})
        a3 = UIAction("click", {"x": 2})
        assert a1 == a2
        assert a1 != a3

    def test_hashable(self):
        a = UIAction("click_submit")
        s = {a}
        assert a in s


# ---------------------------------------------------------------------------
# CausalUIEnvironment (via minimal concrete implementation)
# ---------------------------------------------------------------------------

class _TwoStepEnv(CausalUIEnvironment):
    """Minimal env: step1 → step2 → done."""
    name = "TwoStepEnv"
    max_steps = 5

    def _initial_state(self) -> UIState:
        return UIState(screen="step1")

    def _available_actions(self, state: UIState):
        if state.screen == "step1":
            return [UIAction("next")]
        if state.screen == "step2":
            return [UIAction("finish")]
        return []

    def _transition(self, state: UIState, action: UIAction) -> ActionResult:
        state = state.record_action(action.name)
        if action.name == "next":
            return ActionResult(new_state=state.with_screen("step2"), reward=0.5)
        if action.name == "finish":
            s = state.with_screen("done").with_flag("finished", True)
            return ActionResult(new_state=s, reward=1.0, done=True, success=True)
        return ActionResult(new_state=state, reward=0.0)


class TestCausalUIEnvironment:
    def test_reset_required(self):
        env = _TwoStepEnv()
        with pytest.raises(RuntimeError):
            env.step(UIAction("next"))

    def test_initial_state(self):
        env = _TwoStepEnv()
        state = env.reset()
        assert state.screen == "step1"

    def test_valid_transition(self):
        env = _TwoStepEnv()
        env.reset()
        result = env.step(UIAction("next"))
        assert result.new_state.screen == "step2"
        assert result.reward == 0.5

    def test_full_episode(self):
        env = _TwoStepEnv()
        env.reset()
        env.step(UIAction("next"))
        result = env.step(UIAction("finish"))
        assert result.done
        assert result.success
        assert result.new_state.has_flag("finished")

    def test_invalid_action_penalised(self):
        env = _TwoStepEnv()
        env.reset()
        # "finish" is only available on step2
        result = env.step(UIAction("finish"))
        assert result.reward < 0
        assert not result.done
        assert len(result.new_state.errors) > 0

    def test_max_steps_terminal(self):
        env = _TwoStepEnv()
        env.reset()
        # Exhaust steps by repeatedly trying invalid actions
        for _ in range(env.max_steps):
            env.step(UIAction("next"))
        result = env.step(UIAction("next"))
        assert result.done
        assert not result.success

    def test_available_actions_before_reset(self):
        env = _TwoStepEnv()
        with pytest.raises(RuntimeError):
            env.available_actions()

    def test_steps_counter(self):
        env = _TwoStepEnv()
        env.reset()
        assert env.steps == 0
        env.step(UIAction("next"))
        assert env.steps == 1

    def test_state_property(self):
        env = _TwoStepEnv()
        env.reset()
        state = env.state
        assert state.screen == "step1"


# ---------------------------------------------------------------------------
# CausalUITask
# ---------------------------------------------------------------------------

class TestCausalUITask:
    def _make_task(self):
        env = _TwoStepEnv()
        return CausalUITask(
            env=env,
            goal="Reach done screen",
            is_goal=lambda s: s.screen == "done",
        )

    def test_reset(self):
        task = self._make_task()
        state = task.reset()
        assert state.screen == "step1"

    def test_goal_not_met_mid_episode(self):
        task = self._make_task()
        task.reset()
        result = task.step(UIAction("next"))
        assert not result.done
        assert not result.success

    def test_goal_met(self):
        task = self._make_task()
        task.reset()
        task.step(UIAction("next"))
        result = task.step(UIAction("finish"))
        assert result.done
        assert result.success

    def test_name_and_max_steps(self):
        task = self._make_task()
        assert task.name == "TwoStepEnv"
        assert task.max_steps == 5

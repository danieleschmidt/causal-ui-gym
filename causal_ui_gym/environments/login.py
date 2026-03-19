"""
LoginFlow — a causal UI environment modelling username/password authentication.

Causal structure
----------------
fill_username → fill_password → click_submit → [logged_in]

Causal side-effects enforced:
- click_submit without username  → validation error, stay on login screen
- click_submit without password  → validation error, stay on login screen
- click_submit with wrong creds  → auth error, clear password field
- click_submit with right creds  → navigate to dashboard, set logged_in flag

Optimal path: fill_username → fill_password → click_submit  (3 steps)
"""

from __future__ import annotations

from typing import List

from ..core import (
    ActionResult,
    CausalUIEnvironment,
    CausalUITask,
    UIAction,
    UIState,
)

_VALID_CREDENTIALS = {
    "alice": "hunter2",
    "bob": "passw0rd",
}

_OPTIMAL_STEPS = 3


class _LoginEnv(CausalUIEnvironment):
    name = "LoginFlow"
    max_steps = 15

    def _initial_state(self) -> UIState:
        return UIState(
            screen="login",
            fields={"username": "", "password": ""},
            flags={"logged_in": False},
            errors=[],
            metadata={"optimal_steps": _OPTIMAL_STEPS},
        )

    def _available_actions(self, state: UIState) -> List[UIAction]:
        if state.screen == "login":
            actions: List[UIAction] = [
                UIAction("fill_username", {"value": "alice"}),
                UIAction("fill_username", {"value": "bob"}),
                UIAction("fill_username", {"value": "wrong_user"}),
                UIAction("fill_password", {"value": "hunter2"}),
                UIAction("fill_password", {"value": "passw0rd"}),
                UIAction("fill_password", {"value": "bad_pass"}),
                UIAction("click_submit"),
            ]
            return actions
        if state.screen == "dashboard":
            return [UIAction("click_logout")]
        return []

    def _transition(self, state: UIState, action: UIAction) -> ActionResult:
        state = state.record_action(action.name)
        state = state.clear_errors()

        if action.name == "fill_username":
            new_state = state.with_field("username", action.payload.get("value", ""))
            return ActionResult(new_state=new_state, reward=0.1)

        if action.name == "fill_password":
            new_state = state.with_field("password", action.payload.get("value", ""))
            return ActionResult(new_state=new_state, reward=0.1)

        if action.name == "click_submit":
            username = state.get_field("username")
            password = state.get_field("password")

            if not username:
                s = state.with_error("Username is required.")
                return ActionResult(new_state=s, reward=-0.5)

            if not password:
                s = state.with_error("Password is required.")
                return ActionResult(new_state=s, reward=-0.5)

            expected = _VALID_CREDENTIALS.get(username)
            if expected is None or password != expected:
                s = state.with_error("Invalid credentials.").with_field("password", "")
                return ActionResult(new_state=s, reward=-0.5)

            # Correct credentials — navigate to dashboard
            s = (
                state
                .with_screen("dashboard")
                .with_flag("logged_in", True)
            )
            return ActionResult(new_state=s, reward=1.0, done=True, success=True,
                                info={"optimal_steps": _OPTIMAL_STEPS})

        if action.name == "click_logout":
            s = (
                state
                .with_screen("login")
                .with_flag("logged_in", False)
                .with_field("username", "")
                .with_field("password", "")
            )
            return ActionResult(new_state=s, reward=0.0)

        return ActionResult(new_state=state, reward=-0.1,
                            info={"reason": "unhandled_action"})


def LoginFlow() -> CausalUITask:
    """
    Return a CausalUITask wrapping the login environment.

    Goal: reach the dashboard screen with logged_in=True.
    """
    env = _LoginEnv()
    return CausalUITask(
        env=env,
        goal="Log in successfully and reach the dashboard.",
        is_goal=lambda s: s.screen == "dashboard" and s.has_flag("logged_in"),
    )

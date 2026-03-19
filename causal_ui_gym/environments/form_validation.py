"""
FormValidationFlow — causal UI environment for a multi-field form with validation.

All fields must be valid before submission succeeds.  Each field has its own
validation rule, and submitting too early reveals exactly which fields are
invalid (causal side-effect: submit → validation run → error display).

Fields
------
  email     — must contain "@"
  age       — must be a digit string, 18 ≤ age ≤ 120
  username  — 3–20 alphanumeric chars

Causal structure
----------------
Submitting with any invalid field:
  → errors appear for those fields
  → form stays on "form" screen

Submitting with all fields valid:
  → navigate to "success" screen, set form_submitted flag

Optimal path: fill_email → fill_age → fill_username → click_submit  (4 steps)
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

_OPTIMAL_STEPS = 4


def _validate_email(val: str) -> str:
    """Return error string or empty string if valid."""
    if not val or "@" not in val:
        return "Email must contain '@'."
    return ""


def _validate_age(val: str) -> str:
    if not val.isdigit():
        return "Age must be a number."
    age = int(val)
    if not (18 <= age <= 120):
        return "Age must be between 18 and 120."
    return ""


def _validate_username(val: str) -> str:
    if not (3 <= len(val) <= 20):
        return "Username must be 3–20 characters."
    if not val.isalnum():
        return "Username must be alphanumeric."
    return ""


class _FormEnv(CausalUIEnvironment):
    name = "FormValidationFlow"
    max_steps = 20

    def _initial_state(self) -> UIState:
        return UIState(
            screen="form",
            fields={"email": "", "age": "", "username": ""},
            flags={"form_submitted": False},
            errors=[],
            metadata={"optimal_steps": _OPTIMAL_STEPS},
        )

    def _available_actions(self, state: UIState) -> List[UIAction]:
        if state.screen == "form":
            return [
                # email options
                UIAction("fill_email", {"value": "alice@example.com"}),
                UIAction("fill_email", {"value": "not-an-email"}),
                UIAction("fill_email", {"value": ""}),
                # age options
                UIAction("fill_age", {"value": "25"}),
                UIAction("fill_age", {"value": "17"}),  # underage
                UIAction("fill_age", {"value": "abc"}),  # non-numeric
                # username options
                UIAction("fill_username", {"value": "alice"}),
                UIAction("fill_username", {"value": "ab"}),       # too short
                UIAction("fill_username", {"value": "alice!"}),   # invalid char
                # submit
                UIAction("click_submit"),
            ]
        if state.screen == "success":
            return [UIAction("reset_form")]
        return []

    def _transition(self, state: UIState, action: UIAction) -> ActionResult:
        state = state.record_action(action.name)
        state = state.clear_errors()

        if action.name == "fill_email":
            val = action.payload.get("value", "")
            s = state.with_field("email", val)
            err = _validate_email(val)
            reward = 0.2 if not err else -0.1
            return ActionResult(new_state=s, reward=reward)

        if action.name == "fill_age":
            val = action.payload.get("value", "")
            s = state.with_field("age", val)
            err = _validate_age(val)
            reward = 0.2 if not err else -0.1
            return ActionResult(new_state=s, reward=reward)

        if action.name == "fill_username":
            val = action.payload.get("value", "")
            s = state.with_field("username", val)
            err = _validate_username(val)
            reward = 0.2 if not err else -0.1
            return ActionResult(new_state=s, reward=reward)

        if action.name == "click_submit":
            errors = []
            email_err = _validate_email(state.get_field("email"))
            age_err = _validate_age(state.get_field("age"))
            uname_err = _validate_username(state.get_field("username"))
            if email_err:
                errors.append(email_err)
            if age_err:
                errors.append(age_err)
            if uname_err:
                errors.append(uname_err)

            if errors:
                s = state
                for e in errors:
                    s = s.with_error(e)
                return ActionResult(new_state=s, reward=-0.5 * len(errors),
                                    info={"validation_errors": errors})

            # All valid
            s = (
                state
                .with_screen("success")
                .with_flag("form_submitted", True)
            )
            return ActionResult(new_state=s, reward=1.0, done=True, success=True,
                                info={"optimal_steps": _OPTIMAL_STEPS})

        if action.name == "reset_form":
            s = self._initial_state()
            return ActionResult(new_state=s, reward=0.0)

        return ActionResult(new_state=state, reward=-0.1,
                            info={"reason": "unhandled_action"})


def FormValidationFlow() -> CausalUITask:
    """
    Return a CausalUITask wrapping the form-validation environment.

    Goal: reach the success screen with form_submitted=True.
    """
    env = _FormEnv()
    return CausalUITask(
        env=env,
        goal="Fill all fields correctly and submit the form.",
        is_goal=lambda s: s.screen == "success" and s.has_flag("form_submitted"),
    )

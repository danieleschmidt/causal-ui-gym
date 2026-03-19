"""
Core data structures and base classes for Causal UI Gym.

Design notes:
- UIState is immutable (frozen dataclass) — all transitions return new states.
- CausalUIEnvironment defines a causal graph over UI actions:
    action → trigger side-effects → new UIState
- Wrong-order actions fail (causal precedence is enforced).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# UIState
# ---------------------------------------------------------------------------

@dataclass
class UIState:
    """
    Snapshot of a UI state machine.

    Attributes
    ----------
    screen : str
        Current screen / page name (e.g. "login", "checkout_shipping").
    fields : dict
        Form field values keyed by field name. Values are strings.
    flags : dict
        Boolean-like state flags (e.g. {"logged_in": True, "cart_filled": False}).
    errors : list
        Active error messages displayed to the user.
    history : list
        Sequence of (action_name, screen) tuples visited so far (for path analysis).
    metadata : dict
        Arbitrary environment-specific data.
    """

    screen: str = "start"
    fields: Dict[str, str] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    history: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def with_screen(self, screen: str) -> "UIState":
        """Return a copy with an updated screen."""
        s = copy.deepcopy(self)
        s.screen = screen
        return s

    def with_field(self, name: str, value: str) -> "UIState":
        """Return a copy with one field set."""
        s = copy.deepcopy(self)
        s.fields[name] = value
        return s

    def with_flag(self, name: str, value: bool = True) -> "UIState":
        """Return a copy with a flag set."""
        s = copy.deepcopy(self)
        s.flags[name] = value
        return s

    def with_error(self, msg: str) -> "UIState":
        """Return a copy with an error appended."""
        s = copy.deepcopy(self)
        s.errors.append(msg)
        return s

    def clear_errors(self) -> "UIState":
        """Return a copy with all errors cleared."""
        s = copy.deepcopy(self)
        s.errors = []
        return s

    def record_action(self, action_name: str) -> "UIState":
        """Return a copy with this action appended to the history."""
        s = copy.deepcopy(self)
        s.history.append((action_name, self.screen))
        return s

    def has_flag(self, name: str) -> bool:
        return bool(self.flags.get(name, False))

    def get_field(self, name: str, default: str = "") -> str:
        return self.fields.get(name, default)


# ---------------------------------------------------------------------------
# UIAction
# ---------------------------------------------------------------------------

@dataclass
class UIAction:
    """
    A discrete action an agent can take in the UI.

    Attributes
    ----------
    name : str
        Unique action identifier (e.g. "click_submit", "fill_username").
    payload : dict
        Optional data carried by the action (e.g. {"value": "alice"} for fill).
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.payload:
            return f"{self.name}({self.payload})"
        return self.name

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.payload.items()))))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UIAction):
            return NotImplemented
        return self.name == other.name and self.payload == other.payload


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """
    Result of executing an action in an environment.

    Attributes
    ----------
    new_state : UIState
        State after the action.
    reward : float
        Step reward (positive = progress, negative = mistake, 0 = no-op).
    done : bool
        Whether the task is complete (goal reached or terminal failure).
    success : bool
        True iff the agent reached the goal (only meaningful when done=True).
    info : dict
        Diagnostic information (causal side-effects, messages, etc.).
    """

    new_state: UIState
    reward: float = 0.0
    done: bool = False
    success: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CausalUIEnvironment
# ---------------------------------------------------------------------------

class CausalUIEnvironment:
    """
    Base class for causal UI environments.

    Subclasses implement:
        _initial_state() → UIState
        _available_actions(state) → List[UIAction]
        _transition(state, action) → ActionResult

    The causal structure is encoded in _transition: some actions only succeed
    when certain preconditions hold (flags, filled fields, current screen).
    Attempting actions out of causal order returns a penalty and an error state.
    """

    # Override in subclass
    name: str = "BaseEnvironment"
    max_steps: int = 30

    def __init__(self) -> None:
        self._state: Optional[UIState] = None
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> UIState:
        """Reset to initial state and return it."""
        self._state = self._initial_state()
        self._steps = 0
        return copy.deepcopy(self._state)

    def step(self, action: UIAction) -> ActionResult:
        """
        Execute *action* in the current state.

        Returns ActionResult with the new state, reward, done flag, etc.
        Raises RuntimeError if reset() hasn't been called.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._steps += 1

        # Check step limit
        if self._steps > self.max_steps:
            result = ActionResult(
                new_state=self._state,
                reward=-1.0,
                done=True,
                success=False,
                info={"reason": "max_steps_exceeded"},
            )
            return result

        # Validate action is available
        available = self._available_actions(self._state)
        available_names = {a.name for a in available}
        if action.name not in available_names:
            new_state = self._state.with_error(
                f"Action '{action.name}' is not available on screen '{self._state.screen}'."
            ).record_action(action.name)
            result = ActionResult(
                new_state=new_state,
                reward=-0.5,
                done=False,
                success=False,
                info={"reason": "invalid_action", "available": list(available_names)},
            )
            self._state = new_state
            return result

        result = self._transition(self._state, action)
        self._state = result.new_state
        return result

    def available_actions(self) -> List[UIAction]:
        """Return actions available in the current state."""
        if self._state is None:
            raise RuntimeError("Call reset() before available_actions().")
        return self._available_actions(self._state)

    @property
    def state(self) -> UIState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return copy.deepcopy(self._state)

    @property
    def steps(self) -> int:
        return self._steps

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _initial_state(self) -> UIState:
        raise NotImplementedError

    def _available_actions(self, state: UIState) -> List[UIAction]:
        raise NotImplementedError

    def _transition(self, state: UIState, action: UIAction) -> ActionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CausalUITask
# ---------------------------------------------------------------------------

class CausalUITask:
    """
    Wraps a CausalUIEnvironment with a goal description and goal-check function.

    Usage
    -----
        task = CausalUITask(
            env=LoginFlow(),
            goal="Log in successfully",
            is_goal=lambda s: s.has_flag("logged_in"),
        )
        state = task.reset()
        result = task.step(action)  # ActionResult with done=True when goal reached
    """

    def __init__(
        self,
        env: CausalUIEnvironment,
        goal: str,
        is_goal: Callable[[UIState], bool],
    ) -> None:
        self.env = env
        self.goal = goal
        self.is_goal = is_goal

    def reset(self) -> UIState:
        return self.env.reset()

    def step(self, action: UIAction) -> ActionResult:
        result = self.env.step(action)
        if self.is_goal(result.new_state):
            result.done = True
            result.success = True
            result.reward = max(result.reward, 1.0)
        return result

    def available_actions(self) -> List[UIAction]:
        return self.env.available_actions()

    @property
    def state(self) -> UIState:
        return self.env.state

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def name(self) -> str:
        return self.env.name

    @property
    def max_steps(self) -> int:
        return self.env.max_steps

# Causal UI Gym

A pure-Python gym for testing agent **causal reasoning in UI-like decision tasks**.

Agents navigate UI flows (login, checkout, form validation) where actions have **causal side-effects** — clicking "submit" before filling required fields fails, not because of a rule check, but because the causal preconditions aren't met. Getting to the goal requires understanding the causal order of operations.

No React. No JAX. No Docker. Just Python.

---

## Install

```bash
pip install -e .
```

Requires Python 3.9+ (stdlib only, no third-party dependencies).

---

## Quick Start

```python
from causal_ui_gym import LoginFlow, RandomUIAgent, GreedyUIAgent, EvalHarness

# Run a random agent on the login environment
harness = EvalHarness(n_episodes=20, seed=42, verbose=True)
result = harness.run(RandomUIAgent(seed=0), LoginFlow())
print(result.summary())
```

```
Agent        : RandomUIAgent
Episodes     : 20
Success rate : 25.0%
Mean steps   : 14.3
Mean reward  : 0.128
```

---

## Core Concepts

### `UIState`

Immutable snapshot of a UI state machine.

```python
from causal_ui_gym import UIState

s = UIState(screen="login", fields={"username": "", "password": ""})
s2 = s.with_field("username", "alice").with_screen("dashboard")
# s is unchanged — all methods return new states
```

### `CausalUIEnvironment`

Base class for environments. Subclass it and implement:
- `_initial_state()` → starting `UIState`
- `_available_actions(state)` → `List[UIAction]`
- `_transition(state, action)` → `ActionResult`

The causal structure lives in `_transition`: actions only succeed when preconditions hold.

### `CausalUITask`

Wraps an environment with a goal condition:

```python
from causal_ui_gym import CausalUITask

task = CausalUITask(
    env=my_env,
    goal="Reach the dashboard",
    is_goal=lambda s: s.screen == "dashboard",
)
```

### `UIAgent`

```python
from causal_ui_gym import UIAgent, UIAction, UIState
from typing import List

class MyAgent(UIAgent):
    def act(self, state: UIState, available_actions: List[UIAction]) -> UIAction:
        # your logic here
        ...
```

### `EvalHarness`

```python
harness = EvalHarness(n_episodes=50)
result = harness.run(agent, task)
result.success_rate      # 0.0–1.0
result.mean_steps        # avg steps per episode
result.mean_path_efficiency  # optimal_steps / actual_steps
result.summary()         # human-readable summary
```

---

## Environments

### `LoginFlow`

- **Screen flow:** `login → dashboard`
- **Causal structure:** must fill username, then password, then submit
- **Failure modes:** missing fields, wrong credentials, out-of-order submit
- **Optimal steps:** 3

```python
task = LoginFlow()
state = task.reset()
# Available: fill_username, fill_password, click_submit
```

### `CheckoutFlow`

- **Screen flow:** `cart → shipping → payment → review → confirmation`
- **Causal structure:** each screen gate depends on completing the prior one
- **Failure modes:** empty cart, missing address, invalid card, skipping screens
- **Optimal steps:** 7

```python
task = CheckoutFlow()
```

### `FormValidationFlow`

- **Screen:** single `form` screen with three fields (email, age, username)
- **Causal structure:** each field has validation rules; submit triggers a causal validation pass
- **Failure modes:** invalid email, underage user, non-alphanumeric username
- **Optimal steps:** 4

```python
task = FormValidationFlow()
```

---

## Running All Environments

```python
from causal_ui_gym import (
    LoginFlow, CheckoutFlow, FormValidationFlow,
    RandomUIAgent, EvalHarness,
)

harness = EvalHarness(n_episodes=30)
agent = RandomUIAgent(seed=42)

for task_fn in [LoginFlow, CheckoutFlow, FormValidationFlow]:
    result = harness.run(agent, task_fn())
    print(result.summary())
    print()
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

85 tests, 0 dependencies, ~0.1s.

---

## Extending

Add a new environment in `causal_ui_gym/environments/`:

```python
from causal_ui_gym.core import CausalUIEnvironment, CausalUITask, UIState, UIAction, ActionResult
from typing import List

class MyFlow(CausalUIEnvironment):
    name = "MyFlow"
    max_steps = 20

    def _initial_state(self) -> UIState:
        return UIState(screen="start")

    def _available_actions(self, state: UIState) -> List[UIAction]:
        return [UIAction("do_thing")]

    def _transition(self, state: UIState, action: UIAction) -> ActionResult:
        state = state.record_action(action.name)
        if action.name == "do_thing":
            return ActionResult(
                new_state=state.with_screen("done").with_flag("done", True),
                reward=1.0, done=True, success=True,
            )
        return ActionResult(new_state=state, reward=-0.1)

def my_task() -> CausalUITask:
    return CausalUITask(
        env=MyFlow(),
        goal="Reach done screen",
        is_goal=lambda s: s.has_flag("done"),
    )
```

---

## License

MIT

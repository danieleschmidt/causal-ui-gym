"""
Causal UI Gym — test agents on causal reasoning in UI-like decision tasks.

Core classes:
  UIState             — snapshot of a UI state machine
  CausalUIEnvironment — env where actions have causal side-effects
  CausalUITask        — goal-directed task wrapper
  UIAgent             — base agent interface
  RandomUIAgent       — random action selection
  GreedyUIAgent       — greedy (first-valid) action selection
  EvalHarness         — run agents, collect success rate & path efficiency

Concrete environments:
  LoginFlow           — username/password authentication flow
  CheckoutFlow        — e-commerce checkout funnel
  FormValidationFlow  — form with field-level validation
"""

from .core import (
    UIState,
    UIAction,
    ActionResult,
    CausalUIEnvironment,
    CausalUITask,
)
from .agents import UIAgent, RandomUIAgent, GreedyUIAgent
from .eval import EvalHarness, EvalResult
from .environments import LoginFlow, CheckoutFlow, FormValidationFlow

__all__ = [
    "UIState",
    "UIAction",
    "ActionResult",
    "CausalUIEnvironment",
    "CausalUITask",
    "UIAgent",
    "RandomUIAgent",
    "GreedyUIAgent",
    "EvalHarness",
    "EvalResult",
    "LoginFlow",
    "CheckoutFlow",
    "FormValidationFlow",
]

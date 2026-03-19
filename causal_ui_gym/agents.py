"""
Agent implementations for Causal UI Gym.

UIAgent       — abstract base
RandomUIAgent — picks a uniformly random available action each step
GreedyUIAgent — picks the first available action each step (deterministic)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional

from .core import UIAction, UIState


class UIAgent(ABC):
    """Abstract base class for UI agents."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    @abstractmethod
    def act(self, state: UIState, available_actions: List[UIAction]) -> UIAction:
        """
        Choose an action given the current state and available actions.

        Parameters
        ----------
        state : UIState
            Current environment state.
        available_actions : list[UIAction]
            Actions currently available to the agent.

        Returns
        -------
        UIAction
            The chosen action. Must be one of available_actions.
        """

    def reset(self) -> None:
        """Called at the start of each episode. Override to reset internal state."""


class RandomUIAgent(UIAgent):
    """
    Uniformly random agent — selects a random available action each step.

    Useful as a baseline and for stress-testing environments.
    """

    def act(self, state: UIState, available_actions: List[UIAction]) -> UIAction:
        if not available_actions:
            raise ValueError("No actions available.")
        return self.rng.choice(available_actions)


class GreedyUIAgent(UIAgent):
    """
    Greedy agent — always picks the *first* available action (by list order).

    Deterministic, but dependent on how environments order their action lists.
    Useful for checking that environments have sensible action ordering.
    """

    def act(self, state: UIState, available_actions: List[UIAction]) -> UIAction:
        if not available_actions:
            raise ValueError("No actions available.")
        return available_actions[0]

"""
Evaluation harness for Causal UI Gym.

EvalHarness runs an agent on one or more tasks/environments, collects
success rate and path-efficiency metrics, and returns structured results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .agents import UIAgent
from .core import CausalUIEnvironment, CausalUITask, UIAction, UIState


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Outcome of a single episode."""

    task_name: str
    success: bool
    steps: int
    total_reward: float
    final_state: UIState
    action_history: List[UIAction] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def path_efficiency(self) -> Optional[float]:
        """
        Ratio of minimum-possible steps to actual steps taken.

        Returns None when optimal_steps is not recorded in info.
        A score of 1.0 means the agent took the shortest path; lower = less efficient.
        """
        optimal = self.info.get("optimal_steps")
        if optimal is None or self.steps == 0:
            return None
        return optimal / self.steps


@dataclass
class EvalResult:
    """Aggregated results across all episodes."""

    agent_name: str
    episodes: List[EpisodeResult] = field(default_factory=list)

    # Computed on demand
    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.success for e in self.episodes) / len(self.episodes)

    @property
    def mean_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.steps for e in self.episodes) / len(self.episodes)

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.total_reward for e in self.episodes) / len(self.episodes)

    @property
    def mean_path_efficiency(self) -> Optional[float]:
        effs = [e.path_efficiency for e in self.episodes if e.path_efficiency is not None]
        if not effs:
            return None
        return sum(effs) / len(effs)

    def by_task(self) -> Dict[str, "EvalResult"]:
        """Group episodes by task name."""
        groups: Dict[str, List[EpisodeResult]] = {}
        for ep in self.episodes:
            groups.setdefault(ep.task_name, []).append(ep)
        return {
            name: EvalResult(agent_name=self.agent_name, episodes=eps)
            for name, eps in groups.items()
        }

    def summary(self) -> str:
        lines = [
            f"Agent        : {self.agent_name}",
            f"Episodes     : {len(self.episodes)}",
            f"Success rate : {self.success_rate:.1%}",
            f"Mean steps   : {self.mean_steps:.1f}",
            f"Mean reward  : {self.mean_reward:.3f}",
        ]
        eff = self.mean_path_efficiency
        if eff is not None:
            lines.append(f"Path eff.    : {eff:.3f}")

        by_task = self.by_task()
        if len(by_task) > 1:
            lines.append("")
            lines.append("Per-task breakdown:")
            for task_name, sub in by_task.items():
                lines.append(
                    f"  {task_name:<30} success={sub.success_rate:.1%}  "
                    f"steps={sub.mean_steps:.1f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------

class EvalHarness:
    """
    Run agents against environments/tasks and collect metrics.

    Usage
    -----
        harness = EvalHarness(n_episodes=20, seed=42)
        result = harness.run(agent=RandomUIAgent(seed=0), task=LoginFlow())
        print(result.summary())

    Parameters
    ----------
    n_episodes : int
        Number of episodes to run per (agent, task) pair.
    seed : int, optional
        Base seed for episode-level RNG (ensures reproducibility).
    verbose : bool
        Print per-episode summaries during evaluation.
    """

    def __init__(
        self,
        n_episodes: int = 10,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.n_episodes = n_episodes
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        agent: UIAgent,
        task: Union[CausalUITask, CausalUIEnvironment],
    ) -> EvalResult:
        """
        Run *agent* on *task* for self.n_episodes episodes.

        If *task* is a plain CausalUIEnvironment, it is wrapped in a
        CausalUITask with a trivial is_goal that never fires (the
        environment's own done signal drives termination).
        """
        if isinstance(task, CausalUIEnvironment) and not isinstance(task, CausalUITask):
            task = self._wrap_env(task)

        agent_name = type(agent).__name__
        result = EvalResult(agent_name=agent_name)

        for ep_idx in range(self.n_episodes):
            ep = self._run_episode(agent, task, ep_idx)
            result.episodes.append(ep)
            if self.verbose:
                status = "✓" if ep.success else "✗"
                print(
                    f"[{agent_name}] ep={ep_idx+1}/{self.n_episodes} "
                    f"{status} steps={ep.steps} reward={ep.total_reward:.2f}"
                )

        return result

    def run_many(
        self,
        agent: UIAgent,
        tasks: Sequence[Union[CausalUITask, CausalUIEnvironment]],
    ) -> EvalResult:
        """Run agent on multiple tasks; combine all episodes into one EvalResult."""
        agent_name = type(agent).__name__
        combined = EvalResult(agent_name=agent_name)
        for task in tasks:
            sub = self.run(agent, task)
            combined.episodes.extend(sub.episodes)
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        agent: UIAgent,
        task: CausalUITask,
        ep_idx: int,
    ) -> EpisodeResult:
        agent.reset()
        state = task.reset()

        action_history: List[UIAction] = []
        total_reward = 0.0
        done = False
        success = False
        t0 = time.monotonic()

        while not done:
            available = task.available_actions()
            if not available:
                break  # environment has no moves — terminal state

            action = agent.act(state, available)
            action_history.append(action)

            result = task.step(action)
            total_reward += result.reward
            state = result.new_state
            done = result.done
            success = result.success

            if task.steps >= task.max_steps:
                done = True

        elapsed = time.monotonic() - t0
        return EpisodeResult(
            task_name=task.name,
            success=success,
            steps=task.steps,
            total_reward=total_reward,
            final_state=state,
            action_history=action_history,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _wrap_env(env: CausalUIEnvironment) -> CausalUITask:
        """Wrap a bare environment in a CausalUITask with a never-true goal."""
        return CausalUITask(
            env=env,
            goal="(environment-controlled)",
            is_goal=lambda s: False,
        )

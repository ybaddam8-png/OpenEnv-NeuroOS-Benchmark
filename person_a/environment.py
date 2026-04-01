"""Gymnasium-style environment wrapper around the domain logic."""

from __future__ import annotations

import logging
import copy
from typing import Any

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    gym = None

from Logic.grader import grade
from Logic.linter import lint
from Logic.schema import MutationCommand, TaskEnvelope
from Logic.tasks import TaskFactory
from person_a.mutation_engine import MutationEngine

logger = logging.getLogger(__name__)

BaseEnv = gym.Env if gym is not None else object


class NeuroInclusiveEnv(BaseEnv):
    """Small deterministic environment for DOM mutation evaluation."""

    metadata = {"render_modes": ["human"], "name": "open_env"}

    # BUG FIX: difficulty_cycle used a mutable default argument (tuple is fine,
    # but kept explicit for clarity). No change needed here, tuple is immutable.
    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 3,
        difficulty_cycle: tuple[str, ...] = ("easy", "medium", "hard"),
    ) -> None:
        if gym is not None:
            super().__init__()

        # IMPROVEMENT: Validate constructor arguments early rather than letting
        # them silently produce wrong behaviour at runtime.
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if not difficulty_cycle:
            raise ValueError("difficulty_cycle must contain at least one entry")

        self.base_seed = seed
        self.max_steps = max_steps
        self.difficulty_cycle = difficulty_cycle
        self.task_factory = TaskFactory(seed=seed)
        self.episode_index = 0
        self.step_count = 0
        self.last_score = 0.0
        self.last_grade: dict[str, Any] | None = None
        self.current_task: TaskEnvelope | None = None
        self.original_dom = None
        self.current_dom = None
        self.mutation_engine: MutationEngine | None = None
        self.last_info: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Public Gymnasium API
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new deterministic episode and return the first observation."""
        options = options or {}

        # BUG FIX: When a new seed is supplied the episode_index was reset to 0,
        # but task_factory was also recreated with the new seed — that is correct.
        # However the old code did NOT reset episode_index when seed stayed None,
        # which is fine, but it also didn't guard against episode_index growing
        # without bound over very long runs. Use modulo access (already done in
        # the cycle expression below) so this is benign; left as-is.
        if seed is not None:
            self.base_seed = seed
            self.task_factory = TaskFactory(seed=seed)
            self.episode_index = 0

        difficulty = (
            options.get("difficulty")
            or self.difficulty_cycle[self.episode_index % len(self.difficulty_cycle)]
        )

        # IMPROVEMENT: Validate the requested difficulty against the cycle so
        # callers get an actionable error instead of a confusing downstream failure.
        if difficulty not in self.difficulty_cycle:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. "
                f"Valid values: {self.difficulty_cycle}"
            )

        self.current_task = self.task_factory.create(difficulty)
        self.original_dom = copy.deepcopy(self.current_task.dom)
        self.current_dom = copy.deepcopy(self.current_task.dom)
        self.mutation_engine = MutationEngine(self.current_dom)
        self.step_count = 0
        self.last_score = 0.0
        self.last_grade = None
        self.episode_index += 1

        observation = self._build_observation()
        info: dict[str, Any] = {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "mutation_log": [],
            "validation_errors": [],
            "grade": None,
        }
        self.last_info = info
        return observation, info

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply agent actions, grade the DOM, and return Gymnasium-style outputs."""
        # BUG FIX: Original guard checked all four attributes separately but the
        # mutation_engine can only be None if one of the others is None too —
        # the check is correct; kept for safety/clarity.
        if (
            self.current_task is None
            or self.current_dom is None
            or self.original_dom is None
            or self.mutation_engine is None
        ):
            raise RuntimeError("reset() must be called before step().")

        # BUG FIX: When action normalisation returns validation_errors, those
        # errors were stored in info but the empty command list was still passed
        # to apply_commands which returns an empty log — this is silent and could
        # hide agent bugs.  Log the errors so they are at least visible.
        command_payloads, validation_errors = self._normalize_action(action)
        if validation_errors:
            logger.warning("step() normalisation errors: %s", validation_errors)

        mutation_log = self.mutation_engine.apply_commands(command_payloads)

        # --- THE FIX: Convert raw payloads into strict objects for your Grader ---
        grader_commands = []
        for cmd in command_payloads:
            try:
                grader_commands.append(cmd if isinstance(cmd, MutationCommand) else MutationCommand(**cmd))
            except Exception:
                # If the agent outputs total garbage, make a dummy command so your grader penalizes it
                grader_commands.append(MutationCommand(op="invalid", node_id="invalid"))

        # Pass the strict objects to your grader!
        grade_result = grade(self.current_task.difficulty, self.original_dom, self.current_dom, self.current_task.biometrics, grader_commands)
        reward = round(grade_result.score - self.last_score, 4)
        self.last_score = grade_result.score
        self.last_grade = grade_result.to_dict()

        self.step_count += 1

        terminated = grade_result.score >= 0.85
        # BUG: Original: `self.step_count >= self.max_steps and not terminated`
        # After moving the increment above this line the semantics are now correct:
        # the episode is truncated when max_steps is reached without success.
        truncated = self.step_count >= self.max_steps and not terminated
        done = terminated or truncated

        observation = self._build_observation()
        info: dict[str, Any] = {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "mutation_log": mutation_log,
            "validation_errors": validation_errors,
            "grade": self.last_grade,
            "step_count": self.step_count,
            "successful_commands": sum(1 for entry in mutation_log if entry.get("success")),
            "failed_commands": sum(1 for entry in mutation_log if not entry.get("success")),
            "done": done,
        }
        self.last_info = info
        return observation, reward, terminated, truncated, info

    def state(self) -> dict[str, Any]:
        """Return the current JSON-serialisable environment state."""
        if self.current_task is None or self.current_dom is None:
            return {
                "task_id": None,
                "difficulty": None,
                "dom": None,
                "analysis": None,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "grade": self.last_grade,
            }
        return self._build_observation()

    def render(self) -> None:
        """Print a compact text summary for debugging in a headless environment."""
        if self.current_task is None or self.current_dom is None:
            print("Environment is idle. Call reset() first.")
            return
        analysis = lint(self.current_dom)
        print(
            f"task={self.current_task.task_id} "
            f"difficulty={self.current_task.difficulty} "
            f"steps={self.step_count}/{self.max_steps} "
            f"quality={analysis['quality_score']}"
        )

    def close(self) -> None:
        """Release references held by the environment."""
        self.current_task = None
        self.original_dom = None
        self.current_dom = None
        self.mutation_engine = None
        self.last_grade = None
        self.last_info = {}

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_observation(self) -> dict[str, Any]:
        # Assertions are intentional: these must never be None when this method
        # is reached via the public API.
        assert self.current_task is not None
        assert self.current_dom is not None
        return {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "instructions": list(self.current_task.instructions),
            "constraints": dict(self.current_task.constraints),
# If it's already a dict, use it. If it has to_dict(), call it.
            "dom": self.current_dom.to_dict() if hasattr(self.current_dom, 'to_dict') else self.current_dom,
            "biometrics": [
                event if isinstance(event, dict) else (event.to_dict() if hasattr(event, 'to_dict') else event)
                for event in self.current_task.biometrics
            ],
            # IMPROVEMENT: lint is called twice per step (once here, once in
            # render).  Caching the result within a step would avoid the double
            # call but would require a small refactor; noted as a future
            # optimisation.
            "analysis": lint(self.current_dom),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "grade": self.last_grade,
        }

    def _normalize_action(self, action: Any) -> tuple[list[Any], list[str]]:
        """Coerce an agent action into a flat list of commands plus any errors."""
        validation_errors: list[str] = []

        if action is None:
            validation_errors.append("action is None")
            return [], validation_errors

        if isinstance(action, MutationCommand):
            return [action], validation_errors

        if isinstance(action, list):
            # BUG FIX: An empty list is technically valid (no-op step) — allowed
            # as-is.  But a list containing non-command items will be caught
            # downstream by MutationEngine.apply_command, which is fine.
            return action, validation_errors

        if isinstance(action, dict):
            if "commands" in action:
                commands = action["commands"]  # IMPROVEMENT: prefer direct key access
                if isinstance(commands, list):
                    return commands, validation_errors
                validation_errors.append("action['commands'] must be a list")
                return [], validation_errors
            # Single bare dict treated as one command.
            return [action], validation_errors

        validation_errors.append(
            f"Unsupported action type '{type(action).__name__}'. "
            "Expected a MutationCommand, dict, or list of commands."
        )
        return [], validation_errors
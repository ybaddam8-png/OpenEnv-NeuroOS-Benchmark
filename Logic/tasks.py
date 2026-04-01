"""
tasks.py — Task Factory
========================
Wires together VDOMGenerator + BiometricSimulator + linter into
complete TaskEnvelopes ready to hand to Person A's environment.

Each task is seeded and fully reproducible.
The same (difficulty, seed) pair always produces the same task.

Usage:
    factory = TaskFactory()

    easy_task   = factory.create("easy",   seed=0)
    medium_task = factory.create("medium", seed=0)
    hard_task   = factory.create("hard",   seed=0)

    print(easy_task.to_json())
"""

from __future__ import annotations
from Logic.schema import TaskEnvelope, DOMNode
from Logic.vdom_generator import VDOMGenerator
from Logic.biometric_simulator import BiometricSimulator
from Logic.linter import lint


# ---------------------------------------------------------------------------
# Task instructions (what the agent reads)
# ---------------------------------------------------------------------------

INSTRUCTIONS = {
    "easy": """You are an accessibility engineer.
You have received a DOM tree representing a web interface.
The DOM contains accessibility violations that make it difficult
for users with visual or cognitive impairments to use.

Your job: Issue a list of mutation commands that fix ALL contrast ratio
and ARIA label violations in the DOM.

Rules:
- Only fix contrast_ratio and aria_label violations.
- Do NOT delete or reparent any nodes.
- Do NOT hide nodes that are not genuinely redundant.
- Every mutation must target a real node id from the DOM.

Return a JSON list of mutation commands.
Each command: {"op": "<operation>", "node_id": "<id>", "value": <value>, "reason": "<why>"}
""",

    "medium": """You are an accessibility engineer dealing with a cognitively overloaded interface.

The DOM contains:
- Contrast ratio violations
- Missing ARIA labels
- Redundant/duplicate content that clutters the interface
- Nodes with excessive cognitive weight

Your job: Issue mutation commands that bring the UI into accessibility compliance.
You must fix contrast, ARIA labels, collapse ONLY genuinely redundant nodes,
and reduce cognitive weight on overburdened nodes.

Rules:
- You may use: set_contrast, set_aria_label, collapse, remove_redundancy, set_cognitive_weight
- Only collapse nodes that have duplicate/redundant text (is_redundant=True or identical sibling text)
- Do NOT collapse structural nodes (main, nav, header, footer)
- Do NOT collapse interactive elements (button, input, select)

Return a JSON list of mutation commands.
""",

    "hard": """You are an accessibility engineer responding to a live user stress event.

Alongside the DOM tree, you are provided with a stream of user biometric data
showing cursor velocity, fixation patterns, rage clicks, and cognitive load scores.

The user is clearly struggling. Your mutations must:
1. Fix all accessibility violations (contrast, ARIA, nesting, animation, cognitive weight)
2. Prioritise nodes where the user's stress signals are concentrated
3. Restructure deeply nested elements to reduce cognitive complexity
4. Stop all animations immediately
5. Target the nodes the user was hovering over during high-stress events

Rules:
- You may use all available mutation operations
- Preserve ALL structural tags: main, nav, header, footer, h1-h3
- Do NOT delete any node — only collapse, reparent, or merge
- Every reparent must make the tree SHALLOWER, not deeper
- Respond to the biometric data: if node X shows high rage_click and cognitive_load,
  it must receive at minimum one mutation

Return a JSON list of mutation commands. Include "reason" on each command
explaining which biometric signal drove that decision.
""",
}

CONSTRAINTS = {
    "easy": {
        "allowed_ops":           ["set_contrast", "set_aria_label", "set_font_size"],
        "forbidden_ops":         ["collapse", "reparent", "merge_nodes"],
        "must_preserve_tags":    ["main", "nav", "header", "footer"],
        "min_node_retention":    1.0,   # all nodes must remain
    },
    "medium": {
        "allowed_ops":           ["set_contrast", "set_aria_label", "collapse",
                                  "remove_redundancy", "set_cognitive_weight",
                                  "reduce_sensory_load"],
        "forbidden_ops":         ["reparent", "merge_nodes"],
        "must_preserve_tags":    ["main", "nav", "header", "footer", "h1", "h2"],
        "min_node_retention":    0.75,
    },
    "hard": {
        "allowed_ops":           "all",
        "forbidden_ops":         [],
        "must_preserve_tags":    ["main", "nav", "header", "h1"],
        "min_node_retention":    0.60,
        "biometric_requirement": True,  # agent MUST reference biometric data
    },
}


# ---------------------------------------------------------------------------
# Task Factory
# ---------------------------------------------------------------------------

class TaskFactory:
    """
    Creates reproducible TaskEnvelopes for any difficulty level.

    Parameters
    ----------
    base_seed   : int — base seed; tasks at the same difficulty share the
                  same base_seed but use it differently per difficulty
    """

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    def create(self, difficulty: str, seed: int = None) -> TaskEnvelope:
        """
        Create a complete task envelope for the given difficulty.

        Parameters
        ----------
        difficulty  : "easy" | "medium" | "hard"
        seed        : override seed (defaults to self.base_seed)
        """
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy, medium, or hard")

        s = seed if seed is not None else self.base_seed

        # Generate DOM
        vdom_gen = VDOMGenerator(seed=s)
        dom = vdom_gen.generate(difficulty)

        # Generate biometrics (only meaningful for medium/hard, but always present)
        all_node_ids = [n.id for n in dom.all_nodes()]
        bio_sim = BiometricSimulator(seed=s, node_ids=all_node_ids)
        biometrics = bio_sim.generate(difficulty)

        # Run linter to compute ground truth (stored in task metadata)
        lint_result = lint(dom)
        dom.metadata["initial_lint"] = lint_result.to_dict()

        task_id = f"task_{difficulty}_{s}"

        return TaskEnvelope(
            task_id=task_id,
            difficulty=difficulty,
            dom=dom,
            biometrics=biometrics if difficulty in ("medium", "hard") else [],
            instructions=INSTRUCTIONS[difficulty],
            constraints=CONSTRAINTS[difficulty],
        )

    def create_batch(self, difficulty: str, count: int, start_seed: int = 0) -> list[TaskEnvelope]:
        """
        Create multiple tasks at the same difficulty with sequential seeds.
        Useful for eval runs.
        """
        return [self.create(difficulty, seed=start_seed + i) for i in range(count)]
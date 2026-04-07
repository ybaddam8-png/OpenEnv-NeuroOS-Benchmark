"""
schema.py — Shared JSON DOM Schema for OpenEnv: Neuro-Inclusive UI State Mutator
=================================================================================
THIS IS THE CONTRACT. Person A and Person B both import from this file.
Never change field names or types without coordinating with Person A.

DOM Structure: Recursive nested JSON with children[]
Mutation Format: List of named mutation command dicts
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Core DOM Node
# ---------------------------------------------------------------------------

@dataclass
class DOMNode:
    """
    A single node in the virtual DOM tree.

    Attributes
    ----------
    id          : Unique string identifier, e.g. "n0", "n1_2"
    tag         : HTML-like tag name, e.g. "div", "button", "label", "img"
    text        : Visible text content (empty string if none)
    attributes  : Dict of accessibility-relevant properties (see ATTRIBUTE KEYS below)
    children    : Ordered list of child DOMNodes
    metadata    : Optional dict for task-specific annotations (ground-truth, flags)

    ATTRIBUTE KEYS (all optional, defaults shown):
        contrast_ratio      float   — foreground/background contrast (WCAG needs >= 4.5)
        font_size_px        float   — rendered font size in pixels
        cognitive_weight    float   — estimated cognitive load of this node (0.0 – 1.0)
        is_redundant        bool    — flagged as duplicate/redundant content
        is_hidden           bool    — node is visually hidden (display:none equivalent)
        aria_label          str     — ARIA accessibility label
        role                str     — ARIA role (e.g. "button", "heading", "alert")
        nesting_depth       int     — distance from root (root = 0)
        sensory_load        float   — visual/sensory complexity score (0.0 – 1.0)
        animation_present   bool    — node has CSS animation or transition
    """
    id: str
    tag: str
    text: str = ""
    attributes: dict = field(default_factory=dict)
    children: list[DOMNode] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Recursively convert to plain dict (JSON-serialisable)."""
        return {
            "id": self.id,
            "tag": self.tag,
            "text": self.text,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> DOMNode:
        """Recursively reconstruct DOMNode from a plain dict."""
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            id=d["id"],
            tag=d["tag"],
            text=d.get("text", ""),
            attributes=d.get("attributes", {}),
            children=children,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, s: str) -> DOMNode:
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Tree utilities
    # ------------------------------------------------------------------

    def find_by_id(self, node_id: str) -> Optional[DOMNode]:
        """BFS search for a node by id. Returns None if not found."""
        if self.id == node_id:
            return self
        for child in self.children:
            result = child.find_by_id(node_id)
            if result:
                return result
        return None

    def all_nodes(self) -> list[DOMNode]:
        """Return flat list of all nodes in tree (BFS order)."""
        result = [self]
        for child in self.children:
            result.extend(child.all_nodes())
        return result

    def node_count(self) -> int:
        return len(self.all_nodes())

    def max_depth(self) -> int:
        if not self.children:
            return self.attributes.get("nesting_depth", 0)
        return max(child.max_depth() for child in self.children)


# ---------------------------------------------------------------------------
# Mutation Commands  (what the agent sends back)
# ---------------------------------------------------------------------------

# Supported operation names — mutation engine (Person A) must handle all of these
VALID_OPS = {
    "set_contrast",        # set contrast_ratio on a node
    "set_font_size",       # set font_size_px on a node
    "collapse",            # mark node as is_hidden=True (must preserve node in tree)
    "remove_redundancy",   # set is_redundant=False and strip duplicate text
    "set_aria_label",      # set aria_label attribute
    "set_role",            # set role attribute
    "reduce_sensory_load", # set sensory_load to a lower value
    "stop_animation",      # set animation_present=False
    "reparent",            # move node to a new parent (hard task only)
    "merge_nodes",         # merge two sibling nodes into one (hard task only)
    "set_cognitive_weight",# directly set cognitive_weight on a node
    "delete_node",         # remove node from DOM tree entirely
    "simplify_text",       # reduce complexity of node text
    "disable_autoplay",    # neutralise autoplay distractions
    "remove_animation",    # neutralise animation elements
}

@dataclass
class MutationCommand:
    """
    A single named mutation the agent wants to apply to the DOM.

    Fields
    ------
    op          : Operation name (must be in VALID_OPS)
    node_id     : Target node's id string
    value       : New value for the operation (type depends on op, see below)
    reason      : Optional free-text reason (for logging/explainability)

    Value types by op:
        set_contrast        -> float  (e.g. 5.2)
        set_font_size       -> float  (e.g. 16.0)
        collapse            -> None   (no value needed)
        remove_redundancy   -> None
        set_aria_label      -> str
        set_role            -> str
        reduce_sensory_load -> float  (0.0 – 1.0)
        stop_animation      -> None
        reparent            -> str    (new parent node_id)
        merge_nodes         -> str    (sibling node_id to merge with)
        set_cognitive_weight-> float  (0.0 – 1.0)
        delete_node         -> None
        simplify_text       -> str (optional)
        disable_autoplay    -> None
        remove_animation    -> None
    """
    op: str
    node_id: str
    value: object = None
    reason: str = ""

    def validate(self) -> tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        if self.op not in VALID_OPS:
            return False, f"Unknown op '{self.op}'. Valid ops: {sorted(VALID_OPS)}"
        if not self.node_id or not isinstance(self.node_id, str):
            return False, "node_id must be a non-empty string"
        return True, ""

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "node_id": self.node_id,
            "value": self.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MutationCommand:
        return cls(
            op=d["op"],
            node_id=d["node_id"],
            value=d.get("value"),
            reason=d.get("reason", ""),
        )


# ---------------------------------------------------------------------------
# Task Envelope  (what the environment hands to the agent)
# ---------------------------------------------------------------------------

@dataclass
class TaskEnvelope:
    """
    The full observation handed to the agent at the start of a task.

    Fields
    ------
    task_id         : Unique task identifier string
    difficulty      : "easy" | "medium" | "hard"
    dom             : The initial DOMNode tree (with violations)
    biometrics      : List of biometric event dicts (empty for easy tasks)
    instructions    : Natural language task description for the agent
    constraints     : Dict of task-specific constraints the agent must respect
    """
    task_id: str
    difficulty: str   # "easy" | "medium" | "hard"
    dom: DOMNode
    task_name: str = "neuro-inclusive-audit"
    biometrics: list[dict] = field(default_factory=list)
    instructions: str = ""
    constraints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "task_name": self.task_name,
            "dom": self.dom.to_dict(),
            "biometrics": self.biometrics,
            "instructions": self.instructions,
            "constraints": self.constraints,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> TaskEnvelope:
        return cls(
            task_id=d["task_id"],
            difficulty=d["difficulty"],
            task_name=d.get("task_name", "neuro-inclusive-audit"),
            dom=DOMNode.from_dict(d["dom"]),
            biometrics=d.get("biometrics", []),
            instructions=d.get("instructions", ""),
            constraints=d.get("constraints", {}),
        )


# ---------------------------------------------------------------------------
# Grader Result
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    """
    The output of any grader function.

    Fields
    ------
    score           : Final reward, float in [0.0, 1.0]
    breakdown       : Dict of sub-scores by component (for debugging)
    penalties       : List of penalty dicts {"reason": str, "amount": float}
    exploit_detected: True if agent tried to cheat
    notes           : Human-readable explanation of the score
    """
    score: float
    breakdown: dict = field(default_factory=dict)
    penalties: list[dict] = field(default_factory=list)
    exploit_detected: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "breakdown": self.breakdown,
            "penalties": self.penalties,
            "exploit_detected": self.exploit_detected,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Attribute default helpers  (use these when building nodes)
# ---------------------------------------------------------------------------

DEFAULT_ATTRIBUTES = {
    "contrast_ratio": 4.5,
    "font_size_px": 16.0,
    "cognitive_weight": 0.3,
    "is_redundant": False,
    "is_hidden": False,
    "aria_label": "",
    "aria_hidden": "false",   # "true" = hidden from screen readers (contextual trap)
    "role": "",
    "nesting_depth": 0,
    "sensory_load": 0.2,
    "animation_present": False,
}

def make_attributes(**overrides) -> dict:
    """Return a full attributes dict with defaults, overridden by kwargs."""
    attrs = dict(DEFAULT_ATTRIBUTES)
    attrs.update(overrides)
    return attrs


# ---------------------------------------------------------------------------
# Biometric event schema  (for reference — biometric_simulator.py produces these)
# ---------------------------------------------------------------------------

BIOMETRIC_EVENT_SCHEMA = {
    "timestamp_ms":       int,    # ms since task start
    "cursor_velocity":    float,  # pixels/ms — high = erratic
    "cursor_delta_x":     float,  # x movement since last event
    "cursor_delta_y":     float,  # y movement since last event
    "fixation_duration":  int,    # ms the cursor dwelled in one area
    "rage_click":         bool,   # rapid repeated click detected
    "cognitive_load":     float,  # 0.0 – 100.0 synthetic score
    "scroll_velocity":    float,  # pixels/ms
    "node_hover_id":      str,    # which DOM node id cursor is near (can be "")
}
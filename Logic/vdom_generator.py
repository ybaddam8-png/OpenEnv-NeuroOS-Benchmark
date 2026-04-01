"""
vdom_generator.py — Procedural Virtual DOM Generator
=====================================================
Builds seeded, reproducible DOM trees with injected accessibility violations.
Difficulty controls violation density, tree depth, and structural complexity.

Usage:
    gen = VDOMGenerator(seed=42)
    dom = gen.generate("easy")
    dom = gen.generate("medium")
    dom = gen.generate("hard")
"""

from __future__ import annotations
import random
from Logic.schema import DOMNode, make_attributes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tag pools by semantic category
STRUCTURAL_TAGS = ["div", "section", "article", "aside", "main", "nav", "header", "footer"]
TEXT_TAGS       = ["p", "span", "label", "h1", "h2", "h3", "h4", "li"]
INTERACTIVE_TAGS= ["button", "input", "select", "a", "textarea"]
MEDIA_TAGS      = ["img", "video", "canvas"]

SAMPLE_TEXTS = [
    "Submit form", "Click here for more information", "Important notice",
    "User profile settings", "Notification preferences", "Dashboard overview",
    "Recent activity feed", "Action required", "Update your details",
    "View all items", "Learn more", "Get started today",
    "Privacy policy link", "Terms of service", "Contact support",
    "Enable notifications", "Manage account", "Download report",
    "Filter results", "Sort by date", "Show advanced options",
]

ARIA_LABELS = [
    "Close dialog", "Open menu", "Submit form", "Search",
    "Previous page", "Next page", "Delete item", "Edit entry",
    "Toggle sidebar", "Expand section",
]

ROLES = ["navigation", "main", "complementary", "contentinfo", "banner", "search"]


# ---------------------------------------------------------------------------
# Difficulty profiles
# ---------------------------------------------------------------------------

DIFFICULTY_PROFILES = {
    "easy": {
        "min_nodes":        8,
        "max_nodes":        18,
        "max_depth":        3,
        "violation_rate":   0.35,   # fraction of nodes that get a violation
        "violation_types":  ["low_contrast", "missing_aria"],
        "allow_animation":  False,
        "allow_redundancy": False,
        "max_sensory_load": 0.4,
    },
    "medium": {
        "min_nodes":        20,
        "max_nodes":        40,
        "max_depth":        4,
        "violation_rate":   0.50,
        "violation_types":  ["low_contrast", "missing_aria", "redundancy",
                             "high_cognitive_weight", "high_sensory_load"],
        "allow_animation":  False,
        "allow_redundancy": True,
        "max_sensory_load": 0.7,
    },
    "hard": {
        "min_nodes":        40,
        "max_nodes":        70,
        "max_depth":        7,
        "violation_rate":   0.65,
        "violation_types":  ["low_contrast", "missing_aria", "redundancy",
                             "high_cognitive_weight", "high_sensory_load",
                             "deep_nesting", "animation", "small_font",
                             "contextual_trap"],
        "allow_animation":  True,
        "allow_redundancy": True,
        "max_sensory_load": 0.95,
    },
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class VDOMGenerator:
    """
    Produces seeded, reproducible DOM trees.

    Parameters
    ----------
    seed    : int  — random seed for full reproducibility
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)
        self._node_counter = 0

    def _next_id(self, prefix: str = "n") -> str:
        self._node_counter += 1
        return f"{prefix}{self._node_counter}"

    def generate(self, difficulty: str) -> DOMNode:
        """
        Generate a DOM tree for the given difficulty level.
        Returns a DOMNode root with violations injected per the difficulty profile.
        """
        if difficulty not in DIFFICULTY_PROFILES:
            raise ValueError(f"difficulty must be one of {list(DIFFICULTY_PROFILES.keys())}")

        # Reset counter so same seed+difficulty always gives same IDs
        self._node_counter = 0
        self._rng = random.Random(self.seed)

        profile = DIFFICULTY_PROFILES[difficulty]
        target_nodes = self._rng.randint(profile["min_nodes"], profile["max_nodes"])

        root = self._build_tree(
            depth=0,
            max_depth=profile["max_depth"],
            target_nodes=target_nodes,
            profile=profile,
        )

        # Stamp nesting_depth on every node
        self._stamp_depth(root, 0)

        # Inject violations
        all_nodes = root.all_nodes()
        self._inject_violations(all_nodes, profile)

        # Mark ground truth on metadata
        root.metadata["difficulty"] = difficulty
        root.metadata["seed"] = self.seed
        root.metadata["generated_node_count"] = len(all_nodes)

        return root

    # ------------------------------------------------------------------
    # Tree builder
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        depth: int,
        max_depth: int,
        target_nodes: int,
        profile: dict,
        parent_id: str = "",
    ) -> DOMNode:
        """Recursively build a DOM subtree."""
        node_id = self._next_id()
        tag = self._pick_tag(depth)
        text = self._pick_text(tag)

        node = DOMNode(
            id=node_id,
            tag=tag,
            text=text,
            attributes=make_attributes(
                contrast_ratio=round(self._rng.uniform(3.5, 9.0), 2),
                font_size_px=round(self._rng.uniform(10.0, 24.0), 1),
                cognitive_weight=round(self._rng.uniform(0.1, 0.6), 2),
                sensory_load=round(self._rng.uniform(0.05, profile["max_sensory_load"]), 2),
                animation_present=False,
                nesting_depth=depth,
            ),
        )

        # Decide how many children to sprout
        remaining = target_nodes - self._node_counter
        if depth >= max_depth or remaining <= 0:
            return node

        # Distribute remaining nodes across 1–4 children
        n_children = self._rng.randint(1, min(4, remaining))
        # Each child gets a rough slice of remaining budget
        budget_per_child = max(1, remaining // n_children)

        for _ in range(n_children):
            if self._node_counter >= target_nodes:
                break
            child = self._build_tree(
                depth=depth + 1,
                max_depth=max_depth,
                target_nodes=self._node_counter + budget_per_child,
                profile=profile,
                parent_id=node_id,
            )
            node.children.append(child)

        return node

    # ------------------------------------------------------------------
    # Violation injector
    # ------------------------------------------------------------------

    def _inject_violations(self, nodes: list[DOMNode], profile: dict):
        """
        Walk all nodes and inject violations based on the difficulty profile.
        Marks injected violations in node.metadata["violations"] for ground truth.
        """
        violation_types = profile["violation_types"]
        rate = profile["violation_rate"]

        # Track texts we've already used (for redundancy injection)
        texts_used: list[str] = []

        for node in nodes:
            if self._rng.random() > rate:
                continue  # this node stays clean

            vtype = self._rng.choice(violation_types)
            injected = []

            if vtype == "low_contrast":
                node.attributes["contrast_ratio"] = round(self._rng.uniform(1.5, 4.0), 2)
                injected.append("low_contrast")

            elif vtype == "missing_aria" and node.tag in {"button", "input", "select", "a"}:
                node.attributes["aria_label"] = ""
                injected.append("missing_aria")

            elif vtype == "redundancy" and profile.get("allow_redundancy"):
                if texts_used and self._rng.random() < 0.5:
                    node.text = self._rng.choice(texts_used)
                    node.attributes["is_redundant"] = True
                    injected.append("redundancy")
                elif node.text:
                    texts_used.append(node.text)

            elif vtype == "high_cognitive_weight":
                node.attributes["cognitive_weight"] = round(self._rng.uniform(0.75, 1.0), 2)
                injected.append("high_cognitive_weight")

            elif vtype == "high_sensory_load":
                node.attributes["sensory_load"] = round(self._rng.uniform(0.65, 1.0), 2)
                injected.append("high_sensory_load")

            elif vtype == "deep_nesting":
                # Force extra depth on this node's children
                current_depth = node.attributes.get("nesting_depth", 0)
                node.attributes["nesting_depth"] = max(current_depth, 5)
                injected.append("deep_nesting")

            elif vtype == "animation" and profile.get("allow_animation"):
                node.attributes["animation_present"] = True
                injected.append("animation")

            elif vtype == "small_font" and node.text:
                node.attributes["font_size_px"] = round(self._rng.uniform(6.0, 11.5), 1)
                injected.append("small_font")

            elif vtype == "contextual_trap" and node.children:
                # THE REAL AMBIGUITY TRAP:
                # Parent gets aria_hidden="true" — entire subtree invisible to screen readers.
                # Child looks PERFECT: great contrast, valid aria_label, correct tag.
                # A lazy agent fixes the child's visible attributes and moves on.
                # A correct agent must remove aria_hidden from the PARENT.
                # Linter rule R11 fires on the PARENT — not the child.
                # Grader penalises mutations that target the child but not the parent.
                node.attributes["aria_hidden"] = "true"
                child = node.children[0]
                child.tag = "button"
                child.attributes["aria_label"] = "Submit Payment"   # looks fine
                child.attributes["contrast_ratio"] = 7.5            # looks fine
                child.attributes["font_size_px"] = 16.0             # looks fine
                child.attributes["cognitive_weight"] = 0.2          # looks fine
                node.metadata["violations"] = ["contextual_trap"]   # parent is the bug
                child.metadata["trap_bait"] = True                  # child is the decoy
                injected.append("contextual_trap")

            if injected:
                node.metadata["violations"] = injected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_tag(self, depth: int) -> str:
        if depth == 0:
            return self._rng.choice(["main", "div", "section"])
        elif depth == 1:
            return self._rng.choice(STRUCTURAL_TAGS)
        elif depth <= 3:
            return self._rng.choice(STRUCTURAL_TAGS + TEXT_TAGS)
        else:
            return self._rng.choice(TEXT_TAGS + INTERACTIVE_TAGS)

    def _pick_text(self, tag: str) -> str:
        if tag in STRUCTURAL_TAGS + MEDIA_TAGS:
            return ""
        return self._rng.choice(SAMPLE_TEXTS)

    def _stamp_depth(self, node: DOMNode, depth: int):
        """Correct nesting_depth on all nodes post-build."""
        node.attributes["nesting_depth"] = depth
        for child in node.children:
            self._stamp_depth(child, depth + 1)
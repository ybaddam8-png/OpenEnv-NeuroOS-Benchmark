"""
linter.py — Deterministic Accessibility Rule Checker
=====================================================
Takes a DOMNode tree, returns a LintResult with every violation found
and an overall accessibility score.

This is the ground truth engine. The grader calls this on both the
original and mutated DOMs to compute how much the agent improved.

Rules implemented (all WCAG-inspired, all deterministic):
  R01  contrast_ratio >= 4.5 on any visible text node
  R02  font_size_px >= 12.0 on any visible text node
  R03  nesting_depth <= 4 for any node
  R04  No duplicate text within siblings (redundancy check)
  R05  cognitive_weight per node <= 0.7
  R06  sensory_load per node <= 0.6
  R07  animation_present == False (motion sensitivity)
  R08  aria_label non-empty on interactive nodes (button, input, select)
  R09  role non-empty on landmark nodes (nav, header, footer, main)
  R10  Total visible node count <= 60 (clutter threshold)
  R11  Interactive child inside aria_hidden parent is unreachable (contextual trap)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from Logic.schema import DOMNode


# ---------------------------------------------------------------------------
# Rule constants  (tweak these to adjust difficulty)
# ---------------------------------------------------------------------------

MIN_CONTRAST_RATIO   = 4.5
MIN_FONT_SIZE_PX     = 12.0
MAX_NESTING_DEPTH    = 4
MAX_COGNITIVE_WEIGHT = 0.7
MAX_SENSORY_LOAD     = 0.6
MAX_VISIBLE_NODES    = 60

INTERACTIVE_TAGS     = {"button", "input", "select", "textarea", "a"}
LANDMARK_TAGS        = {"nav", "header", "footer", "main", "aside", "section"}

# Severity weights — used to compute weighted score
SEVERITY = {
    "critical": 1.0,   # blocks accessibility entirely
    "major":    0.6,   # significantly impairs
    "minor":    0.25,  # annoying but workable
}

# Per-rule metadata
RULE_META = {
    "R01": {"desc": "contrast_ratio below minimum",      "severity": "critical"},
    "R02": {"desc": "font_size_px below minimum",        "severity": "major"},
    "R03": {"desc": "nesting depth exceeds maximum",     "severity": "major"},
    "R04": {"desc": "duplicate sibling text (redundant)","severity": "minor"},
    "R05": {"desc": "cognitive_weight too high",         "severity": "major"},
    "R06": {"desc": "sensory_load too high",             "severity": "major"},
    "R07": {"desc": "animation present (motion risk)",   "severity": "critical"},
    "R08": {"desc": "missing aria_label on interactive", "severity": "critical"},
    "R09": {"desc": "missing role on landmark element",  "severity": "minor"},
    "R10": {"desc": "too many visible nodes (clutter)",  "severity": "major"},
    "R11": {"desc": "interactive child inside aria_hidden parent", "severity": "critical"},
}


# ---------------------------------------------------------------------------
# Violation and result types
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    rule:        str        # e.g. "R01"
    node_id:     str        # which node
    description: str        # human-readable
    severity:    str        # "critical" | "major" | "minor"
    actual:      object     # the bad value
    expected:    object     # what it should be
    fix_hint:    str = ""   # what mutation would fix this

    def to_dict(self) -> dict:
        return {
            "rule":        self.rule,
            "node_id":     self.node_id,
            "description": self.description,
            "severity":    self.severity,
            "actual":      self.actual,
            "expected":    self.expected,
            "fix_hint":    self.fix_hint,
        }


@dataclass
class LintResult:
    violations:         list[Violation] = field(default_factory=list)
    accessibility_score: float = 1.0    # 1.0 = perfect, 0.0 = completely broken
    violation_count:    int = 0
    critical_count:     int = 0
    major_count:        int = 0
    minor_count:        int = 0
    total_nodes_checked: int = 0
    notes:              str = ""

    def to_dict(self) -> dict:
        return {
            "accessibility_score":  round(self.accessibility_score, 4),
            "violation_count":      self.violation_count,
            "critical_count":       self.critical_count,
            "major_count":          self.major_count,
            "minor_count":          self.minor_count,
            "total_nodes_checked":  self.total_nodes_checked,
            "notes":                self.notes,
            "violations":           [v.to_dict() for v in self.violations],
        }


# ---------------------------------------------------------------------------
# Main linter
# ---------------------------------------------------------------------------

class AccessibilityLinter:
    """
    Run lint() on a DOMNode tree to get a LintResult.

    Usage:
        linter = AccessibilityLinter()
        result = linter.lint(dom_root)
        print(result.accessibility_score)
        for v in result.violations:
            print(v.rule, v.node_id, v.description)
    """

    def lint(self, root: DOMNode) -> LintResult:
        all_nodes = root.all_nodes()
        violations: list[Violation] = []

        for node in all_nodes:
            if node.attributes.get("is_hidden", False):
                continue  # hidden nodes are exempt from most rules

            violations.extend(self._check_r01(node))
            violations.extend(self._check_r02(node))
            violations.extend(self._check_r03(node))
            violations.extend(self._check_r05(node))
            violations.extend(self._check_r06(node))
            violations.extend(self._check_r07(node))
            violations.extend(self._check_r08(node))
            violations.extend(self._check_r09(node))

        # R04: sibling-level redundancy (needs parent context)
        violations.extend(self._check_r04_tree(root))

        # R10: global clutter (tree-level)
        violations.extend(self._check_r10(root, all_nodes))

        # R11: contextual trap — interactive child inside aria_hidden parent
        violations.extend(self._check_r11_tree(root))

        result = self._compute_score(violations, len(all_nodes))
        return result

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    def _check_r01(self, node: DOMNode) -> list[Violation]:
        """R01: contrast ratio must be >= 4.5 for text-bearing nodes."""
        if not node.text and not node.attributes.get("aria_label"):
            return []
        ratio = node.attributes.get("contrast_ratio", 4.5)
        if ratio < MIN_CONTRAST_RATIO:
            return [Violation(
                rule="R01",
                node_id=node.id,
                description=f"contrast_ratio {ratio:.2f} is below {MIN_CONTRAST_RATIO}",
                severity="critical",
                actual=ratio,
                expected=f">= {MIN_CONTRAST_RATIO}",
                fix_hint=f'set_contrast on {node.id} to a value >= {MIN_CONTRAST_RATIO}',
            )]
        return []

    def _check_r02(self, node: DOMNode) -> list[Violation]:
        """R02: font size must be >= 12px for text nodes."""
        if not node.text:
            return []
        size = node.attributes.get("font_size_px", 16.0)
        if size < MIN_FONT_SIZE_PX:
            return [Violation(
                rule="R02",
                node_id=node.id,
                description=f"font_size_px {size:.1f} is below {MIN_FONT_SIZE_PX}",
                severity="major",
                actual=size,
                expected=f">= {MIN_FONT_SIZE_PX}",
                fix_hint=f'set_font_size on {node.id} to >= {MIN_FONT_SIZE_PX}',
            )]
        return []

    def _check_r03(self, node: DOMNode) -> list[Violation]:
        """R03: nesting depth must not exceed MAX_NESTING_DEPTH."""
        depth = node.attributes.get("nesting_depth", 0)
        if depth > MAX_NESTING_DEPTH:
            return [Violation(
                rule="R03",
                node_id=node.id,
                description=f"nesting_depth {depth} exceeds maximum {MAX_NESTING_DEPTH}",
                severity="major",
                actual=depth,
                expected=f"<= {MAX_NESTING_DEPTH}",
                fix_hint=f'reparent {node.id} to a shallower ancestor',
            )]
        return []

    def _check_r04_tree(self, root: DOMNode) -> list[Violation]:
        """R04: no two visible siblings should have identical non-empty text."""
        violations = []
        self._r04_recurse(root, violations)
        return violations

    def _r04_recurse(self, node: DOMNode, violations: list[Violation]):
        texts_seen: dict[str, str] = {}  # text -> first node_id that had it
        for child in node.children:
            if child.attributes.get("is_hidden", False):
                continue
            t = child.text.strip().lower()
            if t and len(t) > 3:  # ignore very short strings like "ok"
                if t in texts_seen:
                    violations.append(Violation(
                        rule="R04",
                        node_id=child.id,
                        description=f"duplicate sibling text '{child.text[:40]}' (first seen at {texts_seen[t]})",
                        severity="minor",
                        actual=child.text,
                        expected="unique sibling text",
                        fix_hint=f'remove_redundancy on {child.id}',
                    ))
                else:
                    texts_seen[t] = child.id
        for child in node.children:
            self._r04_recurse(child, violations)

    def _check_r05(self, node: DOMNode) -> list[Violation]:
        """R05: cognitive_weight must not exceed 0.7."""
        cw = node.attributes.get("cognitive_weight", 0.3)
        if cw > MAX_COGNITIVE_WEIGHT:
            return [Violation(
                rule="R05",
                node_id=node.id,
                description=f"cognitive_weight {cw:.2f} exceeds {MAX_COGNITIVE_WEIGHT}",
                severity="major",
                actual=cw,
                expected=f"<= {MAX_COGNITIVE_WEIGHT}",
                fix_hint=f'set_cognitive_weight on {node.id} or collapse/simplify',
            )]
        return []

    def _check_r06(self, node: DOMNode) -> list[Violation]:
        """R06: sensory_load must not exceed 0.6."""
        sl = node.attributes.get("sensory_load", 0.2)
        if sl > MAX_SENSORY_LOAD:
            return [Violation(
                rule="R06",
                node_id=node.id,
                description=f"sensory_load {sl:.2f} exceeds {MAX_SENSORY_LOAD}",
                severity="major",
                actual=sl,
                expected=f"<= {MAX_SENSORY_LOAD}",
                fix_hint=f'reduce_sensory_load on {node.id}',
            )]
        return []

    def _check_r07(self, node: DOMNode) -> list[Violation]:
        """R07: animation must not be present (motion sensitivity risk)."""
        if node.attributes.get("animation_present", False):
            return [Violation(
                rule="R07",
                node_id=node.id,
                description="animation_present=True violates motion sensitivity rule",
                severity="critical",
                actual=True,
                expected=False,
                fix_hint=f'stop_animation on {node.id}',
            )]
        return []

    def _check_r08(self, node: DOMNode) -> list[Violation]:
        """R08: interactive elements must have a non-empty aria_label."""
        if node.tag not in INTERACTIVE_TAGS:
            return []
        aria = node.attributes.get("aria_label", "").strip()
        if not aria:
            return [Violation(
                rule="R08",
                node_id=node.id,
                description=f"interactive <{node.tag}> missing aria_label",
                severity="critical",
                actual="",
                expected="non-empty aria_label",
                fix_hint=f'set_aria_label on {node.id}',
            )]
        return []

    def _check_r09(self, node: DOMNode) -> list[Violation]:
        """R09: landmark elements must have a non-empty role."""
        if node.tag not in LANDMARK_TAGS:
            return []
        role = node.attributes.get("role", "").strip()
        if not role:
            return [Violation(
                rule="R09",
                node_id=node.id,
                description=f"landmark <{node.tag}> missing role attribute",
                severity="minor",
                actual="",
                expected="non-empty role",
                fix_hint=f'set_role on {node.id}',
            )]
        return []

    def _check_r11_tree(self, root: DOMNode) -> list[Violation]:
        """
        R11: Contextual trap — an interactive child inside an aria_hidden parent
        is invisible to screen readers even if it looks perfect visually.
        This is the 'Boss Level' trap: the child has good contrast + aria_label,
        but the parent's aria_hidden='true' makes the whole subtree unreachable.
        An agent that fixes the child without fixing the parent scores zero on this node.
        """
        violations = []
        self._r11_recurse(root, parent_aria_hidden=False, violations=violations)
        return violations

    def _r11_recurse(self, node: DOMNode, parent_aria_hidden: bool, violations: list[Violation]):
        is_hidden = str(node.attributes.get("aria_hidden", "false")).lower() == "true"
        # If this node itself is aria_hidden, its interactive children are trapped
        if is_hidden:
            for child in node.children:
                self._check_trapped_interactive(node, child, violations)
        # If a parent was aria_hidden, this whole subtree is already trapped (checked above)
        for child in node.children:
            self._r11_recurse(child, parent_aria_hidden=is_hidden, violations=violations)

    def _check_trapped_interactive(self, parent: DOMNode, child: DOMNode, violations: list[Violation]):
        """Flag interactive children of aria_hidden parents."""
        if child.tag in INTERACTIVE_TAGS:
            violations.append(Violation(
                rule="R11",
                node_id=parent.id,
                description=(
                    f"<{parent.tag}> has aria_hidden='true' but contains interactive "
                    f"<{child.tag}> (id={child.id}) — that element is screen-reader invisible"
                ),
                severity="critical",
                actual=f"aria_hidden=true on parent {parent.id}",
                expected="remove aria_hidden from parent or move interactive child outside",
                fix_hint=f"remove aria_hidden on {parent.id} (not set_aria_label on {child.id})",
            ))

    def _check_r10(self, root: DOMNode, all_nodes: list[DOMNode]) -> list[Violation]:
        """R10: too many visible nodes causes UI clutter."""
        visible = [n for n in all_nodes if not n.attributes.get("is_hidden", False)]
        count = len(visible)
        if count > MAX_VISIBLE_NODES:
            return [Violation(
                rule="R10",
                node_id=root.id,
                description=f"visible node count {count} exceeds clutter threshold {MAX_VISIBLE_NODES}",
                severity="major",
                actual=count,
                expected=f"<= {MAX_VISIBLE_NODES}",
                fix_hint="collapse or remove_redundancy on non-essential nodes",
            )]
        return []

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_score(self, violations: list[Violation], total_nodes: int) -> LintResult:
        """
        Score = 1.0 - (weighted_violation_cost / max_possible_cost)

        Each violation contributes its severity weight.
        We cap max_possible_cost at a reasonable ceiling so the score
        never goes below 0.0 and doesn't saturate at 0 after just a few
        critical violations.
        """
        if not violations:
            return LintResult(
                violations=[],
                accessibility_score=1.0,
                total_nodes_checked=total_nodes,
                notes="No violations found. Perfect accessibility score.",
            )

        cost = sum(SEVERITY[v.severity] for v in violations)

        # Normalise: max expected cost is ~2.0 per node (generous ceiling)
        max_cost = max(total_nodes * 0.8, len(violations) * 0.5, 1.0)
        raw_score = 1.0 - (cost / max_cost)
        score = max(0.0, min(1.0, raw_score))

        critical = [v for v in violations if v.severity == "critical"]
        major    = [v for v in violations if v.severity == "major"]
        minor    = [v for v in violations if v.severity == "minor"]

        return LintResult(
            violations=violations,
            accessibility_score=round(score, 4),
            violation_count=len(violations),
            critical_count=len(critical),
            major_count=len(major),
            minor_count=len(minor),
            total_nodes_checked=total_nodes,
            notes=f"{len(violations)} violations found: {len(critical)} critical, {len(major)} major, {len(minor)} minor.",
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def lint(dom: DOMNode) -> LintResult:
    """Module-level shortcut: lint(dom) -> LintResult."""
    return AccessibilityLinter().lint(dom)
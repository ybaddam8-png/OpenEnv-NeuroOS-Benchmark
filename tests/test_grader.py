"""
tests/test_grader.py — Unit tests for the reward function
==========================================================
Folder structure expected:
    open_env/
        Logic/          <- all source files (schema, grader, linter, etc.)
        tests/          <- this file
        venv/

Run from the open_env/ root:
    pytest tests/test_grader.py -v -s
"""

import pytest
import copy
import sys
import os

# Logic/ is where all source files live (Gemini/Yashwanth folder structure)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "Logic"))

from tasks import TaskFactory
from grader import grade
from schema import MutationCommand, DOMNode


@pytest.fixture
def factory():
    return TaskFactory(base_seed=999)


class TestEasyGrader:
    def test_perfect_score(self, factory):
        """Fixing all contrast and ARIA violations → 1.0."""
        task = factory.create("easy")
        mutated = copy.deepcopy(task.dom)
        for node in mutated.all_nodes():
            if node.attributes.get("contrast_ratio", 5.0) < 4.5:
                node.attributes["contrast_ratio"] = 5.5
            if node.tag in {"button", "input", "select", "a"}:
                if not node.attributes.get("aria_label"):
                    node.attributes["aria_label"] = "Fixed label"
        result = grade("easy", task.dom, mutated)
        assert result.score == 1.0, f"Expected 1.0, got {result.score}"
        assert not result.exploit_detected
        print(f"\n[EASY] Perfect score: {result.score}")

    def test_no_fix_low_score(self, factory):
        """Sending DOM back unchanged → score is low but not zero (violations exist)."""
        task = factory.create("easy")
        result = grade("easy", task.dom, task.dom)
        assert result.score < 0.6, f"Unfixed DOM should score < 0.6, got {result.score}"
        print(f"\n[EASY] No-fix score: {result.score}")

    def test_no_efficiency_penalty_on_easy(self, factory):
        """Easy task has no efficiency penalty even with excess mutations."""
        task = factory.create("easy")
        mutated = copy.deepcopy(task.dom)
        # 200 dummy mutations — easy grader should ignore this
        many_mutations = [
            MutationCommand(op="set_contrast", node_id=task.dom.id, value=5.0)
            for _ in range(200)
        ]
        result = grade("easy", task.dom, mutated, mutation_log=many_mutations)
        # Score driven by fixes, not penalised for mutation count
        assert "efficiency" not in result.notes.lower() or result.score > 0
        print(f"\n[EASY] No efficiency penalty: {result.score}")


class TestMediumGrader:
    def test_partial_score_is_continuous(self, factory):
        """Fixing only contrast → score between 0 and 1, not binary."""
        task = factory.create("medium")
        mutated = copy.deepcopy(task.dom)
        for node in mutated.all_nodes():
            if node.attributes.get("contrast_ratio", 5.0) < 4.5:
                node.attributes["contrast_ratio"] = 5.5
        result = grade("medium", task.dom, mutated)
        assert 0.0 < result.score < 1.0, f"Expected partial score, got {result.score}"
        print(f"\n[MEDIUM] Partial fix score: {result.score} — continuous signal works")

    def test_efficiency_penalty_triggers_on_excess(self, factory):
        """Same fixed DOM: excess mutations score lower than minimal mutations."""
        task = factory.create("medium")
        from linter import lint
        orig_violations = lint(task.dom).violation_count
        mutated = copy.deepcopy(task.dom)
        for node in mutated.all_nodes():
            if node.attributes.get("contrast_ratio", 5.0) < 4.5:
                node.attributes["contrast_ratio"] = 5.5
            if node.tag in {"button", "input", "select", "a"}:
                node.attributes["aria_label"] = "Fixed"
            node.attributes["cognitive_weight"] = 0.3
        minimal = [MutationCommand(op="set_contrast", node_id=task.dom.id, value=5.5)
                   for _ in range(orig_violations)]
        excess  = [MutationCommand(op="set_contrast", node_id=task.dom.id, value=5.5)
                   for _ in range(orig_violations * 20)]
        result_minimal = grade("medium", task.dom, mutated, mutation_log=minimal)
        result_excess  = grade("medium", task.dom, mutated, mutation_log=excess)
        assert result_excess.score < result_minimal.score,             f"Excess ({result_excess.score:.4f}) should < minimal ({result_minimal.score:.4f})"
        assert result_excess.breakdown["efficiency_loss_pct"] > 0
        print(f"[MEDIUM] Minimal: {result_minimal.score:.4f}, Excess: {result_excess.score:.4f}")
    def test_efficiency_no_penalty_within_budget(self, factory):
        """Agent within 3× violation budget pays no efficiency penalty."""
        task = factory.create("medium")
        from linter import lint
        violations = lint(task.dom).violation_count
        within_budget = [
            MutationCommand(op="set_contrast", node_id=task.dom.id, value=5.0)
            for _ in range(violations * 2)   # 2× < 3× threshold
        ]
        mutated = copy.deepcopy(task.dom)
        result = grade("medium", task.dom, mutated, mutation_log=within_budget)
        assert result.breakdown.get("efficiency_loss_pct", 0) == 0.0, \
            "No penalty within budget"
        print(f"\n[MEDIUM] Within-budget no penalty: efficiency_loss={result.breakdown.get('efficiency_loss_pct')}")


class TestHardGrader:
    def test_biometric_cap_applies(self, factory):
        """Agent that ignores biometrics caps at 0.65 even with perfect visual fixes."""
        task = factory.create("hard")
        mutated = copy.deepcopy(task.dom)
        for node in mutated.all_nodes():
            node.attributes["contrast_ratio"] = 5.5
            node.attributes["animation_present"] = False
            if node.tag in {"button", "input", "select", "a"}:
                node.attributes["aria_label"] = "Fixed"
        # Empty mutation_log → no biometric responsiveness
        result = grade("hard", task.dom, mutated, task.biometrics, mutation_log=[])
        assert result.score <= 0.65, f"Biometric cap should apply, got {result.score}"
        assert result.breakdown["bio_capped"] is True
        print(f"\n[HARD] Biometric cap enforced: {result.score}")

    def test_mass_deletion_exploit(self, factory):
        """Deleting all children to fake zero clutter → score = 0.0."""
        task = factory.create("hard")
        mutated = copy.deepcopy(task.dom)
        mutated.children = []
        result = grade("hard", task.dom, mutated, task.biometrics)
        assert result.score == 0.0
        assert result.exploit_detected
        print(f"\n[HARD] Mass deletion caught: score={result.score}")

    def test_root_replacement_exploit(self, factory):
        """Returning a brand-new DOM root → score = 0.0."""
        task = factory.create("hard")
        fake_dom = DOMNode(id="hacked_root", tag="body")
        result = grade("hard", task.dom, fake_dom, task.biometrics)
        assert result.score == 0.0
        assert result.exploit_detected
        print(f"\n[HARD] Root replacement caught: score={result.score}")


class TestContextualTrap:
    def test_trap_detected_by_linter(self, factory):
        """R11 fires when an interactive child is inside an aria_hidden parent."""
        from linter import lint
        # Manually build a trap scenario
        from schema import DOMNode, make_attributes
        child = DOMNode("child1", "button", "Pay Now",
                        make_attributes(aria_label="Submit Payment", contrast_ratio=7.5))
        parent = DOMNode("parent1", "div", "",
                         make_attributes(aria_hidden="true"),
                         children=[child])
        root = DOMNode("root", "main", "", children=[parent])
        result = lint(root)
        r11_violations = [v for v in result.violations if v.rule == "R11"]
        assert len(r11_violations) > 0, "R11 should fire on aria_hidden parent with interactive child"
        assert r11_violations[0].node_id == "parent1"
        print(f"\n[TRAP] R11 correctly fires on parent, not child: {r11_violations[0].description}")

    def test_fixing_child_not_parent_scores_low(self, factory):
        """Agent that fixes the child's aria_label but not the parent aria_hidden gets low score."""
        from schema import DOMNode, make_attributes
        from linter import lint
        child_orig  = DOMNode("c1", "button", "Pay", make_attributes(aria_label="Submit Payment", contrast_ratio=7.5))
        parent_orig = DOMNode("p1", "div", "", make_attributes(aria_hidden="true"), children=[child_orig])
        root_orig   = DOMNode("root", "main", "", children=[parent_orig])

        # Agent "fixes" the child (unnecessary — it was already fine) but ignores parent
        child_mut  = DOMNode("c1", "button", "Pay", make_attributes(aria_label="Submit Payment Fixed", contrast_ratio=8.0))
        parent_mut = DOMNode("p1", "div", "", make_attributes(aria_hidden="true"), children=[child_mut])
        root_mut   = DOMNode("root", "main", "", children=[parent_mut])

        result_bad = grade("easy", root_orig, root_mut)

        # Agent correctly removes aria_hidden from parent
        parent_fix = DOMNode("p1", "div", "", make_attributes(aria_hidden="false"), children=[child_orig])
        root_fix   = DOMNode("root", "main", "", children=[parent_fix])
        result_good = grade("easy", root_orig, root_fix)

        assert result_good.score >= result_bad.score, \
            "Fixing parent should score >= fixing child only"
        print(f"\n[TRAP] Fixing child only: {result_bad.score:.4f}, fixing parent: {result_good.score:.4f}")
"""
preview.py — ASCII Visualizer for the Virtual DOM
==================================================
Color-coded tree view of any generated task.
Shows violations inline and optionally diffs before/after mutations.

Usage:
    cd person_b
    python preview.py                      # medium task, seed 42
    python preview.py easy 0               # easy task, seed 0
    python preview.py hard 7               # hard task, seed 7
    python preview.py medium 42 --diff     # show before + after (simulated fix)
"""

import sys
import os

# Flat path — works in Docker and locally when run from person_b/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Logic.tasks import TaskFactory
from Logic.linter import lint, LintResult
from Logic.schema import DOMNode
import copy


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{C.RESET}"

SEVERITY_COLOR = {
    "critical": C.RED,
    "major":    C.YELLOW,
    "minor":    C.CYAN,
}


# ---------------------------------------------------------------------------
# Tree printer
# ---------------------------------------------------------------------------

def print_dom(dom: DOMNode, label: str = "DOM Tree", show_attrs: bool = False):
    result = lint(dom)
    v_map: dict[str, list] = {}
    for v in result.violations:
        v_map.setdefault(v.node_id, []).append(v)

    print(f"\n{_c(C.BOLD + C.HEADER, f'--- {label} ---')}")
    print(f"{_c(C.BLUE, f'Nodes: {dom.node_count()}  ')}  "
          f"{_c(C.YELLOW, f'Violations: {result.violation_count}')}  "
          f"Score: {_c(C.GREEN if result.accessibility_score > 0.7 else C.RED, f'{result.accessibility_score:.3f}')}")
    print()
    _print_node(dom, v_map, "", is_last=True, show_attrs=show_attrs)
    print()

    if result.violations:
        print(_c(C.BOLD, "Violations summary:"))
        for v in result.violations[:15]:
            col = SEVERITY_COLOR.get(v.severity, C.RESET)
            print(f"  {_c(col, v.rule)} [{v.severity:8s}] {v.node_id:8s}  {v.description}")
            print(f"  {_c(C.DIM, f'  fix: {v.fix_hint}')}")
        if len(result.violations) > 15:
            print(f"  {_c(C.DIM, f'... and {len(result.violations) - 15} more')}")
    print()


def _print_node(node: DOMNode, v_map: dict, prefix: str, is_last: bool, show_attrs: bool):
    connector = "└── " if is_last else "├── "
    violations = v_map.get(node.id, [])

    # Node label
    tag_str  = _c(C.CYAN, f"<{node.tag}>")
    id_str   = _c(C.DIM,  f" #{node.id}")
    text_str = _c(C.DIM,  f' "{node.text[:24]}"') if node.text else ""

    if not violations:
        status = _c(C.GREEN, " ✓")
    else:
        worst    = max(violations, key=lambda v: {"critical": 2, "major": 1, "minor": 0}[v.severity])
        col      = SEVERITY_COLOR[worst.severity]
        rules    = ", ".join(v.rule for v in violations)
        status   = _c(col, f" ✗ [{rules}]")

    print(f"{prefix}{connector}{tag_str}{id_str}{text_str}{status}")

    if show_attrs and node.attributes:
        attr_prefix = prefix + ("    " if is_last else "│   ")
        interesting = {k: v for k, v in node.attributes.items()
                       if k in ("contrast_ratio", "aria_label", "aria_hidden",
                                "cognitive_weight", "animation_present", "is_hidden")}
        for k, v in interesting.items():
            print(f"{attr_prefix}    {_c(C.DIM, f'{k}: {v}')}")

    new_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        _print_node(child, v_map, new_prefix, i == len(node.children) - 1, show_attrs)


# ---------------------------------------------------------------------------
# Diff view
# ---------------------------------------------------------------------------

def print_diff(original: DOMNode, mutated: DOMNode):
    """Show side-by-side violation counts before and after."""
    orig_lint = lint(original)
    mut_lint  = lint(mutated)

    print(_c(C.BOLD + C.HEADER, "\n--- Before / After Diff ---\n"))

    rules = sorted({v.rule for v in orig_lint.violations + mut_lint.violations})
    print(f"  {'Rule':<8}  {'Before':>8}  {'After':>8}  {'Delta':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for rule in rules:
        before = sum(1 for v in orig_lint.violations if v.rule == rule)
        after  = sum(1 for v in mut_lint.violations  if v.rule == rule)
        delta  = after - before
        col    = C.GREEN if delta < 0 else (C.RED if delta > 0 else C.RESET)
        delta_str = _c(col, f"{delta:+d}")
        print(f"  {rule:<8}  {before:>8}  {after:>8}  {delta_str:>8}")

    print()
    score_before = orig_lint.accessibility_score
    score_after  = mut_lint.accessibility_score
    delta_score  = score_after - score_before
    col = C.GREEN if delta_score > 0 else (C.RED if delta_score < 0 else C.RESET)
    print(f"  Score: {score_before:.4f} → {score_after:.4f}  "
          f"({_c(col, f'{delta_score:+.4f}')})")
    print()


# ---------------------------------------------------------------------------
# Simulated fix (for --diff demo)
# ---------------------------------------------------------------------------

def simulate_perfect_fix(dom: DOMNode) -> DOMNode:
    """Simulate an agent that fixes all obvious violations."""
    fixed = copy.deepcopy(dom)
    for node in fixed.all_nodes():
        if node.attributes.get("contrast_ratio", 5.0) < 4.5:
            node.attributes["contrast_ratio"] = 5.5
        if node.tag in {"button", "input", "select", "a"}:
            if not node.attributes.get("aria_label"):
                node.attributes["aria_label"] = "Fixed label"
        if node.attributes.get("animation_present"):
            node.attributes["animation_present"] = False
        if node.attributes.get("cognitive_weight", 0) > 0.7:
            node.attributes["cognitive_weight"] = 0.5
        if node.attributes.get("sensory_load", 0) > 0.6:
            node.attributes["sensory_load"] = 0.4
        if node.attributes.get("font_size_px", 16) < 12:
            node.attributes["font_size_px"] = 14.0
        # Fix contextual trap: remove aria_hidden from parents with interactive children
        has_interactive_child = any(c.tag in {"button","input","select","a"}
                                    for c in node.children)
        if node.attributes.get("aria_hidden") == "true" and has_interactive_child:
            node.attributes["aria_hidden"] = "false"
    return fixed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args      = sys.argv[1:]
    difficulty = args[0] if len(args) > 0 else "medium"
    seed       = int(args[1]) if len(args) > 1 else 42
    show_diff  = "--diff" in args
    show_attrs = "--attrs" in args

    if difficulty not in ("easy", "medium", "hard"):
        print(f"Usage: python preview.py [easy|medium|hard] [seed] [--diff] [--attrs]")
        sys.exit(1)

    factory = TaskFactory(base_seed=seed)
    task    = factory.create(difficulty, seed=seed)

    print(_c(C.BOLD, f"\nOpenEnv Preview — {difficulty.upper()} task (seed={seed})"))
    if task.biometrics:
        from biometric_simulator import BiometricSimulator
        summary = BiometricSimulator.compute_stress_summary(task.biometrics)
        print(f"Biometrics: {len(task.biometrics)} events  "
              f"mean_cog_load={summary['mean_cognitive_load']:.1f}  "
              f"rage_clicks={summary['rage_click_count']}")

    print_dom(task.dom, label=f"{difficulty} DOM (before)", show_attrs=show_attrs)

    if show_diff:
        fixed = simulate_perfect_fix(task.dom)
        print_dom(fixed, label=f"{difficulty} DOM (after simulated fix)", show_attrs=show_attrs)
        print_diff(task.dom, fixed)
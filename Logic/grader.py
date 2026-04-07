"""
grader.py — Deterministic Reward Function (0.0 – 1.0)
======================================================
Three graders: easy, medium, hard.
All return a GradeResult with granular sub-scores and exploit detection.

KEY DESIGN PRINCIPLES:
  1. Never sparse reward — always granular between 0.0 and 1.0
  2. Exploit-proof — deleting the DOM = 0.0, not 1.0
  3. Fixing half the issues ≈ 0.5
  4. Efficiency penalty — spamming mutations beyond 3× violation count costs score
  5. Biometric cap — hard task agents that ignore stress signals cap at 0.65

EXPLOIT CHECKS (all tasks):
  - Root node deleted or replaced          → score = 0.0 hard zero
  - Node count drops below MIN_NODE_RATIO  → -0.4 penalty
  - More than MAX_HIDDEN_RATIO hidden      → -0.35 penalty
  - Critical structural tags removed       → -0.25 per tag
  - Invalid mutation commands              → -0.05 each

EFFICIENCY PENALTY (medium + hard only):
  Budget = violation_count × 3 (two retries per violation, free).
  Beyond the budget, penalty scales up to -20% of score.
  Easy task is exempt — it is a learning task, not a speed test.

Sub-score weights:
  Easy:   contrast_fix(0.70) + aria_fix(0.30)
  Medium: contrast_fix(0.35) + aria_fix(0.15) + redundancy_fix(0.30) + cognitive_fix(0.20)
          + sensory_bonus(up to +0.05) − efficiency_loss(up to 20%)
  Hard:   accessibility_delta(0.40) + structural_integrity(0.20) + biometric_response(0.20)
          + depth_improvement(0.10) + animation_fix(0.10)
          − efficiency_loss(up to 20%), score capped at 0.65 if biometrics ignored
"""

from __future__ import annotations
from Logic.schema import DOMNode, MutationCommand, GradeResult
from Logic.linter import lint, LintResult, MIN_CONTRAST_RATIO


# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

MIN_NODE_RATIO           = 0.6
MAX_HIDDEN_RATIO         = 0.5

PENALTY_BODY_DELETED     = 1.0
PENALTY_MASS_DELETION    = 0.4
PENALTY_MASS_HIDING      = 0.35
PENALTY_STRUCTURAL_LOSS  = 0.25
PENALTY_INVALID_MUTATION = 0.05

CRITICAL_STRUCTURAL_TAGS = {"main", "nav", "header", "footer", "body"}


# ---------------------------------------------------------------------------
# Exploit detector
# ---------------------------------------------------------------------------

def _detect_exploits(
    original: DOMNode,
    mutated: DOMNode,
    mutation_log: list[MutationCommand],
) -> tuple[list[dict], bool]:
    """Returns (penalties_list, exploit_detected)."""
    penalties = []
    exploit   = False

    orig_nodes = original.all_nodes()
    mut_nodes  = mutated.all_nodes()

    # 1. Root must survive
    if mutated.id != original.id:
        penalties.append({"reason": "Root node replaced or deleted",
                          "amount": PENALTY_BODY_DELETED})
        return penalties, True

    # 2. Critical structural tags must survive
    orig_structural = {n.id for n in orig_nodes if n.tag in CRITICAL_STRUCTURAL_TAGS}
    mut_structural  = {n.id for n in mut_nodes  if n.tag in CRITICAL_STRUCTURAL_TAGS}
    lost = orig_structural - mut_structural
    if lost:
        penalties.append({"reason": f"Critical structural nodes removed: {lost}",
                          "amount": PENALTY_STRUCTURAL_LOSS * len(lost)})
        exploit = True

    # 3. Mass deletion
    ratio = len(mut_nodes) / max(len(orig_nodes), 1)
    if ratio < MIN_NODE_RATIO:
        penalties.append({"reason": f"Node count dropped to {ratio:.0%} (min {MIN_NODE_RATIO:.0%})",
                          "amount": PENALTY_MASS_DELETION})
        exploit = True

    # 4. Mass hiding — faking a clean DOM by hiding everything
    hidden      = sum(1 for n in mut_nodes if n.attributes.get("is_hidden", False))
    hidden_ratio = hidden / max(len(mut_nodes), 1)
    if hidden_ratio > MAX_HIDDEN_RATIO:
        penalties.append({"reason": f"{hidden_ratio:.0%} of nodes hidden (max {MAX_HIDDEN_RATIO:.0%})",
                          "amount": PENALTY_MASS_HIDING})
        exploit = True

    # 5. Invalid mutation commands
    for cmd in mutation_log:
        valid, err = cmd.validate()
        if not valid:
            penalties.append({"reason": f"Invalid mutation: {err}",
                              "amount": PENALTY_INVALID_MUTATION})

    return penalties, exploit


def _apply_penalties(raw_score: float, penalties: list[dict]) -> float:
    total = sum(p["amount"] for p in penalties)
    return max(0.0, min(1.0, raw_score - total))


# ---------------------------------------------------------------------------
# Efficiency penalty  (violation-budget-aware — Bug-fixed vs Gemini version)
# ---------------------------------------------------------------------------

def _apply_efficiency_penalty(
    raw_score: float,
    mutation_count: int,
    violation_count: int,
) -> tuple[float, float]:
    """
    Penalises agents that spam far more mutations than violations exist.

    Budget = violation_count × 3  (fix + two retries per violation, free).
    Penalty only starts beyond the free budget. Max penalty = 20% of score.

    Examples with 10 violations (budget = 30):
      15 mutations →  0% penalty   (within budget)
      30 mutations →  0% penalty   (at threshold)
      60 mutations → ~10% penalty
     120 mutations →  20% penalty  (capped)

    WHY NOT node-count-based: a task with 54 nodes but 121 violations needs
    more than 54×2=108 mutations to fix correctly. The original Gemini formula
    would have penalised a correct agent heavily. Violation count is the right
    denominator because it reflects the actual work required.

    Returns (penalised_score, penalty_fraction_applied).
    """
    if violation_count == 0 or mutation_count == 0:
        return raw_score, 0.0

    free_allowance = violation_count * 3
    if mutation_count <= free_allowance:
        return raw_score, 0.0

    excess        = mutation_count - free_allowance
    penalty_ratio = min(0.20, excess / max(violation_count * 10, 1))
    return raw_score * (1.0 - penalty_ratio), round(penalty_ratio, 4)


# ---------------------------------------------------------------------------
# Violation fix-rate helpers
# ---------------------------------------------------------------------------

def _violation_fix_rate(orig_lint: LintResult, mut_lint: LintResult, rule: str) -> float:
    orig_count = sum(1 for v in orig_lint.violations if v.rule == rule)
    if orig_count == 0:
        return 1.0
    mut_count = sum(1 for v in mut_lint.violations if v.rule == rule)
    return max(0, orig_count - mut_count) / orig_count


# ---------------------------------------------------------------------------
# EASY GRADER  (no efficiency penalty)
# ---------------------------------------------------------------------------

def grade_easy(
    original_dom: DOMNode,
    mutated_dom: DOMNode,
    mutation_log: list[MutationCommand] = None,
) -> GradeResult:
    """Fix contrast ratios and missing ARIA labels. No efficiency penalty."""
    mutation_log = mutation_log or []

    penalties, exploit = _detect_exploits(original_dom, mutated_dom, mutation_log)
    if exploit and any(p["amount"] >= PENALTY_BODY_DELETED for p in penalties):
        return GradeResult(score=0.0001, exploit_detected=True, penalties=penalties,
                           notes="Exploit: root tampered. Score = 0.0001.")

    orig_lint    = lint(original_dom)
    mut_lint     = lint(mutated_dom)
    contrast_fix = _violation_fix_rate(orig_lint, mut_lint, "R01")
    aria_fix     = _violation_fix_rate(orig_lint, mut_lint, "R08")
    raw_score    = 0.70 * contrast_fix + 0.30 * aria_fix
    final_score  = _apply_penalties(raw_score, penalties)

    return GradeResult(
        score=max(0.0001, min(0.9999, round(final_score, 4))),
        breakdown={"contrast_fix": round(contrast_fix, 4),
                   "aria_fix":     round(aria_fix, 4),
                   "raw_score":    round(raw_score, 4)},
        penalties=penalties,
        exploit_detected=exploit,
        notes=(f"Easy. Contrast: {contrast_fix:.0%}, ARIA: {aria_fix:.0%}. "
               f"Final: {final_score:.4f}."),
    )


# ---------------------------------------------------------------------------
# MEDIUM GRADER  (efficiency penalty active)
# ---------------------------------------------------------------------------

def grade_medium(
    original_dom: DOMNode,
    mutated_dom: DOMNode,
    mutation_log: list[MutationCommand] = None,
) -> GradeResult:
    """Contrast + ARIA + redundancy + cognitive load. Efficiency penalty active."""
    mutation_log = mutation_log or []

    penalties, exploit = _detect_exploits(original_dom, mutated_dom, mutation_log)
    if exploit and any(p["amount"] >= PENALTY_BODY_DELETED for p in penalties):
        return GradeResult(score=0.0001, exploit_detected=True, penalties=penalties,
                           notes="Exploit: root tampered. Score = 0.0001.")

    orig_lint      = lint(original_dom)
    mut_lint       = lint(mutated_dom)
    contrast_fix   = _violation_fix_rate(orig_lint, mut_lint, "R01")
    aria_fix       = _violation_fix_rate(orig_lint, mut_lint, "R08")
    cognitive_fix  = _violation_fix_rate(orig_lint, mut_lint, "R05")
    redundancy_fix = _score_redundancy_fix(original_dom, mutated_dom, orig_lint, mut_lint)

    raw_score = (0.35 * contrast_fix + 0.15 * aria_fix
                 + 0.30 * redundancy_fix + 0.20 * cognitive_fix)

    # Sensory load bonus (up to +0.05)
    orig_s = sum(1 for v in orig_lint.violations if v.rule == "R06")
    mut_s  = sum(1 for v in mut_lint.violations  if v.rule == "R06")
    sensory_bonus = 0.05 * ((orig_s - mut_s) / orig_s) if orig_s > 0 and mut_s < orig_s else 0.0
    raw_score = min(1.0, raw_score + sensory_bonus)

    after_penalties = _apply_penalties(raw_score, penalties)

    # Efficiency penalty uses violation count as budget denominator
    violation_count = orig_lint.violation_count
    final_score, eff_loss = _apply_efficiency_penalty(
        after_penalties, len(mutation_log), violation_count
    )
    if eff_loss > 0:
        penalties.append({
            "reason": (f"Efficiency: {len(mutation_log)} mutations, "
                       f"{violation_count} violations, budget={violation_count * 3}"),
            "amount": round(eff_loss * after_penalties, 4),
        })

    return GradeResult(
        score=max(0.0001, min(0.9999, round(final_score, 4))),
        breakdown={"contrast_fix":       round(contrast_fix, 4),
                   "aria_fix":           round(aria_fix, 4),
                   "redundancy_fix":     round(redundancy_fix, 4),
                   "cognitive_fix":      round(cognitive_fix, 4),
                   "sensory_bonus":      round(sensory_bonus, 4),
                   "efficiency_loss_pct":round(eff_loss, 4),
                   "raw_score":          round(raw_score, 4)},
        penalties=penalties,
        exploit_detected=exploit,
        notes=(f"Medium. Contrast: {contrast_fix:.0%}, ARIA: {aria_fix:.0%}, "
               f"Redundancy: {redundancy_fix:.0%}, Cognitive: {cognitive_fix:.0%}. "
               f"Efficiency loss: {eff_loss:.0%}. Final: {final_score:.4f}."),
    )


def _score_redundancy_fix(
    original: DOMNode, mutated: DOMNode,
    orig_lint: LintResult, mut_lint: LintResult,
) -> float:
    orig_nodes = {n.id: n for n in original.all_nodes()}
    mut_nodes  = {n.id: n for n in mutated.all_nodes()}
    true_redundant = {
        nid for nid, n in orig_nodes.items()
        if "redundancy" in n.metadata.get("violations", [])
    }
    if not true_redundant:
        return _violation_fix_rate(orig_lint, mut_lint, "R04")

    correctly_fixed = 0
    false_positives = 0
    for nid in orig_nodes:
        if nid not in mut_nodes:
            continue
        hidden = mut_nodes[nid].attributes.get("is_hidden", False)
        if hidden and nid in true_redundant:
            correctly_fixed += 1
        elif hidden and nid not in true_redundant:
            false_positives += 1

    precision  = correctly_fixed / max(len(true_redundant), 1)
    fp_penalty = min(0.5, false_positives * 0.1)
    return max(0.0, precision - fp_penalty)


# ---------------------------------------------------------------------------
# HARD GRADER  (efficiency penalty + biometric cap)
# ---------------------------------------------------------------------------

def grade_hard(
    original_dom: DOMNode,
    mutated_dom: DOMNode,
    biometric_stream: list[dict],
    mutation_log: list[MutationCommand] = None,
) -> GradeResult:
    """Full accessibility + structural integrity + biometric responsiveness."""
    mutation_log = mutation_log or []

    penalties, exploit = _detect_exploits(original_dom, mutated_dom, mutation_log)
    if exploit and any(p["amount"] >= PENALTY_BODY_DELETED for p in penalties):
        return GradeResult(score=0.0001, exploit_detected=True, penalties=penalties,
                           notes="Exploit: root tampered. Score = 0.0001.")

    orig_lint = lint(original_dom)
    mut_lint  = lint(mutated_dom)

    if orig_lint.accessibility_score >= 1.0:
        accessibility_delta = 1.0
    else:
        improvement   = mut_lint.accessibility_score - orig_lint.accessibility_score
        max_possible  = 1.0 - orig_lint.accessibility_score
        accessibility_delta = max(0.0, improvement / max(max_possible, 0.001))

    structural_integrity = _score_structural_integrity(original_dom, mutated_dom)
    biometric_response   = _score_biometric_response(
        original_dom, mutated_dom, mutation_log, biometric_stream
    )
    depth_improvement = _violation_fix_rate(orig_lint, mut_lint, "R03")
    animation_fix     = _violation_fix_rate(orig_lint, mut_lint, "R07")

    raw_score = (0.40 * accessibility_delta
                 + 0.20 * structural_integrity
                 + 0.20 * biometric_response
                 + 0.10 * depth_improvement
                 + 0.10 * animation_fix)

    # Biometric cap
    bio_capped = False
    if biometric_response < 0.2:
        raw_score  = min(raw_score, 0.65)
        bio_capped = True
        penalties.append({"reason": "Biometric signals ignored (response < 0.2). Capped at 0.65.",
                          "amount": 0.0})

    after_penalties = _apply_penalties(raw_score, penalties)

    violation_count = orig_lint.violation_count
    final_score, eff_loss = _apply_efficiency_penalty(
        after_penalties, len(mutation_log), violation_count
    )
    if eff_loss > 0:
        penalties.append({
            "reason": (f"Efficiency: {len(mutation_log)} mutations, "
                       f"{violation_count} violations, budget={violation_count * 3}"),
            "amount": round(eff_loss * after_penalties, 4),
        })

    return GradeResult(
        score=max(0.0001, min(0.9999, round(final_score, 4))),
        breakdown={"accessibility_delta":  round(accessibility_delta, 4),
                   "structural_integrity": round(structural_integrity, 4),
                   "biometric_response":   round(biometric_response, 4),
                   "depth_improvement":    round(depth_improvement, 4),
                   "animation_fix":        round(animation_fix, 4),
                   "efficiency_loss_pct":  round(eff_loss, 4),
                   "bio_capped":           bio_capped,
                   "raw_score":            round(raw_score, 4)},
        penalties=penalties,
        exploit_detected=exploit,
        notes=(f"Hard. Accessibility Δ: {accessibility_delta:.0%}, "
               f"Structural: {structural_integrity:.0%}, Biometric: {biometric_response:.0%}, "
               f"Depth: {depth_improvement:.0%}, Animation: {animation_fix:.0%}. "
               f"Efficiency loss: {eff_loss:.0%}. Final: {final_score:.4f}."),
    )


def _score_structural_integrity(original: DOMNode, mutated: DOMNode) -> float:
    orig_nodes  = original.all_nodes()
    mut_nodes   = mutated.all_nodes()
    important   = {"h1", "h2", "h3", "button", "input", "nav", "main", "header"}
    orig_imp    = {n.tag for n in orig_nodes} & important
    mut_imp     = {n.tag for n in mut_nodes}  & important
    if not orig_imp:
        return 1.0
    preserved   = len(mut_imp) / len(orig_imp)
    count_score = min(1.0, (len(mut_nodes) / max(len(orig_nodes), 1)) / MIN_NODE_RATIO)
    return round(0.7 * preserved + 0.3 * count_score, 4)


def _score_biometric_response(
    original: DOMNode, mutated: DOMNode,
    mutation_log: list[MutationCommand],
    biometric_stream: list[dict],
) -> float:
    if not biometric_stream:
        return 0.5
    stressed: dict[str, int] = {}
    for ev in biometric_stream:
        if ev.get("cognitive_load", 0) > 65 or ev.get("rage_click", False):
            nid = ev.get("node_hover_id", "")
            if nid:
                stressed[nid] = stressed.get(nid, 0) + 1
    if not stressed:
        return 0.5
    if not mutation_log:
        return 0.0
    on_stressed  = sum(1 for cmd in mutation_log if cmd.node_id in stressed)
    responsiveness = on_stressed / len(mutation_log)
    top_3      = sorted(stressed, key=lambda k: -stressed[k])[:3]
    top_bonus  = min(0.15, sum(1 for cmd in mutation_log if cmd.node_id in top_3) * 0.05)
    return min(1.0, round(responsiveness + top_bonus, 4))


# ---------------------------------------------------------------------------
# NEW TASK GRADERS
# ---------------------------------------------------------------------------

def grade_cognitive_load(
    original_dom: DOMNode,
    mutated_dom: DOMNode,
    mutation_log: list[MutationCommand] = None,
) -> GradeResult:
    mutation_log = mutation_log or []
    penalties, exploit = _detect_exploits(original_dom, mutated_dom, mutation_log)
    if exploit and any(p["amount"] >= PENALTY_BODY_DELETED for p in penalties):
        return GradeResult(score=0.0001, exploit_detected=True, penalties=penalties,
                           notes="Exploit: root tampered. Score = 0.0001.")

    orig_nodes = {n.id: n for n in original_dom.all_nodes()}
    
    redundant_target = [nid for nid, n in orig_nodes.items() if "redundancy" in n.metadata.get("violations", [])]
    dense_text_target = [nid for nid, n in orig_nodes.items() if "high_cognitive_weight" in n.metadata.get("violations", [])]

    correct_deletes = sum(1 for cmd in mutation_log if cmd.op == "delete_node" and cmd.node_id in redundant_target)
    correct_simplifies = sum(1 for cmd in mutation_log if cmd.op in ("simplify_text", "set_cognitive_weight") and cmd.node_id in dense_text_target)

    del_score = correct_deletes / max(len(redundant_target), 1) if redundant_target else 1.0
    simp_score = correct_simplifies / max(len(dense_text_target), 1) if dense_text_target else 1.0

    raw_score = 0.5 * del_score + 0.5 * simp_score
    final_score = _apply_penalties(raw_score, penalties)

    return GradeResult(
        score=max(0.0001, min(0.9999, round(final_score, 4))),
        breakdown={"del_score": round(del_score, 4), "simp_score": round(simp_score, 4), "raw_score": round(raw_score, 4)},
        penalties=penalties,
        exploit_detected=exploit,
        notes=f"Cognitive Load. Final: {final_score:.4f}.",
    )

def grade_sensory_overload(
    original_dom: DOMNode,
    mutated_dom: DOMNode,
    mutation_log: list[MutationCommand] = None,
) -> GradeResult:
    mutation_log = mutation_log or []
    penalties, exploit = _detect_exploits(original_dom, mutated_dom, mutation_log)
    if exploit and any(p["amount"] >= PENALTY_BODY_DELETED for p in penalties):
        return GradeResult(score=0.0001, exploit_detected=True, penalties=penalties,
                           notes="Exploit: root tampered. Score = 0.0001.")

    orig_nodes = {n.id: n for n in original_dom.all_nodes()}
    
    autoplay_target = [nid for nid, n in orig_nodes.items() if "autoplay_video" in n.metadata.get("violations", [])]
    animation_target = [nid for nid, n in orig_nodes.items() if "animation" in n.metadata.get("violations", [])]

    correct_autoplay = sum(1 for cmd in mutation_log if cmd.op == "disable_autoplay" and cmd.node_id in autoplay_target)
    correct_animation = sum(1 for cmd in mutation_log if cmd.op in ("remove_animation", "stop_animation") and cmd.node_id in animation_target)

    auto_score = correct_autoplay / max(len(autoplay_target), 1) if autoplay_target else 1.0
    anim_score = correct_animation / max(len(animation_target), 1) if animation_target else 1.0

    raw_score = 0.5 * auto_score + 0.5 * anim_score
    final_score = _apply_penalties(raw_score, penalties)

    return GradeResult(
        score=max(0.0001, min(0.9999, round(final_score, 4))),
        breakdown={"auto_score": round(auto_score, 4), "anim_score": round(anim_score, 4), "raw_score": round(raw_score, 4)},
        penalties=penalties,
        exploit_detected=exploit,
        notes=f"Sensory Overload. Final: {final_score:.4f}.",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade(
    task_name: str,
    difficulty: str,
    original_dom: DOMNode,
    mutated_dom: DOMNode,
    biometric_stream: list[dict] = None,
    mutation_log: list[MutationCommand] = None,
) -> GradeResult:
    """grade(task_name, difficulty, orig, mutated, bio?, mutations?) -> GradeResult"""
    mutation_log     = mutation_log or []
    biometric_stream = biometric_stream or []
    
    if task_name == "cognitive-load-reduction":
        return grade_cognitive_load(original_dom, mutated_dom, mutation_log)
    if task_name == "sensory-overload-prevention":
        return grade_sensory_overload(original_dom, mutated_dom, mutation_log)

    if difficulty == "easy":
        return grade_easy(original_dom, mutated_dom, mutation_log)
    elif difficulty == "medium":
        return grade_medium(original_dom, mutated_dom, mutation_log)
    elif difficulty == "hard":
        return grade_hard(original_dom, mutated_dom, biometric_stream, mutation_log)
    else:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Must be easy/medium/hard.")
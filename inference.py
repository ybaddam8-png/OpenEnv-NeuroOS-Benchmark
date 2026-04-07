import os
import json
import socket
import requests
import httpx
from typing import List, Optional, Set, Tuple
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
ENV_URL      = os.getenv("ENV_URL",      "http://127.0.0.1:7860")

TASK_NAME = "neuro-inclusive-audit"
BENCHMARK = "NEXUS-NeuroOS"
MAX_STEPS = 8

# ---------------------------------------------------------------------------
# Logging (unchanged)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# OpenAI client (unchanged)
# ---------------------------------------------------------------------------
def build_openai_client() -> OpenAI:
    host = API_BASE_URL.replace("https://", "").replace("http://", "").split("/")[0]
    try:
        ip = socket.gethostbyname(host)
        print(f"[DEBUG] DNS resolved {host} -> {ip}", flush=True)
    except Exception as exc:
        print(f"[DEBUG] DNS resolution failed: {exc}", flush=True)

    transport = httpx.HTTPTransport(retries=3, local_address="0.0.0.0")
    http_client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(120.0, connect=10.0, write=10.0, pool=5.0),
    )
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        http_client=http_client,
        max_retries=2,
    )

# ---------------------------------------------------------------------------
# Environment health check (unchanged)
# ---------------------------------------------------------------------------
def check_env_health() -> bool:
    try:
        res = requests.get(f"{ENV_URL}/", timeout=5)
        return res.status_code < 500
    except Exception:
        return False

# ---------------------------------------------------------------------------
# FIXED FALLBACK – always finds any untried (node_id, op) pair
# ---------------------------------------------------------------------------
def get_fallback_actions(obs: dict, attempted_set: Set[Tuple[str, str]], max_actions: int = 4) -> List[dict]:
    nodes = obs.get("nodes", [])
    if not nodes:
        return []

    actions = []
    # Priority order for ops (heuristic)
    ops_priority = ["set_aria_label", "set_contrast", "set_font_size"]

    # First pass: use heuristics (missing aria_label, low contrast, small font)
    for node in nodes:
        nid = node.get("id", "node0")
        if not node.get("aria_label") and (nid, "set_aria_label") not in attempted_set:
            label = f"{node.get('type', 'element').capitalize()}_{nid}"
            actions.append({"op": "set_aria_label", "node_id": nid, "value": label})
        elif node.get("contrast", 99) < 4.5 and (nid, "set_contrast") not in attempted_set:
            actions.append({"op": "set_contrast", "node_id": nid, "value": 4.5})
        elif node.get("font_size", 99) < 16 and (nid, "set_font_size") not in attempted_set:
            actions.append({"op": "set_font_size", "node_id": nid, "value": 16})
        if len(actions) >= max_actions:
            return actions[:max_actions]

    # Second pass: if still need more, add ANY (node, op) not yet attempted
    # Iterate in deterministic order (node ids sorted, ops in priority order)
    if len(actions) < max_actions:
        for node in nodes:
            nid = node.get("id", "node0")
            for op in ops_priority:
                key = (nid, op)
                if key not in attempted_set:
                    # Choose a sensible default value for the op
                    if op == "set_contrast":
                        val = 4.5
                    elif op == "set_aria_label":
                        val = f"{node.get('type', 'element').capitalize()}_{nid}"
                    else:  # set_font_size
                        val = 16
                    actions.append({"op": op, "node_id": nid, "value": val})
                    if len(actions) >= max_actions:
                        return actions[:max_actions]

    return actions[:max_actions]

# ---------------------------------------------------------------------------
# Deduplicate actions (unchanged)
# ---------------------------------------------------------------------------
def deduplicate_actions(actions: List[dict], attempted_set: Set[Tuple[str, str]]) -> List[dict]:
    unique = []
    for a in actions:
        key = (a.get("node_id"), a.get("op"))
        if key not in attempted_set:
            unique.append(a)
    return unique

# ---------------------------------------------------------------------------
# Prompt (includes attempted_set)
# ---------------------------------------------------------------------------
def build_prompt(obs: dict, reward_history: List[float], action_history: List[list], attempted_set: Set[Tuple[str, str]]) -> str:
    last_reward = reward_history[-1] if reward_history else None
    last_action = action_history[-1] if action_history else None

    if last_reward is None:
        feedback = "First step. Fix the most critical accessibility issues."
    elif last_reward > 0:
        feedback = f"Last reward: +{last_reward:.2f}. Good. Fix OTHER nodes not yet touched."
    elif last_reward < 0:
        feedback = f"Last reward: {last_reward:.2f} (PENALTY). Do NOT repeat: {json.dumps(last_action)}."
    else:
        feedback = "Last reward: 0.0. Those nodes are already fixed. Move to different nodes/ops."

    nodes = [
        {"id": n.get("id"), "type": n.get("type"),
         "contrast": n.get("contrast"), "aria_label": n.get("aria_label"),
         "font_size": n.get("font_size")}
        for n in obs.get("nodes", [])
    ]

    attempted_list = [{"node_id": nid, "op": op} for (nid, op) in attempted_set]

    return (
        "You are a Neuro-Inclusive UI Auditor. Maximise accessibility reward.\n\n"
        f"NODES: {json.dumps(nodes)}\n"
        f"ALREADY ATTEMPTED (do NOT repeat these): {json.dumps(attempted_list)}\n"
        f"FEEDBACK: {feedback}\n\n"
        "RULES:\n"
        "- set_contrast: value>=4.5 where contrast<4.5\n"
        "- set_aria_label: descriptive string where aria_label is null\n"
        "- set_font_size: value>=16 where font_size<16\n"
        "- Never repeat a node_id+op pair already in history\n"
        "- Max 4 actions per step\n\n"
        'OUTPUT: valid JSON only, no markdown.\n'
        '{"actions":[{"op":"set_contrast","node_id":"n1","value":4.5}]}'
    )

# ---------------------------------------------------------------------------
# LLM caller (unchanged)
# ---------------------------------------------------------------------------
def get_llm_action(
    client: OpenAI,
    obs: dict,
    reward_history: List[float],
    action_history: List[list],
    attempted_set: Set[Tuple[str, str]],
) -> Optional[list]:
    prompt = build_prompt(obs, reward_history, action_history, attempted_set)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
            timeout=120.0,
        )
        content = (completion.choices[0].message.content or "{}").strip()
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else content
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        raw = json.loads(content).get("actions", [])
        clean = [
            {k: v for k, v in a.items() if k in ("op", "node_id", "value")}
            for a in raw if "op" in a and "node_id" in a
        ]
        return clean if clean else None
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return None

# ---------------------------------------------------------------------------
# Safe JSON parser (unchanged)
# ---------------------------------------------------------------------------
def safe_parse(res: requests.Response, label: str) -> Optional[dict]:
    if not res.text.strip():
        return None
    try:
        return res.json()
    except json.JSONDecodeError:
        return None

# ---------------------------------------------------------------------------
# Main – with corrected early termination logic
# ---------------------------------------------------------------------------
def main() -> None:
    rewards:        List[float] = []
    action_history: List[list]  = []
    attempted_set:  Set[Tuple[str, str]] = set()
    steps_taken = 0
    score       = 0.0
    success     = False
    done        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    if not check_env_health():
        print("[DEBUG] Environment unreachable. Aborting.", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    client = build_openai_client()

    try:
        reset_res = requests.post(f"{ENV_URL}/reset", json={}, timeout=10)
        reset_res.raise_for_status()
        obs = safe_parse(reset_res, "/reset") or {}

        for step in range(1, MAX_STEPS + 1):
            # 1. Get raw actions (LLM or fallback)
            raw_actions = get_llm_action(client, obs, rewards, action_history, attempted_set)
            if raw_actions is None:
                raw_actions = get_fallback_actions(obs, attempted_set)
                print(f"[DEBUG] Step {step}: LLM unavailable — using smart fallback.", flush=True)

            # 2. Deduplicate: remove any already attempted
            actions = deduplicate_actions(raw_actions, attempted_set)

            # 3. If no actions after dedup, try to generate fresh ones
            if not actions:
                print(f"[DEBUG] Step {step}: No new actions after dedup. Generating fresh fallback actions.", flush=True)
                actions = get_fallback_actions(obs, attempted_set)
                if not actions:
                    # No legal actions left – episode ends naturally
                    print("[DEBUG] No remaining untried (node, op) pairs. Ending episode.", flush=True)
                    break

            # 4. Record these actions as attempted
            for a in actions:
                attempted_set.add((a.get("node_id"), a.get("op")))

            action_history.append(actions)
            action_str = json.dumps(actions).replace(" ", "")

            # 5. Send to environment
            step_res = requests.post(
                f"{ENV_URL}/step",
                json={"actions": actions},
                timeout=10,
            )

            if step_res.status_code == 422:
                print(f"[DEBUG] Step {step}: 422 — payload={json.dumps({'actions': actions})}", flush=True)
                break

            step_res.raise_for_status()
            result = safe_parse(step_res, f"/step {step}")
            if result is None:
                break

            obs    = result.get("observation", {})
            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))
            error  = result.get("error", None)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score   = min(max(sum(rewards) / len(rewards), 0.0), 1.0) if rewards else 0.0
        success = done

    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
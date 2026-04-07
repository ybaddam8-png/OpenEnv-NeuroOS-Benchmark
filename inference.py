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
def get_fallback_actions(obs: dict, attempted_set: Set[Tuple[str, str]], task_name: str, max_actions: int = 4) -> List[dict]:
    nodes = obs.get("nodes", []) or []
    nodes = obs.get("dom", {}).get("children", []) if "dom" in obs and not nodes else nodes
    
    # Flatten the dom to find nodes if they are structured nested
    def flatten_dom(node, acc=None):
        if acc is None: acc = []
        if isinstance(node, dict):
            acc.append(node)
            for child in node.get("children", []):
                flatten_dom(child, acc)
        return acc
    
    if "dom" in obs:
        nodes = flatten_dom(obs["dom"])

    if not nodes:
        return []

    actions = []
    
    if task_name == "cognitive-load-reduction":
        ops_priority = ["delete_node", "simplify_text"]
    elif task_name == "sensory-overload-prevention":
        ops_priority = ["disable_autoplay", "remove_animation", "stop_animation"]
    else:
        ops_priority = ["set_aria_label", "set_contrast", "set_font_size"]

    # First pass: use heuristics (missing aria_label, low contrast, small font)
    for node in nodes:
        nid = node.get("id", "node0")
        attrs = node.get("attributes", {})
        
        if task_name == "cognitive-load-reduction":
            if attrs.get("is_redundant", False) and (nid, "delete_node") not in attempted_set:
                actions.append({"op": "delete_node", "node_id": nid})
            elif attrs.get("cognitive_weight", 0) > 0.5 and (nid, "simplify_text") not in attempted_set:
                actions.append({"op": "simplify_text", "node_id": nid, "value": "Simplified text"})
        elif task_name == "sensory-overload-prevention":
            if attrs.get("autoplay", False) and (nid, "disable_autoplay") not in attempted_set:
                actions.append({"op": "disable_autoplay", "node_id": nid})
            elif attrs.get("animated", False) and (nid, "remove_animation") not in attempted_set:
                actions.append({"op": "remove_animation", "node_id": nid})
        else:
            if not attrs.get("aria_label") and (nid, "set_aria_label") not in attempted_set:
                label = f"{node.get('tag', 'element').capitalize()}_{nid}"
                actions.append({"op": "set_aria_label", "node_id": nid, "value": label})
            elif attrs.get("contrast_ratio", 99) < 4.5 and (nid, "set_contrast") not in attempted_set:
                actions.append({"op": "set_contrast", "node_id": nid, "value": 4.5})
            elif attrs.get("font_size_px", 99) < 16 and (nid, "set_font_size") not in attempted_set:
                actions.append({"op": "set_font_size", "node_id": nid, "value": 16})
        
        if len(actions) >= max_actions:
            return actions[:max_actions]

    # Second pass: if still need more, add ANY (node, op) not yet attempted
    if len(actions) < max_actions:
        for node in nodes:
            nid = node.get("id", "node0")
            for op in ops_priority:
                key = (nid, op)
                if key not in attempted_set:
                    val = None
                    if op == "set_contrast":
                        val = 4.5
                    elif op == "set_aria_label":
                        val = f"{node.get('tag', 'element').capitalize()}_{nid}"
                    elif op == "set_font_size":
                        val = 16
                    elif op == "simplify_text":
                        val = "Simplified text"
                    actions.append({"op": op, "node_id": nid, "value": val} if val is not None else {"op": op, "node_id": nid})
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
def build_prompt(obs: dict, reward_history: List[float], action_history: List[list], attempted_set: Set[Tuple[str, str]], task_name: str) -> str:
    last_reward = reward_history[-1] if reward_history else None
    last_action = action_history[-1] if action_history else None

    if last_reward is None:
        feedback = "First step. Fix the most critical issues."
    elif last_reward > 0:
        feedback = f"Last reward: +{last_reward:.2f}. Good. Fix OTHER nodes not yet touched."
    elif last_reward < 0:
        feedback = f"Last reward: {last_reward:.2f} (PENALTY). Do NOT repeat: {json.dumps(last_action)}."
    else:
        feedback = "Last reward: 0.0. Those nodes are already fixed. Move to different nodes/ops."

    nodes = []
    # Similar flatten logic for LLM observation
    def flatten_dom(node, acc=None):
        if acc is None: acc = []
        if isinstance(node, dict):
            acc.append(node)
            for child in node.get("children", []):
                flatten_dom(child, acc)
        return acc
    
    dom_nodes = flatten_dom(obs.get("dom", {})) if "dom" in obs else obs.get("nodes", [])
    
    for n in dom_nodes:
        attrs = n.get("attributes", {})
        nodes.append({
            "id": n.get("id"), "tag": n.get("tag"),
            "contrast": attrs.get("contrast_ratio"), 
            "aria_label": attrs.get("aria_label"),
            "font_size": attrs.get("font_size_px"),
            "cognitive_weight": attrs.get("cognitive_weight"),
            "is_redundant": attrs.get("is_redundant"),
            "autoplay": attrs.get("autoplay"),
            "animated": attrs.get("animated")
        })

    attempted_list = [{"node_id": nid, "op": op} for (nid, op) in attempted_set]

    return (
        f"You are navigating task: {task_name}. Maximise reward.\n\n"
        f"NODES: {json.dumps(nodes)}\n"
        f"ALREADY ATTEMPTED (do NOT repeat these): {json.dumps(attempted_list)}\n"
        f"FEEDBACK: {feedback}\n\n"
        "RULES:\n"
        "- Max 4 actions per step\n"
        "- Never repeat a node_id+op pair already in history\n\n"
        'OUTPUT: valid JSON only, no markdown.\n'
        '{"actions":[{"op":"op_name","node_id":"n1"}]}'
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
    task_name: str
) -> Optional[list]:
    prompt = build_prompt(obs, reward_history, action_history, attempted_set, task_name)
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
# Main – Evaluate All 3 tasks
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task_name: str) -> None:
    rewards:        List[float] = []
    action_history: List[list]  = []
    attempted_set:  Set[Tuple[str, str]] = set()
    steps_taken = 0
    score       = 0.0
    success     = False
    done        = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_res = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name}, timeout=10)
        reset_res.raise_for_status()
        obs = safe_parse(reset_res, "/reset") or {}
        obs = obs.get("observation", obs)

        for step in range(1, MAX_STEPS + 1):
            raw_actions = get_llm_action(client, obs, rewards, action_history, attempted_set, task_name)
            if raw_actions is None:
                raw_actions = get_fallback_actions(obs, attempted_set, task_name)
                print(f"[DEBUG] Step {step}: LLM unavailable or error — using smart fallback.", flush=True)

            actions = deduplicate_actions(raw_actions, attempted_set)

            if not actions:
                print(f"[DEBUG] Step {step}: No new actions after dedup. Generating fresh fallback actions.", flush=True)
                actions = get_fallback_actions(obs, attempted_set, task_name)
                if not actions:
                    print("[DEBUG] No remaining untried (node, op) pairs. Ending episode.", flush=True)
                    break

            for a in actions:
                attempted_set.add((a.get("node_id"), a.get("op")))

            action_history.append(actions)
            action_str = json.dumps(actions).replace(" ", "")

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
            done   = bool(result.get("done", result.get("terminated", False) or result.get("truncated", False)))
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

def main() -> None:
    if not check_env_health():
        print("[DEBUG] Environment unreachable. Aborting.", flush=True)
        return

    client = build_openai_client()
    
    tasks_to_test = [
        "neuro-inclusive-audit",
        "cognitive-load-reduction",
        "sensory-overload-prevention"
    ]
    
    for t in tasks_to_test:
        run_task(client, t)

if __name__ == "__main__":
    main()
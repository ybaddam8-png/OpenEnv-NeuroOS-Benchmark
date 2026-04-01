"""Run a deterministic baseline agent over the OpenEnv benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from person_a.environment import NeuroInclusiveEnv


class DeterministicBaselineAgent:
    """Simple heuristic agent that targets obvious lint issues."""

    def act(self, observation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        commands: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for node in self._flatten(observation["dom"]):
            node_id = node["id"]
            attributes = node.get("attributes", {})
            tag = str(node.get("tag", "")).lower()

            def add_command(op: str, value: Any, reason: str) -> None:
                key = (op, node_id)
                if key in seen_pairs:
                    return
                seen_pairs.add(key)
                commands.append({"op": op, "node_id": node_id, "value": value, "reason": reason})

            contrast_ratio = self._coerce_float(attributes.get("contrast_ratio"), default=7.0)
            if contrast_ratio < 4.5:
                add_command("set_contrast", 7.0, "Improve text and control contrast.")

            if self._is_animated(attributes):
                add_command("stop_animation", None, "Stop distracting animation.")

            if self._is_interactive(tag, attributes):
                aria_label = attributes.get("aria_label")
                if not isinstance(aria_label, str) or not aria_label.strip():
                    add_command("set_aria_label", self._label_for_node(node), "Add an accessible label.")

                role = attributes.get("role")
                if not isinstance(role, str) or not role.strip():
                    add_command("set_role", self._default_role(tag), "Add a missing semantic role.")

            sensory_load = max(
                self._coerce_float(attributes.get("sensory_load"), default=0.0),
                self._coerce_float(attributes.get("motion_level"), default=0.0),
            )
            if sensory_load > 0.6:
                add_command("reduce_sensory_load", {"target_load": 0.2}, "Reduce sensory overload.")

            cognitive_weight = self._coerce_float(attributes.get("cognitive_weight"), default=0.0)
            if cognitive_weight > 0.65:
                add_command("set_cognitive_weight", 0.35, "Reduce cognitive burden.")

            if bool(attributes.get("redundant")):
                add_command("remove_redundancy", None, "Remove repeated or noisy text.")

            font_size = self._coerce_float(attributes.get("font_size"), default=16.0)
            if font_size < 14.0:
                add_command("set_font_size", 16, "Increase small text for readability.")

            if len(commands) >= 8:
                break

        return {"commands": commands}

    def _flatten(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = [node]
        for child in node.get("children", []):
            nodes.extend(self._flatten(child))
        return nodes

    def _is_interactive(self, tag: str, attributes: dict[str, Any]) -> bool:
        role = str(attributes.get("role", "")).strip().lower()
        return tag in {"button", "a", "input", "select", "textarea"} or bool(attributes.get("interactive")) or role in {
            "button",
            "link",
            "textbox",
        }

    def _is_animated(self, attributes: dict[str, Any]) -> bool:
        animation_name = str(attributes.get("animation", "")).strip().lower()
        return bool(attributes.get("animated")) or bool(attributes.get("flashing")) or (animation_name and animation_name not in {"none", "static"})

    def _label_for_node(self, node: dict[str, Any]) -> str:
        text = str(node.get("text", "")).strip()
        if text:
            return text
        placeholder = str(node.get("attributes", {}).get("placeholder", "")).strip()
        if placeholder:
            return placeholder
        tag = str(node.get("tag", "control")).strip().lower()
        return f"{tag} control"

    def _default_role(self, tag: str) -> str:
        mapping = {
            "button": "button",
            "a": "link",
            "input": "textbox",
            "select": "listbox",
            "textarea": "textbox",
        }
        return mapping.get(tag, "group")

    def _coerce_float(self, value: Any, default: float) -> float:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return default
        return default


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_path = project_root / "results.jsonl"

    env = NeuroInclusiveEnv(seed=7, max_steps=3)
    agent = DeterministicBaselineAgent()
    episode_results: list[dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as handle:
        for episode in range(10):
            observation, reset_info = env.reset()
            terminated = False
            truncated = False
            cumulative_reward = 0.0
            command_count = 0
            errors: list[str] = []
            final_info = reset_info

            while not (terminated or truncated):
                action = agent.act(observation)
                command_count += len(action["commands"])
                observation, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                final_info = info
                errors.extend(
                    entry["message"]
                    for entry in info["mutation_log"]
                    if not entry.get("success")
                )
                if not action["commands"]:
                    break

            result = {
                "episode": episode + 1,
                "task_id": observation["task_id"],
                "difficulty": observation["difficulty"],
                "reward": round(cumulative_reward, 4),
                "done": terminated or truncated,
                "terminated": terminated,
                "truncated": truncated,
                "num_commands": command_count,
                "errors": errors,
                "grade_score": final_info.get("grade", {}).get("score") if isinstance(final_info.get("grade"), dict) else None,
            }
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            episode_results.append(result)

    env.close()

    avg_reward = round(sum(item["reward"] for item in episode_results) / max(len(episode_results), 1), 4)
    solved = sum(1 for item in episode_results if item["terminated"])
    truncated_count = sum(1 for item in episode_results if item["truncated"])
    print(
        f"episodes={len(episode_results)} avg_reward={avg_reward} "
        f"terminated={solved} truncated={truncated_count} results={results_path.name}"
    )


if __name__ == "__main__":
    main()

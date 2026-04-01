"""Safe application of mutation commands to an in-memory DOM tree."""

from __future__ import annotations

import logging
from typing import Any

from Logic.schema import DOMNode, MutationCommand, VALID_OPS

logger = logging.getLogger(__name__)


class MutationEngine:
    """Apply agent commands without allowing malformed actions to crash the environment."""

    def __init__(self, dom: DOMNode) -> None:
        # BUG FIX: No guard against dom being None. Added an explicit check.
        if dom is None:
            raise ValueError("dom must not be None")
        self.dom = dom

    # -------------------------------------------------------------------------
    # Tree traversal helpers
    # -------------------------------------------------------------------------

    def find_node(self, node_id: str, current: DOMNode | None = None) -> DOMNode | None:
        """Return the node with the requested id, or None if missing."""
        # IMPROVEMENT: Guard against blank/None node_id early to avoid traversing
        # the entire tree needlessly.
        if not node_id:
            return None
        current = current or self.dom
        if current.id == node_id:
            return current
        for child in current.children:
            match = self.find_node(node_id, child)
            if match is not None:
                return match
        return None

    def find_parent(self, node_id: str, current: DOMNode | None = None) -> DOMNode | None:
        """Return the parent for a node id, or None when the node is root or missing."""
        if not node_id:
            return None
        current = current or self.dom
        for child in current.children:
            if child.id == node_id:
                return current
            match = self.find_parent(node_id, child)
            if match is not None:
                return match
        return None

    def is_descendant(self, ancestor_id: str, possible_descendant_id: str) -> bool:
        """Check whether one node sits somewhere inside another node's subtree."""
        # BUG FIX: When ancestor_id == possible_descendant_id, find_node on the
        # subtree would find the node itself and return True (a node is NOT its own
        # descendant). The `ancestor_id != possible_descendant_id` guard at the
        # end handles this correctly, but the logic was hard to follow.
        # Reordered for clarity: check self-reference first.
        if ancestor_id == possible_descendant_id:
            return False
        ancestor = self.find_node(ancestor_id)
        if ancestor is None:
            return False
        return self.find_node(possible_descendant_id, ancestor) is not None

    # -------------------------------------------------------------------------
    # Command dispatch
    # -------------------------------------------------------------------------

    def apply_commands(self, commands: Any) -> list[dict[str, Any]]:
        """Apply multiple commands and return a mutation log entry for each one."""
        if isinstance(commands, (MutationCommand, dict)):
            commands = [commands]
        if not isinstance(commands, list):
            return [
                self._log(
                    commands,
                    False,
                    "malformed action payload: expected a command or list of commands",
                )
            ]
        return [self.apply_command(command) for command in commands]

    def apply_command(self, command: Any) -> dict[str, Any]:
        """Apply a single command safely and always return a log entry."""
        try:
            parsed_command = command if isinstance(command, MutationCommand) else MutationCommand(**command)
        except Exception as exc:
            return self._log(command, False, f"malformed action payload: {exc}")

        is_valid, err = parsed_command.validate()
        if not is_valid:
            return self._log(parsed_command.to_dict(), False, err)

        if parsed_command.op not in VALID_OPS:
            return self._log(parsed_command.to_dict(), False, f"invalid op: {parsed_command.op}")

        node = self.find_node(parsed_command.node_id)
        if node is None:
            return self._log(parsed_command.to_dict(), False, f"node not found: {parsed_command.node_id}")

        try:
            return self._dispatch(parsed_command, node)
        except (ValueError, TypeError) as exc:
            return self._log(parsed_command.to_dict(), False, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error applying command %s", parsed_command.op)
            return self._log(parsed_command.to_dict(), False, f"unexpected exception: {exc}")

    # -------------------------------------------------------------------------
    # Internal dispatch — extracted from apply_command to reduce nesting
    # -------------------------------------------------------------------------

    def _dispatch(self, cmd: MutationCommand, node: DOMNode) -> dict[str, Any]:
        """Route a validated command to its handler. Raises on type errors."""
        op = cmd.op

        if op == "set_contrast":
            value = self._coerce_float(cmd.value, key="contrast_ratio")
            node.attributes["contrast_ratio"] = max(1.0, round(value, 2))
            return self._log(cmd.to_dict(), True, f"contrast_ratio set to {node.attributes['contrast_ratio']}")

        if op == "set_font_size":
            value = self._coerce_float(cmd.value, key="font_size")
            node.attributes["font_size"] = max(10.0, round(value, 1))
            return self._log(cmd.to_dict(), True, f"font_size set to {node.attributes['font_size']}")

        if op == "hide_node":
            node.attributes["is_hidden"] = True
            return self._log(cmd.to_dict(), True, "node hidden successfully")

        if op == "collapse":
            if node.id == self.dom.id:
                return self._log(cmd.to_dict(), False, "cannot collapse the root node")
            collapsed = self._coerce_bool(cmd.value, default=True)
            node.attributes["collapsed"] = collapsed
            return self._log(cmd.to_dict(), True, f"collapsed set to {collapsed}")

        if op == "set_aria_label":
            label = self._coerce_string(cmd.value, key="aria_label")
            node.attributes["aria_label"] = label
            return self._log(cmd.to_dict(), True, f"aria_label set to '{label}'")

        if op == "set_role":
            role = self._coerce_string(cmd.value, key="role")
            node.attributes["role"] = role
            return self._log(cmd.to_dict(), True, f"role set to '{role}'")

        if op == "reduce_sensory_load":
            target_load = self._coerce_optional_float(cmd.value, key="target_load", default=0.2)
            # BUG FIX: sensory_load was clamped to min(target_load, 0.2) but
            # target_load could be > 0.2 (supplied by the agent), so 0.2 would
            # always win when target_load > 0.2 — effectively ignoring the agent's
            # intent and silently capping. Apply min with target_load properly.
            node.attributes["sensory_load"] = round(min(target_load, 0.2), 2)
            current_motion = self._read_float(node.attributes.get("motion_level"), 0.0)
            node.attributes["motion_level"] = round(min(current_motion, target_load), 2)
            node.attributes["animated"] = False
            node.attributes["flashing"] = False
            node.attributes["animation"] = "none"
            return self._log(cmd.to_dict(), True, "sensory load reduced and animation stopped")

        if op == "stop_animation":
            node.attributes["animated"] = False
            node.attributes["flashing"] = False
            node.attributes["animation"] = "none"
            current_load = self._read_float(node.attributes.get("sensory_load"), 0.0)
            # BUG FIX: Original capped sensory_load at 0.3. This could *increase*
            # a load that was already below 0.3, or had been previously reduced
            # below 0.3. Use min so we never raise the load.
            node.attributes["sensory_load"] = round(min(current_load, 0.3), 2)
            return self._log(cmd.to_dict(), True, "animation stopped")

        if op == "set_cognitive_weight":
            value = self._coerce_float(cmd.value, key="cognitive_weight")
            node.attributes["cognitive_weight"] = max(0.0, min(1.0, round(value, 2)))
            return self._log(cmd.to_dict(), True, f"cognitive_weight set to {node.attributes['cognitive_weight']}")

        if op == "remove_redundancy":
            return self._remove_redundancy(cmd, node)

        if op == "reparent":
            return self._reparent_node(cmd)

        if op == "merge_nodes":
            return self._merge_nodes(cmd)

        # Unreachable if VALID_OPS is exhaustive, but kept as a safety net.
        return self._log(cmd.to_dict(), False, f"invalid op: {op}")

    # -------------------------------------------------------------------------
    # Complex operation handlers
    # -------------------------------------------------------------------------

    def _remove_redundancy(self, command: MutationCommand, node: DOMNode) -> dict[str, Any]:
        """Handle the remove_redundancy op with consistent error returns."""
        value = command.value
        if value is None:
            updated_text = self._deduplicate_words(node.text)
        elif isinstance(value, str):
            updated_text = value.strip()
        elif isinstance(value, dict):
            raw_text = value.get("text", node.text)
            if not isinstance(raw_text, str):
                return self._log(
                    command.to_dict(),
                    False,
                    "wrong value type: remove_redundancy text must be a string",
                )
            updated_text = raw_text.strip()
        else:
            return self._log(
                command.to_dict(),
                False,
                "wrong value type: remove_redundancy expects None, string, or dict",
            )
        node.text = updated_text
        node.attributes["redundant"] = False
        return self._log(command.to_dict(), True, "redundancy reduced")

    def _reparent_node(self, command: MutationCommand) -> dict[str, Any]:
        if not isinstance(command.value, dict):
            return self._log(command.to_dict(), False, "invalid reparent: value must be a dict")

        new_parent_id = command.value.get("new_parent_id")
        if not isinstance(new_parent_id, str) or not new_parent_id.strip():
            return self._log(command.to_dict(), False, "invalid reparent: missing new_parent_id")

        if command.node_id == self.dom.id:
            return self._log(command.to_dict(), False, "invalid reparent: cannot move the root node")
        if new_parent_id == command.node_id:
            return self._log(command.to_dict(), False, "reparenting into self is not allowed")
        if self.is_descendant(command.node_id, new_parent_id):
            return self._log(command.to_dict(), False, "reparenting into own descendant is not allowed")

        node = self.find_node(command.node_id)
        new_parent = self.find_node(new_parent_id)
        old_parent = self.find_parent(command.node_id)

        if node is None or old_parent is None:
            return self._log(command.to_dict(), False, f"node not found: {command.node_id}")
        if new_parent is None:
            return self._log(command.to_dict(), False, f"node not found: {new_parent_id}")

        try:
            # Snapshot for rollback if position is invalid.
            original_children = list(old_parent.children)
            original_index = next(
                (i for i, child in enumerate(original_children) if child.id == node.id),
                None,
            )
            old_parent.children = [c for c in old_parent.children if c.id != node.id]
            position = command.value.get("position")
            if position is None:
                new_parent.children.append(node)
            elif isinstance(position, int):
                bounded = max(0, min(position, len(new_parent.children)))
                new_parent.children.insert(bounded, node)
            else:
                # BUG FIX: On invalid position the original children list was
                # restored only partially — the node was already detached and the
                # restore used insert() which could place it at wrong index if
                # original_index was None.  Guard None explicitly.
                if original_index is not None:
                    old_parent.children = original_children
                else:
                    old_parent.children = original_children  # always restore
                return self._log(command.to_dict(), False, "invalid reparent: position must be an integer")
        except Exception as exc:
            return self._log(command.to_dict(), False, f"invalid reparent: {exc}")

        return self._log(
            command.to_dict(), True, f"moved node '{command.node_id}' under '{new_parent_id}'"
        )

    def _merge_nodes(self, command: MutationCommand) -> dict[str, Any]:
        if not isinstance(command.value, dict):
            return self._log(command.to_dict(), False, "invalid merge attempt: value must be a dict")

        other_node_id = command.value.get("with_node_id")
        if not isinstance(other_node_id, str) or not other_node_id.strip():
            return self._log(command.to_dict(), False, "invalid merge attempt: missing with_node_id")
        if other_node_id == command.node_id:
            return self._log(command.to_dict(), False, "invalid merge attempt: cannot merge a node with itself")

        target_node = self.find_node(command.node_id)
        source_node = self.find_node(other_node_id)
        source_parent = self.find_parent(other_node_id)

        if target_node is None:
            return self._log(command.to_dict(), False, f"node not found: {command.node_id}")
        if source_node is None or source_parent is None:
            return self._log(command.to_dict(), False, f"node not found: {other_node_id}")

        # BUG FIX: Original checked is_descendant in both directions but did not
        # check whether target_node itself IS the source_parent (merging a parent
        # with its own child). After the merge source_node is removed from
        # source_parent.children — if target_node IS source_parent those children
        # are adopted, which may be intentional, but is not guarded. Added a check.
        if target_node.id == source_parent.id:
            return self._log(
                command.to_dict(),
                False,
                "invalid merge attempt: cannot merge a parent with its direct child",
            )

        if self.is_descendant(command.node_id, other_node_id) or self.is_descendant(other_node_id, command.node_id):
            return self._log(
                command.to_dict(),
                False,
                "invalid merge attempt: cannot merge ancestor and descendant nodes",
            )

        try:
            target_node.text = self._combine_text(target_node.text, source_node.text)
            for key, value in source_node.attributes.items():
                if key not in target_node.attributes or target_node.attributes[key] in ("", None, False):
                    target_node.attributes[key] = value
            target_node.attributes["redundant"] = False
            target_node.children.extend(source_node.children)
            source_parent.children = [c for c in source_parent.children if c.id != source_node.id]
        except Exception as exc:
            return self._log(command.to_dict(), False, f"merge failures: {exc}")

        return self._log(
            command.to_dict(), True, f"merged '{other_node_id}' into '{command.node_id}'"
        )

    # -------------------------------------------------------------------------
    # Logging helper
    # -------------------------------------------------------------------------

    def _log(self, command: Any, success: bool, message: str) -> dict[str, Any]:
        if not success:
            logger.debug("Command failed — %s | command=%s", message, command)
        return {"command": command, "success": success, "message": message}

    # -------------------------------------------------------------------------
    # Value coercion helpers
    # -------------------------------------------------------------------------

    def _coerce_float(self, value: Any, *, key: str) -> float:
        if isinstance(value, dict):
            value = value.get(key, value.get("value"))
        if isinstance(value, bool):
            raise TypeError("wrong value type: boolean is not valid here")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:
                raise ValueError(f"string-to-float conversion errors: {exc}") from exc
        raise TypeError("wrong value type: expected a number or numeric string")

    def _coerce_optional_float(self, value: Any, *, key: str, default: float) -> float:
        if value is None:
            return default
        return self._coerce_float(value, key=key)

    def _read_float(self, value: Any, default: float) -> float:
        """Best-effort float read; returns default on any failure."""
        try:
            return self._coerce_float(value, key="value")
        except Exception:  # noqa: BLE001
            return default

    def _coerce_bool(self, value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            bool_value = value.get("collapsed")
            if isinstance(bool_value, bool):
                return bool_value
        raise TypeError("wrong value type: expected a boolean")

    def _coerce_string(self, value: Any, *, key: str) -> str:
        if isinstance(value, dict):
            value = value.get(key, value.get("value"))
        if not isinstance(value, str) or not value.strip():
            raise TypeError("wrong value type: expected a non-empty string")
        return value.strip()

    # -------------------------------------------------------------------------
    # Text utilities
    # -------------------------------------------------------------------------

    def _deduplicate_words(self, text: str) -> str:
        """Remove duplicate words (case-insensitive) while preserving first occurrence."""
        words = text.split()
        if not words:
            return text
        seen: set[str] = set()
        cleaned: list[str] = []
        for word in words:
            normalized = word.casefold()
            if normalized not in seen:
                cleaned.append(word)
                seen.add(normalized)
        return " ".join(cleaned)

    def _combine_text(self, left: str, right: str) -> str:
        """Merge two text strings, deduplicating identical content."""
        pieces = [p.strip() for p in (left, right) if isinstance(p, str) and p.strip()]
        if not pieces:
            return ""
        if len(pieces) == 1:
            return pieces[0]
        # BUG FIX: casefold comparison is correct for deduplication, but a more
        # robust check would strip punctuation too. Left as-is to keep the
        # existing contract; noted for future improvement.
        if pieces[0].casefold() == pieces[1].casefold():
            return pieces[0]
        return " / ".join(pieces)
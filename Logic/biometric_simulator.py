"""
biometric_simulator.py — Synthetic User Biometric Event Generator
=================================================================
Produces seeded, reproducible streams of biometric events that simulate
a real user experiencing cognitive stress when navigating a cluttered UI.

Easy   → calm, regular cursor movement, low cognitive load
Medium → moderate stress, occasional pauses, some rage clicks
Hard   → erratic cursor, high cognitive load spikes, frequent rage clicks,
         unpredictable fixation patterns

Each event matches BAIOMETRIC_EVENT_SCHEMA defined in schema.py.

Usage:
    sim = BiometricSimulator(seed=42)
    events = sim.generate("easy",   duration_ms=5000)
    events = sim.generate("medium", duration_ms=10000)
    events = sim.generate("hard",   duration_ms=15000)
"""

from __future__ import annotations
import random
import math
from Logic.schema import BIOMETRIC_EVENT_SCHEMA  # imported for reference only


# ---------------------------------------------------------------------------
# Difficulty profiles for biometrics
# ---------------------------------------------------------------------------

BIO_PROFILES = {
    "easy": {
        "event_interval_ms":        (200, 400),    # (min, max) ms between events
        "base_velocity":            (0.5, 2.0),    # pixels/ms — calm cursor
        "velocity_spike_chance":    0.03,          # rarely spikes
        "velocity_spike_magnitude": (3.0, 6.0),
        "fixation_duration_ms":     (300, 800),    # long fixations = reading calmly
        "rage_click_chance":        0.01,
        "cognitive_load_base":      (10.0, 30.0),  # low baseline
        "cognitive_load_spike":     (35.0, 55.0),
        "cognitive_spike_chance":   0.05,
        "scroll_velocity":          (0.0, 1.5),
        "duration_ms":              6000,
    },
    "medium": {
        "event_interval_ms":        (100, 300),
        "base_velocity":            (1.5, 4.0),
        "velocity_spike_chance":    0.12,
        "velocity_spike_magnitude": (6.0, 12.0),
        "fixation_duration_ms":     (150, 500),
        "rage_click_chance":        0.07,
        "cognitive_load_base":      (30.0, 55.0),
        "cognitive_load_spike":     (60.0, 80.0),
        "cognitive_spike_chance":   0.15,
        "scroll_velocity":          (0.0, 3.0),
        "duration_ms":              10000,
    },
    "hard": {
        "event_interval_ms":        (50, 180),     # rapid-fire events
        "base_velocity":            (3.0, 8.0),    # erratic baseline
        "velocity_spike_chance":    0.30,
        "velocity_spike_magnitude": (12.0, 25.0),  # extreme spikes
        "fixation_duration_ms":     (50, 250),     # short, fractured attention
        "rage_click_chance":        0.20,
        "cognitive_load_base":      (55.0, 80.0),
        "cognitive_load_spike":     (82.0, 100.0),
        "cognitive_spike_chance":   0.35,
        "scroll_velocity":          (0.5, 8.0),    # frantic scrolling
        "duration_ms":              15000,
    },
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class BiometricSimulator:
    """
    Generates a timestamped stream of biometric events.

    Parameters
    ----------
    seed        : int  — for reproducibility
    node_ids    : list[str]  — optional list of DOM node IDs to reference
                  in node_hover_id (pass from your generated DOM)
    """

    def __init__(self, seed: int = 0, node_ids: list[str] = None):
        self.seed = seed
        self.node_ids = node_ids or []
        self._rng = random.Random(seed)

    def generate(
        self,
        difficulty: str,
        duration_ms: int = None,
        node_ids: list[str] = None,
    ) -> list[dict]:
        """
        Generate a full biometric event stream.

        Parameters
        ----------
        difficulty  : "easy" | "medium" | "hard"
        duration_ms : total simulation window in ms (uses profile default if None)
        node_ids    : DOM node IDs to reference (overrides constructor list)

        Returns
        -------
        list of dicts, each matching BIOMETRIC_EVENT_SCHEMA
        """
        if difficulty not in BIO_PROFILES:
            raise ValueError(f"difficulty must be one of {list(BIO_PROFILES.keys())}")

        self._rng = random.Random(self.seed)
        profile = BIO_PROFILES[difficulty]
        ids = node_ids or self.node_ids

        total_ms = duration_ms or profile["duration_ms"]
        events = []
        t = 0
        prev_x, prev_y = 400.0, 300.0  # start near center of a 800x600 viewport

        # State variables for temporal correlation
        current_cog_load = self._rng.uniform(*profile["cognitive_load_base"])
        in_rage_sequence = False
        rage_count = 0

        while t < total_ms:
            # Advance time
            interval = self._rng.randint(*profile["event_interval_ms"])
            t += interval

            # ---- Cursor velocity ----
            if self._rng.random() < profile["velocity_spike_chance"]:
                velocity = self._rng.uniform(*profile["velocity_spike_magnitude"])
            else:
                velocity = self._rng.uniform(*profile["base_velocity"])

            # ---- Cursor position delta (correlated with velocity) ----
            angle = self._rng.uniform(0, 2 * math.pi)
            delta_x = round(velocity * math.cos(angle) * (interval / 100), 2)
            delta_y = round(velocity * math.sin(angle) * (interval / 100), 2)
            prev_x = max(0, min(1280, prev_x + delta_x))
            prev_y = max(0, min(900,  prev_y + delta_y))

            # ---- Fixation duration ----
            fixation = self._rng.randint(*profile["fixation_duration_ms"])
            # Short fixation if cursor is moving fast
            if velocity > profile["velocity_spike_magnitude"][0] * 0.8:
                fixation = min(fixation, 100)

            # ---- Rage clicks ----
            if in_rage_sequence and rage_count > 0:
                rage_click = True
                rage_count -= 1
                if rage_count == 0:
                    in_rage_sequence = False
            elif self._rng.random() < profile["rage_click_chance"]:
                rage_click = True
                # Start a rage sequence of 2–5 rapid clicks
                rage_count = self._rng.randint(1, 4)
                in_rage_sequence = True
            else:
                rage_click = False

            # ---- Cognitive load (mean-reverting with spikes) ----
            revert_speed = 0.08
            target_load = sum(profile["cognitive_load_base"]) / 2
            current_cog_load += (target_load - current_cog_load) * revert_speed
            if self._rng.random() < profile["cognitive_spike_chance"]:
                current_cog_load = self._rng.uniform(*profile["cognitive_load_spike"])
            current_cog_load = max(0.0, min(100.0, current_cog_load))

            # ---- Scroll velocity ----
            scroll_v = self._rng.uniform(*profile["scroll_velocity"])

            # ---- Node hover (which DOM node the cursor is over) ----
            hover_id = self._rng.choice(ids) if ids and self._rng.random() < 0.6 else ""

            events.append({
                "timestamp_ms":      t,
                "cursor_velocity":   round(velocity, 3),
                "cursor_delta_x":    delta_x,
                "cursor_delta_y":    delta_y,
                "fixation_duration": fixation,
                "rage_click":        rage_click,
                "cognitive_load":    round(current_cog_load, 2),
                "scroll_velocity":   round(scroll_v, 3),
                "node_hover_id":     hover_id,
            })

        return events

    # ------------------------------------------------------------------
    # Analysis helpers  (used by the grader)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stress_summary(events: list[dict]) -> dict:
        """
        Compute summary statistics from a biometric event stream.
        The hard-task grader uses this to check if the agent responded
        to the right stress signals.

        Returns dict with:
            mean_cognitive_load
            max_cognitive_load
            rage_click_count
            rage_click_rate        (rage clicks / total events)
            mean_cursor_velocity
            max_cursor_velocity
            high_stress_windows    list of (start_ms, end_ms) with cog_load > 70
            top_stressed_nodes     list of node_ids most hovered during high stress
        """
        if not events:
            return {}

        cog_loads = [e["cognitive_load"] for e in events]
        velocities = [e["cursor_velocity"] for e in events]
        rage_clicks = [e for e in events if e["rage_click"]]

        # Find high-stress windows (cognitive_load > 70)
        high_stress = [e for e in events if e["cognitive_load"] > 70.0]
        windows = []
        if high_stress:
            win_start = high_stress[0]["timestamp_ms"]
            win_end = high_stress[0]["timestamp_ms"]
            for ev in high_stress[1:]:
                if ev["timestamp_ms"] - win_end < 1000:  # within 1s
                    win_end = ev["timestamp_ms"]
                else:
                    windows.append((win_start, win_end))
                    win_start = ev["timestamp_ms"]
                    win_end = ev["timestamp_ms"]
            windows.append((win_start, win_end))

        # Nodes hovered during high-stress windows
        stressed_node_counts: dict[str, int] = {}
        for ev in high_stress:
            nid = ev.get("node_hover_id", "")
            if nid:
                stressed_node_counts[nid] = stressed_node_counts.get(nid, 0) + 1
        top_stressed = sorted(stressed_node_counts, key=lambda k: -stressed_node_counts[k])[:5]

        return {
            "mean_cognitive_load":  round(sum(cog_loads) / len(cog_loads), 2),
            "max_cognitive_load":   round(max(cog_loads), 2),
            "rage_click_count":     len(rage_clicks),
            "rage_click_rate":      round(len(rage_clicks) / len(events), 4),
            "mean_cursor_velocity": round(sum(velocities) / len(velocities), 3),
            "max_cursor_velocity":  round(max(velocities), 3),
            "high_stress_windows":  windows,
            "top_stressed_nodes":   top_stressed,
        }
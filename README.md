---
title: NEXUS NeuroOS Benchmark
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---


# NEXUS NeuroOS: OpenEnv Benchmark

## 🌍 Environment Description & Motivation
Modern web interfaces are often designed with high sensory density, which can be overwhelming or entirely inaccessible for neurodivergent individuals (e.g., those with ADHD, Autism, or Dyslexia). 

**NEXUS NeuroOS** is a reinforcement learning environment built on the OpenEnv standard that tasks an AI agent with dynamically optimizing a user interface. The agent acts as a "Neuro-Inclusive UI Auditor." Its motivation is to minimize **Sensory Load** and **Cognitive Weight** while maintaining functional usability. This serves as a critical, automated safety gate before enterprise UI deployments, ensuring digital spaces are accessible to all cognitive profiles.

## 🔍 Observation and Action Spaces

### Observation Space
The environment returns a tightly typed JSON object representing the current state of the UI and the user:
* `dom` (Dict): The current Virtual DOM tree, including node IDs, types, contrast levels, and active CSS states.
* `biometrics` (Dict): Simulated real-time stress indicators of the user (e.g., cognitive load spikes).
* `instructions` (String): The overarching goal for the current task (e.g., "Reduce sensory load on the checkout form").

### Action Space
The agent must output a JSON array of mutation commands to alter the DOM.
* `op` (String): The operation to perform (e.g., `set_contrast`, `remove_node`, `simplify_text`).
* `node_id` (String): The target element in the VDOM.
* `value` (Float/String): The new parameter to apply.
* *Example:* `[{"op": "set_contrast", "node_id": "submit_btn", "value": 7.0}]`

## 📋 Task Descriptions & Difficulty

The environment features three deterministic tasks, evaluated by a mathematical grader that penalizes destructive actions (like deleting the whole page to achieve zero sensory load) and rewards precise accessibility improvements. The grader returns a continuous reward signal between `0.0` and `1.0`.

1. **Easy: Contrast Compliance**
   * *Objective:* Adjust the contrast ratio of critical navigation buttons to meet WCAG AAA standards without altering the layout.
   * *Grader Logic:* Calculates contrast ratios against background nodes; rewards values >= 7.0.
2. **Medium: Form De-cluttering**
   * *Objective:* Simplify a highly dense registration form. The agent must successfully identify and remove redundant decorative elements while preserving all required input fields.
   * *Grader Logic:* Penalizes deletion of `<input>` nodes; rewards deletion of non-functional visual noise.
3. **Hard: Dynamic Persona Adaptation (ADHD Profile)**
   * *Objective:* Execute a complete UI overhaul of a dashboard. The agent must increase structural padding, mute background animations, and group related items to minimize cognitive switching penalties, all while maintaining a functional state.
   * *Grader Logic:* Complex reward function balancing structural integrity with a massive reduction in the `cognitive_weight` metric.

## 🚀 Setup and Usage Instructions

### Docker Execution (Recommended for HF Spaces)
This environment is fully containerized and serves a FastAPI backend to comply with OpenEnv automated ping standards.

```bash
# Build the Docker image locally
docker build -t neuro-os-env .

# Run the container (Exposes port 7860)
docker run -p 7860:7860 neuro-os-env

📊 Baseline Scores
Running our baseline gpt-3.5-turbo agent against the deterministic grader yielded the following scores (bounded between 0.0 and 1.0):

Task 1 (Easy): 0.85

Task 2 (Medium): 0.62

Task 3 (Hard): 0.30

Average Overall Score: 0.59


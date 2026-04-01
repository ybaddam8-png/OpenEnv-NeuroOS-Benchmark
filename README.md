# OpenEnv Neuro-Inclusive UI Mutation Benchmark

This project simulates cluttered and inaccessible UI DOM trees, allows an agent to repair them with structured mutation commands, and grades whether the result improves accessibility, sensory load, and cognitive load.

## Repository Structure

```text
open_env/
├── Logic/        # Domain dataclasses, task generation, linting, biometrics, grading
├── person_a/     # Safe mutation engine and Gymnasium-style environment wrapper
├── Dockerfile    # Headless container runtime
├── openenv.yaml  # Environment metadata/config
├── README.md
└── run_eval.py   # Baseline evaluation entry point
```

## Local Run

From the `open_env` directory:

```bash
python run_eval.py
```

This writes `results.jsonl` in the project root and prints a short summary.

## Docker Run

Build and run from the `open_env` directory:

```bash
docker build -t open-env .
docker run --rm open-env
```

The container runs `python run_eval.py` by default.

## Notes

- The benchmark is deterministic for a fixed seed.
- The baseline agent is intentionally simple and heuristic.
- The environment uses Gymnasium-style method signatures without requiring external services.

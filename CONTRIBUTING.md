# Contributing to RLMAPF2

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run tests before opening a pull request:

```bash
pytest -q tests
python3 -m compileall -q train.py eval.py rlmapf2.py
```

## Development Guidelines

- Keep changes scoped and include a short rationale in the PR description.
- Prefer config-driven changes (`configs/train/*.yaml`, `configs/eval/*.yaml`) over hardcoded values.
- Avoid committing generated artifacts (videos, checkpoints, experiment outputs).
- Keep backward compatibility for documented CLI flags unless there is a strong reason to break it.

## Pull Request Checklist

- [ ] Code compiles and tests pass locally.
- [ ] README or docs were updated when behavior changed.
- [ ] New config keys are documented.
- [ ] No large generated files were added.

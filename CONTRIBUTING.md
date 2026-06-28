# Contributing to bMRI

Thanks for your interest. bMRI is a small focused library, contributions
are welcome.

## Setup

```bash
git clone https://github.com/ludgerradke/bMRI.git
cd bMRI
uv sync --extra dev
uv run pre-commit install
```

## Tests

```bash
uv run pytest --no-cov -q
```

If you change the fitting code, add a phantom test in
`test/test_fitting_accuracy.py` that proves your change still recovers
known ground-truth values.

## Style

We use `ruff` for formatting and linting. The pre-commit hook handles
this automatically. Type hints are encouraged, `mypy` is wired up but
not yet strict.

## Pull requests

- Keep PRs small and focused
- Add a phantom test for any new fitting model
- Run `uv run pytest` locally before pushing
- Describe what the PR does and why in the body

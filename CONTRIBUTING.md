# Contributing to TowerMind

Thank you for your interest in contributing to **TowerMind**! We welcome contributions to our Gymnasium tower defense environment, Reinforcement Learning models, LLM meta-controller architecture, benchmarks, and analytics dashboard.

---

## Code of Conduct

Please be respectful, collaborative, and constructive when submitting issues, PRs, and participating in discussions.

---

## Getting Started

1. **Fork the Repository** on GitHub.
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/codehub001/TowerMind.git
   cd TowerMind
   ```
3. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[all]"
   ```
4. **Create a Feature Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```

---

## Development & Code Quality Guidelines

We maintain high code quality standards:
- **Linting & Formatting**: We use [Ruff](https://github.com/astral-sh/ruff).
  ```bash
  ruff check .
  ruff format --check .
  ```
- **Automated Tests**: All tests must pass before opening a PR.
  ```bash
  pytest -v
  ```
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat(...)`: New features or modules
  - `fix(...)`: Bug fixes
  - `docs(...)`: Documentation updates
  - `test(...)`: Unit and integration tests
  - `refactor(...)`: Code refactoring without behavioral change
  - `chore(...)`: Tooling, configuration, or dependency updates

---

## Submitting a Pull Request

1. Ensure all 57+ unit tests pass.
2. Push your branch to GitHub:
   ```bash
   git push origin feat/your-feature-name
   ```
3. Open a Pull Request against the `main` branch with a clear summary of your changes.
